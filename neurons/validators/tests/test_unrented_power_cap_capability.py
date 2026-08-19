"""DAH-2715 — unrented incentive GPU power cap gate.

An unrented executor whose container provably cannot apply a GPU power cap
(no CAP_SYS_ADMIN, or /dev/nvidiactl not owned by root — see DAH-2705) forfeits
the unrented rental incentive while staying active. Enforcement is gated by
ENABLE_UNRENTED_POWER_CAP_LIMIT; while the flag is off the breach is only logged
(shadow mode) and the payout is unchanged.
"""

import logging
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default

CAPS_WITH_SYS_ADMIN = "000001ffffffffff"   # sysbox / privileged container: every capability
CAPS_WITHOUT_SYS_ADMIN = "00000000a80425fb"  # prod 128.140.36.181: default docker caps
NOBODY_UID = 65534  # what sysbox maps /dev/nvidiactl to (prod 88.22.127.152)


def _build_incentive() -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})


def _make_job(
    *,
    cap_eff: str | None = CAPS_WITHOUT_SYS_ADMIN,
    owner_uid: int | None = 0,
    is_rented: bool = False,
    spec: dict | None = None,
) -> JobResult:
    if spec is None:
        spec = {}
        if cap_eff is not None:
            spec["container_cap_eff"] = cap_eff
        if owner_uid is not None:
            spec["nvidiactl_owner_uid"] = owner_uid

    return JobResult(
        spec=spec,
        executor_info=ExecutorSSHInfo(
            uuid="exec-1",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
            price_per_gpu=1.0,
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model=H200,
        gpu_count=8,
        is_rented=is_rented,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


def test_missing_cap_sys_admin_is_flagged():
    # Arrange — prod case: default docker capabilities, device owned by root
    incentive = _build_incentive()

    # Act
    incapable = incentive._power_cap_incapable(_make_job())

    # Assert
    assert incapable is not None
    assert incapable.container_cap_eff == CAPS_WITHOUT_SYS_ADMIN
    assert incapable.nvidiactl_owner_uid == 0


def test_device_not_owned_by_root_is_flagged():
    # Arrange — prod sysbox case: every capability held, but the device is mapped away
    incentive = _build_incentive()

    # Act
    incapable = incentive._power_cap_incapable(
        _make_job(cap_eff=CAPS_WITH_SYS_ADMIN, owner_uid=NOBODY_UID)
    )

    # Assert
    assert incapable is not None
    assert incapable.nvidiactl_owner_uid == NOBODY_UID


def test_capable_executor_is_not_flagged():
    # Arrange — both conditions met, which is what `nvidia-smi -pl` needs
    incentive = _build_incentive()

    # Act
    incapable = incentive._power_cap_incapable(
        _make_job(cap_eff=CAPS_WITH_SYS_ADMIN, owner_uid=0)
    )

    # Assert
    assert incapable is None


UNPROVEN_SPECS: list[dict] = [
    {},                                                             # validator older than DAH-2705
    {"container_cap_eff": CAPS_WITHOUT_SYS_ADMIN},                   # uid reading missing
    {"nvidiactl_owner_uid": 0},                                      # cap mask missing
    {"container_cap_eff": None, "nvidiactl_owner_uid": 0},           # probe reported nothing
    {"container_cap_eff": "not-hex", "nvidiactl_owner_uid": 0},      # unparsable mask
    {"container_cap_eff": CAPS_WITHOUT_SYS_ADMIN, "nvidiactl_owner_uid": "0"},  # uid as a string
    {"container_cap_eff": CAPS_WITHOUT_SYS_ADMIN, "nvidiactl_owner_uid": True},  # bool is not a uid
    {"container_cap_eff": ["ffff"], "nvidiactl_owner_uid": 0},       # mask came back as a list
]


@pytest.mark.parametrize("spec", UNPROVEN_SPECS)
def test_unproven_probe_is_never_flagged(spec):
    # Fail open: only a proven false is penalized. The scrape is written on the miner's
    # machine, so a wrong type is a missing reading, not a breach.
    incentive = _build_incentive()

    assert incentive._power_cap_incapable(_make_job(spec=spec)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", UNPROVEN_SPECS)
async def test_malformed_scrape_never_breaks_scoring(monkeypatch, spec):
    # Raising out of calculate_executor_score would cost EVERY miner this cycle's weights.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(spec=spec))

    assert result.eligible_for_rental_share is True


def test_enforcement_defaults_to_shadow_mode():
    # Rollout contract: first deploy must be shadow-only, so the flag default
    # (not the env-resolved value) must stay False.
    default = type(settings).model_fields["ENABLE_UNRENTED_POWER_CAP_LIMIT"].default

    assert default is False


@pytest.mark.asyncio
async def test_shadow_mode_keeps_rental_eligibility(monkeypatch):
    # Arrange — incapable container but flag off → shadow only
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_LIMIT", False)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — still eligible for the unrented rental pool, nothing told to the miner
    assert result.eligible_for_rental_share is True
    assert "cannot_apply_gpu_power_cap" not in "\n".join(result.incentive_logs)


@pytest.mark.asyncio
async def test_shadow_mode_emits_the_measurement_log(monkeypatch, caplog):
    # The shadow log is the whole deliverable of the first deploy: the rollout decision
    # is made from these fields, so they are a dashboard contract.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_LIMIT", False)
    incentive = _build_incentive()

    with caplog.at_level(logging.INFO):
        await incentive.calculate_executor_score(_make_job())

    breach = next(
        r.msg for r in caplog.records
        if hasattr(r.msg, "extra") and r.msg.extra.get("reason") == "cannot_apply_gpu_power_cap"
    )
    assert "shadow only - flag off" in breach.message
    assert breach.extra["container_cap_eff"] == CAPS_WITHOUT_SYS_ADMIN
    assert breach.extra["nvidiactl_owner_uid"] == 0
    assert breach.extra["enforced"] is False
    assert breach.extra["pool"] == "rental_kept_shadow"


@pytest.mark.asyncio
async def test_enforced_drops_rental_eligibility(monkeypatch):
    # Arrange — incapable container and flag on
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — excluded from the rental pool, no mining either (active but no incentive)
    assert result.eligible_for_rental_share is False
    assert result.mining_score == 0


@pytest.mark.asyncio
async def test_enforced_appends_customer_facing_incentive_log(monkeypatch):
    # DAH-2327: the zero-incentive reason must reach the customer-facing incentive log,
    # carrying both readings so the provider can tell which of the two to fix.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(
        _make_job(cap_eff=CAPS_WITH_SYS_ADMIN, owner_uid=NOBODY_UID)
    )

    log = "\n".join(result.incentive_logs)
    assert "cannot_apply_gpu_power_cap" in log
    assert CAPS_WITH_SYS_ADMIN in log
    assert str(NOBODY_UID) in log
    assert [reason.reason for reason in result.zero_incentive_reasons] == ["cannot_apply_gpu_power_cap"]


@pytest.mark.asyncio
async def test_capable_executor_keeps_eligibility_when_enforced(monkeypatch):
    # Arrange — flag on but the container can apply a cap
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(
        _make_job(cap_eff=CAPS_WITH_SYS_ADMIN, owner_uid=0)
    )

    # Assert
    assert result.eligible_for_rental_share is True


@pytest.mark.asyncio
async def test_rented_executor_is_not_gated(monkeypatch):
    # A rented machine earns from the rented path and must not be touched by this gate.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_POWER_CAP_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(is_rented=True))

    assert "cannot_apply_gpu_power_cap" not in "\n".join(result.incentive_logs)
