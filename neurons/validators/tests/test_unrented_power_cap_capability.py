"""DAH-2715 — unrented incentive GPU power cap gate.

An unrented executor whose container provably cannot apply a GPU power cap
(no CAP_SYS_ADMIN, or /dev/nvidiactl not owned by root — see DAH-2705) forfeits
the unrented rental incentive while staying active. Enforcement is ON by default
and can be switched off with ENABLE_UNRENTED_POWER_CAP_LIMIT; with the flag off
the breach is only logged (shadow mode) and the payout is unchanged.
"""

from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default

CAPS_WITHOUT_SYS_ADMIN = "00000000a80425fb"  # prod 128.140.36.181: default docker caps


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


def _logged_reasons(caplog) -> list[str]:
    # the reason codes of the structured lines captured so far; _m keeps them off the message
    return [r.msg.extra.get("reason") for r in caplog.records if hasattr(r.msg, "extra")]


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


