"""DAH-2787 — unrented incentive filler container entry gate.

While an idle executor earns the unrented incentive it runs Lium's own job in a filler
container. A visit inside that container from the host (`docker exec`, `nsenter`) forfeits the
unrented incentive while the node stays active. The gate ships in shadow mode: without
ENABLE_UNRENTED_FILLER_ENTRY_LIMIT the visit is only logged and the payout is unchanged.

The validator execs into a filler itself while the container is being created, and never enters
one afterwards, so only a visit FILLER_ENTRY_GRACE_SECONDS after the container started is judged.
"""

import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo

from core.config import settings
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default

FILLER_CONTAINER = "filler_9f2c"
LONG_AFTER_START = 3600.0
DURING_CREATION = 5.0


def _build_incentive() -> RentalPriceIncentive:
    return RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})


def _entry(
    seconds_after_start: float = LONG_AFTER_START,
    kind: str = "docker_exec",
    pid: int | None = None,
    command: str = "kill -STOP 1584",
) -> dict[str, Any]:
    return {
        "container": FILLER_CONTAINER,
        "kind": kind,
        "pid": pid,
        "seconds_after_start": seconds_after_start,
        "command": command,
    }


def _make_job(
    *,
    entries: list[Any] | None = None,
    is_rented: bool = False,
    spec: dict | None = None,
) -> JobResult:
    if spec is None:
        spec = {"filler_entries": [_entry()] if entries is None else entries}

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


def test_a_docker_exec_into_the_filler_is_flagged():
    # Arrange — prod 109.174.15.2: a timer execs in and SIGSTOPs our miner
    incentive = _build_incentive()

    # Act
    entry = incentive._filler_container_entry(_make_job())

    # Assert
    assert entry is not None
    assert entry.container_name == FILLER_CONTAINER
    assert entry.kind == "docker_exec"
    assert entry.command == "kill -STOP 1584"


def test_an_open_session_is_flagged_too():
    # An `nsenter` never reaches the docker daemon; the live process is the only proof of it.
    incentive = _build_incentive()
    entries = [_entry(kind="open_session", pid=200, command="bash")]

    entry = incentive._filler_container_entry(_make_job(entries=entries))

    assert entry is not None
    assert entry.kind == "open_session"
    assert entry.pid == 200


def test_our_own_exec_while_the_container_is_created_is_not_flagged():
    # The validator injects the public keys and bootstraps sshd right after `docker run`.
    incentive = _build_incentive()

    entry = incentive._filler_container_entry(
        _make_job(entries=[_entry(seconds_after_start=DURING_CREATION)])
    )

    assert entry is None


def test_the_latest_visit_is_the_one_reported():
    # The provider is shown the visit that costs them the incentive, not a random one.
    incentive = _build_incentive()
    entries = [_entry(seconds_after_start=1000.0), _entry(seconds_after_start=9000.0)]

    entry = incentive._filler_container_entry(_make_job(entries=entries))

    assert entry is not None
    assert entry.seconds_after_start == 9000.0


UNPROVEN_SPECS: list[dict] = [
    {},                                                    # validator older than DAH-2787
    {"filler_entries": []},                                # probe ran, nobody was inside
    {"filler_entries": None},                              # probe reported nothing
    {"filler_entries": "bash"},                            # not a list
    {"filler_entries": [{}]},                              # entry without any reading
    {"filler_entries": ["bash"]},                          # entry is not an object
    {"filler_entries": [{**_entry(), "kind": "guessed"}]},  # a shape this gate does not know
    {"filler_entries": [{**_entry(), "kind": None}]},
    {"filler_entries": [{**_entry(), "seconds_after_start": "3600"}]},  # reading as a string
    {"filler_entries": [{**_entry(), "seconds_after_start": None}]},
    {"filler_entries": [{**_entry(), "container": 1}]},    # container name is not a name
    {"filler_entries": [{**_entry(), "pid": "200"}]},      # pid as a string
    {"filler_entries": [{**_entry(), "pid": 200}]},        # a finished exec names no process
    {"filler_entries": [{**_entry(kind="open_session"), "pid": None}]},  # a session must name one
    {"filler_entries": [{**_entry(kind="open_session"), "pid": 0}]},
    {"filler_entries": [{**_entry(kind="open_session"), "pid": -1}]},
    {"filler_entries": [{**_entry(), "seconds_after_start": 10**400}]},  # too large for a float
    {"filler_entries": [{**_entry(), "seconds_after_start": float("inf")}]},
    {"filler_entries": [{**_entry(), "seconds_after_start": float("nan")}]},
    {"filler_entry_scrape_error": "docker api /events returned HTTP 500"},
]


@pytest.mark.parametrize("spec", UNPROVEN_SPECS)
def test_an_unproven_probe_is_never_flagged(spec):
    # Fail open: only a proven visit is penalized. The scrape runs on the miner's machine,
    # so a missing or wrongly typed reading is an inability to measure, not a breach.
    incentive = _build_incentive()

    assert incentive._filler_container_entry(_make_job(spec=spec)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", UNPROVEN_SPECS)
async def test_a_malformed_scrape_never_breaks_scoring(monkeypatch, spec):
    # Raising out of calculate_executor_score would cost EVERY miner this cycle's weights.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FILLER_ENTRY_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(spec=spec))

    assert result.eligible_for_rental_share is True


def test_an_estimated_job_result_carries_no_spec():
    # estimate_executor builds a spec-less JobResult for every GPU model every cycle.
    incentive = _build_incentive()
    job = _make_job()
    job.spec = None

    assert incentive._filler_container_entry(job) is None


def test_the_gate_ships_in_shadow_mode():
    # Rollout contract: the detection is new, so the first deploy only measures.
    default = type(settings).model_fields["ENABLE_UNRENTED_FILLER_ENTRY_LIMIT"].default

    assert default is False


@pytest.mark.asyncio
async def test_shadow_mode_keeps_rental_eligibility(monkeypatch):
    # Arrange — somebody was inside the filler but the flag is off
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FILLER_ENTRY_LIMIT", False)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — still eligible for the unrented rental pool, nothing told to the miner
    assert result.eligible_for_rental_share is True
    assert "filler_container_entered" not in "\n".join(result.incentive_logs)


@pytest.mark.asyncio
async def test_shadow_mode_emits_the_measurement_log(monkeypatch, caplog):
    # The shadow log is the whole deliverable of the first deploy: the rollout decision is
    # made from these fields, so they are a dashboard contract.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FILLER_ENTRY_LIMIT", False)
    incentive = _build_incentive()

    with caplog.at_level(logging.INFO):
        await incentive.calculate_executor_score(_make_job())

    breach = next(
        r.msg for r in caplog.records
        if hasattr(r.msg, "extra") and r.msg.extra.get("reason") == "filler_container_entered"
    )
    assert "shadow only - flag off" in breach.message
    assert breach.extra["filler_container"] == FILLER_CONTAINER
    assert breach.extra["entry_kind"] == "docker_exec"
    assert breach.extra["entry_command"] == "kill -STOP 1584"
    assert breach.extra["entry_seconds_after_start"] == LONG_AFTER_START
    assert breach.extra["enforced"] is False
    assert breach.extra["pool"] == "rental_kept_shadow"


@pytest.mark.asyncio
async def test_enforced_drops_rental_eligibility(monkeypatch):
    # Arrange — somebody was inside the filler and the flag is on
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FILLER_ENTRY_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job())

    # Assert — excluded from the rental pool, no mining either (active but no incentive)
    assert result.eligible_for_rental_share is False
    assert result.mining_score == 0


@pytest.mark.asyncio
async def test_enforced_appends_customer_facing_incentive_log(monkeypatch):
    # DAH-2327: the zero-incentive reason must reach the customer-facing incentive log,
    # carrying the visit so the provider can see what was judged.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FILLER_ENTRY_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job())

    log = "\n".join(result.incentive_logs)
    assert "filler_container_entered" in log
    assert FILLER_CONTAINER in log
    assert [reason.reason for reason in result.zero_incentive_reasons] == ["filler_container_entered"]


@pytest.mark.asyncio
async def test_a_clean_filler_keeps_eligibility_when_enforced(monkeypatch):
    # Arrange — flag on, nobody entered the container
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FILLER_ENTRY_LIMIT", True)
    incentive = _build_incentive()

    # Act
    result = await incentive.calculate_executor_score(_make_job(entries=[]))

    # Assert
    assert result.eligible_for_rental_share is True


@pytest.mark.asyncio
async def test_a_rented_executor_is_not_gated(monkeypatch):
    # A rented machine earns from the rented path and must not be touched by this gate.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_FILLER_ENTRY_LIMIT", True)
    incentive = _build_incentive()

    result = await incentive.calculate_executor_score(_make_job(is_rented=True))

    assert "filler_container_entered" not in "\n".join(result.incentive_logs)
