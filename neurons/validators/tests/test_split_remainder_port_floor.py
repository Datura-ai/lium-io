"""DAH-3698 — the free remainder of a partially rented split node earns no idle pay when the
node has fewer free ports than the marketplace floor.

`PortCountCheck` exempts a rented executor from `MIN_PORT_COUNT` (the tenant already holds the
ports), so a part-rented split node with 2 free ports still reaches scoring, and DAH-2467 then
pays its free GPUs from the unrented pool. The platform lists a node only with
`available_port_count >= MIN_PORT_COUNT` and the rent path refuses a pod below it, so nobody
can rent that remainder. The remainder is still expanded (the rented portion keeps earning in
the mining pool) but forfeits the unrented incentive with a named reason once
ENABLE_UNRENTED_PORT_FLOOR_FOR_SPLIT_REMAINDER is on; off (the default) only logs the shortfall.
"""

import logging
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.miner_incentive_log import ZeroIncentiveReason
from incentive.rental_price import RentalPriceIncentive
from services.const import MIN_PORT_COUNT
from services.task_service import JobResult

from core.config import settings

H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default
MINER_HOTKEY = "miner-hotkey-1"


def _make_job(
    *,
    available_port_count: int | None = 2,
    gpu_count: int = 8,
    rented_gpu_count: int | None = 4,
    is_rented: bool = True,
    spec: dict | None = None,
) -> JobResult:
    if spec is None:
        spec = {}
        if available_port_count is not None:
            spec["available_port_count"] = available_port_count
    return JobResult(
        spec=spec,
        executor_info=ExecutorSSHInfo(
            uuid="exec-split-ports",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model=H200,
        gpu_count=gpu_count,
        is_rented=is_rented,
        rented_gpu_count=rented_gpu_count,
        supports_gpu_splitting=True,
        gpu_splitting_min_count=1,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


def _build_incentive(job: JobResult) -> RentalPriceIncentive:
    redis_service = AsyncMock()
    redis_service.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    return RentalPriceIncentive(
        IncentiveConfig(),
        redis_service,
        {MINER_HOTKEY: [job]},
        {H200: job.gpu_count},
    )


def _reason_codes(job: JobResult) -> list[str]:
    return [reason.reason for reason in job.zero_incentive_reasons]


# ── the measurement ───────────────────────────────────────────────────────────


def test_remainder_with_two_free_ports_is_port_limited():
    # Arrange — the prod case: 4 of 8 GPUs rented, the tenant holds all but 2 verified ports.
    incentive = _build_incentive(_make_job(available_port_count=2))
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[MINER_HOTKEY]

    # Act
    limited = incentive._port_limited_remainder(remainder)

    # Assert
    assert limited is not None
    assert limited.available_port_count == 2
    assert limited.required == MIN_PORT_COUNT


def test_remainder_at_the_floor_is_not_port_limited():
    # Arrange — exactly MIN_PORT_COUNT free ports: the platform lists it and the rent path accepts it.
    incentive = _build_incentive(_make_job(available_port_count=MIN_PORT_COUNT))
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[MINER_HOTKEY]

    # Act / Assert
    assert incentive._port_limited_remainder(remainder) is None


def test_rented_portion_is_never_port_limited():
    # Arrange — the rented GPUs earn in the mining pool whatever the free-port count.
    incentive = _build_incentive(_make_job(available_port_count=0))
    incentive._expand_partially_rented_split_results()
    rented_portion, _ = incentive.job_results[MINER_HOTKEY]

    # Act / Assert
    assert incentive._port_limited_remainder(rented_portion) is None


def test_whole_idle_node_is_out_of_scope():
    # Arrange — an idle (unrented) node below the floor never reaches scoring: PortCountCheck fails
    # it. The gate is for the remainder only; a whole idle result is left to that check.
    incentive = _build_incentive(_make_job(available_port_count=2, is_rented=False, rented_gpu_count=None))

    # Act / Assert
    assert incentive._port_limited_remainder(incentive.job_results[MINER_HOTKEY][0]) is None


@pytest.mark.parametrize(
    "spec",
    [
        None,  # synthetic / estimated result: no scrape at all
        {},  # a validator older than PortCountCheck's spec write
        {"available_port_count": None},
        {"available_port_count": "two"},  # unreadable: the miner's machine wrote the scrape
        {"available_port_count": True},  # bool passes isinstance(int); must not read as 1
    ],
)
def test_unreadable_port_count_fails_open(spec):
    # Arrange
    job = _make_job(spec=spec if spec is not None else {})
    if spec is None:
        job.spec = None
    incentive = _build_incentive(job)
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[MINER_HOTKEY]

    # Act / Assert — nobody loses incentive over telemetry
    assert incentive._port_limited_remainder(remainder) is None


# ── the scoring decision ──────────────────────────────────────────────────────


async def _score(job: JobResult, rental_share: float = 0.1) -> RentalPriceIncentive:
    incentive = _build_incentive(job)
    incentive._calculate_rental_share = AsyncMock(return_value=rental_share)
    await incentive.calculate_mining_scores()
    return incentive


@pytest.mark.asyncio
async def test_default_settings_pay_the_port_limited_remainder_exactly_as_before(caplog):
    # Rollout contract (SO §70): a money-path gate ships in shadow mode. With the settings object
    # untouched (no monkeypatch of the flag) a 2-port remainder must receive the same idle pay
    # as the computation the gate never sees — the same node with MIN_PORT_COUNT free ports —
    # and the scorer must say so in the shadow line. Precondition, not the assertion: the
    # settings object as the test process resolved it carries the shipped default.
    assert settings.ENABLE_UNRENTED_PORT_FLOOR_FOR_SPLIT_REMAINDER is False, (
        "ENABLE_UNRENTED_PORT_FLOOR_FOR_SPLIT_REMAINDER is on in this environment — "
        "the default-off contract cannot be measured here"
    )

    # Arrange — the baseline: an identical node at the floor, which no gate touches.
    baseline_job = _make_job(available_port_count=MIN_PORT_COUNT, gpu_count=8, rented_gpu_count=4)
    baseline = await _score(baseline_job)
    assert baseline_job.incentive_idle > 0  # the baseline really pays the free GPUs

    # Act — the prod case under default settings.
    job = _make_job(available_port_count=2, gpu_count=8, rented_gpu_count=4)
    with caplog.at_level(logging.INFO):
        incentive = await _score(job)

    # Assert — every payout figure equals the unflagged computation.
    assert job.incentive_idle == pytest.approx(baseline_job.incentive_idle)
    assert job.incentive_rented == pytest.approx(baseline_job.incentive_rented)
    assert job.incentive == pytest.approx(baseline_job.incentive)
    assert job.mining_score == pytest.approx(baseline_job.mining_score)
    assert incentive.unrented_count_by_bucket == baseline.unrented_count_by_bucket
    assert _reason_codes(job) == _reason_codes(baseline_job)
    assert ZeroIncentiveReason.PORT_LIMITED_REMAINDER.value not in _reason_codes(job)

    # …and the shortfall is still visible: one shadow line, not enforced, with the reason code.
    breach = _port_floor_breach_lines(caplog)
    assert len(breach) == 1
    assert breach[0].extra["enforced"] is False
    assert breach[0].extra["reason"] == ZeroIncentiveReason.PORT_LIMITED_REMAINDER
    assert breach[0].extra["available_port_count"] == 2
    assert "shadow only - flag off" in breach[0].message


@pytest.mark.asyncio
async def test_port_limited_remainder_earns_no_idle_pay_and_the_rented_portion_is_unchanged(monkeypatch):
    # Arrange — 8-GPU split node, 4 rented, 2 free ports; rental share pinned for determinism.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_PORT_FLOOR_FOR_SPLIT_REMAINDER", True)
    job = _make_job(available_port_count=2, gpu_count=8, rented_gpu_count=4)
    incentive = _build_incentive(job)
    monkeypatch.setattr(incentive, "_calculate_rental_share", AsyncMock(return_value=0.1))

    # Act
    await incentive.calculate_mining_scores()

    # Assert — merged back into ONE result; the rented 4 GPUs earned in the mining pool exactly
    # as before (mining_score on 4 of 8 GPUs), the free 4 earned nothing.
    assert incentive.job_results[MINER_HOTKEY] == [job]
    assert job.gpu_count == 8
    assert job.mining_score == pytest.approx(1.0 * 0.3 * 4 / 8)
    assert job.incentive_rented == pytest.approx(incentive.mining_share)
    assert job.incentive_idle == 0.0
    assert job.incentive == pytest.approx(job.incentive_rented)
    assert job.node_state_at_cycle == "mixed"

    # The free GPUs were NOT counted in the unrented tier: "what earns = what is listed".
    assert incentive.unrented_count_by_bucket.get(("H200", 1)) is None

    # The reason reaches the backend under the merged executor, and the miner-facing log names it.
    assert _reason_codes(job) == [ZeroIncentiveReason.PORT_LIMITED_REMAINDER.value]
    reason = job.zero_incentive_reasons[0]
    assert reason.context["available_port_count"] == 2
    assert reason.context["required_port_count"] == MIN_PORT_COUNT
    assert "2 free port" in reason.message_for_miner
    assert "port_limited_remainder" in job.full_log_text


@pytest.mark.asyncio
async def test_remainder_with_enough_ports_is_paid_as_before(monkeypatch):
    # Arrange — the same node with 3 free ports: DAH-2467 behaviour unchanged.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_PORT_FLOOR_FOR_SPLIT_REMAINDER", True)
    job = _make_job(available_port_count=MIN_PORT_COUNT, gpu_count=8, rented_gpu_count=4)
    incentive = _build_incentive(job)
    monkeypatch.setattr(incentive, "_calculate_rental_share", AsyncMock(return_value=0.1))

    # Act
    await incentive.calculate_mining_scores()

    # Assert — both pools paid, the 4 free GPUs counted in the min-count bucket.
    assert job.mining_score == pytest.approx(1.0 * 0.3 * 4 / 8)
    assert job.incentive_idle == pytest.approx(0.1)
    assert job.incentive == pytest.approx(job.incentive_rented + job.incentive_idle)
    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == 4
    assert ZeroIncentiveReason.PORT_LIMITED_REMAINDER.value not in _reason_codes(job)


@pytest.mark.asyncio
async def test_flag_off_only_logs_the_shortfall(monkeypatch, caplog):
    # Arrange — shadow mode: the payout is unchanged and the would-be exclusion is logged.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_PORT_FLOOR_FOR_SPLIT_REMAINDER", False)
    job = _make_job(available_port_count=2, gpu_count=8, rented_gpu_count=4)
    incentive = _build_incentive(job)
    monkeypatch.setattr(incentive, "_calculate_rental_share", AsyncMock(return_value=0.1))

    # Act
    with caplog.at_level(logging.INFO):
        await incentive.calculate_mining_scores()

    # Assert — payout unchanged, no reason shipped, one shadow line with the dashboard fields.
    assert job.incentive_idle == pytest.approx(0.1)
    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == 4
    assert ZeroIncentiveReason.PORT_LIMITED_REMAINDER.value not in _reason_codes(job)
    breach = _port_floor_breach_lines(caplog)
    assert len(breach) == 1
    assert "shadow only - flag off" in breach[0].message
    assert breach[0].extra["enforced"] is False
    assert breach[0].extra["pool"] == "rental_kept_shadow"


@pytest.mark.asyncio
async def test_enforced_log_line_names_the_reason_and_the_pool(monkeypatch, caplog):
    # Arrange — the mitigation for the pay change: the scorer's structured line carries the reason
    # code so Loki / the provider dashboard (DAH-3699) can explain the zero.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_PORT_FLOOR_FOR_SPLIT_REMAINDER", True)
    job = _make_job(available_port_count=2, gpu_count=8, rented_gpu_count=4)
    incentive = _build_incentive(job)
    monkeypatch.setattr(incentive, "_calculate_rental_share", AsyncMock(return_value=0.1))

    # Act
    with caplog.at_level(logging.INFO):
        await incentive.calculate_mining_scores()

    # Assert
    breach = _port_floor_breach_lines(caplog)
    assert len(breach) == 1
    assert "shadow only" not in breach[0].message
    assert breach[0].extra["executor_id"] == "exec-split-ports"
    assert breach[0].extra["available_port_count"] == 2
    assert breach[0].extra["required_port_count"] == MIN_PORT_COUNT
    assert breach[0].extra["enforced"] is True
    assert breach[0].extra["pool"] == "rental_excluded"


def _port_floor_breach_lines(caplog) -> list:
    return [
        r.msg for r in caplog.records
        if hasattr(r.msg, "extra")
        and r.msg.extra.get("reason") == ZeroIncentiveReason.PORT_LIMITED_REMAINDER
    ]
