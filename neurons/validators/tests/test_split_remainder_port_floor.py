"""DAH-3698 — the free remainder of a partially rented split node earns idle pay only for the
minimum-size splits that have a usable, unique free port (owner rule, 26 Oct 2026), and nothing
below the marketplace floor of MIN_PORT_COUNT free ports.

`PortCountCheck` exempts a rented executor from `MIN_PORT_COUNT` (the tenant already holds the
ports), so a part-rented split node with 2 free ports still reaches scoring, and DAH-2467 then
pays its free GPUs from the unrented pool. The platform lists a node only with
`available_port_count >= MIN_PORT_COUNT` and the rent path refuses a pod below it, so nobody
can rent that remainder. Above the floor each simultaneous renter of a minimum-size split needs its
own port, so a remainder with fewer ports than splits is paid for as many splits as it has ports.
"""

import logging
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.miner_incentive_log import ZeroIncentiveReason
from incentive.rental_price import RentalPriceIncentive
from services.const import DEFAULT_JOB_OWNER_LIUM, MIN_PORT_COUNT
from services.task_service import JobResult


H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default
MINER_HOTKEY = "miner-hotkey-1"


def _make_job(
    *,
    available_port_count: int | None = 2,
    gpu_count: int = 8,
    rented_gpu_count: int | None = 4,
    is_rented: bool = True,
    spec: dict | None = None,
    min_count: int = 1,
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
        gpu_splitting_min_count=min_count,
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
    assert limited.required_port_count == 4
    assert limited.paid_split_count == 0  # below the rent path's floor no split is rentable
    assert limited.unpaid_gpu_count == 4


def test_remainder_with_a_port_per_split_is_not_port_limited():
    # Arrange — 4 free 1-GPU splits, 4 free ports: every split can be rented at once.
    incentive = _build_incentive(_make_job(available_port_count=4))
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


def test_lium_filler_remainder_below_the_floor_is_port_limited():
    # Arrange — a Lium filler on the free GPUs does not make them rentable on Lium
    incentive = _build_incentive(_make_job(available_port_count=2))
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[MINER_HOTKEY]
    remainder.default_job_owner = DEFAULT_JOB_OWNER_LIUM

    # Act
    port_limited = incentive._port_limited_remainder(remainder)

    # Assert
    assert port_limited is not None
    assert port_limited.available_port_count == 2


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
async def test_port_limited_remainder_earns_no_idle_pay_and_the_rented_portion_is_unchanged(monkeypatch):
    # Arrange — 8-GPU split node, 4 rented, 2 free ports; rental share pinned for determinism.
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
    assert reason.context["required_port_count"] == 4  # one per free split
    assert "2 free port" in reason.message_for_miner
    assert "port_limited_remainder" in job.full_log_text


@pytest.mark.asyncio
async def test_remainder_with_enough_ports_is_paid_as_before(monkeypatch):
    # Arrange — the same node with a free port for each of its 4 free 1-GPU splits: DAH-2467 unchanged.
    job = _make_job(available_port_count=4, gpu_count=8, rented_gpu_count=4)
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
async def test_enforced_log_line_names_the_reason_and_the_pool(monkeypatch, caplog):
    # Arrange — the mitigation for the pay change: the scorer's structured line carries the reason
    # code so Loki / the provider dashboard (DAH-3699) can explain the zero.
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
    assert breach[0].extra["required_port_count"] == 4
    assert breach[0].extra["paid_split_count"] == 0
    assert breach[0].extra["enforced"] is True
    assert breach[0].extra["pool"] == "rental_excluded"


def _port_floor_breach_lines(caplog) -> list:
    return [
        r.msg for r in caplog.records
        if hasattr(r.msg, "extra")
        and r.msg.extra.get("reason") == ZeroIncentiveReason.PORT_LIMITED_REMAINDER
    ]


# ── one port per minimum-size split (owner rule, 26 Oct 2026) ─────────────────


async def _score_split(ports, *, gpu_count, rented, min_count=1, spec_extra=None):
    job = _make_job(available_port_count=ports, gpu_count=gpu_count, rented_gpu_count=rented, min_count=min_count)
    job.spec.update(spec_extra or {})
    incentive = await _score(job)
    return job, incentive


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ports", "paid_gpus"),
    [(8, 8), (7, 7), (MIN_PORT_COUNT, MIN_PORT_COUNT), (MIN_PORT_COUNT - 1, 0)],
)
async def test_8x1_remainder_is_paid_one_split_per_free_port(ports, paid_gpus):
    # 9 GPUs offered as 1-GPU splits, 1 rented: 8 free splits need 8 free ports for 8 renters.
    job, incentive = await _score_split(ports, gpu_count=9, rented=1)

    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == (paid_gpus or None)
    assert job.gpu_count == 9  # the merged row still reports the whole node
    if paid_gpus == 8:
        assert ZeroIncentiveReason.PORT_LIMITED_REMAINDER.value not in _reason_codes(job)
    else:
        assert _reason_codes(job) == [ZeroIncentiveReason.PORT_LIMITED_REMAINDER.value]
        assert job.zero_incentive_reasons[0].context["paid_split_count"] == paid_gpus
        assert job.zero_incentive_reasons[0].context["gpu_count"] == 8 - paid_gpus
    assert (job.incentive_idle > 0) is (paid_gpus > 0)


@pytest.mark.asyncio
async def test_partial_remainder_is_paid_pro_rata_against_a_fully_ported_twin():
    full, full_incentive = await _score_split(8, gpu_count=9, rented=1)
    partial, partial_incentive = await _score_split(7, gpu_count=9, rented=1)
    key = ("H200", 1)
    assert partial_incentive._weighted_rate_sum_by_bucket[key] == pytest.approx(
        full_incentive._weighted_rate_sum_by_bucket[key] * 7 / 8
    )
    assert partial.mining_score == pytest.approx(full.mining_score)  # the rented GPU is untouched


@pytest.mark.parametrize(
    ("ports", "unpaid_gpus"),
    [(4, None), (3, 2), (2, 8)],
)
def test_4x2_remainder_needs_a_port_per_2_gpu_split(ports, unpaid_gpus):
    # 10 GPUs in 2-GPU splits, 2 rented: 8 free GPUs are 4 splits, so 4 ports pay them all.
    incentive = _build_incentive(_make_job(available_port_count=ports, gpu_count=10, rented_gpu_count=2, min_count=2))
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[MINER_HOTKEY]

    limited = incentive._port_limited_remainder(remainder)

    assert (limited.unpaid_gpu_count if limited else None) == unpaid_gpus


def test_leftover_gpus_below_one_split_ride_with_the_last_paid_split():
    # 5 free GPUs in 2-GPU splits = 2 splits (the 5th GPU joins a 3-GPU rental); 3 ports pay all
    incentive = _build_incentive(_make_job(available_port_count=3, gpu_count=8, rented_gpu_count=3, min_count=2))
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[MINER_HOTKEY]
    assert incentive._port_limited_remainder(remainder) is None


@pytest.mark.parametrize(
    ("rented", "unpaid_gpus"),
    [(8, 2), (2, None), (1, None)],
    ids=["below_min", "equal_min", "above_min"],
)
def test_remainder_below_one_whole_split_earns_no_idle_pay(rented, unpaid_gpus):
    # 10-GPU node, 8-GPU minimum, 3 free ports: 2 free GPUs are no rentable split, 8 and 9 are one
    incentive = _build_incentive(
        _make_job(available_port_count=3, gpu_count=10, rented_gpu_count=rented, min_count=8)
    )
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[MINER_HOTKEY]

    limited = incentive._port_limited_remainder(remainder)

    assert (limited.unpaid_gpu_count if limited else None) == unpaid_gpus
    if limited:
        assert limited.paid_split_count == 0


def test_ports_held_by_the_rental_are_not_free_ports():
    # PortCountCheck publishes only ports no pod holds: 6 verified free on a node whose tenant
    # holds 3 more. The 6 free ones are all the 8 free splits can use.
    incentive = _build_incentive(
        _make_job(spec={"available_port_count": 6, "verified_ports": [20001, 20002, 20003, 20004, 20005, 20006]},
                  gpu_count=9, rented_gpu_count=1)
    )
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[MINER_HOTKEY]
    limited = incentive._port_limited_remainder(remainder)
    assert limited is not None
    assert (limited.available_port_count, limited.paid_split_count, limited.unpaid_gpu_count) == (6, 6, 2)


def test_colliding_ports_count_once():
    # two mappings publish the same external port: it can serve one renter only
    incentive = _build_incentive(
        _make_job(spec={"available_port_count": 8, "verified_ports": [20001, 20001, 20002, 20002, 20003, 20004, 20005, 20006]},
                  gpu_count=9, rented_gpu_count=1)
    )
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[MINER_HOTKEY]
    limited = incentive._port_limited_remainder(remainder)
    assert limited is not None
    assert (limited.available_port_count, limited.paid_split_count) == (6, 6)


@pytest.mark.asyncio
async def test_a_first_failed_port_verification_does_not_cost_the_remainder():
    # PortConnectivityCheck marks the first failed verification inside its window; the gate skips it
    job, incentive = await _score_split(0, gpu_count=9, rented=1, spec_extra={"port_verify_first_miss": True})
    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == 8
    assert ZeroIncentiveReason.PORT_LIMITED_REMAINDER.value not in _reason_codes(job)


@pytest.mark.asyncio
async def test_a_repeated_failed_verification_counts():
    job, incentive = await _score_split(0, gpu_count=9, rented=1)
    assert incentive.unrented_count_by_bucket.get(("H200", 1)) is None
    assert _reason_codes(job) == [ZeroIncentiveReason.PORT_LIMITED_REMAINDER.value]


@pytest.mark.asyncio
async def test_partial_log_line_carries_the_split_counts(caplog):
    with caplog.at_level(logging.INFO):
        await _score_split(5, gpu_count=9, rented=1)
    breach = _port_floor_breach_lines(caplog)
    assert len(breach) == 1
    extra = breach[0].extra
    assert (extra["available_port_count"], extra["required_port_count"]) == (5, 8)
    assert (extra["free_split_count"], extra["paid_split_count"], extra["unpaid_gpu_count"]) == (8, 5, 3)
    assert extra["pool"] == "rental_partial"
