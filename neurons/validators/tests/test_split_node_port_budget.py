"""DAH-3698 (owner, 22 Sep 2026) — a GPU-split node's free GPUs earn idle pay only up to the
number its free verified ports can back at the marketplace floor.

The platform gives every pod MIN_PORT_COUNT ports (`prepare_ports_data`, the rent path's 409
below it) and rents a split node in bundles of `gpu_splitting_min_count` GPUs, so a node with
`available_port_count` free ports can start `available_port_count // MIN_PORT_COUNT` more
pods. Free GPUs beyond `bundles * gpu_splitting_min_count` cannot be rented until a tenant
leaves or the provider opens more ports; with ENABLE_UNRENTED_PORT_BUDGET_FOR_SPLIT_GPUS on
they earn no idle pay (the backed GPUs and the rented GPUs earn as before); off (the default)
only logs the shortfall. lium-io#1414's floor (fewer than MIN_PORT_COUNT free ports → no idle
pay at all) runs first and stays as it is.
"""

import logging
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.miner_incentive_log import ZeroIncentiveReason
from incentive.rental_price import PORT_UNBACKED_GPUS_EVENT, RentalPriceIncentive
from services.const import DEFAULT_JOB_OWNER_LIUM, MIN_PORT_COUNT
from services.task_service import JobResult

from core.config import settings

H200 = "NVIDIA H200"  # base model H200 is rental-eligible by default
SPLIT_HOTKEY = "miner-hotkey-split"
PLAIN_HOTKEY = "miner-hotkey-plain"


def _make_job(
    *,
    uuid: str = "exec-split-budget",
    available_port_count: int | None = 5,
    gpu_count: int = 8,
    rented_gpu_count: int | None = 4,
    is_rented: bool = True,
    supports_gpu_splitting: bool = True,
    gpu_splitting_min_count: int | None = 1,
    spec: dict | None = None,
) -> JobResult:
    if spec is None:
        spec = {}
        if available_port_count is not None:
            spec["available_port_count"] = available_port_count
    return JobResult(
        spec=spec,
        executor_info=ExecutorSSHInfo(
            uuid=uuid,
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
        supports_gpu_splitting=supports_gpu_splitting,
        gpu_splitting_min_count=gpu_splitting_min_count,
        collateral_deposited=True,
        sysbox_runtime=True,
    )


def _plain_idle_job(gpu_count: int = 1) -> JobResult:
    # an ordinary idle node of the same model, no splitting, plenty of ports: the yardstick
    return _make_job(
        uuid="exec-plain-idle",
        available_port_count=12,
        gpu_count=gpu_count,
        rented_gpu_count=None,
        is_rented=False,
        supports_gpu_splitting=False,
        gpu_splitting_min_count=None,
    )


def _build_incentive(*jobs_by_hotkey: tuple[str, JobResult]) -> RentalPriceIncentive:
    redis_service = AsyncMock()
    redis_service.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    job_results = {hotkey: [job] for hotkey, job in jobs_by_hotkey}
    total = sum(job.gpu_count for _, job in jobs_by_hotkey)
    return RentalPriceIncentive(IncentiveConfig(), redis_service, job_results, {H200: total})


def _remainder_of(incentive: RentalPriceIncentive, hotkey: str = SPLIT_HOTKEY) -> JobResult:
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[hotkey]
    return remainder


def _budget_lines(caplog) -> list:
    return [
        r.msg for r in caplog.records
        if hasattr(r.msg, "extra") and r.msg.extra.get("event") == PORT_UNBACKED_GPUS_EVENT
    ]


# ── the measurement ───────────────────────────────────────────────────────────


def test_remainder_with_five_ports_backs_one_of_four_free_gpus():
    # Arrange — 4 of 8 GPUs rented, 5 free ports, 1-GPU bundles: one more pod can start.
    remainder = _remainder_of(_build_incentive((SPLIT_HOTKEY, _make_job(available_port_count=5))))

    # Act
    shortfall = RentalPriceIncentive._port_budget_shortfall(remainder)

    # Assert
    assert shortfall is not None
    assert shortfall.available_port_count == 5
    assert shortfall.ports_per_bundle == MIN_PORT_COUNT
    assert shortfall.gpu_splitting_min_count == 1
    assert shortfall.free_gpu_count == 4
    assert shortfall.backed_gpu_count == 1
    assert shortfall.unbacked_gpu_count == 3


def test_remainder_with_ports_for_every_free_gpu_has_no_shortfall():
    # Arrange — 4 free GPUs, 12 free ports: four 1-GPU pods fit.
    remainder = _remainder_of(_build_incentive((SPLIT_HOTKEY, _make_job(available_port_count=4 * MIN_PORT_COUNT))))

    # Act / Assert
    assert RentalPriceIncentive._port_budget_shortfall(remainder) is None


def test_bundles_of_two_gpus_are_counted_per_bundle():
    # Arrange — 5 free GPUs, 2-GPU bundles, 6 ports: two bundles (4 GPUs) fit, 1 GPU is beyond them.
    job = _make_job(available_port_count=6, gpu_count=8, rented_gpu_count=3, gpu_splitting_min_count=2)
    remainder = _remainder_of(_build_incentive((SPLIT_HOTKEY, job)))

    # Act
    shortfall = RentalPriceIncentive._port_budget_shortfall(remainder)

    # Assert
    assert shortfall is not None
    assert (shortfall.backed_gpu_count, shortfall.unbacked_gpu_count) == (4, 1)


def test_whole_idle_split_node_is_measured_too():
    # Arrange — an idle 8-GPU node that splits into 1-GPU pods with 6 open ports: two pods fit.
    job = _make_job(available_port_count=6, is_rented=False, rented_gpu_count=None)

    # Act
    shortfall = RentalPriceIncentive._port_budget_shortfall(job)

    # Assert
    assert shortfall is not None
    assert shortfall.free_gpu_count == 8
    assert (shortfall.backed_gpu_count, shortfall.unbacked_gpu_count) == (2, 6)


def test_whole_node_without_real_splitting_is_out_of_scope():
    # Arrange — min count equal to the GPU count: the node rents as one pod, MIN_PORT_COUNT covers it.
    job = _make_job(available_port_count=3, is_rented=False, rented_gpu_count=None, gpu_splitting_min_count=8)

    # Act / Assert
    assert RentalPriceIncentive._port_budget_shortfall(job) is None


def test_node_that_does_not_split_is_out_of_scope():
    job = _make_job(
        available_port_count=3, is_rented=False, rented_gpu_count=None,
        supports_gpu_splitting=False, gpu_splitting_min_count=None,
    )
    assert RentalPriceIncentive._port_budget_shortfall(job) is None


def test_rented_portion_is_never_measured():
    # Arrange — the rented 4 GPUs stay in the mining pool whatever the port count.
    incentive = _build_incentive((SPLIT_HOTKEY, _make_job(available_port_count=5)))
    incentive._expand_partially_rented_split_results()
    rented_portion, _ = incentive.job_results[SPLIT_HOTKEY]

    # Act / Assert
    assert RentalPriceIncentive._port_budget_shortfall(rented_portion) is None


def test_lium_filler_remainder_is_never_measured():
    # Arrange — the free GPUs run a Lium filler, which holds their ports: filler revenue, not idle pay
    remainder = _remainder_of(_build_incentive((SPLIT_HOTKEY, _make_job(available_port_count=5))))
    remainder.default_job_owner = DEFAULT_JOB_OWNER_LIUM

    # Act / Assert
    assert RentalPriceIncentive._port_budget_shortfall(remainder) is None


@pytest.mark.parametrize(
    "spec",
    [
        None,  # synthetic / estimated result: no scrape at all
        {},  # a validator older than PortCountCheck's spec write
        {"available_port_count": None},
        {"available_port_count": "five"},  # unreadable: the miner's machine wrote the scrape
        {"available_port_count": True},  # bool passes isinstance(int); must not read as 1
    ],
)
def test_unreadable_port_count_fails_open(spec):
    # Arrange
    job = _make_job(spec=spec if spec is not None else {})
    if spec is None:
        job.spec = None
    remainder = _remainder_of(_build_incentive((SPLIT_HOTKEY, job)))

    # Act / Assert — nobody loses incentive over telemetry
    assert RentalPriceIncentive._port_budget_shortfall(remainder) is None


# ── the scoring decision ──────────────────────────────────────────────────────


async def _score(incentive: RentalPriceIncentive, rental_share: float = 0.1) -> RentalPriceIncentive:
    incentive._calculate_rental_share = AsyncMock(return_value=rental_share)
    await incentive.calculate_mining_scores()
    return incentive


@pytest.mark.asyncio
async def test_default_settings_pay_every_free_gpu_exactly_as_before(caplog):
    # Rollout contract (SO §70): a money-path rule ships in shadow mode. With the settings
    # object untouched, the port-limited node's 4 free GPUs and a plain idle 1-GPU node share
    # the unrented pool 4 : 1 — the computation the rule never touched — and the scorer says
    # so in one shadow line. Precondition, not the assertion: the shipped default is off.
    assert settings.ENABLE_UNRENTED_PORT_BUDGET_FOR_SPLIT_GPUS is False, (
        "ENABLE_UNRENTED_PORT_BUDGET_FOR_SPLIT_GPUS is on in this environment — "
        "the default-off contract cannot be measured here"
    )
    split_job = _make_job(available_port_count=5)
    plain_job = _plain_idle_job()

    # Act
    with caplog.at_level(logging.INFO):
        incentive = await _score(_build_incentive((SPLIT_HOTKEY, split_job), (PLAIN_HOTKEY, plain_job)))

    # Assert — 4 + 1 GPUs in the 1× tier, idle pay in proportion, nothing withheld.
    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == 5
    assert split_job.incentive_idle == pytest.approx(0.1 * 4 / 5)
    assert plain_job.incentive_idle == pytest.approx(0.1 * 1 / 5)
    assert split_job.port_unbacked_gpu_count == 0
    assert "port_unbacked_gpus" not in split_job.full_log_text
    assert split_job.zero_incentive_reasons == []

    # …and the shortfall is still visible: one shadow line with the counts.
    lines = _budget_lines(caplog)
    assert len(lines) == 1
    assert lines[0].extra["enforced"] is False
    assert lines[0].extra["pool"] == "rental_kept_shadow"
    assert (lines[0].extra["backed_gpu_count"], lines[0].extra["unbacked_gpu_count"]) == (1, 3)
    assert "shadow only - flag off" in lines[0].message


@pytest.mark.asyncio
async def test_flag_on_pays_idle_only_for_the_backed_gpus(monkeypatch, caplog):
    # Arrange — the same two nodes with the flag on: the split node's 5 ports back 1 GPU, so it
    # and the plain 1-GPU node now share the pool 1 : 1.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_PORT_BUDGET_FOR_SPLIT_GPUS", True)
    split_job = _make_job(available_port_count=5)
    plain_job = _plain_idle_job()

    # Act
    with caplog.at_level(logging.INFO):
        incentive = await _score(_build_incentive((SPLIT_HOTKEY, split_job), (PLAIN_HOTKEY, plain_job)))

    # Assert — the tier counts the backed GPU only; idle pay halves; the plain node gains.
    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == 2
    assert split_job.incentive_idle == pytest.approx(0.1 * 1 / 2)
    assert plain_job.incentive_idle == pytest.approx(0.1 * 1 / 2)

    # The rented 4 GPUs earn in the mining pool exactly as before, the merged result is whole.
    assert incentive.job_results[SPLIT_HOTKEY] == [split_job]
    assert split_job.gpu_count == 8
    assert split_job.mining_score == pytest.approx(1.0 * 0.3 * 4 / 9)
    assert split_job.incentive == pytest.approx(split_job.incentive_rented + split_job.incentive_idle)
    assert split_job.node_state_at_cycle == "mixed"

    # Paid, for fewer GPUs: a report line in the job log, no zero reason; the formula snapshot
    # carries the unbacked count so the stored numbers reproduce the payout.
    assert split_job.zero_incentive_reasons == []
    assert "port_unbacked_gpus" in split_job.full_log_text
    assert "covers 1 of the 4 free GPU(s)" in split_job.full_log_text
    assert split_job.incentive_formula_inputs["unrented"]["port_unbacked_gpu_count"] == 3
    assert split_job.incentive_formula_inputs["unrented"]["gpu_count"] == 4

    lines = _budget_lines(caplog)
    assert len(lines) == 1
    assert lines[0].extra["enforced"] is True
    assert lines[0].extra["pool"] == "rental_partial"
    assert lines[0].extra["is_split_remainder"] is True
    assert "shadow only" not in lines[0].message


@pytest.mark.asyncio
async def test_flag_on_whole_idle_split_node_is_paid_for_the_backed_gpus(monkeypatch):
    # Arrange — an idle 8-GPU node splitting into 1-GPU pods with 6 open ports (2 pods fit),
    # next to a plain idle 1-GPU node: the 1× tier counts 2 + 1 and the pool splits 2 : 1.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_PORT_BUDGET_FOR_SPLIT_GPUS", True)
    split_job = _make_job(available_port_count=6, is_rented=False, rented_gpu_count=None)
    plain_job = _plain_idle_job()

    # Act
    incentive = await _score(_build_incentive((SPLIT_HOTKEY, split_job), (PLAIN_HOTKEY, plain_job)))

    # Assert
    assert split_job.port_unbacked_gpu_count == 6
    assert split_job.idle_payable_gpu_count == 2
    assert split_job.gpu_count == 8
    assert split_job.incentive_idle == pytest.approx(0.1 * 2 / 3)
    assert plain_job.incentive_idle == pytest.approx(0.1 * 1 / 3)
    assert split_job.incentive_formula_inputs["port_unbacked_gpu_count"] == 6
    assert "covers 2 of the 8 free GPU(s)" in split_job.full_log_text


@pytest.mark.asyncio
async def test_flag_on_node_with_ports_for_every_gpu_is_paid_as_before(monkeypatch, caplog):
    # Arrange — 4 free GPUs, 12 free ports: DAH-2467 behaviour unchanged, no line at all.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_PORT_BUDGET_FOR_SPLIT_GPUS", True)
    split_job = _make_job(available_port_count=4 * MIN_PORT_COUNT)
    plain_job = _plain_idle_job()

    # Act
    with caplog.at_level(logging.INFO):
        incentive = await _score(_build_incentive((SPLIT_HOTKEY, split_job), (PLAIN_HOTKEY, plain_job)))

    # Assert
    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == 5
    assert split_job.incentive_idle == pytest.approx(0.1 * 4 / 5)
    assert split_job.port_unbacked_gpu_count == 0
    assert _budget_lines(caplog) == []
    assert "port_unbacked_gpus" not in split_job.full_log_text


@pytest.mark.asyncio
async def test_remainder_below_the_floor_is_left_to_the_port_floor_gate(monkeypatch, caplog):
    # Arrange — 2 free ports: lium-io#1414's gate withholds all idle pay first; this rule
    # writes no second line and no second job-log entry for the same GPUs.
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_PORT_FLOOR_FOR_SPLIT_REMAINDER", True)
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_PORT_BUDGET_FOR_SPLIT_GPUS", True)
    split_job = _make_job(available_port_count=2)

    # Act
    with caplog.at_level(logging.INFO):
        await _score(_build_incentive((SPLIT_HOTKEY, split_job)))

    # Assert
    assert split_job.incentive_idle == 0.0
    assert [r.reason for r in split_job.zero_incentive_reasons] == [ZeroIncentiveReason.PORT_LIMITED_REMAINDER.value]
    assert _budget_lines(caplog) == []
    assert "port_unbacked_gpus" not in split_job.full_log_text


def test_idle_payable_gpu_count_defaults_to_gpu_count():
    # Every result the rule never touches pays for all its GPUs: the field is 0 by default.
    job = _plain_idle_job(gpu_count=4)
    assert job.port_unbacked_gpu_count == 0
    assert job.idle_payable_gpu_count == 4
