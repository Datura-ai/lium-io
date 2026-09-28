"""B300 idle pay (Fish, 28 Sep 2026): 1.25 USD/GPU-h, 64 GPUs in the 8× bucket, and a per-model soft limit rate.

Two 8× B300 hosts listed at 12.90 and 12.95 USD/GPU-h were paid idle at 6.40 while Lium's filler
earned 0.84-1.30 on idle B300s. They kept the idle pay under the soft price limit because the
served ceiling is p90 8.637 × soft_limit_price_rate 1.5 = 12.9555. The idle rate drops to what the
filler earns, the 8× bucket doubles, and `UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL` lets one
model's ceiling differ from the shared rate. Every other GPU type keeps its rate, caps and ceiling.
"""

from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG

from core.config import (
    Settings,
    settings,
    shared_client,
    validate_soft_price_limit_rates_only_tighten,
)
import incentive.config as incentive_config
from incentive.config import MAX_UNRENTED_GPUS_BY_TYPE, RENTAL_PRICES_PER_HOUR, IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

B300_AC = "NVIDIA B300 SXM6 AC"
B300_PC = "NVIDIA B300 SXM6 PC"
H200 = "NVIDIA H200"
RTX_PRO_6000_SERVER = "NVIDIA RTX PRO 6000 Blackwell Server Edition"

# Prod shared config, 28 Sep 2026 07:5xZ (GET https://lium.io/api/v1/shared-config).
PROD_B300_P90 = 8.637
PROD_H200_P90 = 3.842
PROD_SOFT_LIMIT_PRICE_RATE = 1.5

# 72 h filler earnings on idle B300 (the only filler on B300), GPU-hours and USD/GPU-h.
FILLER_B300_EARNINGS = {B300_AC: (1006, 1.30), B300_PC: (94, 0.84)}

OTHER_ELIGIBLE_CAPS = {
    "B200": {1: 10, 8: 64},
    "H200": {1: 10, 8: 64},
    "H100": {1: 10, 8: 64},
    "RTX 4090": {1: 10, 8: 64},
    "A100": {1: 10, 8: 64},
    "RTX A6000": {1: 10, 8: 64},
    "RTX 3090": {1: 10, 8: 64},
    "RTX 5090": {1: 10, 8: 64},
    "RTX 6000 Ada Generation": {1: 10, 8: 64},
    "RTX PRO 6000": {1: 10, 8: 64},
    "L40S": {1: 10, 8: 64},
    "L40": {1: 10, 8: 64},
}


def _job(executor_id: str, gpu_model: str, gpu_count: int, price_per_gpu: float | None = None) -> JobResult:
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid=executor_id, address="10.0.0.1", port=8080,
            ssh_username="root", ssh_port=22,
            python_path="/usr/bin/python3", root_dir="/tmp",
            price_per_gpu=price_per_gpu,
        ),
        score=1.0, job_score=1.0, job_batch_id="b300-batch",
        log_status="success", log_text="ok",
        gpu_model=gpu_model, gpu_count=gpu_count, is_rented=False,
        collateral_deposited=True, sysbox_runtime=True,
    )


def _serve_prod_soft_limit(monkeypatch) -> None:
    new_cfg = shared_client.config.model_copy(
        update={
            "machine_prices_p90": {B300_AC: PROD_B300_P90, B300_PC: PROD_B300_P90, H200: PROD_H200_P90},
            "soft_limit_price_rate": PROD_SOFT_LIMIT_PRICE_RATE,
        }
    )
    monkeypatch.setattr(shared_client, "_config", new_cfg)


async def _score(job_results: dict[str, list[JobResult]]) -> RentalPriceIncentive:
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)
    gpu_totals: dict[str, int] = {}
    for jobs in job_results.values():
        for job in jobs:
            gpu_totals[job.gpu_model] = gpu_totals.get(job.gpu_model, 0) + job.gpu_count
    incentive = RentalPriceIncentive(IncentiveConfig(), redis, job_results, total_gpu_model_count_map=gpu_totals)
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 500.0
    price_provider.get_alpha_rate.return_value = 0.5
    incentive.price_provider = price_provider
    await incentive.calculate_mining_scores()
    return incentive


# ── idle rate ───────────────────────────────────────────────────────────────


def test_b300_idle_rate_is_at_or_under_what_the_filler_earns_on_it() -> None:
    gpu_hours = sum(hours for hours, _ in FILLER_B300_EARNINGS.values())
    usd = sum(hours * rate for hours, rate in FILLER_B300_EARNINGS.values())
    weighted_filler_rate = usd / gpu_hours  # 1386.76 / 1100 = 1.2607

    assert weighted_filler_rate == pytest.approx(1.2607, abs=1e-4)
    assert incentive_config.B300_IDLE_USD_PER_GPU_HOUR == 1.25
    assert weighted_filler_rate / incentive_config.B300_IDLE_USD_PER_GPU_HOUR >= 1.0


def test_both_b300_names_are_paid_the_idle_rate_and_no_other_gpu_moves() -> None:
    prices = IncentiveConfig().rental_prices_per_hour
    upstream = DEFAULT_SHARED_CONFIG.machine_prices

    assert prices[B300_AC] == prices[B300_PC] == 1.25
    # the installed lium-core may not carry the PC alias yet; the pin adds it
    moved = {gpu for gpu in upstream if RENTAL_PRICES_PER_HOUR[gpu] != upstream[gpu]}
    assert B300_AC in moved
    assert moved <= {B300_AC, B300_PC, RTX_PRO_6000_SERVER}


# ── 8× bucket ───────────────────────────────────────────────────────────────


def test_b300_8x_bucket_holds_64_gpus_and_the_1x_bucket_stays_at_4() -> None:
    assert IncentiveConfig().max_unrented_gpus["B300"] == {1: 4, 8: 64}


def test_no_other_gpu_type_cap_changes() -> None:
    caps = IncentiveConfig().max_unrented_gpus
    for gpu_type, cap in OTHER_ELIGIBLE_CAPS.items():
        assert caps[gpu_type] == cap, gpu_type
    assert {gpu_type for gpu_type, cap in MAX_UNRENTED_GPUS_BY_TYPE.items() if cap} == set(OTHER_ELIGIBLE_CAPS) | {"B300"}


@pytest.mark.asyncio
async def test_eight_idle_8x_b300_are_paid_in_full_at_1_25() -> None:
    jobs = {f"miner_{i}": [_job(f"exec-8x-{i}", B300_AC, 8)] for i in range(8)}

    incentive = await _score(jobs)

    assert incentive.unrented_count_by_bucket[("B300", 8)] == 64
    assert incentive.cap_multiplier_by_bucket[("B300", 8)] == pytest.approx(1.0)
    for (job,) in jobs.values():
        assert job.max_cap == 64
        assert job.cap_dilution_applied is False
        assert job.hourly_rate == 1.25
        assert job.effective_rate == pytest.approx(1.25)


@pytest.mark.asyncio
async def test_a_ninth_idle_8x_b300_dilutes_the_bucket_to_64_of_72() -> None:
    jobs = {f"miner_{i}": [_job(f"exec-8x-{i}", B300_AC, 8)] for i in range(9)}

    incentive = await _score(jobs)

    assert incentive.cap_multiplier_by_bucket[("B300", 8)] == pytest.approx(64 / 72)
    for (job,) in jobs.values():
        assert job.cap_dilution_applied is True
        assert job.effective_rate == pytest.approx(1.25 * 64 / 72)


@pytest.mark.asyncio
async def test_an_idle_8x_h200_is_still_paid_its_own_rate_under_a_64_gpu_cap() -> None:
    jobs = {f"miner_{i}": [_job(f"exec-h200-{i}", H200, 8)] for i in range(9)}

    incentive = await _score(jobs)

    assert incentive.cap_multiplier_by_bucket[("H200", 8)] == pytest.approx(64 / 72)
    for (job,) in jobs.values():
        assert job.max_cap == 64
        assert job.hourly_rate == DEFAULT_SHARED_CONFIG.machine_prices[H200]


# ── soft price limit ────────────────────────────────────────────────────────


def test_the_12_95_listing_sits_under_the_served_b300_ceiling(monkeypatch) -> None:
    _serve_prod_soft_limit(monkeypatch)
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})

    assert PROD_B300_P90 * PROD_SOFT_LIMIT_PRICE_RATE == pytest.approx(12.9555)
    assert incentive._is_over_soft_price_limit(_job("dublin", B300_AC, 8, 12.95)) is False
    assert incentive._is_over_soft_price_limit(_job("newburyport", B300_AC, 8, 12.90)) is False


@pytest.mark.parametrize("gpu_model", [B300_AC, B300_PC])
def test_a_b300_rate_override_puts_the_12_95_listing_over_its_ceiling(monkeypatch, gpu_model: str) -> None:
    _serve_prod_soft_limit(monkeypatch)
    monkeypatch.setattr(settings, "UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", {"B300": 1.1})
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})

    # ceiling 8.637 × 1.1 = 9.5007
    assert incentive._is_over_soft_price_limit(_job("dublin", gpu_model, 8, 12.95)) is True
    assert incentive._is_over_soft_price_limit(_job("at-ceiling", gpu_model, 8, 9.50)) is False


def test_a_b300_rate_override_leaves_other_gpu_types_on_the_shared_rate(monkeypatch) -> None:
    _serve_prod_soft_limit(monkeypatch)
    monkeypatch.setattr(settings, "UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", {"B300": 1.1})
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})

    # H200 ceiling stays 3.842 × 1.5 = 5.763, not 3.842 × 1.1 = 4.2262
    assert incentive._is_over_soft_price_limit(_job("h200", H200, 8, 5.0)) is False
    assert incentive._is_over_soft_price_limit(_job("h200", H200, 8, 5.77)) is True


@pytest.mark.asyncio
async def test_enforced_b300_override_drops_idle_pay_and_quotes_its_own_rate(monkeypatch) -> None:
    _serve_prod_soft_limit(monkeypatch)
    monkeypatch.setattr(settings, "UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", {"B300": 1.1})
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_SOFT_PRICE_LIMIT", True)
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})

    result = await incentive.calculate_executor_score(_job("dublin", B300_AC, 8, 12.95))

    log = "\n".join(result.incentive_logs)
    assert result.eligible_for_rental_share is False
    assert "x 1.1)" in log


@pytest.mark.asyncio
async def test_b300_override_is_shadow_only_while_the_flag_is_off(monkeypatch) -> None:
    _serve_prod_soft_limit(monkeypatch)
    monkeypatch.setattr(settings, "UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", {"B300": 1.1})
    monkeypatch.setattr(settings, "ENABLE_UNRENTED_SOFT_PRICE_LIMIT", False)
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})

    result = await incentive.calculate_executor_score(_job("dublin", B300_AC, 8, 12.95))

    assert result.eligible_for_rental_share is True


def test_the_override_is_empty_by_default() -> None:
    field = type(settings).model_fields["UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL"]
    assert field.default_factory() == {}


def test_the_override_is_read_from_the_environment_as_json(monkeypatch) -> None:
    monkeypatch.setenv("UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", '{"B300": 1.1}')

    assert Settings().UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL == {"B300": 1.1}


@pytest.mark.parametrize("rate", ["0.0", "-1.1", "NaN", "Infinity", "-Infinity", "1e400"])
def test_a_rate_that_is_not_finite_and_above_0_is_refused(monkeypatch, rate: str) -> None:
    monkeypatch.setenv("UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", f'{{"B300": {rate}}}')

    with pytest.raises(ValueError, match="must be a finite number above 0"):
        Settings()


@pytest.mark.parametrize("key", ["b300", B300_AC, "B301", ""])
def test_a_key_that_is_not_a_base_model_is_refused(monkeypatch, key: str) -> None:
    monkeypatch.setenv("UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", f'{{"{key}": 1.1}}')

    with pytest.raises(ValueError, match="is not a base model"):
        Settings()


@pytest.mark.parametrize("raw", ["", "not json", '["B300", 1.1]', '{"B300": "fast"}'])
def test_a_malformed_override_stops_the_validator(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", raw)

    with pytest.raises(Exception, match="UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL"):
        Settings()


def test_a_json_null_override_is_no_override(monkeypatch) -> None:
    monkeypatch.setenv("UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", "null")

    assert Settings().UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL == {}


def test_the_proposed_b300_value_loads(monkeypatch) -> None:
    monkeypatch.setenv("UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", '{"B300": 1.1, "H200": 1.5}')

    assert Settings().UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL == {"B300": 1.1, "H200": 1.5}


@pytest.mark.parametrize("raw", ['{"B300": true}', '{"B300": false}'])
def test_a_boolean_rate_is_refused(monkeypatch, raw: str) -> None:
    monkeypatch.setenv("UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", raw)

    with pytest.raises(ValueError, match="must be a number"):
        Settings()


@pytest.mark.parametrize("rate", [3.0, 1e308, 1.5000001])
def test_a_rate_above_the_served_rate_stops_the_validator(rate: float) -> None:
    with pytest.raises(ValueError, match="may only tighten"):
        validate_soft_price_limit_rates_only_tighten({"B300": rate}, PROD_SOFT_LIMIT_PRICE_RATE)


@pytest.mark.parametrize(
    "overrides", [{"B300": 1.1}, {"B300": 1.1, "H200": PROD_SOFT_LIMIT_PRICE_RATE}, {}]
)
def test_a_rate_at_or_below_the_served_rate_is_accepted(overrides: dict[str, float]) -> None:
    validate_soft_price_limit_rates_only_tighten(overrides, PROD_SOFT_LIMIT_PRICE_RATE)


@pytest.mark.parametrize(("rate", "price_per_gpu"), [(3.0, 20.00), (1e308, 1e300)])
def test_a_looser_override_that_skipped_the_startup_check_keeps_the_served_ceiling(
    monkeypatch, rate: float, price_per_gpu: float
) -> None:
    _serve_prod_soft_limit(monkeypatch)
    monkeypatch.setattr(settings, "UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", {"B300": rate})
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})

    assert incentive._soft_limit_price_rate(B300_AC) == PROD_SOFT_LIMIT_PRICE_RATE
    assert incentive._is_over_soft_price_limit(_job("dublin", B300_AC, 8, price_per_gpu)) is True


def test_a_served_rate_that_drops_below_the_override_after_startup_wins(monkeypatch) -> None:
    _serve_prod_soft_limit(monkeypatch)
    monkeypatch.setattr(
        shared_client,
        "_config",
        shared_client.config.model_copy(update={"soft_limit_price_rate": 1.0}),
    )
    monkeypatch.setattr(settings, "UNRENTED_SOFT_PRICE_LIMIT_RATE_BY_BASE_MODEL", {"B300": 1.1})
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})

    # ceiling 8.637 × 1.0, not 8.637 × 1.1 = 9.5007
    assert incentive._soft_limit_price_rate(B300_AC) == 1.0
    assert incentive._is_over_soft_price_limit(_job("dublin", B300_AC, 8, 9.00)) is True
