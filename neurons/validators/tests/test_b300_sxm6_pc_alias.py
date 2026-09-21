"""`NVIDIA B300 SXM6 PC` is listed as the B300 SXM6 AC card's alias: same price, family, rate and sizes.

Two providers reported the name on 21 Sep 2026, one of them onboarding an 8-card host: a card that
reports `NVIDIA B300 SXM6 PC` fails `GpuModelValidCheck` and the VRAM precheck because only the AC
spelling is in the tables (NVIDIA's public name table lists only the AC spelling too). The tables
derive the PC entry from the AC entry, so the two cannot diverge; these tests hold that shape: the
PC value equals the AC value in every table, the PC name is never a literal row of its own, and a
spelling nobody added still fails.
"""
from __future__ import annotations

import inspect
import re
from types import ModuleType
from unittest.mock import AsyncMock

import pytest
from incentive import config as incentive_config
from incentive.config import BASE_GPU_MAP, IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from incentive.utils import get_hourly_rate
from services.const import GPU_MODEL_RATES
from services.gpu_precheck import (
    UnsupportedGpuModelError,
    VramRangeMismatchError,
    precheck_gpu_spec,
)
from services.gpu_spec_table import normalize_gpu_model
from services.task.checks.gpu_model_valid import GpuModelValidCheck
from services.task.checks.gpu_vram_precheck import GpuVramPrecheck
from services.task.messages import GpuModelMessages
from services.task_service import JobResult

from services import const, gpu_spec_table
from tests.helpers import build_context_config, build_services, build_state

AC = "NVIDIA B300 SXM6 AC"
PC = "NVIDIA B300 SXM6 PC"
# A third spelling nobody has added: the negative control for every "the PC name passes" test.
UNLISTED = "NVIDIA B300 SXM6 LC"
# What each card of an 8x B300 SXM6 AC host reports through NVML (gpu_spec_table's `observed`).
OBSERVED_NVML_MB = 275040
# A B200's NVML reading: the size a PC label must not be allowed to carry.
B200_NVML_MB = 183359

TABLE_MODULES = [incentive_config, const, gpu_spec_table]


def _tables_naming_the_ac_card(module: ModuleType) -> dict[str, dict]:
    """Every module-level dict keyed by GPU name that carries the AC entry — the tables to keep in step."""
    return {name: value for name, value in vars(module).items() if isinstance(value, dict) and AC in value}


# --- The sync guard: every table with the AC name has the PC name at the same value -----------------
@pytest.mark.parametrize("module", TABLE_MODULES, ids=lambda module: module.__name__)
def test_every_table_naming_the_ac_card_names_the_pc_card_with_the_same_value(module: ModuleType) -> None:
    tables = _tables_naming_the_ac_card(module)
    assert tables, f"{module.__name__} has no table with {AC!r}; the parametrisation is stale"
    for table_name, table in tables.items():
        assert PC in table, f"{module.__name__}.{table_name} has {AC!r} but not {PC!r}"
        assert table[PC] == table[AC], (
            f"{module.__name__}.{table_name}: {PC!r} = {table[PC]!r} but {AC!r} = {table[AC]!r}"
        )


def test_the_guard_reads_the_four_tables_the_validator_decides_from() -> None:
    """The discovery above is not vacuous: it sees the price pin, the family map, the emission rate
    and the VRAM size table. A new table that names the AC card joins the guard by itself."""
    assert {"RENTAL_PRICES_PER_HOUR", "BASE_GPU_MAP"} <= set(_tables_naming_the_ac_card(incentive_config))
    assert {"GPU_MODEL_RATES"} <= set(_tables_naming_the_ac_card(const))
    assert {"GPU_VRAM_SIZES_MB"} <= set(_tables_naming_the_ac_card(gpu_spec_table))


@pytest.mark.parametrize("module", TABLE_MODULES, ids=lambda module: module.__name__)
def test_the_pc_name_is_derived_from_the_ac_row_never_a_literal_of_its_own(module: ModuleType) -> None:
    """A re-price of the AC card has to move the PC name with it, so no table may spell the PC name
    as a key with a value of its own — the alias is an assignment from the AC entry."""
    source = inspect.getsource(module)
    assert not re.search(r'"NVIDIA B300 SXM6 PC"\s*:', source), (
        f"{module.__name__} spells {PC!r} as a literal row; derive it from {AC!r} instead"
    )
    assert re.search(r'\["NVIDIA B300 SXM6 PC"\]\s*=.*\["NVIDIA B300 SXM6 AC"\]', source)


def test_the_pc_name_is_not_normalised_into_the_ac_name() -> None:
    """The executor's native verifier rebuilds `machine_info` from the name NVML reports, so the
    validator must send the driver's own spelling: an alias to the AC name in NORMALIZATION_MAP would
    break the work-proof's key derivation. The PC name is a table row, not an alias."""
    assert normalize_gpu_model(PC) == PC
    assert PC not in gpu_spec_table.NORMALIZATION_MAP


# --- Incentive: one class, one price, one cap ------------------------------------------------------
@pytest.mark.parametrize("gpu_count", [1, 8])
def test_pc_and_ac_resolve_to_the_same_hourly_rate(gpu_count: int) -> None:
    config = IncentiveConfig()
    pc_rate = get_hourly_rate(PC, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour)
    ac_rate = get_hourly_rate(AC, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour)

    assert pc_rate == ac_rate == incentive_config.RENTAL_PRICES_PER_HOUR[AC]


def test_pc_maps_to_the_b300_family_and_its_buckets() -> None:
    config = IncentiveConfig()

    assert BASE_GPU_MAP[PC] == BASE_GPU_MAP[AC] == "B300"
    assert config.max_unrented_gpus[BASE_GPU_MAP[PC]] == {1: 4, 8: 32}
    assert "B300" in config.rental_incentive_gpu_types


async def _run_with_production_config(job_results: dict[str, list[JobResult]]) -> RentalPriceIncentive:
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)

    total_gpus = sum(job.gpu_count for jobs in job_results.values() for job in jobs)
    incentive = RentalPriceIncentive(
        IncentiveConfig(), redis, job_results,
        total_gpu_model_count_map={AC: total_gpus // 2, PC: total_gpus - total_gpus // 2},
    )
    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 500.0
    price_provider.get_alpha_rate.return_value = 0.5
    incentive.price_provider = price_provider

    await incentive.calculate_mining_scores()
    return incentive


@pytest.mark.asyncio
async def test_idle_pc_and_ac_single_cards_share_one_b300_bucket(make_pcc_job) -> None:
    """Six idle 1x AC nodes and six idle 1x PC nodes are twelve B300 cards in the `("B300", 1)`
    bucket: every one is paid 4/12 (DAH-3601's cap of 4). Two buckets, or a PC node the algorithm
    could not place, would show up as a different count or multiplier."""
    jobs = {f"ac_{i}": [make_pcc_job(f"exec-ac-{i}", AC, 1)] for i in range(6)}
    jobs.update({f"pc_{i}": [make_pcc_job(f"exec-pc-{i}", PC, 1)] for i in range(6)})

    incentive = await _run_with_production_config(jobs)

    assert incentive.unrented_count_by_bucket[("B300", 1)] == 12
    assert incentive.cap_multiplier_by_bucket[("B300", 1)] == pytest.approx(4 / 12)
    for jobs_of_miner in jobs.values():
        result = jobs_of_miner[0]
        assert result.count_bucket == 1
        assert result.max_cap == 4
        assert result.hourly_rate == incentive_config.RENTAL_PRICES_PER_HOUR[AC]
        assert result.effective_rate == pytest.approx(result.hourly_rate * 4 / 12)


# --- Verification: the PC name passes the two gates the AC name passes; a third spelling fails ------
def test_precheck_accepts_a_pc_card_reporting_what_an_ac_card_reports() -> None:
    assert precheck_gpu_spec(PC, OBSERVED_NVML_MB) is None
    assert gpu_spec_table.get_expected_vram_windows(PC) == gpu_spec_table.get_expected_vram_windows(AC)
    assert gpu_spec_table.matmul_probe_vram_mb(PC) == gpu_spec_table.matmul_probe_vram_mb(AC)


def test_precheck_still_rejects_a_pc_label_on_a_smaller_card() -> None:
    with pytest.raises(VramRangeMismatchError):
        precheck_gpu_spec(PC, B200_NVML_MB)


def test_a_spelling_nobody_added_still_fails_every_table() -> None:
    assert UNLISTED not in GPU_MODEL_RATES
    assert UNLISTED not in BASE_GPU_MAP
    assert UNLISTED not in IncentiveConfig().rental_prices_per_hour
    with pytest.raises(UnsupportedGpuModelError):
        precheck_gpu_spec(UNLISTED, OBSERVED_NVML_MB)


def _gpu_state(gpu_model: str, gpu_count: int = 8):
    details = [
        {"name": gpu_model, "uuid": f"GPU-{index}", "capacity": OBSERVED_NVML_MB} for index in range(gpu_count)
    ]
    return build_state(
        specs={"gpu": {"count": gpu_count, "details": details}},
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        gpu_details=details,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gpu_model,expected_pass,expected_reason",
    [
        (AC, True, GpuModelMessages.MODEL_OK.reason),
        (PC, True, GpuModelMessages.MODEL_OK.reason),
        (UNLISTED, False, GpuModelMessages.MODEL_UNSUPPORTED.reason),
    ],
)
async def test_gpu_model_valid_check_with_the_production_rates(
    gpu_model: str, expected_pass: bool, expected_reason: str, context_factory
) -> None:
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(gpu_model_rates=GPU_MODEL_RATES),
        state=_gpu_state(gpu_model),
    )

    result = await GpuModelValidCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_model,expected_pass", [(AC, True), (PC, True), (UNLISTED, False)])
async def test_gpu_vram_precheck_on_an_eight_card_host(gpu_model: str, expected_pass: bool, context_factory) -> None:
    ctx = context_factory(state=_gpu_state(gpu_model))

    result = await GpuVramPrecheck().run(ctx)

    assert result.passed is expected_pass
