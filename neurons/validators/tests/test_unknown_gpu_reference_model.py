"""DAH-5088: an unknown NVIDIA model passes validation and is scored as its reference model.

The reference is the cheapest model in services/gpu_reference_table.py whose nominal VRAM is
within 0.90-1.10x of the card's reported VRAM, else the cheapest with the largest size below it,
else none (still unsupported).
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import services.gpu_reference_table as gpu_reference_table
from neurons.validators.src.services.task.checks.gpu_model_valid import GpuModelValidCheck
from neurons.validators.src.services.task.messages import GpuModelMessages as Msg
from neurons.validators.src.services.task.result_handler import ResultHandler
from neurons.validators.src.services.task.score_calculator import calculate_scores
from services.const import GPU_MODEL_RATES
from services.gpu_precheck import UnsupportedGpuModelError, precheck_gpu_spec
from services.gpu_reference_table import GPU_REFERENCE_TABLE
from services.gpu_spec_table import (
    GPU_VRAM_SIZES_MB,
    VRAM_FLOOR_RATIO,
    matmul_probe_vram_mb,
    reference_gpu_model,
)

from core.config import shared_client
from tests.helpers import build_context_config, build_services, build_state, default_executor
from tests.test_idle_slot_caps_by_tier import _run_with_production_config
from tests.test_rental_price_incentive_flow import _make_pcc_job

UNKNOWN = "NVIDIA RTX Future 9000"
# lium-platform keeps a byte-identical apps/lium/backend/apps/server/src/core/gpu_reference_table.py
# and pins the same hash
GPU_REFERENCE_TABLE_SHA256 = "032eff33bb4a278f4cbb54ed887da0cd2003ac6cdd2e8d4923b7920cca1b139c"
PRICES = {model: price for model, (price, _) in GPU_REFERENCE_TABLE.items()}
CANDIDATES = list(GPU_REFERENCE_TABLE)


def _in_band(model: str, vram_mb: int) -> bool:
    return any(0.90 * vram_mb <= size <= 1.10 * vram_mb for size in GPU_VRAM_SIZES_MB[model])


def test_the_reference_table_is_the_one_lium_platform_pins():
    table = Path(gpu_reference_table.__file__).read_bytes()
    assert hashlib.sha256(table).hexdigest() == GPU_REFERENCE_TABLE_SHA256


def test_every_candidate_is_a_scored_model_with_the_registry_sizes():
    for model, (_, sizes_gb) in GPU_REFERENCE_TABLE.items():
        assert model in GPU_MODEL_RATES
        assert sorted(size * 1024 for size in sizes_gb) == sorted(GPU_VRAM_SIZES_MB[model])
    assert set(GPU_MODEL_RATES) <= set(GPU_REFERENCE_TABLE)


@pytest.mark.parametrize("vram_mb", [8000, 15352, 23028, 46068, 81559, 140000, 183359, 275040])
def test_an_unknown_card_gets_the_cheapest_known_model_of_its_memory(vram_mb):
    reference = reference_gpu_model(UNKNOWN, vram_mb)
    in_band = [m for m in CANDIDATES if _in_band(m, vram_mb)]
    assert reference in in_band
    cheapest = min(PRICES[m] for m in in_band)
    assert PRICES[reference] == cheapest
    assert reference == min(m for m in in_band if PRICES[m] == cheapest)


# the same cases as lium-platform's src/test/test_unknown_gpu_reference_model.py (MiB, as
# nvidia-smi reports): the backend lists and the validator pays an unknown card as one model
PARITY_CASES = [
    (6144, "NVIDIA GeForce RTX 2060"),
    (11264, "NVIDIA GeForce GTX 1080 Ti"),
    (16303, "NVIDIA Quadro RTX 5000"),
    (20000, "NVIDIA RTX A4500"),
    (24564, "NVIDIA Tesla P40"),
    (30000, "NVIDIA GeForce RTX 5090 D"),
    (40960, "NVIDIA CMP 170HX"),
    (49140, "NVIDIA Quadro RTX 8000"),
    (81559, "NVIDIA A800 80GB PCIe"),
    (97871, "NVIDIA H800 NVL"),
    (143771, "NVIDIA H200 NVL"),
    (183359, "NVIDIA B200"),
    (409600, "NVIDIA B300 SXM6 AC"),
]


@pytest.mark.parametrize(("vram_mb", "reference"), PARITY_CASES)
def test_the_reference_matches_the_backends(vram_mb, reference):
    assert reference_gpu_model(UNKNOWN, vram_mb) == reference


def test_with_none_that_close_it_falls_back_to_the_largest_size_below():
    vram_mb = 400_000
    assert not [m for m in CANDIDATES if _in_band(m, vram_mb)]
    largest = max(size for m in CANDIDATES for size in GPU_VRAM_SIZES_MB[m] if size < vram_mb)
    reference = reference_gpu_model(UNKNOWN, vram_mb)
    assert largest in GPU_VRAM_SIZES_MB[reference]
    at_largest = [m for m in CANDIDATES if largest in GPU_VRAM_SIZES_MB[m]]
    assert PRICES[reference] == min(PRICES[m] for m in at_largest)


@pytest.mark.parametrize(
    "name,vram_mb",
    [
        ("AMD Instinct MI300X", 196_608),
        (UNKNOWN, 1000),
        (UNKNOWN, None),
        (UNKNOWN, 0),
        (None, 24_576),
    ],
)
def test_no_reference_for_a_non_nvidia_name_or_too_little_memory(name, vram_mb):
    assert reference_gpu_model(name, vram_mb) is None


def test_a_known_name_is_its_own_reference_whatever_its_vram():
    assert reference_gpu_model("NVIDIA H100 80GB HBM3", 8000) == "NVIDIA H100 80GB HBM3"
    assert reference_gpu_model("Tesla T4", None) == "NVIDIA T4 Tensor Core GPU"


@pytest.mark.parametrize(("name", "anchor"), [("NVIDIA A100-PCIE-40GB", 0.37), ("NVIDIA A100-SXM4-40GB", 0.47)])
def test_an_a100_40gb_is_paid_as_itself_not_as_an_a100_80gb_or_a_mining_card(name, anchor):
    from incentive.config import BASE_GPU_MAP, MAX_UNRENTED_GPUS_BY_TYPE, IncentiveConfig
    from incentive.utils import get_hourly_rate

    config = IncentiveConfig()
    assert reference_gpu_model(name, 40960) == name
    assert BASE_GPU_MAP[name] == "A100 40GB" and MAX_UNRENTED_GPUS_BY_TYPE["A100 40GB"] == {1: 10, 8: 64}
    for gpu_count in (1, 8):
        assert get_hourly_rate(name, gpu_count, config.gpu_count_custom_prices, config.rental_prices_per_hour) == anchor
    assert PRICES[name] == anchor


def test_precheck_and_matmul_sizing_take_the_reported_vram_of_an_unknown_card():
    precheck_gpu_spec(UNKNOWN, 23028)
    with pytest.raises(UnsupportedGpuModelError):
        precheck_gpu_spec("AMD Instinct MI300X", 196_608)
    assert matmul_probe_vram_mb(UNKNOWN, 23028) == round(23028 * VRAM_FLOOR_RATIO)
    assert matmul_probe_vram_mb(UNKNOWN) is None
    assert matmul_probe_vram_mb("AMD Instinct MI300X", 196_608) is None


def _model_ctx(context_factory, details: list[dict]):
    state = build_state(
        specs={"gpu": {"count": len(details), "details": details}},
        gpu_count=len(details),
        gpu_details=details,
    )
    return context_factory(
        services=build_services(),
        config=build_context_config(gpu_model_rates=GPU_MODEL_RATES),
        state=state,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "details,reason",
    [
        ([{"name": UNKNOWN, "capacity": 23028}] * 2, Msg.MODEL_OK.reason),
        ([{"name": "AMD Instinct MI300X", "capacity": 196_608}], Msg.MODEL_UNSUPPORTED.reason),
        ([{"name": UNKNOWN, "capacity": 1000}], Msg.MODEL_UNSUPPORTED.reason),
        (
            [
                {"name": UNKNOWN, "capacity": 23028},
                {"name": "NVIDIA GeForce RTX 4090", "capacity": 23028},
            ],
            Msg.MODEL_MIXED.reason,
        ),
        (
            [{"name": UNKNOWN, "capacity": 23028}, {"name": UNKNOWN, "capacity": 81559}],
            Msg.MODEL_MIXED.reason,
        ),
    ],
)
async def test_gpu_validate_model_passes_an_unknown_nvidia_card(context_factory, details, reason):
    result = await GpuModelValidCheck().run(_model_ctx(context_factory, details))
    assert result.event.reason_code == reason
    assert result.passed is (reason == Msg.MODEL_OK.reason)
    if result.passed:
        assert result.event.what_we_saw["reference_gpu_model"] == reference_gpu_model(
            UNKNOWN, 23028
        )


@pytest.mark.asyncio
async def test_the_result_is_scored_as_the_reference_and_keeps_the_card_name_in_its_specs(
    context_factory,
):
    details = [{"name": UNKNOWN, "capacity": 23028, "uuid": "GPU-1"}]
    state = build_state(
        specs={"gpu": {"count": 1, "details": details}},
        gpu_details=details,
        gpu_model_count=f"{UNKNOWN}:1",
    )
    ctx = context_factory(state=state)
    result = await ResultHandler(redis_service=MagicMock(), dry_run=True).handle_result(
        context=ctx,
        miner_info=SimpleNamespace(miner_hotkey="hk", job_batch_id="batch"),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="",
        success=True,
    )
    reference = reference_gpu_model(UNKNOWN, 23028)
    assert result.gpu_model == reference
    assert result.spec["gpu"]["details"][0]["name"] == UNKNOWN

    # idle pay and its cap: the unknown card's result is paid exactly as an honest reference card
    jobs = {
        "unknown": [_make_pcc_job("unknown", result.gpu_model, 1)],
        "honest": [_make_pcc_job("honest", reference, 1)],
    }
    await _run_with_production_config(jobs)
    unknown, honest = jobs["unknown"][0], jobs["honest"][0]
    assert unknown.eligible_for_rental_share is honest.eligible_for_rental_share
    assert unknown.effective_rate == honest.effective_rate


def test_an_unknown_card_has_no_hard_price_limit(context_factory):
    # the reference (the cheapest same-memory model) is a payout floor, not a fair price (the backend skips it too)
    details = [{"name": UNKNOWN, "capacity": 23028}]
    reference_limit = (
        PRICES[reference_gpu_model(UNKNOWN, 23028)] * shared_client.config.machine_max_price_rate
    )
    executor = default_executor().model_copy(update={"price_per_gpu": reference_limit * 10})
    state = build_state(
        specs={"network": {"ema_verifyx_download_speed": 500.0}},
        gpu_model=UNKNOWN,
        gpu_details=details,
    )

    assert calculate_scores(context_factory(executor=executor, state=state), rented=True)[0] == 1.0
