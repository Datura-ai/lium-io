"""DAH-5508: memory-modded cards (RTX 4090 48GB, RTX 4080 / 4080 SUPER 32GB) are their own models, named from VRAM."""

import json

import pytest

from incentive.config import BASE_GPU_MAP, MAX_UNRENTED_GPUS_BY_TYPE, RENTAL_PRICES_PER_HOUR
from services.const import GPU_MODEL_RATES
from services.gpu_precheck import VramRangeMismatchError, precheck_gpu_spec
from services.gpu_spec_table import (
    GPU_VRAM_SIZES_MB,
    MODDED_VRAM_MODELS,
    RTX_4090,
    RTX_4090_48GB,
    classify_gpu_model,
    driver_gpu_name,
    matmul_probe_vram_mb,
)
from services.task.checks.gpu_model_truth import EXPECTED_CC_ONLY
from services.task.checks.machine_spec_scrape import _normalize_gpu_details
from services.task.result_handler import _canonicalize_published_gpu_names

RTX_4080 = "NVIDIA GeForce RTX 4080"
RTX_4080_32GB = "NVIDIA GeForce RTX 4080 32GB"
RTX_4080_SUPER_32GB = "NVIDIA GeForce RTX 4080 SUPER 32GB"


@pytest.mark.parametrize(
    "raw, vram_mb, model",
    [
        (RTX_4090, 49140, RTX_4090_48GB),
        (RTX_4090, 46068, RTX_4090_48GB),
        (RTX_4080, 32607, RTX_4080_32GB),
        ("NVIDIA GeForce RTX 4080 SUPER", 32607, RTX_4080_SUPER_32GB),
    ],
)
def test_a_modded_card_is_its_own_model_and_passes_both_prechecks(raw, vram_mb, model):
    assert classify_gpu_model(raw, vram_mb) == model
    precheck_gpu_spec(model, vram_mb)  # the validator pipeline passes the classified name
    precheck_gpu_spec(raw, vram_mb)  # the provider preflight passes the driver's name


@pytest.mark.parametrize("raw, vram_mb", [(RTX_4090, 24564), (RTX_4090, 23028), (RTX_4080, 16376)])
def test_a_factory_card_is_unchanged(raw, vram_mb):
    assert classify_gpu_model(raw, vram_mb) == raw
    precheck_gpu_spec(raw, vram_mb)


@pytest.mark.parametrize("raw, vram_mb", [(RTX_4090, 36864), (RTX_4090, 32768), (RTX_4090, 65536), (RTX_4080, 24576)])
def test_an_odd_capacity_is_still_rejected(raw, vram_mb):
    with pytest.raises(VramRangeMismatchError):
        precheck_gpu_spec(classify_gpu_model(raw, vram_mb), vram_mb)
    with pytest.raises(VramRangeMismatchError):
        precheck_gpu_spec(raw, vram_mb)


def test_a_card_reporting_a_modded_name_is_judged_by_its_vram():
    assert classify_gpu_model(RTX_4090_48GB, 24564) == RTX_4090
    assert classify_gpu_model(RTX_4090_48GB, 49140) == RTX_4090_48GB
    assert classify_gpu_model(RTX_4080_32GB, 16376) == RTX_4080


def test_other_cards_of_that_memory_are_untouched():
    assert classify_gpu_model("NVIDIA L40", 46068) == "NVIDIA L40"
    assert classify_gpu_model("NVIDIA GeForce RTX 4090 D", 49140) == "NVIDIA GeForce RTX 4090 D"
    assert classify_gpu_model("NVIDIA GeForce RTX 5090", 32607) == "NVIDIA GeForce RTX 5090"


@pytest.mark.parametrize("model", sorted(MODDED_VRAM_MODELS))
def test_every_modded_model_has_its_own_economics_and_checks(model):
    base, size_mb = MODDED_VRAM_MODELS[model]
    family = BASE_GPU_MAP[model]
    assert GPU_MODEL_RATES[model] == 0.0  # no rental history yet
    assert RENTAL_PRICES_PER_HOUR[model] > RENTAL_PRICES_PER_HOUR[base]
    assert family != BASE_GPU_MAP[base]
    assert MAX_UNRENTED_GPUS_BY_TYPE[family] == {1: 10, 8: 64}  # the default tier
    assert EXPECTED_CC_ONLY[model] == "8.9"
    assert driver_gpu_name(model) == base
    assert matmul_probe_vram_mb(model) == round(size_mb * 0.90)  # the all-card probe uses the modded size


@pytest.mark.parametrize("model", sorted(MODDED_VRAM_MODELS))
def test_the_all_card_probe_sizes_a_raw_base_name_by_its_modded_reading(model):
    # review: the probe gets the raw driver name, so a 32 GB claim was challenged as a factory 16 GB card
    base, size_mb = MODDED_VRAM_MODELS[model]
    factory_mb = min(GPU_VRAM_SIZES_MB[base])

    assert matmul_probe_vram_mb(base, size_mb - 12) == round(size_mb * 0.90)
    assert matmul_probe_vram_mb(base, factory_mb - 12) == round(factory_mb * 0.90)


def test_the_standard_cards_keep_their_numbers():
    assert GPU_MODEL_RATES[RTX_4090] == 0.05
    assert RENTAL_PRICES_PER_HOUR[RTX_4090_48GB] == 0.48
    assert RENTAL_PRICES_PER_HOUR[RTX_4080_32GB] == RENTAL_PRICES_PER_HOUR[RTX_4080_SUPER_32GB] == 0.26
    assert MAX_UNRENTED_GPUS_BY_TYPE["RTX 4090"] == {1: 10, 8: 64}
    assert driver_gpu_name(RTX_4090) == RTX_4090


def test_scrape_state_and_published_specs_name_the_modded_card():
    details = [{"name": RTX_4090, "capacity": 49140}, {"name": RTX_4090, "capacity": 24564}]
    expected = [RTX_4090_48GB, RTX_4090]
    assert [d["name"] for d in _normalize_gpu_details(details)] == expected
    published = _canonicalize_published_gpu_names({"gpu": {"details": details}})
    assert [d["name"] for d in published["gpu"]["details"]] == expected


def test_the_native_challenge_is_keyed_on_the_drivers_name():
    from unittest.mock import MagicMock

    from services.matrix_validation_service import ValidationService

    service = ValidationService.__new__(ValidationService)
    service.wrapper = MagicMock()
    service.wrapper.getCipherText.return_value = "cipher"
    spec = {"gpu": {"count": 1, "details": [{"name": RTX_4090_48GB, "capacity": 49140, "uuid": "GPU-1"}]}}

    challenge = service.prepare_matmul_challenge(spec, {})

    sent = service.wrapper.generateChallenge.call_args.args[2]
    assert json.loads(sent)["gpu_model"] == RTX_4090  # what the executor's libdmcompverify reconstructs
    challenge.close()


@pytest.mark.asyncio
async def test_published_estimates_cover_every_modded_model(monkeypatch):
    from incentive import rental_price

    async def fake_estimate(config, redis_service, snapshot, params):
        return {"gpu_model": params.gpu_model}

    monkeypatch.setattr(rental_price, "estimate_executor", fake_estimate)
    estimates = await rental_price.precompute_all_estimates(None, None, None)
    assert set(MODDED_VRAM_MODELS) <= set(estimates)
    assert len(estimates) == len(BASE_GPU_MAP)
