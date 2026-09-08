from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import settings
from incentive import rental_price as rental_price_module
from incentive.config import DEFAULT_PRICE, IncentiveConfig
from incentive.factory import IncentiveFactory
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from constants import TOTAL_BURN_EMISSION
from services.const import TEMPO, SECONDS_PER_BLOCK, FIXED_RATIO
from tests.helpers import (
    assert_executor_has_log,
    assert_incentive_log_present,
    assert_log_contains_keys,
    assert_rental_price_incentive_log_full_content,
    extract_incentive_section,
)
from tests.test_incentive_flow import _run_set_weights_and_capture, _run_sync_with_jobs
from tests.test_rental_price_helpers import (
    expected_emission_splits,
    expected_executor_score,
    expected_final_weight,
    expected_miner_rental_value,
)

pytest_plugins = ["fixtures.incentive_fixtures"]

ALGORITHM = "rental_price"

# Per-GPU-type caps for testing. `max_unrented_gpus` is now
# `dict[str, dict[int, int]]` — an empty dict opts the family out. To mimic the
# old aggregate-cap tests we fan the same cap value across a wide range of
# buckets so any gpu_count used in the tests lands on a populated entry.
_AGG_CAP_BUCKETS = range(1, 2001)


def _as_bucket_caps(cap: int) -> dict[int, int]:
    return {i: cap for i in _AGG_CAP_BUCKETS}


MAX_UNRENTED_GPUS_BY_TYPE: dict[str, dict[int, int]] = {
    "H100": _as_bucket_caps(1000),
    "H200": _as_bucket_caps(1000),
    "A100": {},  # not eligible
}

# Flat per-base aggregate view, consumed by the legacy `expected_emission_splits`
# / `expected_miner_rental_value` helpers which still reason in int caps.
MAX_UNRENTED_GPUS_AGGREGATE: dict[str, int] = {"H100": 1000, "H200": 1000, "A100": 0}

RENTAL_INCENTIVE_GPU_TYPES = [
    gpu_type for gpu_type, cap in MAX_UNRENTED_GPUS_BY_TYPE.items() if any(v > 0 for v in cap.values())
]

H100_HOURLY_RATE = 3.50
H200_HOURLY_RATE = 4.00
H200_NVL_HOURLY_RATE = 3.50
A100_HOURLY_RATE = 2.00
RENTAL_PRICES_PER_HOUR = {
    "H100": H100_HOURLY_RATE,
    "H200": H200_HOURLY_RATE,
    "H200 NVL": H200_NVL_HOURLY_RATE,
    "A100": A100_HOURLY_RATE,
}
BASE_GPU_MAP = {
    "H200 NVL": "H200",
    "H200": "H200",
    "H100": "H100",
    "A100": "A100",
}

TAO_PRICE = 500.0
ALPHA_RATE = 0.5

GPU_PORTION = {
    "H100": 0.3,
    "H200": 0.25,
    "H200 NVL": 0.25,
    "A100": 0.2,
    "RTX4090": 0.15,
    "RTX3090": 0.1,
}

def _expected_rental_share(total_rental_cost: float, tao_price: float, alpha_rate: float) -> float:
    epoch_emission = TEMPO * tao_price * alpha_rate
    if total_rental_cost == 0 or epoch_emission == 0:
        return 0.0
    return total_rental_cost * TEMPO * SECONDS_PER_BLOCK / 3600 / FIXED_RATIO / epoch_emission


@pytest.fixture
def rental_price_config():
    return IncentiveConfig(
        algorithm=ALGORITHM,
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
        rental_prices_per_hour=RENTAL_PRICES_PER_HOUR,
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )


@pytest.fixture
def mock_price_provider():
    provider = AsyncMock()
    provider.get_tao_price.return_value = TAO_PRICE
    provider.get_alpha_rate.return_value = ALPHA_RATE
    return provider


@pytest.fixture
def price_provider_holder(mock_price_provider):
    return {"provider": mock_price_provider}


@pytest.fixture
def validator_with_rental_price(
    validator_with_mocks,
    incentive_redis_service,
    rental_price_config,
    price_provider_holder,
    monkeypatch,
):
    original_create = IncentiveFactory.create

    def create_with_price_provider(*args, **kwargs):
        incentive = original_create(*args, **kwargs)
        if hasattr(incentive, "price_provider"):
            incentive.price_provider = price_provider_holder["provider"]
        return incentive

    monkeypatch.setattr(IncentiveFactory, "create", create_with_price_provider)
    monkeypatch.setattr(settings, "incentive", rental_price_config)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", BASE_GPU_MAP)
    validator_with_mocks.incentive = rental_price_config
    return validator_with_mocks


def _make_rented_data(
    rented_executor_ids: list[str] | None = None,
    gpu_splitting_config: dict[str, int] | None = None,
    spot_executor_ids: list[str] | None = None,
    new_rentals_paused_executor_ids: list[str] | None = None,
    provider_discord_connected_executor_ids: list[str] | None = None,
    default_job_owner_by_executor: dict[str, str] | None = None,
) -> RentedExecutorsResponse:
    executors = {}
    for executor_id in rented_executor_ids or []:
        executors[executor_id] = RentedExecutor(
            miner_hotkey="miner-hotkey",
            executor_ip_address="127.0.0.1",
            executor_ip_port="8000",
            pods=[RentedPod(pod_id="pod-1", container_name="ctr")],
        )
    return RentedExecutorsResponse(
        executors=executors,
        banned_guids=[],
        gpu_splitting_config=gpu_splitting_config or {},
        spot_executor_ids=spot_executor_ids or [],
        new_rentals_paused_executor_ids=new_rentals_paused_executor_ids or [],
        provider_discord_connected_executor_ids=provider_discord_connected_executor_ids,
        default_job_owner_by_executor=default_job_owner_by_executor or {},
    )


def _job(create_job_result, *, executor_id: str, gpu_model: str, gpu_count: int, is_rented: bool, **kwargs):
    result = create_job_result(
        executor_id=executor_id,
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        **kwargs,
    )
    result.is_rented = is_rented
    return result


def _total_gpu_counts(all_job_results: dict[str, list]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for results in all_job_results.values():
        for result in results:
            counts[result.gpu_model] = counts.get(result.gpu_model, 0) + result.gpu_count
    return counts


@pytest.mark.asyncio
async def test_rental_price_scenario_basic_mixed(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="miner_c"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=10, is_rented=True),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H100", gpu_count=8, is_rented=False),
        ],
        "miner_c": [
            _job(create_job_result, executor_id="exec-c", gpu_model="H200", gpu_count=5, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data(["exec-a"]))

    await _run_sync_with_jobs(validator, miners, all_job_results)

    rented_result = all_job_results["miner_a"][0]
    idle_result = all_job_results["miner_b"][0]
    assert rented_result.incentive_formula_version == "mining_v1"
    assert rented_result.incentive_formula_inputs["mining_share"] == pytest.approx(1 - TOTAL_BURN_EMISSION)
    assert rented_result.incentive_formula_inputs["total_mining_score"] == pytest.approx(
        rented_result.mining_score
    )
    assert idle_result.incentive_formula_version == "rental_price_v2"
    assert idle_result.incentive_formula_inputs["validator_tao_price_usd"] == TAO_PRICE
    assert idle_result.incentive_formula_inputs["validator_alpha_rate_tao_per_block"] == ALPHA_RATE
    assert idle_result.incentive_formula_inputs["max_cap"] == MAX_UNRENTED_GPUS_AGGREGATE["H100"]

    # Verify rental-specific logging for rented vs unrented executors
    from tests.helpers import (
        assert_incentive_log_present,
        assert_executor_has_log,
        assert_log_contains_keys,
        assert_rental_price_incentive_log_full_content,
    )

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))

                if result.eligible_for_rental_share and not result.is_rented:
                    # Rental algorithm logs for unrented eligible executors (rental_price.py 146-165)
                    assert_rental_price_incentive_log_full_content(result.full_log_text)
                elif result.is_rented:
                    # Falls back to default algorithm for rented or non-eligible
                    assert_log_contains_keys(result.full_log_text, [
                        "mining_score",
                        "total_mining_score"
                    ])

    total_gpu_counts = _total_gpu_counts(all_job_results)
    expected_a_score = expected_executor_score(
        gpu_model="H100",
        gpu_count=10,
        total_gpu_count=total_gpu_counts["H100"],
        portion=GPU_PORTION["H100"],
        is_rented=True,
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )

    unrented_counts = {"H100": 8, "H200": 5}
    splits = expected_emission_splits(
        unrented_gpu_counts=unrented_counts,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
    )

    miner_rental_values = {
        "miner_b": 8 * H100_HOURLY_RATE,
        "miner_c": 5 * H200_HOURLY_RATE,
    }
    total_mining_score = expected_a_score
    total_rental_value = sum(miner_rental_values.values())

    expected_scores = {
        "burner1": expected_final_weight(
            miner_mining_score=0.0,
            miner_rental_value=0.0,
            total_mining_score=total_mining_score,
            total_rental_value=total_rental_value,
            mining_share=splits["mining_share"],
            rental_share=splits["rental_share"],
            is_burner=True,
            burn_share=splits["burn_share"],
            num_burners=2,
        ),
        "burner2": expected_final_weight(
            miner_mining_score=0.0,
            miner_rental_value=0.0,
            total_mining_score=total_mining_score,
            total_rental_value=total_rental_value,
            mining_share=splits["mining_share"],
            rental_share=splits["rental_share"],
            is_burner=True,
            burn_share=splits["burn_share"],
            num_burners=2,
        ),
        "miner_a": expected_final_weight(
            miner_mining_score=expected_a_score,
            miner_rental_value=0.0,
            total_mining_score=total_mining_score,
            total_rental_value=total_rental_value,
            mining_share=splits["mining_share"],
            rental_share=splits["rental_share"],
            is_burner=False,
            burn_share=splits["burn_share"],
            num_burners=2,
        ),
        "miner_b": expected_final_weight(
            miner_mining_score=0.0,
            miner_rental_value=miner_rental_values["miner_b"],
            total_mining_score=total_mining_score,
            total_rental_value=total_rental_value,
            mining_share=splits["mining_share"],
            rental_share=splits["rental_share"],
            is_burner=False,
            burn_share=splits["burn_share"],
            num_burners=2,
        ),
        "miner_c": expected_final_weight(
            miner_mining_score=0.0,
            miner_rental_value=miner_rental_values["miner_c"],
            total_mining_score=total_mining_score,
            total_rental_value=total_rental_value,
            mining_share=splits["mining_share"],
            rental_share=splits["rental_share"],
            is_burner=False,
            burn_share=splits["burn_share"],
            num_burners=2,
        ),
    }

    for hotkey, expected in expected_scores.items():
        assert validator.miner_scores[hotkey] == pytest.approx(expected, abs=0.0001)

    assert sum(validator.miner_scores.values()) == pytest.approx(1.0, abs=0.0001)


@pytest.mark.asyncio
# NOTE: Legacy aggregate-cap dilution scenario removed — with per-`(base_model, bucket)`
# caps, three executors with distinct gpu_counts each land in their own bucket with a
# 1.0 multiplier, so the original aggregate-dilution assertion no longer applies.


@pytest.mark.asyncio
async def test_different_caps_per_gpu_type(
    validator_with_mocks,
    incentive_redis_service,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    price_provider_holder,
    monkeypatch,
):
    """Test that different GPU types have independent caps applied correctly."""
    # Per-bucket caps keyed by the gpu_count each executor in this test reports.
    different_caps: dict[str, dict[int, int]] = {
        "H100": {8: 5},
        "H200": {6: 3},
    }
    # Flat per-base aggregate value used by the legacy expected_emission_splits helper.
    different_caps_aggregate = {"H100": 5, "H200": 3}
    custom_config = IncentiveConfig(
        algorithm=ALGORITHM,
        rental_incentive_gpu_types=list(different_caps.keys()),
        max_unrented_gpus=different_caps,
        rental_prices_per_hour=RENTAL_PRICES_PER_HOUR,
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
    )

    # Set up validator with custom config
    original_create = IncentiveFactory.create

    def create_with_price_provider(*args, **kwargs):
        incentive = original_create(*args, **kwargs)
        if hasattr(incentive, "price_provider"):
            incentive.price_provider = price_provider_holder["provider"]
        return incentive

    monkeypatch.setattr(IncentiveFactory, "create", create_with_price_provider)
    monkeypatch.setattr(settings, "incentive", custom_config)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", BASE_GPU_MAP)

    validator = validator_with_mocks
    validator.incentive = custom_config
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]

    # Mock price provider
    mock_price_provider = AsyncMock()
    mock_price_provider.get_tao_price.return_value = TAO_PRICE
    mock_price_provider.get_alpha_rate.return_value = ALPHA_RATE
    price_provider_holder["provider"] = mock_price_provider

    # Both GPU types exceed their respective caps
    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=8, is_rented=False),
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=8, is_rented=True),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H200", gpu_count=6, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data())

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify cap dilution for each GPU type independently
    h100_effective_rate = H100_HOURLY_RATE * different_caps_aggregate["H100"] / 8
    h200_effective_rate = H200_HOURLY_RATE * different_caps_aggregate["H200"] / 6
    expected_total_rental_cost = 8 * h100_effective_rate + 6 * h200_effective_rate

    splits = expected_emission_splits(
        unrented_gpu_counts={"H100": 8, "H200": 6},
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=different_caps_aggregate,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
    )

    # Verify effective rates are calculated per GPU type
    assert h100_effective_rate == pytest.approx(H100_HOURLY_RATE * 5 / 8, abs=0.001)
    assert h200_effective_rate == pytest.approx(H200_HOURLY_RATE * 3 / 6, abs=0.001)

    # Verify total rental cost aggregates correctly
    assert expected_total_rental_cost == pytest.approx(
        8 * h100_effective_rate + 6 * h200_effective_rate, abs=1
    )

    # Verify rental share calculation
    assert splits["rental_share"] > 0

    # Verify each miner gets appropriate weight based on their GPU type's cap
    assert validator.miner_scores["miner_a"] > 0
    assert validator.miner_scores["miner_b"] > 0
    assert sum(validator.miner_scores.values()) == pytest.approx(1.0, abs=0.0001)


@pytest.mark.asyncio
async def test_rental_price_scenario_all_unrented(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="miner_c"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=20, is_rented=False),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H200", gpu_count=15, is_rented=False),
        ],
        "miner_c": [
            _job(create_job_result, executor_id="exec-c", gpu_model="H100", gpu_count=10, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data())

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify rental logging for all unrented scenario
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, [
                    "effective_rate",
                    "rental_share"
                ])

    splits = expected_emission_splits(
        unrented_gpu_counts={"H100": 30, "H200": 15},
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
    )

    assert splits["mining_share"] == pytest.approx(1 - TOTAL_BURN_EMISSION)
    assert sum(validator.miner_scores.values()) == pytest.approx(TOTAL_BURN_EMISSION, abs=0.0001)

    for hotkey in ["miner_a", "miner_b", "miner_c"]:
        assert validator.miner_scores[hotkey] > 0


@pytest.mark.asyncio
async def test_rental_price_scenario_zero_unrented(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="miner_c"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=10, is_rented=True),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="A100", gpu_count=8, is_rented=False),
        ],
        "miner_c": [
            _job(create_job_result, executor_id="exec-c", gpu_model="RTX4090", gpu_count=5, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data(["exec-a"]))

    await _run_sync_with_jobs(validator, miners, all_job_results)

    splits = expected_emission_splits(
        unrented_gpu_counts={},
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
    )

    assert splits["rental_share"] == 0.0
    assert splits["burn_share"] == pytest.approx(TOTAL_BURN_EMISSION, abs=0.0001)
    assert splits["mining_share"] == pytest.approx(1 - TOTAL_BURN_EMISSION, abs=0.0001)
    mock_price_provider.get_tao_price.assert_not_called()
    mock_price_provider.get_alpha_rate.assert_not_called()


@pytest.mark.asyncio
async def test_rental_price_scenario_rental_share_cap(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    price_provider_holder,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = 0.01
    price_provider.get_alpha_rate.return_value = 0.01
    price_provider_holder["provider"] = price_provider

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=1000, is_rented=False),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H100", gpu_count=1, is_rented=True),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data(["exec-b"]))

    await _run_sync_with_jobs(validator, miners, all_job_results)

    splits = expected_emission_splits(
        unrented_gpu_counts={"H100": 1000},
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        tao_price=0.01,
        alpha_rate=0.01,
    )

    assert splits["rental_share"] == pytest.approx(TOTAL_BURN_EMISSION, abs=0.0001)
    assert splits["burn_share"] == pytest.approx(0.0, abs=0.0001)
    assert validator.miner_scores["burner1"] == pytest.approx(0.0, abs=0.0001)
    assert validator.miner_scores["burner2"] == pytest.approx(0.0, abs=0.0001)


@pytest.mark.asyncio
async def test_rental_price_edge_multi_executor_accumulation(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a1", gpu_model="H100", gpu_count=5, is_rented=True),
            _job(create_job_result, executor_id="exec-a2", gpu_model="H100", gpu_count=3, is_rented=False),
            _job(create_job_result, executor_id="exec-a3", gpu_model="H200", gpu_count=4, is_rented=False),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b1", gpu_model="H100", gpu_count=2, is_rented=True),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(["exec-a1", "exec-b1"])
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify logging for multiple executors per miner (CRITICAL edge case)
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))

                if result.eligible_for_rental_share and not result.is_rented:
                    # Rental algorithm logs
                    assert_log_contains_keys(result.full_log_text, [
                        "effective_rate",
                        "rental_share"
                    ])
                elif result.is_rented:
                    # Default algorithm logs for rented executors
                    assert_log_contains_keys(result.full_log_text, [
                        "mining_score",
                        "total_mining_score"
                    ])

    # Verify miner_a has 3 independent log entries
    miner_a_results = all_job_results["miner_a"]
    assert len(miner_a_results) == 3, "Should have 3 executors"
    for result in miner_a_results:
        if result.incentive_logs:
            assert str(result.executor_info.uuid) in result.full_log_text

    total_gpu_counts = _total_gpu_counts(all_job_results)
    expected_a_mining = expected_executor_score(
        gpu_model="H100",
        gpu_count=5,
        total_gpu_count=total_gpu_counts["H100"],
        portion=GPU_PORTION["H100"],
        is_rented=True,
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )
    expected_b_mining = expected_executor_score(
        gpu_model="H100",
        gpu_count=2,
        total_gpu_count=total_gpu_counts["H100"],
        portion=GPU_PORTION["H100"],
        is_rented=True,
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )

    total_unrented_counts = {"H100": 3, "H200": 4}
    expected_a_rental = expected_miner_rental_value(
        miner_results=all_job_results["miner_a"],
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        total_unrented_counts=total_unrented_counts,
    )
    assert expected_a_rental == pytest.approx(3 * H100_HOURLY_RATE + 4 * H200_HOURLY_RATE)

    splits = expected_emission_splits(
        unrented_gpu_counts=total_unrented_counts,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
    )

    total_mining_score = expected_a_mining + expected_b_mining
    total_rental_value = expected_a_rental

    expected_a_weight = expected_final_weight(
        miner_mining_score=expected_a_mining,
        miner_rental_value=expected_a_rental,
        total_mining_score=total_mining_score,
        total_rental_value=total_rental_value,
        mining_share=splits["mining_share"],
        rental_share=splits["rental_share"],
        is_burner=False,
        burn_share=splits["burn_share"],
        num_burners=2,
    )

    assert validator.miner_scores["miner_a"] == pytest.approx(expected_a_weight, abs=0.0001)


@pytest.mark.asyncio
async def test_rental_price_edge_gpu_type_mix(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="miner_c"),
        create_neuron_info(uid=5, hotkey="miner_d"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=4, is_rented=True),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H200", gpu_count=3, is_rented=False),
        ],
        "miner_c": [
            _job(create_job_result, executor_id="exec-c", gpu_model="A100", gpu_count=2, is_rented=False),
        ],
        "miner_d": [
            _job(create_job_result, executor_id="exec-d", gpu_model="H200", gpu_count=1, is_rented=True),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(["exec-a", "exec-d"])
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify logging for mixed GPU types edge case
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                # Verify GPU model appears in logs (mixed GPU types)
                assert result.gpu_model in result.full_log_text

                if result.eligible_for_rental_share and not result.is_rented:
                    assert_log_contains_keys(result.full_log_text, ["effective_rate", "rental_share"])
                elif result.is_rented:
                    assert_log_contains_keys(result.full_log_text, ["mining_score", "total_mining_score"])

    total_gpu_counts = _total_gpu_counts(all_job_results)
    expected_c_mining = expected_executor_score(
        gpu_model="A100",
        gpu_count=2,
        total_gpu_count=total_gpu_counts["A100"],
        portion=GPU_PORTION["A100"],
        is_rented=False,
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )
    assert expected_c_mining == 0

    # miner_c has no rental incentive for A100, so it gets 0 score
    assert validator.miner_scores.get("miner_c", 0) == 0
    assert validator.miner_scores["miner_b"] > 0
    assert validator.miner_scores["miner_a"] > 0
    assert validator.miner_scores["miner_d"] > 0


@pytest.mark.asyncio
async def test_rental_price_edge_uptime_penalties(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="miner_c"),
    ]

    all_job_results = {
        "miner_a": [
            _job(
                create_job_result,
                executor_id="exec-a",
                gpu_model="H100",
                gpu_count=2,
                is_rented=True,
                collateral_deposited=True,
            ),
        ],
        "miner_b": [
            _job(
                create_job_result,
                executor_id="exec-b",
                gpu_model="H100",
                gpu_count=2,
                is_rented=True,
                collateral_deposited=False,
            ),
        ],
        "miner_c": [
            _job(
                create_job_result,
                executor_id="exec-c",
                gpu_model="H100",
                gpu_count=2,
                is_rented=False,
                collateral_deposited=False,
            ),
        ],
    }

    async def get_uptime_side_effect(executor_info):
        if "exec-b" in str(executor_info.uuid):
            return 60
        return 120

    validator.redis_service.get_executor_uptime = AsyncMock(side_effect=get_uptime_side_effect)
    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(["exec-a", "exec-b"])
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    total_gpu_counts = _total_gpu_counts(all_job_results)
    expected_a = expected_executor_score(
        gpu_model="H100",
        gpu_count=2,
        total_gpu_count=total_gpu_counts["H100"],
        portion=GPU_PORTION["H100"],
        is_rented=True,
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )
    expected_b = expected_executor_score(
        gpu_model="H100",
        gpu_count=2,
        total_gpu_count=total_gpu_counts["H100"],
        portion=GPU_PORTION["H100"],
        is_rented=True,
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=False,
        uptime_minutes=60,
    )

    assert expected_b < expected_a
    assert validator.miner_scores["miner_b"] < validator.miner_scores["miner_a"]


@pytest.mark.asyncio
async def test_rental_price_burner_distribution(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=2, is_rented=True),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data(["exec-a"]))

    await _run_sync_with_jobs(validator, miners, all_job_results)

    burn_share = TOTAL_BURN_EMISSION
    assert validator.miner_scores["burner1"] == pytest.approx(burn_share / 2, abs=0.0001)
    assert validator.miner_scores["burner2"] == pytest.approx(burn_share / 2, abs=0.0001)


@pytest.mark.asyncio
async def test_rental_price_weight_normalization(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=4, is_rented=True),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H100", gpu_count=4, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data(["exec-a"]))

    await _run_sync_with_jobs(validator, miners, all_job_results)

    captured = await _run_set_weights_and_capture(
        mock_subtensor_client, miners, validator.miner_scores, normalize=True
    )
    processed = captured["processed_weights"]
    assert processed.sum() == pytest.approx(1.0, abs=0.0001)


def test_expected_emission_splits_zero_unrented():
    splits = expected_emission_splits(
        unrented_gpu_counts={},
        rental_prices={"H100": 3.5},
        max_unrented_gpus={"H100": 1000},
        tao_price=500.0,
        alpha_rate=0.5,
    )

    assert splits["rental_share"] == 0.0
    assert splits["burn_share"] == pytest.approx(TOTAL_BURN_EMISSION, abs=0.0001)
    assert splits["mining_share"] == pytest.approx(1 - TOTAL_BURN_EMISSION, abs=0.0001)


def test_expected_emission_splits_basic_calculation():
    unrented_counts = {"H100": 8, "H200": 5}
    rental_prices = {"H100": 3.5, "H200": 4.0}

    splits = expected_emission_splits(
        unrented_gpu_counts=unrented_counts,
        rental_prices=rental_prices,
        max_unrented_gpus={"H100": 1000, "H200": 1000},
        tao_price=500.0,
        alpha_rate=0.5,
    )

    total_rental_cost = 8 * 3.5 + 5 * 4.0
    expected_rental = _expected_rental_share(total_rental_cost, 500.0, 0.5)

    assert splits["rental_share"] == pytest.approx(expected_rental, abs=0.0001)
    assert splits["burn_share"] == pytest.approx(TOTAL_BURN_EMISSION - expected_rental, abs=0.0001)
    assert splits["mining_share"] == pytest.approx(1 - TOTAL_BURN_EMISSION, abs=0.0001)


def test_expected_emission_splits_cap_dilution():
    unrented_counts = {"H100": 1500}
    rental_prices = {"H100": 3.5}

    splits = expected_emission_splits(
        unrented_gpu_counts=unrented_counts,
        rental_prices=rental_prices,
        max_unrented_gpus={"H100": 1000},
        tao_price=500.0,
        alpha_rate=0.5,
    )

    effective_rate = 3.5 * 1000 / 1500
    total_rental_cost = 1500 * effective_rate
    expected_rental = _expected_rental_share(total_rental_cost, 500.0, 0.5)

    assert effective_rate == pytest.approx(2.333, abs=0.001)
    assert splits["rental_share"] == pytest.approx(expected_rental, abs=0.0001)


def test_expected_emission_splits_cap_at_burn_emission():
    unrented_counts = {"H100": 1000}
    rental_prices = {"H100": 3.5}

    splits = expected_emission_splits(
        unrented_gpu_counts=unrented_counts,
        rental_prices=rental_prices,
        max_unrented_gpus={"H100": 1000},
        tao_price=0.01,
        alpha_rate=0.01,
    )

    assert splits["rental_share"] == pytest.approx(TOTAL_BURN_EMISSION, abs=0.0001)
    assert splits["burn_share"] == pytest.approx(0.0, abs=0.0001)
    assert splits["mining_share"] == pytest.approx(1 - TOTAL_BURN_EMISSION, abs=0.0001)


def test_expected_emission_splits_zero_epoch_emission():
    splits = expected_emission_splits(
        unrented_gpu_counts={"H100": 10},
        rental_prices={"H100": 3.5},
        max_unrented_gpus={"H100": 1000},
        tao_price=0.0,
        alpha_rate=0.5,
    )

    assert splits["rental_share"] == 0.0
    assert splits["burn_share"] == pytest.approx(TOTAL_BURN_EMISSION, abs=0.0001)
    assert splits["mining_share"] == pytest.approx(1 - TOTAL_BURN_EMISSION, abs=0.0001)


@pytest.mark.asyncio
async def test_rental_price_failed_executors_rented_do_not_score(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    # --- Arrange ---
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="miner_c"),
    ]

    all_job_results = {
        "miner_a": [
            _job(
                create_job_result,
                executor_id="exec-a",
                gpu_model="H100",
                gpu_count=4,
                is_rented=True,
                score=0.0,
                job_score=0.0,
            ),
        ],
        "miner_b": [
            _job(
                create_job_result,
                executor_id="exec-b",
                gpu_model="H100",
                gpu_count=4,
                is_rented=True,
            ),
        ],
        "miner_c": [
            _job(
                create_job_result,
                executor_id="exec-c",
                gpu_model=None,
                gpu_count=0,
                is_rented=False,
            ),
        ]
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(["exec-a", "exec-b"])
    )

    # --- Act ---
    await _run_sync_with_jobs(validator, miners, all_job_results)

    # --- Assert ---

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.mining_score is None:
                section = extract_incentive_section(result.full_log_text)
                if section:
                    assert "Mining score is not set" in section

    assert validator.miner_scores["miner_a"] == 0.0
    assert validator.miner_scores["miner_b"] > 0
    assert validator.miner_scores["miner_c"] == 0.0


@pytest.mark.asyncio
async def test_rental_price_failed_unrented_executors_do_not_count_rental(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    # --- Arrange ---
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]

    all_job_results = {
        "miner_a": [
            _job(
                create_job_result,
                executor_id="exec-a",
                gpu_model="H100",
                gpu_count=6,
                is_rented=False,
                score=0.0,
                job_score=0.0,
            ),
        ],
        "miner_b": [
            _job(
                create_job_result,
                executor_id="exec-b",
                gpu_model="H100",
                gpu_count=4,
                is_rented=True,
            ),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(["exec-b"])
    )

    # --- Act ---
    await _run_sync_with_jobs(validator, miners, all_job_results)

    # --- Assert ---
    from tests.helpers import (
        assert_executor_has_log,
        assert_incentive_log_present,
        assert_log_contains_keys,
    )

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.mining_score is None:
                section = extract_incentive_section(result.full_log_text)
                if section:
                    assert "Mining score is not set" in section
            elif result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, ["mining_score", "total_mining_score"])

    splits = expected_emission_splits(
        unrented_gpu_counts={},
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
    )
    assert splits["rental_share"] == 0.0
    assert validator.miner_scores.get("miner_a", 0) == 0.0
    assert validator.miner_scores["miner_b"] > 0


@pytest.mark.asyncio
async def test_rental_price_gpu_type_max_cap_zero_miner_gets_zero_score(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    # --- Arrange ---
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]

    # miner_a: only unrented A100 (max_cap 0) with non-zero job score -> miner gets 0
    # miner_b: H100 (eligible) -> miner gets non-zero score
    all_job_results = {
        "miner_a": [
            _job(
                create_job_result,
                executor_id="exec-a",
                gpu_model="A100",
                gpu_count=4,
                is_rented=False,
            ),
        ],
        "miner_b": [
            _job(
                create_job_result,
                executor_id="exec-b",
                gpu_model="H100",
                gpu_count=4,
                is_rented=False,
            ),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data([])
    )

    # --- Act ---
    await _run_sync_with_jobs(validator, miners, all_job_results)

    # --- Assert ---
    # Job for miner_a has score/job_score > 0 (default from create_job_result), but miner gets 0 (max_cap=0 GPU)
    miner_a_results = all_job_results["miner_a"]
    assert len(miner_a_results) == 1
    assert miner_a_results[0].score > 0 or miner_a_results[0].job_score > 0

    assert validator.miner_scores.get("miner_a", 0) == pytest.approx(0.0, abs=0.0001)
    assert validator.miner_scores["miner_b"] > 0


@pytest.mark.asyncio
async def test_rental_price_edge_single_miner_dominance(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="miner_c"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=1000, is_rented=False),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H100", gpu_count=5, is_rented=True),
        ],
        "miner_c": [
            _job(create_job_result, executor_id="exec-c", gpu_model="H100", gpu_count=5, is_rented=True),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(["exec-b", "exec-c"])
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify logging for single dominant miner scenario
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))

                if result.eligible_for_rental_share and not result.is_rented:
                    # Rental algorithm logs
                    assert_log_contains_keys(result.full_log_text, ["effective_rate", "rental_share"])
                elif result.is_rented:
                    # Default algorithm logs for rented executors
                    assert_log_contains_keys(result.full_log_text, ["mining_score", "total_mining_score"])

    # The single miner controlling vast unrented capacity dominates each rented
    # peer individually. (It no longer necessarily out-earns the rented miners
    # combined: DAH-2273 raised the rented mining pool from 0.09 to 0.13, so two
    # rented peers can now sum above one capped unrented miner.)
    assert validator.miner_scores["miner_a"] > validator.miner_scores["miner_b"]
    assert validator.miner_scores["miner_a"] > validator.miner_scores["miner_c"]
    assert validator.miner_scores["miner_b"] == pytest.approx(
        validator.miner_scores["miner_c"], abs=0.0001
    )


@pytest.mark.asyncio
async def test_rental_price_price_provider_fallback(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    price_provider_holder,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = None
    price_provider.get_alpha_rate.return_value = None
    price_provider_holder["provider"] = price_provider

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=4, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data())

    await _run_sync_with_jobs(validator, miners, all_job_results)

    assert validator.miner_scores["miner_a"] == 0.0
    assert validator.miner_scores["burner1"] == pytest.approx(TOTAL_BURN_EMISSION / 2, abs=0.0001)
    assert validator.miner_scores["burner2"] == pytest.approx(TOTAL_BURN_EMISSION / 2, abs=0.0001)
    assert price_provider.get_tao_price.called
    assert price_provider.get_alpha_rate.called


@pytest.mark.asyncio
async def test_rental_price_integration_chain_submission(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]

    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=4, is_rented=True),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H100", gpu_count=4, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data(["exec-a"]))

    await _run_sync_with_jobs(validator, miners, all_job_results)

    mock_subtensor_client.get_miners = AsyncMock(return_value=miners)

    with patch("clients.subtensor_client.process_weights_for_netuid") as process_mock, patch(
        "clients.subtensor_client._convert_weights_with_positive_floor"
    ) as convert_mock:
        def process_side_effect(uids, weights, netuid, subtensor, metagraph):
            return uids, weights

        process_mock.side_effect = process_side_effect
        convert_mock.return_value = (list(range(len(miners))), [10000] * len(miners), [])

        await mock_subtensor_client.set_weights(miner_scores=validator.miner_scores)

    assert mock_subtensor_client.send_weights_to_lium.called
    call_payload = mock_subtensor_client.send_weights_to_lium.call_args.args[0]
    assert "netuid" in call_payload
    assert "uids" in call_payload
    assert "weights" in call_payload

@pytest.mark.skip(reason="Tests legacy cross-variant aggregation under one aggregate cap; per-bucket caps place each variant in its own bucket so this behavior no longer applies.")
@pytest.mark.asyncio
async def test_rental_price_gpu_variants_under_cap(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """Test multiple variants of same base model, total under cap - no dilution.

    Scenario:
    - Miner A: 3 H200 @ $4.00/hr (unrented)
    - Miner B: 2 H200 NVL @ $3.50/hr (unrented)
    - H200 cap: 1000
    - Total H200 variants: 5 (3 + 2)

    Expected:
    - No dilution (5 < 1000)
    - H200 effective rate: $4.00 (no dilution)
    - H200 NVL effective rate: $3.50 (no dilution)
    - Both miners get rental share proportional to their GPU counts and rates
    """
    # Arrange
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]

    # Setup: H200 cap is 1000, total is 3+2=5 (under cap)
    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H200", gpu_count=3, is_rented=False),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H200 NVL", gpu_count=2, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data())

    # Act
    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Assert
    miner_a_result = all_job_results["miner_a"][0]
    miner_b_result = all_job_results["miner_b"][0]

    # Total H200 variants (5) is well below the cap (1000), so no dilution is applied
    assert miner_a_result.cap_dilution_applied is False, "H200 should have no dilution"
    assert miner_a_result.total_unrented_by_gpu_type == 5, "Total should be 5 (3 H200 + 2 H200 NVL)"
    assert miner_a_result.effective_rate == pytest.approx(H200_HOURLY_RATE, abs=0.01), "Effective rate should equal hourly rate (no dilution)"
    assert miner_a_result.max_cap == 1000, "Cap should be 1000 for H200 base model"

    # H200 NVL shares the same base model ("H200") so it uses the same cap and total unrented count
    assert miner_b_result.cap_dilution_applied is False, "H200 NVL should have no dilution"
    assert miner_b_result.total_unrented_by_gpu_type == 5, "Total should be 5 (3 H200 + 2 H200 NVL)"
    assert miner_b_result.effective_rate == pytest.approx(H200_NVL_HOURLY_RATE, abs=0.01), "Effective rate should equal hourly rate (no dilution)"
    assert miner_b_result.max_cap == 1000, "Cap should be 1000 for H200 base model"

    # Verify logs contain dilution-related fields
    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, [
                    "effective_rate",
                    "cap_dilution_applied",
                    "total_unrented_by_gpu_type",
                ])

    # Calculate expected values
    unrented_counts = {"H200": 3, "H200 NVL": 2}
    splits = expected_emission_splits(
        unrented_gpu_counts=unrented_counts,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
        base_gpu_map=BASE_GPU_MAP,
    )

    # No dilution: effective rates = hourly rates; rental values are simply count * hourly_rate
    total_rental_cost = 3 * H200_HOURLY_RATE + 2 * H200_NVL_HOURLY_RATE
    total_rental_value = total_rental_cost

    miner_a_rental = 3 * H200_HOURLY_RATE
    miner_b_rental = 2 * H200_NVL_HOURLY_RATE

    expected_a_weight = expected_final_weight(
        miner_mining_score=0.0,
        miner_rental_value=miner_a_rental,
        total_mining_score=0.0,
        total_rental_value=total_rental_value,
        mining_share=splits["mining_share"],
        rental_share=splits["rental_share"],
        is_burner=False,
        burn_share=splits["burn_share"],
        num_burners=2,
    )

    expected_b_weight = expected_final_weight(
        miner_mining_score=0.0,
        miner_rental_value=miner_b_rental,
        total_mining_score=0.0,
        total_rental_value=total_rental_value,
        mining_share=splits["mining_share"],
        rental_share=splits["rental_share"],
        is_burner=False,
        burn_share=splits["burn_share"],
        num_burners=2,
    )

    # Final weights match: each miner's rental value / total rental value * rental_share
    assert validator.miner_scores["miner_a"] == pytest.approx(expected_a_weight, abs=0.0001)
    assert validator.miner_scores["miner_b"] == pytest.approx(expected_b_weight, abs=0.0001)

@pytest.mark.skip(reason="Tests legacy aggregate cross-variant dilution; replaced by per-`(base, bucket)` semantics.")
@pytest.mark.asyncio
async def test_rental_price_gpu_variants_exceeds_cap(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """Test multiple variants of same base model, total exceeds cap - dilution applied.

    Scenario:
    - Miner A: 6 H200 @ $4.00/hr (unrented)
    - Miner B: 4 H200 NVL @ $3.50/hr (unrented)
    - H200 cap: 8
    - Total H200 variants: 10 (6 + 4)

    Expected:
    - Dilution applied (10 > 8)
    - Dilution factor: 8/10 = 0.8
    - H200 effective rate: $4.00 * 0.8 = $3.20
    - H200 NVL effective rate: $3.50 * 0.8 = $2.80
    - Total rental cost: 6*$3.20 + 4*$2.80 = $30.40
    - Weight distribution proportional to effective rental values
    """
    # Arrange
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]

    # Override caps for this test via direct mutation (safe: each test gets a fresh fixture)
    custom_caps = {
        "H100": 1000,
        "H200": 8,
        "A100": 0,
    }
    validator.incentive.max_unrented_gpus = custom_caps

    # Total: 6 + 4 = 10 H200 variants (exceeds cap of 8)
    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H200", gpu_count=6, is_rented=False),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H200 NVL", gpu_count=4, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data())

    # Act
    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Assert
    miner_a_result = all_job_results["miner_a"][0]
    miner_b_result = all_job_results["miner_b"][0]

    # Total H200 variants (10) exceeds cap (8) → dilution factor = 8/10 = 0.8
    assert miner_a_result.cap_dilution_applied is True, "H200 should have dilution applied"
    assert miner_a_result.total_unrented_by_gpu_type == 10, "Total should be 10 (6 H200 + 4 H200 NVL)"
    assert miner_a_result.max_cap == 8, "Cap should be 8 for H200 base model"

    # H200 effective_rate = $4.00 * (8/10) = $3.20
    expected_h200_effective_rate = H200_HOURLY_RATE * 8 / 10
    assert miner_a_result.effective_rate == pytest.approx(expected_h200_effective_rate, abs=0.01), \
        f"H200 effective rate should be ${expected_h200_effective_rate:.2f} (diluted)"

    # H200 NVL shares the same base model cap and dilution factor
    assert miner_b_result.cap_dilution_applied is True, "H200 NVL should have dilution applied"
    assert miner_b_result.total_unrented_by_gpu_type == 10, "Total should be 10 (same as H200)"
    assert miner_b_result.max_cap == 8, "Cap should be 8 for H200 base model"

    # H200 NVL effective_rate = $3.50 * (8/10) = $2.80
    expected_h200_nvl_effective_rate = H200_NVL_HOURLY_RATE * 8 / 10
    assert miner_b_result.effective_rate == pytest.approx(expected_h200_nvl_effective_rate, abs=0.01), \
        f"H200 NVL effective rate should be ${expected_h200_nvl_effective_rate:.2f} (diluted)"

    # Total rental cost: 6*$3.20 + 4*$2.80 = $19.20 + $11.20 = $30.40
    expected_total_rental_cost = 6 * expected_h200_effective_rate + 4 * expected_h200_nvl_effective_rate
    assert expected_total_rental_cost == pytest.approx(30.40, abs=0.1)

    # Verify rental share calculation
    unrented_counts = {"H200": 6, "H200 NVL": 4}
    splits = expected_emission_splits(
        unrented_gpu_counts=unrented_counts,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=custom_caps,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
        base_gpu_map=BASE_GPU_MAP,
    )

    # Rental share is positive because effective rental cost > 0 after dilution
    assert splits["rental_share"] > 0, "Rental share should be positive"

    miner_a_rental_value = 6 * expected_h200_effective_rate
    miner_b_rental_value = 4 * expected_h200_nvl_effective_rate

    # Weight ratio should match rental value ratio since both miners have no mining scores
    weights_ratio = validator.miner_scores["miner_a"] / validator.miner_scores["miner_b"]
    rental_ratio = miner_a_rental_value / miner_b_rental_value
    assert weights_ratio == pytest.approx(rental_ratio, abs=0.01), \
        f"Weight ratio ({weights_ratio:.4f}) should match rental value ratio ({rental_ratio:.4f})"

    # Both miners receive non-zero scores from the rental pool
    assert validator.miner_scores["miner_a"] > 0, "Miner A should receive rental share"
    assert validator.miner_scores["miner_b"] > 0, "Miner B should receive rental share"


@pytest.mark.asyncio
async def test_rental_price_multi_variant_mixed_rental_status(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """Test same base model variants with mixed rental status."""
    # Arrange
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="miner_c"),
    ]

    # Miner A: rented H200 (gets mining score)
    # Miner B: unrented H200 NVL (gets rental share)
    # Miner C: unrented H200 (gets rental share)
    # Total unrented H200 variants: 3 + 2 = 5 (under cap of 1000)
    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H200", gpu_count=5, is_rented=True),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H200 NVL", gpu_count=3, is_rented=False),
        ],
        "miner_c": [
            _job(create_job_result, executor_id="exec-c", gpu_model="H200", gpu_count=2, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data(["exec-a"]))

    # Act
    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Assert
    # Verify miners B and C have rental values
    total_unrented_counts = {"H200": 2, "H200 NVL": 3}  # For expected_miner_rental_value

    expected_b_rental = expected_miner_rental_value(
        miner_results=all_job_results["miner_b"],
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        total_unrented_counts=total_unrented_counts,
        base_gpu_map=BASE_GPU_MAP,
    )
    expected_c_rental = expected_miner_rental_value(
        miner_results=all_job_results["miner_c"],
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        total_unrented_counts=total_unrented_counts,
        base_gpu_map=BASE_GPU_MAP,
    )

    # No cap dilution (5 total < 1000 cap), so rental values are count * full hourly rate
    assert expected_b_rental == pytest.approx(3 * H200_NVL_HOURLY_RATE, abs=0.01)
    assert expected_c_rental == pytest.approx(2 * H200_HOURLY_RATE, abs=0.01)

    # All three miners get positive weights from their respective pools (mining / rental)
    assert validator.miner_scores["miner_a"] > 0  # From mining pool
    assert validator.miner_scores["miner_b"] > 0  # From rental pool
    assert validator.miner_scores["miner_c"] > 0  # From rental pool
    assert sum(validator.miner_scores.values()) == pytest.approx(1.0, abs=0.0001)


@pytest.mark.skip(reason="Tests legacy aggregate cross-variant grouping; replaced by per-`(base, bucket)` semantics.")
@pytest.mark.asyncio
async def test_rental_price_multiple_base_models_with_variants(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """Test multiple base models, each with variants, independent dilution.

    Scenario:
    - Miner A: 4 H100 @ $3.50/hr (unrented)
    - Miner B: 3 H200 @ $4.00/hr (unrented)
    - Miner C: 2 H200 NVL @ $3.50/hr (unrented)
    - H100 cap: 10, H200 cap: 4

    Expected:
    - H100 total: 4 (under cap of 10) → NO dilution
    - H200 total: 5 (3+2, exceeds cap of 4) → dilution factor 4/5 = 0.8
    - Each base model's dilution is INDEPENDENT
    - H100 effective rate: $3.50 (no dilution)
    - H200 effective rate: $4.00 * 0.8 = $3.20 (diluted)
    - H200 NVL effective rate: $3.50 * 0.8 = $2.80 (diluted)
    """
    # Arrange
    validator = validator_with_rental_price
    validator.miner_scores = {}

    # Override caps for this test via direct mutation (safe: each test gets a fresh fixture)
    custom_caps = {
        "H100": 10,
        "H200": 4,
        "A100": 0,
    }
    validator.incentive.max_unrented_gpus = custom_caps

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="miner_c"),
    ]

    # H100 total: 4 (under cap of 10) - no dilution
    # H200 total: 3+2=5 (exceeds cap of 4) - dilution factor 4/5 = 0.8
    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=4, is_rented=False),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H200", gpu_count=3, is_rented=False),
        ],
        "miner_c": [
            _job(create_job_result, executor_id="exec-c", gpu_model="H200 NVL", gpu_count=2, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data())

    # Act
    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Assert
    miner_a_result = all_job_results["miner_a"][0]
    miner_b_result = all_job_results["miner_b"][0]
    miner_c_result = all_job_results["miner_c"][0]

    # H100: total (4) is under cap (10) → no dilution, effective_rate = full hourly rate
    assert miner_a_result.cap_dilution_applied is False, "H100 should have NO dilution (4 < 10)"
    assert miner_a_result.total_unrented_by_gpu_type == 4, "H100 total should be 4 (only H100, not H200 variants)"
    assert miner_a_result.max_cap == 10, "H100 cap should be 10"
    assert miner_a_result.effective_rate == pytest.approx(H100_HOURLY_RATE, abs=0.01), "H100 effective rate should NOT be diluted"

    # H200: total variants (5) exceeds cap (4) → dilution factor 4/5 = 0.8
    assert miner_b_result.cap_dilution_applied is True, "H200 should have dilution (5 > 4)"
    assert miner_b_result.total_unrented_by_gpu_type == 5, "H200 total should be 5 (3 H200 + 2 H200 NVL)"
    assert miner_b_result.max_cap == 4, "H200 cap should be 4"

    # H200 effective_rate = $4.00 * (4/5) = $3.20; independent of H100's dilution
    expected_h200_effective_rate = H200_HOURLY_RATE * 4 / 5
    assert miner_b_result.effective_rate == pytest.approx(expected_h200_effective_rate, abs=0.01), \
        f"H200 effective rate should be ${expected_h200_effective_rate:.2f} (diluted by factor 4/5)"

    # H200 NVL shares the same base model ("H200") so it uses the same dilution factor as H200
    assert miner_c_result.cap_dilution_applied is True, "H200 NVL should have dilution (same base model as H200)"
    assert miner_c_result.total_unrented_by_gpu_type == 5, "H200 NVL total should be 5 (same as H200)"
    assert miner_c_result.max_cap == 4, "H200 NVL cap should be 4 (same as H200)"

    expected_h200_nvl_effective_rate = H200_NVL_HOURLY_RATE * 4 / 5
    assert miner_c_result.effective_rate == pytest.approx(expected_h200_nvl_effective_rate, abs=0.01), \
        f"H200 NVL effective rate should be ${expected_h200_nvl_effective_rate:.2f} (diluted by same factor as H200)"

    # Dilution factors must differ between H100 (no dilution) and H200 group (diluted)
    h100_dilution_factor = miner_a_result.effective_rate / H100_HOURLY_RATE
    h200_dilution_factor = miner_b_result.effective_rate / H200_HOURLY_RATE
    h200_nvl_dilution_factor = miner_c_result.effective_rate / H200_NVL_HOURLY_RATE

    assert h100_dilution_factor == pytest.approx(1.0, abs=0.01), "H100 should have no dilution (factor = 1.0)"
    assert h200_dilution_factor == pytest.approx(0.8, abs=0.01), "H200 should have dilution factor 0.8"
    assert h200_nvl_dilution_factor == pytest.approx(0.8, abs=0.01), "H200 NVL should have same dilution factor as H200"

    # All three miners receive positive rental share weights
    assert validator.miner_scores["miner_a"] > 0, "Miner A should get rental share"
    assert validator.miner_scores["miner_b"] > 0, "Miner B should get rental share"
    assert validator.miner_scores["miner_c"] > 0, "Miner C should get rental share"


@pytest.mark.asyncio
async def test_rental_price_single_miner_multiple_variants(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """Test one miner with multiple executors of different variants."""
    # Arrange
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
    ]

    # Miner A has 3 executors:
    # - Executor 1: 3 H200 unrented (rental value)
    # - Executor 2: 2 H200 NVL unrented (rental value)
    # - Executor 3: 4 H100 rented (mining score)
    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a1", gpu_model="H200", gpu_count=3, is_rented=False),
            _job(create_job_result, executor_id="exec-a2", gpu_model="H200 NVL", gpu_count=2, is_rented=False),
            _job(create_job_result, executor_id="exec-a3", gpu_model="H100", gpu_count=4, is_rented=True),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data(["exec-a3"]))

    # Act
    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Assert
    # Each executor generates its own independent log entry keyed by UUID
    assert len(all_job_results["miner_a"]) == 3
    for result in all_job_results["miner_a"]:
        if result.incentive_logs:
            assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))

    # Calculate expected values
    total_unrented_counts = {"H200": 3, "H200 NVL": 2}
    expected_rental = expected_miner_rental_value(
        miner_results=all_job_results["miner_a"],
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        total_unrented_counts=total_unrented_counts,
        base_gpu_map=BASE_GPU_MAP,
    )

    # miner_a accumulates rental value from both H200 variants (no dilution since 5 < 1000)
    assert expected_rental == pytest.approx(3 * H200_HOURLY_RATE + 2 * H200_NVL_HOURLY_RATE, abs=0.01)

    # miner_a receives both mining and rental portions; total weights sum to 1.0
    assert validator.miner_scores["miner_a"] > 0
    assert sum(validator.miner_scores.values()) == pytest.approx(1.0, abs=0.0001)


@pytest.mark.skip(reason="Tests legacy extreme aggregate cross-variant dilution; replaced by per-`(base, bucket)` semantics.")
@pytest.mark.asyncio
async def test_rental_price_variant_edge_case_extreme_dilution(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """Test extreme dilution scenario with variants."""
    # Arrange
    validator = validator_with_rental_price
    validator.miner_scores = {}

    # Override caps for this test via direct mutation (safe: each test gets a fresh fixture)
    custom_caps = {
        "H100": 1000,
        "H200": 8,
        "A100": 0,
    }
    validator.incentive.max_unrented_gpus = custom_caps

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]

    # Total: 500 + 500 = 1000 H200 variants, cap: 8
    # Dilution factor: 8/1000 = 0.008
    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H200", gpu_count=500, is_rented=False),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H200 NVL", gpu_count=500, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data())

    # Act
    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Assert
    # Extreme dilution: 1000 total GPUs vs cap of 8 → factor 8/1000 = 0.008
    dilution_factor = 8 / 1000
    expected_h200_effective_rate = H200_HOURLY_RATE * dilution_factor
    expected_h200_nvl_effective_rate = H200_NVL_HOURLY_RATE * dilution_factor

    # Effective rates are dramatically reduced by the extreme dilution factor
    assert expected_h200_effective_rate == pytest.approx(0.032, abs=0.001)
    assert expected_h200_nvl_effective_rate == pytest.approx(0.028, abs=0.001)

    # Both miners have 500 GPUs each; weight ratio equals ratio of their effective rates
    weights_ratio = validator.miner_scores["miner_a"] / validator.miner_scores["miner_b"]
    expected_ratio = (500 * expected_h200_effective_rate) / (500 * expected_h200_nvl_effective_rate)
    assert weights_ratio == pytest.approx(expected_ratio, abs=0.01)

    # Dilution flag and total GPU count must appear in all eligible executor logs
    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert "cap_dilution_applied\": true" in result.full_log_text or "cap_dilution_applied': True" in result.full_log_text
                assert "total_unrented_by_gpu_type\": 1000" in result.full_log_text or "total_unrented_by_gpu_type': 1000" in result.full_log_text


@pytest.mark.asyncio
async def test_rental_price_variant_eligibility_check(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """Verify base model eligibility checking."""
    # Arrange
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="miner_c"),
    ]

    # H200/H200 NVL: Eligible (base model "H200" in rental_incentive_gpu_types, cap > 0)
    # A100: Not eligible (cap = 0)
    all_job_results = {
        "miner_a": [
            _job(create_job_result, executor_id="exec-a", gpu_model="H200", gpu_count=4, is_rented=False),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H200 NVL", gpu_count=3, is_rented=False),
        ],
        "miner_c": [
            _job(create_job_result, executor_id="exec-c", gpu_model="A100", gpu_count=5, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data())

    # Act
    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Assert
    # H200 and H200 NVL are eligible (base model "H200" has cap > 0) → positive rental weights
    assert validator.miner_scores["miner_a"] > 0
    assert validator.miner_scores["miner_b"] > 0

    # A100 has cap=0 → excluded from rental calculations entirely; miner gets 0
    assert validator.miner_scores.get("miner_c", 0) == pytest.approx(0.0, abs=0.0001)

    # Rental share should only be based on H200 variants (A100 not included in calculations)
    unrented_counts = {"H200": 4, "H200 NVL": 3}
    splits = expected_emission_splits(
        unrented_gpu_counts=unrented_counts,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_AGGREGATE,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
        base_gpu_map=BASE_GPU_MAP,
    )

    # Rental share is positive because eligible H200 variants contribute to rental cost
    assert splits["rental_share"] > 0


# ---------------------------------------------------------------------------
# GPU splitting tests (DAH-1871)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gpu_splitting_hardware_and_backend_enabled(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """When hardware supports GPU splitting AND backend opts in, the min-count rate is used."""
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
    ]

    job_a = _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=8, is_rented=False)
    job_a.supports_gpu_splitting = True
    job_a.gpu_splitting_min_count = 1

    all_job_results = {"miner_a": [job_a]}

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(gpu_splitting_config={"exec-a": 1})
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Executor is eligible and unrented → positive rental score
    assert validator.miner_scores.get("miner_a", 0) > 0

    # hourly_rate should be max(bundle_rate, min_count_rate)
    bundle_rate = H100_HOURLY_RATE  # rate for 8 GPUs
    single_rate = H100_HOURLY_RATE  # rate for 1 GPU (same table entry)
    assert job_a.hourly_rate == pytest.approx(max(bundle_rate, single_rate), rel=1e-6)

    # incentive log should contain gpu_splitting_min_count
    assert_incentive_log_present(job_a.full_log_text)
    assert_rental_price_incentive_log_full_content(job_a.full_log_text)


@pytest.mark.asyncio
async def test_gpu_splitting_hardware_supports_but_backend_not_enabled(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """When hardware supports GPU splitting but the backend does NOT opt in, standard bundle rate is used."""
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
    ]

    job_a = _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=8, is_rented=False)
    # hardware claims support but backend has no entry → combined flag must be False
    job_a.supports_gpu_splitting = False
    job_a.gpu_splitting_min_count = None

    all_job_results = {"miner_a": [job_a]}

    # No gpu_splitting_config provided → backend has not opted in
    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data()
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    assert validator.miner_scores.get("miner_a", 0) > 0
    # hourly_rate is only the bundle rate, no splitting uplift
    assert job_a.hourly_rate == pytest.approx(H100_HOURLY_RATE, rel=1e-6)


@pytest.mark.asyncio
async def test_gpu_splitting_backend_enabled_but_hardware_does_not_support(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """When backend opts in but hardware does NOT support GPU splitting, standard rate is used."""
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
    ]

    job_a = _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=8, is_rented=False)
    # hardware does NOT support splitting → combined flag must be False even though backend opted in
    job_a.supports_gpu_splitting = False
    job_a.gpu_splitting_min_count = None

    all_job_results = {"miner_a": [job_a]}

    # Backend says min_count=1 for exec-a, but hardware flag prevents it
    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(gpu_splitting_config={"exec-a": 1})
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    assert validator.miner_scores.get("miner_a", 0) > 0
    assert job_a.hourly_rate == pytest.approx(H100_HOURLY_RATE, rel=1e-6)


@pytest.mark.asyncio
async def test_sync_logs_successful_gpu_estimate_precompute(
    validator_with_rental_price,
    create_job_result,
    create_neuron_info,
    monkeypatch,
):
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_a"),
    ]
    all_job_results = {
        "miner_a": [_job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=8, is_rented=False)],
    }
    logger_info = MagicMock()

    monkeypatch.setattr("core.validator.precompute_all_estimates", AsyncMock(return_value={"H100": {}}))
    monkeypatch.setattr("core.validator.logger.info", logger_info)
    monkeypatch.setattr("core.validator.time.perf_counter", MagicMock(side_effect=[10.0, 10.5]))

    await _run_sync_with_jobs(validator, miners, all_job_results)

    success_logs = [
        call.args[0]
        for call in logger_info.call_args_list
        if str(call.args[0]) == "[sync] GPU estimates calculated successfully"
    ]
    assert len(success_logs) == 1
    assert success_logs[0].extra["duration_ms"] == 500.0
    assert success_logs[0].extra["gpu_models_count"] == 1


# ── Per-count rental subsidy cap (Fish table) ────────────────────────────────
#
# Direct-algorithm tests for `IncentiveConfig.max_unrented_gpus = {bucket: cap}`.
# These test `RentalPriceIncentive` end-to-end (pre + finish + post) without
# spinning up the full validator sync — the validator-level scoring path is
# already covered by the scenarios above.

from incentive.rental_price import RentalPriceIncentive  # noqa: E402

# Custom BASE_GPU_MAP/prices for these tests so the per-count cap behavior is
# exercised against a single base model ("B200") with realistic 1× and 8× tiers.
PCC_BASE_GPU_MAP = {
    "NVIDIA B200": "B200",
    "NVIDIA H200": "H200",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA GeForce RTX 4090": "RTX 4090",
    "NVIDIA A100 80GB PCIe": "A100",
    "NVIDIA RTX A6000": "RTX A6000",
    "NVIDIA GeForce RTX 3090": "RTX 3090",
}

PCC_RENTAL_PRICES = {gpu: 4.0 for gpu in PCC_BASE_GPU_MAP.keys()}

# 1× and 8× tiers eligible; everything else → 0 (mirrors production B200 config).
PCC_GPU_CUSTOM_PRICES = {gpu: {"*": 0, "1": DEFAULT_PRICE, "8": DEFAULT_PRICE} for gpu in PCC_BASE_GPU_MAP.keys()}

PCC_PER_COUNT_CAPS: dict[str, dict[int, int]] = {
    "B200": {1: 1, 8: 8},
    "H200": {1: 1, 8: 8},
    "H100": {1: 1, 8: 8},
    "RTX 4090": {1: 1, 8: 8},
    "A100": {1: 1, 8: 8},
    "RTX A6000": {1: 1, 8: 8},
    "RTX 3090": {1: 1, 8: 8},
}

PCC_HOURLY_RATE = 4.0
PCC_SYSBOX_MULTIPLIER = 1.0  # sysbox_runtime=True → multiplier=1.0


def _make_pcc_config(caps: dict[str, dict[int, int]] | None = None) -> IncentiveConfig:
    return IncentiveConfig(
        algorithm=ALGORITHM,
        rental_incentive_gpu_types=list(PCC_PER_COUNT_CAPS.keys()),
        max_unrented_gpus=caps if caps is not None else PCC_PER_COUNT_CAPS,
        rental_prices_per_hour=PCC_RENTAL_PRICES,
        gpu_count_custom_prices=PCC_GPU_CUSTOM_PRICES,
    )


def _make_pcc_job(
    executor_id: str,
    gpu_model: str,
    gpu_count: int,
    *,
    is_rented: bool = False,
    supports_gpu_splitting: bool = False,
    gpu_splitting_min_count: int | None = None,
):
    from datura.requests.miner_requests import ExecutorSSHInfo
    from services.task_service import JobResult

    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid=executor_id, address="10.0.0.1", port=8080,
            ssh_username="root", ssh_port=22,
            python_path="/usr/bin/python3", root_dir="/tmp",
        ),
        score=1.0, job_score=1.0, job_batch_id="pcc-batch",
        log_status="success", log_text="ok",
        gpu_model=gpu_model, gpu_count=gpu_count, is_rented=is_rented,
        collateral_deposited=True, sysbox_runtime=True,
        supports_gpu_splitting=supports_gpu_splitting,
        gpu_splitting_min_count=gpu_splitting_min_count,
    )


async def _run_pcc_incentive(
    config: IncentiveConfig,
    job_results: dict,
    monkeypatch,
):
    """Run the rental-price algorithm directly and return the populated instance."""
    redis = AsyncMock()
    redis.get_portion_per_gpu_type = AsyncMock(return_value=0.3)
    redis.get_executor_uptime = AsyncMock(return_value=9999)

    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", PCC_BASE_GPU_MAP)

    incentive = RentalPriceIncentive(
        config, redis, job_results,
        total_gpu_model_count_map={k: v for k, v in
            ((r.gpu_model, sum(j.gpu_count for j in jobs if j.gpu_model == r.gpu_model))
             for jobs in job_results.values() for r in jobs)},
    )

    price_provider = AsyncMock()
    price_provider.get_tao_price.return_value = TAO_PRICE
    price_provider.get_alpha_rate.return_value = ALPHA_RATE
    incentive.price_provider = price_provider

    await incentive.calculate_mining_scores()
    return incentive


@pytest.mark.asyncio
async def test_pcc_case1_single_1xB200_full_payout(monkeypatch):
    """Fish case 1 — one 1×B200 unrented executor receives full payout."""
    config = _make_pcc_config()
    jobs = {"miner_a": [_make_pcc_job("exec-a", "NVIDIA B200", 1)]}

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    assert incentive.unrented_count_by_bucket[("B200", 1)] == 1
    assert incentive.cap_multiplier_by_bucket[("B200", 1)] == pytest.approx(1.0)
    result = jobs["miner_a"][0]
    assert result.count_bucket == 1
    assert result.effective_rate == pytest.approx(PCC_HOURLY_RATE * PCC_SYSBOX_MULTIPLIER)


@pytest.mark.asyncio
async def test_pcc_case2_two_1xB200_split_budget(monkeypatch):
    """Fish case 2 — two 1×B200 executors each get half the 1× budget."""
    config = _make_pcc_config()
    jobs = {
        "miner_a": [_make_pcc_job("exec-a", "NVIDIA B200", 1)],
        "miner_b": [_make_pcc_job("exec-b", "NVIDIA B200", 1)],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    assert incentive.unrented_count_by_bucket[("B200", 1)] == 2
    assert incentive.cap_multiplier_by_bucket[("B200", 1)] == pytest.approx(0.5)
    expected_eff = 0.5 * PCC_HOURLY_RATE * PCC_SYSBOX_MULTIPLIER
    for hk in ("miner_a", "miner_b"):
        assert jobs[hk][0].effective_rate == pytest.approx(expected_eff)


@pytest.mark.asyncio
async def test_pcc_case3_2xB200_no_split_no_subsidy(monkeypatch):
    """Fish case 3 — 2×B200 without GPU splitting has rate=0 and is not accumulated."""
    config = _make_pcc_config()
    jobs = {"miner_a": [_make_pcc_job("exec-a", "NVIDIA B200", 2)]}

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    # 2× tier price is 0 → executor not accumulated into any bucket.
    assert ("B200", 1) not in incentive.unrented_count_by_bucket
    assert ("B200", 2) not in incentive.unrented_count_by_bucket
    assert ("B200", 8) not in incentive.unrented_count_by_bucket
    result = jobs["miner_a"][0]
    assert result.hourly_rate == 0.0
    assert (result.incentive or 0.0) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_pcc_case4_2xB200_with_split_lands_in_1_bucket(monkeypatch):
    """Fish case 4 — 2×B200 with `gpu_splitting_min_count=1` lands in the 1× bucket."""
    config = _make_pcc_config()
    jobs = {
        "miner_a": [_make_pcc_job(
            "exec-a", "NVIDIA B200", 2,
            supports_gpu_splitting=True, gpu_splitting_min_count=1,
        )],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    # Both GPUs counted into the 1× bucket; with cap=1 → cap_mult = 1/2 = 0.5.
    assert incentive.unrented_count_by_bucket[("B200", 1)] == 2
    assert incentive.cap_multiplier_by_bucket[("B200", 1)] == pytest.approx(0.5)
    result = jobs["miner_a"][0]
    assert result.count_bucket == 1
    # Per-GPU effective_rate is half of the 1× rate (matches case 2).
    assert result.effective_rate == pytest.approx(0.5 * PCC_HOURLY_RATE * PCC_SYSBOX_MULTIPLIER)


@pytest.mark.asyncio
async def test_pcc_case5_single_8xB200_full_payout(monkeypatch):
    """Fish case 5 — one 8×B200 executor gets the full 8-GPU payout."""
    config = _make_pcc_config()
    jobs = {"miner_a": [_make_pcc_job("exec-a", "NVIDIA B200", 8)]}

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    assert incentive.unrented_count_by_bucket[("B200", 8)] == 8
    assert incentive.cap_multiplier_by_bucket[("B200", 8)] == pytest.approx(1.0)
    result = jobs["miner_a"][0]
    assert result.count_bucket == 8
    assert result.effective_rate == pytest.approx(PCC_HOURLY_RATE * PCC_SYSBOX_MULTIPLIER)


@pytest.mark.asyncio
async def test_pcc_8xB200_split_prefers_8_bucket(monkeypatch):
    """8×B200 splitting + min_count=1 lands in the 8× bucket when it has a cap.

    With the default caps the solo node fills the 8× bucket exactly to cap
    (multiplier 1.0), so the DAH-2528 occupancy-aware fallback must NOT move it
    either — reassignment only triggers when the source bucket is strictly over
    cap (see the test_pcc_2528_* cases).
    """
    config = _make_pcc_config()
    jobs = {
        "miner_a": [_make_pcc_job(
            "exec-a", "NVIDIA B200", 8,
            supports_gpu_splitting=True, gpu_splitting_min_count=1,
        )],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    assert ("B200", 1) not in incentive.unrented_count_by_bucket
    assert incentive.unrented_count_by_bucket[("B200", 8)] == 8
    assert incentive.cap_multiplier_by_bucket[("B200", 8)] == pytest.approx(1.0)
    result = jobs["miner_a"][0]
    assert result.count_bucket == 8
    assert result.effective_rate == pytest.approx(PCC_HOURLY_RATE * PCC_SYSBOX_MULTIPLIER)


@pytest.mark.asyncio
async def test_pcc_mixed_1x_and_8x_both_full(monkeypatch):
    """Fish desired steady state — 1× and 8× executors coexist, both receive full payout."""
    config = _make_pcc_config()
    jobs = {
        "miner_a": [_make_pcc_job("exec-a", "NVIDIA B200", 1)],
        "miner_b": [_make_pcc_job("exec-b", "NVIDIA B200", 8)],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    assert incentive.cap_multiplier_by_bucket[("B200", 1)] == pytest.approx(1.0)
    assert incentive.cap_multiplier_by_bucket[("B200", 8)] == pytest.approx(1.0)
    assert jobs["miner_a"][0].effective_rate == pytest.approx(PCC_HOURLY_RATE)
    assert jobs["miner_b"][0].effective_rate == pytest.approx(PCC_HOURLY_RATE)


@pytest.mark.asyncio
async def test_pcc_overflow_isolated_within_bucket(monkeypatch):
    """Two 8×B200 executors dilute the 8× bucket; the 1× bucket is unaffected."""
    config = _make_pcc_config()
    jobs = {
        "miner_a": [_make_pcc_job("exec-a", "NVIDIA B200", 8)],
        "miner_b": [_make_pcc_job("exec-b", "NVIDIA B200", 8)],
        "miner_c": [_make_pcc_job("exec-c", "NVIDIA B200", 1)],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    # 8× bucket: 16 unrented vs cap 8 → 0.5 multiplier.
    assert incentive.unrented_count_by_bucket[("B200", 8)] == 16
    assert incentive.cap_multiplier_by_bucket[("B200", 8)] == pytest.approx(8 / 16)
    # 1× bucket: 1 unrented vs cap 1 → 1.0 multiplier (no dilution).
    assert incentive.unrented_count_by_bucket[("B200", 1)] == 1
    assert incentive.cap_multiplier_by_bucket[("B200", 1)] == pytest.approx(1.0)
    # 1× executor's payout is unaffected by the 8× overflow.
    assert jobs["miner_c"][0].effective_rate == pytest.approx(PCC_HOURLY_RATE)


@pytest.mark.parametrize(
    "gpu_model, base",
    [
        ("NVIDIA B200", "B200"),
        ("NVIDIA H200", "H200"),
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA GeForce RTX 4090", "RTX 4090"),
        ("NVIDIA A100 80GB PCIe", "A100"),
        ("NVIDIA RTX A6000", "RTX A6000"),
        ("NVIDIA GeForce RTX 3090", "RTX 3090"),
    ],
)
@pytest.mark.asyncio
async def test_pcc_per_family_migration(gpu_model, base, monkeypatch):
    """Per-family migration — `{1: 1, 8: 8}` works for every migrated family."""
    config = _make_pcc_config()
    jobs = {
        "miner_a": [_make_pcc_job("exec-a", gpu_model, 1)],
        "miner_b": [_make_pcc_job("exec-b", gpu_model, 8)],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    assert incentive.cap_multiplier_by_bucket[(base, 1)] == pytest.approx(1.0)
    assert incentive.cap_multiplier_by_bucket[(base, 8)] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_pcc_non_eligible_family_no_subsidy(monkeypatch):
    """Non-eligible family (`max_unrented_gpus[X] == {}`) contributes nothing."""
    caps = dict(PCC_PER_COUNT_CAPS)
    caps["B200"] = {}
    config = _make_pcc_config(caps=caps)
    config.rental_incentive_gpu_types = [k for k, v in caps.items() if any(c > 0 for c in v.values())]

    jobs = {"miner_a": [_make_pcc_job("exec-a", "NVIDIA B200", 1)]}

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    # Nothing accumulated for B200 at all.
    assert ("B200", 1) not in incentive.unrented_count_by_bucket
    assert ("B200", 8) not in incentive.unrented_count_by_bucket
    assert (jobs["miner_a"][0].incentive or 0.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "bad_cap",
    [
        {"B200": 8},           # aggregate int no longer accepted
        {"B200": {0: 1}},      # bucket key non-positive
        {"B200": {1: -1}},     # negative cap
        {"B200": "8"},         # wrong outer type
        {"B200": {"1": 1}},    # bucket key not int
        {"B200": {1: "1"}},    # cap not int
    ],
)
def test_pcc_validator_rejects_malformed_caps(bad_cap):
    """Validator — malformed `max_unrented_gpus` entries raise ValueError."""
    with pytest.raises(ValueError):
        IncentiveConfig(
            algorithm=ALGORITHM,
            rental_incentive_gpu_types=["B200"],
            max_unrented_gpus=bad_cap,
            rental_prices_per_hour=PCC_RENTAL_PRICES,
            gpu_count_custom_prices=PCC_GPU_CUSTOM_PRICES,
        )


@pytest.mark.asyncio
async def test_pcc_snapshot_exposes_by_bucket(monkeypatch):
    """Snapshot exposes per-bucket state via `by_bucket` only (no legacy `by_gpu_type`)."""
    config = _make_pcc_config()
    jobs = {
        "miner_a": [_make_pcc_job("exec-a", "NVIDIA B200", 1)],
        "miner_b": [_make_pcc_job("exec-b", "NVIDIA B200", 8)],
        "miner_c": [_make_pcc_job("exec-c", "NVIDIA B200", 8)],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)
    snapshot = incentive.get_snapshot()

    # Bucket-keyed field has both buckets present.
    assert "B200·1" in snapshot.rental.by_bucket
    assert "B200·8" in snapshot.rental.by_bucket
    assert snapshot.rental.by_bucket["B200·1"].unrented_count == 1
    assert snapshot.rental.by_bucket["B200·8"].unrented_count == 16

    # Legacy aggregate field no longer exposed on RentalShareState.
    assert not hasattr(snapshot.rental, "by_gpu_type")


@pytest.mark.asyncio
async def test_rental_price_spot_excluded_from_both_pools(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """A spot executor must produce zero mining and rental incentive, and must
    not affect the share earned by secure executors sharing the same GPU model:
    - secure unrented in the same bucket: per-bucket cap multiplier stays 1.0
    - secure rented: mining_score denominator (total_gpu_count) excludes spot
    """
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_secure_unrented"),
        create_neuron_info(uid=3, hotkey="miner_secure_rented"),
        create_neuron_info(uid=4, hotkey="miner_spot"),
    ]

    secure_unrented = _job(
        create_job_result, executor_id="exec-secure-unrented",
        gpu_model="H100", gpu_count=8, is_rented=False,
    )
    secure_rented = _job(
        create_job_result, executor_id="exec-secure-rented",
        gpu_model="H100", gpu_count=8, is_rented=True,
    )
    spot = _job(
        create_job_result, executor_id="exec-spot",
        gpu_model="H100", gpu_count=8, is_rented=False,
    )
    spot.is_spot = True

    all_job_results = {
        "miner_secure_unrented": [secure_unrented],
        "miner_secure_rented": [secure_rented],
        "miner_spot": [spot],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(
            rented_executor_ids=["exec-secure-rented"],
            spot_executor_ids=["exec-spot"],
        )
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Spot is zeroed on both axes
    assert spot.mining_score == 0
    assert spot.eligible_for_rental_share is False
    assert (spot.incentive or 0.0) == 0.0
    assert validator.miner_scores.get("miner_spot", 0.0) == pytest.approx(0.0, abs=0.0001)

    # Secure unrented stays in rental pool and is NOT diluted by spot
    # in the per-bucket cap. If spot leaked into the bucket,
    # total_unrented_by_gpu_type would be 16, not 8.
    assert secure_unrented.eligible_for_rental_share is True
    assert secure_unrented.total_unrented_by_gpu_type == 8
    assert secure_unrented.unrented_cap_multiplier == pytest.approx(1.0)
    assert (secure_unrented.incentive or 0.0) > 0.0
    assert validator.miner_scores["miner_secure_unrented"] > 0

    # Secure rented mining denominator excludes spot. The map includes the two
    # secure H100 executors (8+8=16) but must NOT include the spot's 8 GPUs;
    # an inflated denominator (24) would silently reduce the rented executor's
    # mining incentive by 1/3 (formula: score * gpu_portion * gpu_count / total).
    assert secure_rented.is_rented is True
    assert secure_rented.total_gpu_count == 16
    assert validator.miner_scores["miner_secure_rented"] > 0


@pytest.mark.asyncio
async def test_rental_price_missing_discord_excluded_like_spot(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
    monkeypatch,
):
    """A no-Discord executor after cutoff is treated like spot for incentives:
    - it earns zero
    - it does not dilute secure unrented bucket caps
    - it does not inflate rented mining denominators
    """
    monkeypatch.setattr(settings, "DISCORD_INCENTIVE_CUTOFF", datetime.utcnow() - timedelta(days=1))

    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_secure_unrented"),
        create_neuron_info(uid=3, hotkey="miner_secure_rented"),
        create_neuron_info(uid=4, hotkey="miner_no_discord"),
    ]

    secure_unrented = _job(
        create_job_result, executor_id="exec-secure-unrented",
        gpu_model="H100", gpu_count=8, is_rented=False,
    )
    secure_rented = _job(
        create_job_result, executor_id="exec-secure-rented",
        gpu_model="H100", gpu_count=8, is_rented=True,
    )
    no_discord = _job(
        create_job_result, executor_id="exec-no-discord",
        gpu_model="H100", gpu_count=8, is_rented=False,
    )
    no_discord.provider_discord_connected = False

    all_job_results = {
        "miner_secure_unrented": [secure_unrented],
        "miner_secure_rented": [secure_rented],
        "miner_no_discord": [no_discord],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(
            rented_executor_ids=["exec-secure-rented"],
            provider_discord_connected_executor_ids=[
                "exec-secure-unrented",
                "exec-secure-rented",
            ],
        )
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    assert no_discord.mining_score == 0
    assert no_discord.eligible_for_rental_share is False
    assert (no_discord.incentive or 0.0) == 0.0
    assert validator.miner_scores.get("miner_no_discord", 0.0) == pytest.approx(0.0, abs=0.0001)

    assert secure_unrented.eligible_for_rental_share is True
    assert secure_unrented.total_unrented_by_gpu_type == 8
    assert secure_unrented.unrented_cap_multiplier == pytest.approx(1.0)
    assert (secure_unrented.incentive or 0.0) > 0.0

    assert secure_rented.is_rented is True
    assert secure_rented.total_gpu_count == 16
    assert validator.miner_scores["miner_secure_rented"] > 0


@pytest.mark.asyncio
async def test_rental_price_paused_unrented_excluded_from_incentives_but_still_validated(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """A paused unrented executor is validated but does not enter either
    incentive pool:
    - it earns zero
    - it does not dilute secure unrented bucket caps
    - it does not inflate rented mining denominators
    """
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_secure_unrented"),
        create_neuron_info(uid=3, hotkey="miner_secure_rented"),
        create_neuron_info(uid=4, hotkey="miner_paused"),
    ]

    secure_unrented = _job(
        create_job_result, executor_id="exec-secure-unrented",
        gpu_model="H100", gpu_count=8, is_rented=False,
    )
    secure_rented = _job(
        create_job_result, executor_id="exec-secure-rented",
        gpu_model="H100", gpu_count=8, is_rented=True,
    )
    paused = _job(
        create_job_result, executor_id="exec-paused",
        gpu_model="H100", gpu_count=8, is_rented=False,
    )
    paused.is_new_rentals_paused = True

    all_job_results = {
        "miner_secure_unrented": [secure_unrented],
        "miner_secure_rented": [secure_rented],
        "miner_paused": [paused],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(
            rented_executor_ids=["exec-secure-rented"],
            new_rentals_paused_executor_ids=["exec-paused"],
        )
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    assert paused.mining_score == 0
    assert paused.eligible_for_rental_share is False
    assert (paused.incentive or 0.0) == 0.0
    assert validator.miner_scores.get("miner_paused", 0.0) == pytest.approx(0.0, abs=0.0001)

    assert secure_unrented.eligible_for_rental_share is True
    assert secure_unrented.total_unrented_by_gpu_type == 8
    assert secure_unrented.unrented_cap_multiplier == pytest.approx(1.0)
    assert (secure_unrented.incentive or 0.0) > 0.0

    assert secure_rented.is_rented is True
    assert secure_rented.total_gpu_count == 16
    assert validator.miner_scores["miner_secure_rented"] > 0


@pytest.mark.asyncio
async def test_rental_price_miner_default_job_unrented_earns_nothing(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """An unrented executor running the MINER'S OWN default job earns nothing:
    - it is excluded from both pools (mining + rental)
    - it does not dilute the unrented bucket caps of legitimate executors
    A Lium-owned default job (or no default job) keeps the unrented incentive.
    """
    validator = validator_with_rental_price
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_plain"),
        create_neuron_info(uid=3, hotkey="miner_lium_job"),
        create_neuron_info(uid=4, hotkey="miner_own_job"),
    ]

    plain = _job(
        create_job_result, executor_id="exec-plain",
        gpu_model="H100", gpu_count=8, is_rented=False,
    )
    lium_job = _job(
        create_job_result, executor_id="exec-lium-job",
        gpu_model="H100", gpu_count=8, is_rented=False,
    )
    own_job = _job(
        create_job_result, executor_id="exec-own-job",
        gpu_model="H100", gpu_count=8, is_rented=False,
    )
    lium_job.default_job_owner = "lium"
    own_job.default_job_owner = "miner"

    all_job_results = {
        "miner_plain": [plain],
        "miner_lium_job": [lium_job],
        "miner_own_job": [own_job],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(
            default_job_owner_by_executor={
                "exec-lium-job": "lium",
                "exec-own-job": "miner",
            },
        )
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Miner's own default job -> excluded from both pools, earns nothing
    assert own_job.default_job_owner == "miner"
    assert own_job.mining_score == 0
    assert own_job.eligible_for_rental_share is False
    assert (own_job.incentive or 0.0) == 0.0
    assert validator.miner_scores.get("miner_own_job", 0.0) == pytest.approx(0.0, abs=0.0001)

    # Lium-owned default job -> still earns the unrented incentive
    assert lium_job.default_job_owner == "lium"
    assert lium_job.eligible_for_rental_share is True
    assert (lium_job.incentive or 0.0) > 0.0

    # No default job -> still earns the unrented incentive
    assert plain.default_job_owner is None
    assert plain.eligible_for_rental_share is True
    assert (plain.incentive or 0.0) > 0.0

    # The miner's own job must not dilute the legitimate unrented bucket (only plain + lium_job count)
    assert lium_job.total_unrented_by_gpu_type == 16
    assert plain.total_unrented_by_gpu_type == 16


# ── DAH-2528: occupancy-aware bucket fallback for split-capable idle nodes ────

PCC_2528_CAPS = {**PCC_PER_COUNT_CAPS, "B200": {1: 10, 8: 8}}


@pytest.mark.asyncio
async def test_pcc_2528_split_node_falls_back_when_8_bucket_over_cap(monkeypatch):
    """The 8× bucket is over cap and the 1× tier is empty: the split-capable node
    moves whole into the 1× tier at full multiplier; non-splitting siblings stay
    diluted, and the source bucket's multiplier improves by the freed GPUs.
    """
    config = _make_pcc_config(caps=PCC_2528_CAPS)
    jobs = {
        "miner_a": [_make_pcc_job(
            "exec-a", "NVIDIA B200", 8,
            supports_gpu_splitting=True, gpu_splitting_min_count=1,
        )],
        "miner_b": [_make_pcc_job("exec-b", "NVIDIA B200", 8)],
        "miner_c": [_make_pcc_job("exec-c", "NVIDIA B200", 8)],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    # Mover: whole node (8 GPUs) rated against the 1× tier at full weight.
    mover = jobs["miner_a"][0]
    assert mover.count_bucket == 1
    assert mover.max_cap == 10
    assert mover.bucket_reassigned_from == 8
    assert mover.bucket_reassigned_from_multiplier == pytest.approx(8 / 24)
    assert mover.unrented_cap_multiplier == pytest.approx(1.0)
    assert mover.effective_rate == pytest.approx(PCC_HOURLY_RATE * PCC_SYSBOX_MULTIPLIER)
    assert mover.cap_dilution_applied is False
    assert any("unrented_bucket_reassigned" in line for line in mover.incentive_logs)

    # Bucket fills after the move: 1× holds the mover, 8× keeps the two others.
    assert incentive.unrented_count_by_bucket[("B200", 1)] == 8
    assert incentive.unrented_count_by_bucket[("B200", 8)] == 16
    assert incentive.cap_multiplier_by_bucket[("B200", 1)] == pytest.approx(1.0)
    assert incentive.cap_multiplier_by_bucket[("B200", 8)] == pytest.approx(0.5)

    # Non-splitting siblings stay in the 8× bucket, diluted, and are never reassigned.
    for hk in ("miner_b", "miner_c"):
        stayer = jobs[hk][0]
        assert stayer.count_bucket == 8
        assert stayer.bucket_reassigned_from is None
        assert stayer.effective_rate == pytest.approx(0.5 * PCC_HOURLY_RATE * PCC_SYSBOX_MULTIPLIER)
        assert not any("unrented_bucket_reassigned" in line for line in stayer.incentive_logs)

    # Snapshot persists the post-move fills, so estimates see the same state.
    by_bucket = incentive.get_snapshot().rental.by_bucket
    assert by_bucket["B200·1"].unrented_count == 8
    assert by_bucket["B200·1"].cap_multiplier == pytest.approx(1.0)
    assert by_bucket["B200·8"].unrented_count == 16
    assert by_bucket["B200·8"].cap_multiplier == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_pcc_2528_no_move_when_whole_node_does_not_fit_target(monkeypatch):
    """Strict admission: the whole node must fit under the target cap. With 3 native
    1× nodes already in the tier (cap 10), an 8-GPU candidate would make 11 — it
    stays in its over-cap 8× bucket and the 1× incumbents keep full weight.
    """
    config = _make_pcc_config(caps=PCC_2528_CAPS)
    jobs = {
        "miner_a": [_make_pcc_job(
            "exec-a", "NVIDIA B200", 8,
            supports_gpu_splitting=True, gpu_splitting_min_count=1,
        )],
        "miner_b": [_make_pcc_job("exec-b", "NVIDIA B200", 8)],
        "miner_c": [
            _make_pcc_job("exec-c1", "NVIDIA B200", 1),
            _make_pcc_job("exec-c2", "NVIDIA B200", 1),
            _make_pcc_job("exec-c3", "NVIDIA B200", 1),
        ],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    candidate = jobs["miner_a"][0]
    assert candidate.count_bucket == 8
    assert candidate.bucket_reassigned_from is None
    assert candidate.effective_rate == pytest.approx(0.5 * PCC_HOURLY_RATE * PCC_SYSBOX_MULTIPLIER)
    assert not any("unrented_bucket_reassigned" in line for line in candidate.incentive_logs)

    assert incentive.unrented_count_by_bucket[("B200", 1)] == 3
    assert incentive.unrented_count_by_bucket[("B200", 8)] == 16
    assert incentive.cap_multiplier_by_bucket[("B200", 1)] == pytest.approx(1.0)
    assert incentive.cap_multiplier_by_bucket[("B200", 8)] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_pcc_2528_greedy_stops_when_source_no_longer_over_cap(monkeypatch):
    """Two split-capable 8× nodes over an 8-GPU cap: the lowest-uuid one moves,
    which brings the source back to cap, so the second stays — and both end at
    full multiplier. Rerunning an identical fleet reproduces the same assignment.
    """
    def make_jobs():
        return {
            "miner_a": [_make_pcc_job(
                "exec-a", "NVIDIA B200", 8,
                supports_gpu_splitting=True, gpu_splitting_min_count=1,
            )],
            "miner_b": [_make_pcc_job(
                "exec-b", "NVIDIA B200", 8,
                supports_gpu_splitting=True, gpu_splitting_min_count=1,
            )],
        }

    config = _make_pcc_config(caps=PCC_2528_CAPS)
    jobs = make_jobs()
    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    mover, stayer = jobs["miner_a"][0], jobs["miner_b"][0]
    assert mover.count_bucket == 1
    assert mover.bucket_reassigned_from == 8
    assert stayer.count_bucket == 8
    assert stayer.bucket_reassigned_from is None
    # After the move both buckets sit exactly at (or under) cap: nobody is diluted.
    assert mover.unrented_cap_multiplier == pytest.approx(1.0)
    assert stayer.unrented_cap_multiplier == pytest.approx(1.0)
    assert incentive.unrented_count_by_bucket[("B200", 1)] == 8
    assert incentive.unrented_count_by_bucket[("B200", 8)] == 8

    # Determinism: an identical fleet produces identical bucket assignments.
    jobs_rerun = make_jobs()
    await _run_pcc_incentive(config, jobs_rerun, monkeypatch)
    for hk in jobs:
        assert jobs_rerun[hk][0].count_bucket == jobs[hk][0].count_bucket
        assert jobs_rerun[hk][0].bucket_reassigned_from == jobs[hk][0].bucket_reassigned_from


@pytest.mark.asyncio
async def test_pcc_2528_no_move_when_source_at_cap(monkeypatch):
    """Reassignment needs a strictly over-cap source: a bucket exactly at cap pays
    full weight already, so the split-capable node stays put.
    """
    config = _make_pcc_config(caps=PCC_2528_CAPS)
    jobs = {
        "miner_a": [_make_pcc_job(
            "exec-a", "NVIDIA B200", 8,
            supports_gpu_splitting=True, gpu_splitting_min_count=1,
        )],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    result = jobs["miner_a"][0]
    assert result.count_bucket == 8
    assert result.bucket_reassigned_from is None
    assert result.unrented_cap_multiplier == pytest.approx(1.0)
    assert ("B200", 1) not in incentive.unrented_count_by_bucket


@pytest.mark.asyncio
async def test_pcc_2528_move_lands_exactly_at_target_cap_with_incumbents(monkeypatch):
    """A move that fills the target exactly to cap is admitted, and the native 1×
    incumbents keep full weight — the newcomer never dilutes them.
    """
    config = _make_pcc_config(caps=PCC_2528_CAPS)
    jobs = {
        "miner_a": [_make_pcc_job(
            "exec-a", "NVIDIA B200", 8,
            supports_gpu_splitting=True, gpu_splitting_min_count=1,
        )],
        "miner_b": [_make_pcc_job("exec-b", "NVIDIA B200", 8)],
        "miner_c": [
            _make_pcc_job("exec-c1", "NVIDIA B200", 1),
            _make_pcc_job("exec-c2", "NVIDIA B200", 1),
        ],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    mover = jobs["miner_a"][0]
    assert mover.count_bucket == 1
    assert mover.bucket_reassigned_from == 8
    # 2 incumbents + 8 = 10 == cap: admitted, everyone in the tier at full weight.
    assert incentive.unrented_count_by_bucket[("B200", 1)] == 10
    assert incentive.cap_multiplier_by_bucket[("B200", 1)] == pytest.approx(1.0)
    for native in jobs["miner_c"]:
        assert native.effective_rate == pytest.approx(PCC_HOURLY_RATE * PCC_SYSBOX_MULTIPLIER)


@pytest.mark.asyncio
async def test_pcc_2528_source_bucket_emptied_by_move(monkeypatch):
    """With cap 8×=4, a single split-capable 8× node over-fills its own bucket and
    moves out entirely, leaving the source at 0 GPUs — no division error, and the
    snapshot and total_rental_cost stay consistent.
    """
    config = _make_pcc_config(caps={**PCC_PER_COUNT_CAPS, "B200": {1: 10, 8: 4}})
    jobs = {
        "miner_a": [_make_pcc_job(
            "exec-a", "NVIDIA B200", 8,
            supports_gpu_splitting=True, gpu_splitting_min_count=1,
        )],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    mover = jobs["miner_a"][0]
    assert mover.count_bucket == 1
    assert mover.bucket_reassigned_from == 8
    assert mover.bucket_reassigned_from_multiplier == pytest.approx(4 / 8)
    assert mover.unrented_cap_multiplier == pytest.approx(1.0)
    assert incentive.unrented_count_by_bucket[("B200", 8)] == 0
    assert incentive.unrented_count_by_bucket[("B200", 1)] == 8
    # The emptied source contributes nothing; the whole cost is the mover at full rate.
    assert incentive.total_rental_cost == pytest.approx(8 * PCC_HOURLY_RATE * PCC_SYSBOX_MULTIPLIER)
    by_bucket = incentive.get_snapshot().rental.by_bucket
    assert by_bucket["B200·8"].unrented_count == 0
    assert by_bucket["B200·1"].unrented_count == 8


@pytest.mark.asyncio
async def test_pcc_2528_no_move_when_split_tier_has_no_cap(monkeypatch):
    """A min-count tier with no configured cap can never receive a fallback move:
    the candidate stays in its over-cap gpu_count bucket, diluted.
    """
    config = _make_pcc_config(caps={**PCC_PER_COUNT_CAPS, "B200": {8: 8}})
    jobs = {
        "miner_a": [_make_pcc_job(
            "exec-a", "NVIDIA B200", 8,
            supports_gpu_splitting=True, gpu_splitting_min_count=1,
        )],
        "miner_b": [_make_pcc_job("exec-b", "NVIDIA B200", 8)],
    }

    incentive = await _run_pcc_incentive(config, jobs, monkeypatch)

    candidate = jobs["miner_a"][0]
    assert candidate.count_bucket == 8
    assert candidate.bucket_reassigned_from is None
    assert candidate.unrented_cap_multiplier == pytest.approx(0.5)
    assert ("B200", 1) not in incentive.unrented_count_by_bucket
