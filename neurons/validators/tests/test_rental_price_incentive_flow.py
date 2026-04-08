from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import settings
from incentive import rental_price as rental_price_module
from incentive.config import DEFAULT_PRICE, IncentiveConfig
from incentive.factory import IncentiveFactory
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.const import TEMPO, SECONDS_PER_BLOCK, FIXED_RATIO, TOTAL_BURN_EMISSION
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

# Per-GPU-type caps for testing (cap 0 = eligible for rental logic but no rental share / 0 miner score)
MAX_UNRENTED_GPUS_BY_TYPE = {
    "H100": 1000,
    "H200": 1000,
    "A100": 0,
}

RENTAL_INCENTIVE_GPU_TYPES = [
    gpu_type for gpu_type, cap in MAX_UNRENTED_GPUS_BY_TYPE.items() if cap > 0
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
        rental_incentive_gpu_types=[
            gpu_type for gpu_type, cap in MAX_UNRENTED_GPUS_BY_TYPE.items() if cap > 0
        ],
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
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
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
async def test_rental_price_scenario_cap_dilution(
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
            _job(create_job_result, executor_id="exec-a", gpu_model="H100", gpu_count=600, is_rented=False),
        ],
        "miner_b": [
            _job(create_job_result, executor_id="exec-b", gpu_model="H100", gpu_count=500, is_rented=False),
        ],
        "miner_c": [
            _job(create_job_result, executor_id="exec-c", gpu_model="H100", gpu_count=400, is_rented=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data())

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify cap dilution appears in logs
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                # Verify cap_dilution_applied is logged (key feature of this test)
                assert_log_contains_keys(result.full_log_text, [
                    "effective_rate",
                    "total_unrented_by_gpu_type",
                    "cap_dilution_applied",
                    "rental_share"
                ])

    total_unrented = 1500
    expected_effective_rate = H100_HOURLY_RATE * MAX_UNRENTED_GPUS_BY_TYPE["H100"] / total_unrented
    expected_total_rental_cost = total_unrented * expected_effective_rate

    splits = expected_emission_splits(
        unrented_gpu_counts={"H100": total_unrented},
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
    )

    assert expected_effective_rate == pytest.approx(2.333, abs=0.001)
    assert expected_total_rental_cost == pytest.approx(3500, abs=1)
    assert splits["rental_share"] == pytest.approx(0.1146, abs=0.001)

    weights = {
        "miner_a": validator.miner_scores["miner_a"],
        "miner_b": validator.miner_scores["miner_b"],
        "miner_c": validator.miner_scores["miner_c"],
    }
    assert weights["miner_a"] / weights["miner_b"] == pytest.approx(600 / 500, abs=0.01)
    assert weights["miner_b"] / weights["miner_c"] == pytest.approx(500 / 400, abs=0.01)


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
    # Create custom config with different caps for H100 and H200
    different_caps = {
        "H100": 5,
        "H200": 3,
    }
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
    h100_effective_rate = H100_HOURLY_RATE * different_caps["H100"] / 8
    h200_effective_rate = H200_HOURLY_RATE * different_caps["H200"] / 6
    expected_total_rental_cost = 8 * h100_effective_rate + 6 * h200_effective_rate

    splits = expected_emission_splits(
        unrented_gpu_counts={"H100": 8, "H200": 6},
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=different_caps,
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
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
    )

    assert splits["mining_share"] == pytest.approx(0.09)
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
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
    )

    assert splits["rental_share"] == 0.0
    assert splits["burn_share"] == pytest.approx(TOTAL_BURN_EMISSION, abs=0.0001)
    assert splits["mining_share"] == pytest.approx(0.09, abs=0.0001)
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
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
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
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
        total_unrented_counts=total_unrented_counts,
    )
    assert expected_a_rental == pytest.approx(3 * H100_HOURLY_RATE + 4 * H200_HOURLY_RATE)

    splits = expected_emission_splits(
        unrented_gpu_counts=total_unrented_counts,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
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
    from tests.helpers import extract_incentive_section

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
        extract_incentive_section,
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
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
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

    assert validator.miner_scores["miner_a"] > validator.miner_scores["miner_b"] + validator.miner_scores["miner_c"]
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
        "clients.subtensor_client.convert_weights_and_uids_for_emit"
    ) as convert_mock:
        def process_side_effect(uids, weights, netuid, subtensor, metagraph):
            return uids, weights

        process_mock.side_effect = process_side_effect
        convert_mock.return_value = (list(range(len(miners))), [10000] * len(miners))

        await mock_subtensor_client.set_weights(miner_scores=validator.miner_scores)

    assert mock_subtensor_client.send_weights_to_lium.called
    call_payload = mock_subtensor_client.send_weights_to_lium.call_args.args[0]
    assert "netuid" in call_payload
    assert "uids" in call_payload
    assert "weights" in call_payload

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
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
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
    # Verify miner A has mining score (rented → uses default algorithm)
    total_gpu_counts = _total_gpu_counts(all_job_results)
    expected_a_mining = expected_executor_score(
        gpu_model="H200",
        gpu_count=5,
        total_gpu_count=total_gpu_counts["H200"],
        portion=GPU_PORTION["H200"],
        is_rented=True,
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )

    # Verify miners B and C have rental values
    unrented_counts = {"H200 NVL": 3, "H200": 2}
    total_unrented_counts = {"H200": 2, "H200 NVL": 3}  # For expected_miner_rental_value

    expected_b_rental = expected_miner_rental_value(
        miner_results=all_job_results["miner_b"],
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
        total_unrented_counts=total_unrented_counts,
        base_gpu_map=BASE_GPU_MAP,
    )
    expected_c_rental = expected_miner_rental_value(
        miner_results=all_job_results["miner_c"],
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
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

    # Verify total rental cost: H100 undiluted + H200 variants diluted
    expected_total_rental_cost = (
        4 * H100_HOURLY_RATE +  # H100: no dilution
        3 * expected_h200_effective_rate +  # H200: diluted
        2 * expected_h200_nvl_effective_rate  # H200 NVL: diluted
    )

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
    total_gpu_counts = _total_gpu_counts(all_job_results)
    expected_mining = expected_executor_score(
        gpu_model="H100",
        gpu_count=4,
        total_gpu_count=total_gpu_counts["H100"],
        portion=GPU_PORTION["H100"],
        is_rented=True,
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )

    total_unrented_counts = {"H200": 3, "H200 NVL": 2}
    expected_rental = expected_miner_rental_value(
        miner_results=all_job_results["miner_a"],
        rental_incentive_gpu_types=RENTAL_INCENTIVE_GPU_TYPES,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
        total_unrented_counts=total_unrented_counts,
        base_gpu_map=BASE_GPU_MAP,
    )

    # miner_a accumulates rental value from both H200 variants (no dilution since 5 < 1000)
    assert expected_rental == pytest.approx(3 * H200_HOURLY_RATE + 2 * H200_NVL_HOURLY_RATE, abs=0.01)

    # miner_a receives both mining and rental portions; total weights sum to 1.0
    assert validator.miner_scores["miner_a"] > 0
    assert sum(validator.miner_scores.values()) == pytest.approx(1.0, abs=0.0001)


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
        max_unrented_gpus=MAX_UNRENTED_GPUS_BY_TYPE,
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


# ── CVM (TDX-attested) incentive tests ───────────────────────────────────────

CVM_RATE_MULTIPLIER = 1.5
CVM_MINING_MULTIPLIER = 1.3

# Config with CVM variants enabled alongside regular GPU types
MAX_UNRENTED_GPUS_CVM = {
    **MAX_UNRENTED_GPUS_BY_TYPE,
    "H100 CVM": 8,
    "H200 CVM": 8,
}


@pytest.fixture
def rental_price_config_cvm(rental_price_config):
    """IncentiveConfig with CVM variants enabled."""
    return IncentiveConfig(
        algorithm=ALGORITHM,
        rental_incentive_gpu_types=[
            gpu_type for gpu_type, cap in MAX_UNRENTED_GPUS_CVM.items() if cap > 0
        ],
        max_unrented_gpus=MAX_UNRENTED_GPUS_CVM,
        rental_prices_per_hour=RENTAL_PRICES_PER_HOUR,
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
        cvm_rate_multiplier=CVM_RATE_MULTIPLIER,
        cvm_mining_multiplier=CVM_MINING_MULTIPLIER,
    )


@pytest.fixture
def validator_with_cvm(
    validator_with_mocks,
    incentive_redis_service,
    rental_price_config_cvm,
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
    monkeypatch.setattr(settings, "incentive", rental_price_config_cvm)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", BASE_GPU_MAP)
    validator_with_mocks.incentive = rental_price_config_cvm
    return validator_with_mocks


def _cvm_job(create_job_result, *, executor_id: str, gpu_model: str, gpu_count: int,
             is_rented: bool, tdx_attestation_passed: bool = False, **kwargs):
    result = create_job_result(executor_id=executor_id, gpu_model=gpu_model, gpu_count=gpu_count, **kwargs)
    result.is_rented = is_rented
    result.tdx_attestation_passed = tdx_attestation_passed
    return result


@pytest.mark.asyncio
async def test_cvm_unrented_uses_separate_bucket(
    validator_with_cvm,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """CVM unrented executor routes to '<base> CVM' bucket, not plain base bucket."""
    validator = validator_with_cvm
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_cvm"),
        create_neuron_info(uid=3, hotkey="miner_plain"),
    ]

    # Both miners have unrented H100 GPUs within the cap (8 GPUs), but one is CVM
    all_job_results = {
        "miner_cvm": [
            _cvm_job(create_job_result, executor_id="exec-cvm", gpu_model="H100", gpu_count=8,
                     is_rented=False, tdx_attestation_passed=True),
        ],
        "miner_plain": [
            _cvm_job(create_job_result, executor_id="exec-plain", gpu_model="H100", gpu_count=8,
                     is_rented=False, tdx_attestation_passed=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data([]))

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Both miners should earn rental share (each within their own cap)
    assert validator.miner_scores.get("miner_cvm", 0) > 0, "CVM miner should earn rental share"
    assert validator.miner_scores.get("miner_plain", 0) > 0, "Plain miner should earn rental share"

    # CVM miner earns more due to cvm_rate_multiplier applied to hourly_rate
    # Effective rate for CVM: H100_HOURLY_RATE * CVM_RATE_MULTIPLIER * cap_mult
    # Effective rate for plain: H100_HOURLY_RATE * cap_mult
    assert validator.miner_scores["miner_cvm"] > validator.miner_scores["miner_plain"], (
        "CVM miner should earn more than plain miner due to cvm_rate_multiplier"
    )
    # Ratio should equal CVM_RATE_MULTIPLIER since both have equal GPU counts and are within cap
    ratio = validator.miner_scores["miner_cvm"] / validator.miner_scores["miner_plain"]
    assert ratio == pytest.approx(CVM_RATE_MULTIPLIER, rel=0.01)


@pytest.mark.asyncio
async def test_cvm_caps_are_independent_from_plain(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    price_provider_holder,
    monkeypatch,
):
    """CVM and plain GPU-type caps dilute independently — CVM is not penalized by non-CVM abundance.

    Scenario: plain H100 pool is over cap (100 GPUs, cap=8) → heavy dilution.
              CVM H100 pool is exactly at cap (8 GPUs, cap=8) → no dilution.
    If they shared a bucket, the CVM executor would also be diluted. Because they're separate,
    CVM still gets full (undiluted) rate.
    """
    # Use a config where the plain H100 cap is small (8) so over-cap dilution is visible
    small_cap_max = {
        "H100": 8,
        "H100 CVM": 8,
        "H200": 8,
        "A100": 0,
    }
    cvm_config = IncentiveConfig(
        algorithm=ALGORITHM,
        rental_incentive_gpu_types=[k for k, v in small_cap_max.items() if v > 0],
        max_unrented_gpus=small_cap_max,
        rental_prices_per_hour=RENTAL_PRICES_PER_HOUR,
        gpu_count_custom_prices={"*": {"*": DEFAULT_PRICE}},
        cvm_rate_multiplier=CVM_RATE_MULTIPLIER,
        cvm_mining_multiplier=CVM_MINING_MULTIPLIER,
    )

    original_create = IncentiveFactory.create

    def create_with_price_provider(*args, **kwargs):
        incentive = original_create(*args, **kwargs)
        if hasattr(incentive, "price_provider"):
            incentive.price_provider = price_provider_holder["provider"]
        return incentive

    monkeypatch.setattr(IncentiveFactory, "create", create_with_price_provider)
    monkeypatch.setattr(settings, "incentive", cvm_config)
    monkeypatch.setattr(rental_price_module, "BASE_GPU_MAP", BASE_GPU_MAP)
    validator = validator_with_mocks
    validator.incentive = cvm_config
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_cvm"),
        create_neuron_info(uid=3, hotkey="miner_plain"),
    ]

    # Plain H100: 100 GPUs, cap=8 → diluted (cap_multiplier=8/100=0.08)
    # CVM H100:   8 GPUs,  cap=8 → no dilution (cap_multiplier=1.0)
    PLAIN_GPU_COUNT = 100
    CVM_GPU_COUNT = 8
    PLAIN_CAP = 8

    all_job_results = {
        "miner_cvm": [
            _cvm_job(create_job_result, executor_id="exec-cvm", gpu_model="H100", gpu_count=CVM_GPU_COUNT,
                     is_rented=False, tdx_attestation_passed=True),
        ],
        "miner_plain": [
            _cvm_job(create_job_result, executor_id="exec-plain", gpu_model="H100", gpu_count=PLAIN_GPU_COUNT,
                     is_rented=False, tdx_attestation_passed=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data([]))

    await _run_sync_with_jobs(validator, miners, all_job_results)

    cvm_score = validator.miner_scores.get("miner_cvm", 0)
    plain_score = validator.miner_scores.get("miner_plain", 0)

    assert cvm_score > 0
    assert plain_score > 0

    # CVM: 8 GPUs * H100_rate * 1.5 (CVM mult) * 1.0 (no dilution) = 8 * 3.5 * 1.5 = 42
    # Plain: 100 GPUs * H100_rate * (8/100) (diluted to cap) = 100 * 3.5 * 0.08 = 28
    # total = 70; CVM fraction = 42/70 = 0.6; Plain fraction = 28/70 = 0.4
    cvm_effective = CVM_GPU_COUNT * H100_HOURLY_RATE * CVM_RATE_MULTIPLIER * 1.0
    plain_effective = PLAIN_GPU_COUNT * H100_HOURLY_RATE * (PLAIN_CAP / PLAIN_GPU_COUNT)
    expected_cvm_fraction = cvm_effective / (cvm_effective + plain_effective)

    total_rental_score = cvm_score + plain_score
    actual_cvm_fraction = cvm_score / total_rental_score
    assert actual_cvm_fraction == pytest.approx(expected_cvm_fraction, rel=0.01)


@pytest.mark.asyncio
async def test_cvm_rented_gets_mining_multiplier(
    validator_with_cvm,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """Rented CVM executor's mining score is multiplied by cvm_mining_multiplier."""
    validator = validator_with_cvm
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_cvm"),
        create_neuron_info(uid=3, hotkey="miner_plain"),
    ]

    # Both have same GPU count and GPU model, rented; only CVM miner has TDX attestation
    all_job_results = {
        "miner_cvm": [
            _cvm_job(create_job_result, executor_id="exec-cvm", gpu_model="H100", gpu_count=1,
                     is_rented=True, tdx_attestation_passed=True),
        ],
        "miner_plain": [
            _cvm_job(create_job_result, executor_id="exec-plain", gpu_model="H100", gpu_count=1,
                     is_rented=True, tdx_attestation_passed=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(["exec-cvm", "exec-plain"])
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    cvm_score = validator.miner_scores.get("miner_cvm", 0)
    plain_score = validator.miner_scores.get("miner_plain", 0)

    assert cvm_score > 0
    assert plain_score > 0
    # CVM gets cvm_mining_multiplier applied to its mining_score, so its final weight is proportionally higher
    # mining_score_cvm = base * CVM_MINING_MULTIPLIER
    # mining_score_plain = base
    # total = base * (1 + CVM_MINING_MULTIPLIER)
    # incentive_cvm / incentive_plain = CVM_MINING_MULTIPLIER
    ratio = cvm_score / plain_score
    assert ratio == pytest.approx(CVM_MINING_MULTIPLIER, rel=0.01)


@pytest.mark.asyncio
async def test_cvm_falls_back_to_plain_when_no_cvm_cap_configured(
    validator_with_rental_price,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
    mock_price_provider,
):
    """CVM executor with GPU type that has no CVM variant configured falls back to plain bucket."""
    validator = validator_with_rental_price  # config has NO CVM variants
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_cvm"),
        create_neuron_info(uid=3, hotkey="miner_plain"),
    ]

    # Both unrented H100; CVM flag is set but no "H100 CVM" in config → falls back to "H100"
    all_job_results = {
        "miner_cvm": [
            _cvm_job(create_job_result, executor_id="exec-cvm", gpu_model="H100", gpu_count=1,
                     is_rented=False, tdx_attestation_passed=True),
        ],
        "miner_plain": [
            _cvm_job(create_job_result, executor_id="exec-plain", gpu_model="H100", gpu_count=1,
                     is_rented=False, tdx_attestation_passed=False),
        ],
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(return_value=_make_rented_data([]))

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Both compete in the same "H100" bucket, same rate → equal scores
    cvm_score = validator.miner_scores.get("miner_cvm", 0)
    plain_score = validator.miner_scores.get("miner_plain", 0)

    assert cvm_score > 0
    assert plain_score > 0
    assert cvm_score == pytest.approx(plain_score, rel=0.001)
