from unittest.mock import AsyncMock, patch

import pytest

from core.config import settings
from incentive.config import IncentiveConfig
from incentive.factory import IncentiveFactory
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.const import TEMPO, SECONDS_PER_BLOCK, FIXED_RATIO, TOTAL_BURN_EMISSION
from tests.test_incentive_flow import _run_set_weights_and_capture, _run_sync_with_jobs
from tests.test_rental_price_helpers import (
    expected_emission_splits,
    expected_executor_score,
    expected_final_weight,
    expected_miner_rental_value,
)

pytest_plugins = ["fixtures.incentive_fixtures"]

ALGORITHM = "rental_price"
ELIGIBLE_GPU_TYPES = ["H100", "H200"]
MAX_UNRENTED_GPUS = 1000

H100_HOURLY_RATE = 3.50
H200_HOURLY_RATE = 4.00
RENTAL_PRICES_PER_HOUR = {
    "H100": H100_HOURLY_RATE,
    "H200": H200_HOURLY_RATE,
}

TAO_PRICE = 500.0
ALPHA_RATE = 0.5

GPU_PORTION = {
    "H100": 0.3,
    "H200": 0.25,
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
        eligible_gpu_types=ELIGIBLE_GPU_TYPES,
        max_unrented_gpus=MAX_UNRENTED_GPUS,
        rental_prices_per_hour=RENTAL_PRICES_PER_HOUR,
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
    validator_with_mocks.incentive = rental_price_config
    return validator_with_mocks


def _make_rented_data(rented_executor_ids: list[str] | None = None) -> RentedExecutorsResponse:
    executors = {}
    for executor_id in rented_executor_ids or []:
        executors[executor_id] = RentedExecutor(
            miner_hotkey="miner-hotkey",
            executor_ip_address="127.0.0.1",
            executor_ip_port="8000",
            pods=[RentedPod(pod_id="pod-1", container_name="ctr")],
        )
    return RentedExecutorsResponse(executors=executors, banned_guids=[])


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

    total_gpu_counts = _total_gpu_counts(all_job_results)
    expected_a_score = expected_executor_score(
        gpu_model="H100",
        gpu_count=10,
        total_gpu_count=total_gpu_counts["H100"],
        portion=GPU_PORTION["H100"],
        is_rented=True,
        eligible_gpu_types=ELIGIBLE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )

    unrented_counts = {"H100": 8, "H200": 5}
    splits = expected_emission_splits(
        unrented_gpu_counts=unrented_counts,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS,
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

    total_unrented = 1500
    expected_effective_rate = H100_HOURLY_RATE * MAX_UNRENTED_GPUS / total_unrented
    expected_total_rental_cost = total_unrented * expected_effective_rate

    splits = expected_emission_splits(
        unrented_gpu_counts={"H100": total_unrented},
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS,
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

    splits = expected_emission_splits(
        unrented_gpu_counts={"H100": 30, "H200": 15},
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS,
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
        max_unrented_gpus=MAX_UNRENTED_GPUS,
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
        max_unrented_gpus=MAX_UNRENTED_GPUS,
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

    total_gpu_counts = _total_gpu_counts(all_job_results)
    expected_a_mining = expected_executor_score(
        gpu_model="H100",
        gpu_count=5,
        total_gpu_count=total_gpu_counts["H100"],
        portion=GPU_PORTION["H100"],
        is_rented=True,
        eligible_gpu_types=ELIGIBLE_GPU_TYPES,
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
        eligible_gpu_types=ELIGIBLE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )

    total_unrented_counts = {"H100": 3, "H200": 4}
    expected_a_rental = expected_miner_rental_value(
        miner_results=all_job_results["miner_a"],
        eligible_gpu_types=ELIGIBLE_GPU_TYPES,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS,
        total_unrented_counts=total_unrented_counts,
    )
    assert expected_a_rental == pytest.approx(3 * H100_HOURLY_RATE + 4 * H200_HOURLY_RATE)

    splits = expected_emission_splits(
        unrented_gpu_counts=total_unrented_counts,
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS,
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

    total_gpu_counts = _total_gpu_counts(all_job_results)
    expected_a_mining = expected_executor_score(
        gpu_model="H100",
        gpu_count=4,
        total_gpu_count=total_gpu_counts["H100"],
        portion=GPU_PORTION["H100"],
        is_rented=True,
        eligible_gpu_types=ELIGIBLE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )
    expected_c_mining = expected_executor_score(
        gpu_model="A100",
        gpu_count=2,
        total_gpu_count=total_gpu_counts["A100"],
        portion=GPU_PORTION["A100"],
        is_rented=False,
        eligible_gpu_types=ELIGIBLE_GPU_TYPES,
        sysbox_runtime=True,
        collateral_deposited=True,
        uptime_minutes=120,
    )
    assert expected_c_mining > 0

    assert validator.miner_scores["miner_c"] > 0
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
        eligible_gpu_types=ELIGIBLE_GPU_TYPES,
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
        eligible_gpu_types=ELIGIBLE_GPU_TYPES,
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
        max_unrented_gpus=1000,
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
        max_unrented_gpus=1000,
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
        max_unrented_gpus=1000,
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
        max_unrented_gpus=1000,
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
        max_unrented_gpus=1000,
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
    }

    validator.backend_client.get_all_rented_executors = AsyncMock(
        return_value=_make_rented_data(["exec-a", "exec-b"])
    )

    await _run_sync_with_jobs(validator, miners, all_job_results)

    assert validator.miner_scores["miner_a"] == 0.0
    assert validator.miner_scores["miner_b"] > 0


@pytest.mark.asyncio
async def test_rental_price_failed_unrented_executors_do_not_count_rental(
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

    await _run_sync_with_jobs(validator, miners, all_job_results)

    splits = expected_emission_splits(
        unrented_gpu_counts={},
        rental_prices=RENTAL_PRICES_PER_HOUR,
        max_unrented_gpus=MAX_UNRENTED_GPUS,
        tao_price=TAO_PRICE,
        alpha_rate=ALPHA_RATE,
    )

    assert splits["rental_share"] == 0.0
    assert validator.miner_scores.get("miner_a", 0) == 0.0
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
