from unittest.mock import AsyncMock

import pytest

from core.config import settings
from incentive import rental_price as rental_price_module
from incentive.config import DEFAULT_PRICE, IncentiveConfig
from incentive.factory import IncentiveFactory
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from constants import TOTAL_BURN_EMISSION
from tests.helpers import (
    extract_incentive_section,
)
from tests.test_incentive_flow import _run_set_weights_and_capture, _run_sync_with_jobs
from tests.test_rental_price_helpers import (
    expected_emission_splits,
    expected_executor_score,
    expected_final_weight,
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


# ---------------------------------------------------------------------------
# GPU splitting tests (DAH-1871)
# ---------------------------------------------------------------------------


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


