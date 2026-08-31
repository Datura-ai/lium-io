import logging
import pytest
from unittest.mock import AsyncMock, patch

from constants import TOTAL_BURN_EMISSION

BASE_JOB_SCORE = 1.0
BURNER_COUNT = 2
MINING_ALLOCATION = 1 - TOTAL_BURN_EMISSION
SYSBOX_PENALTY_PORTION = 0.1
UPTIME_PENALTY_PORTION = 0.2
UPTIME_REQUIRED_MINUTES = 120

GPU_PORTION = {
    "H100": 0.3,
    "H200": 0.25,
    "A100": 0.2,
    "RTX4090": 0.15,
    "RTX3090": 0.1,
}


def expected_score(
    *,
    portion: float,
    gpu_count: int,
    total_gpu_count: int,
    sysbox_runtime: bool = True,
    collateral_deposited: bool = True,
    uptime_minutes: int | None = None,
) -> float:
    score = BASE_JOB_SCORE * portion * gpu_count / total_gpu_count
    if not sysbox_runtime:
        score *= 1 - SYSBOX_PENALTY_PORTION
    if not collateral_deposited:
        uptime_ratio = min(1.0, (uptime_minutes or 0) / UPTIME_REQUIRED_MINUTES)
        score *= 1 - UPTIME_PENALTY_PORTION + UPTIME_PENALTY_PORTION * uptime_ratio
    return score


pytest_plugins = ["fixtures.incentive_fixtures"]


def _floor_log_extras(caplog):
    return [
        {"floor_hotkeys": record.msg.extra["floor_hotkeys"], "floor_count": record.msg.extra["floor_count"]}
        for record in caplog.records
        if hasattr(record.msg, "extra") and "floor_hotkeys" in record.msg.extra
    ]


def _weights_by_uid(captured):
    return {
        int(uid): float(weight)
        for uid, weight in zip(captured["uids"], captured["weights"])
    }


async def _run_set_weights_and_capture(
    mock_subtensor_client, miners, miner_scores, *, normalize=False, active_hotkeys=None
):
    captured = {}

    def capture_weights(uids, weights, netuid, subtensor, metagraph):
        captured["uids"] = uids
        captured["weights"] = weights
        if not normalize:
            return uids, weights
        total = weights.sum()
        if total > 0:
            normalized = weights / total
        else:
            normalized = weights
        captured["processed_weights"] = normalized
        return uids, normalized

    with patch("clients.subtensor_client.process_weights_for_netuid", side_effect=capture_weights):
        mock_subtensor_client.get_miners = AsyncMock(return_value=miners)
        await mock_subtensor_client.set_weights(
            miner_scores=miner_scores, active_hotkeys=active_hotkeys
        )

    return captured


async def _run_sync_with_jobs(validator, miners, job_results_by_hotkey, *, job_batch_id="2024-01-01 00:00:00"):
    async def request_job_side_effect(
        payload,
        encrypted_files,
        rented_data,
        default_docker_image_digests,
        executor_image_snapshot,
    ):
        hotkey = payload.miner_hotkey
        results = job_results_by_hotkey.get(hotkey, [])
        return {
            "miner_hotkey": hotkey,
            "miner_coldkey": "test_coldkey",
            "results": results,
        }

    validator.subtensor_client.get_miners = AsyncMock(return_value=miners)
    validator.subtensor_client.should_set_weights = AsyncMock(return_value=False)
    validator.subtensor_client.get_time_from_block = AsyncMock(return_value=job_batch_id)
    validator.miner_service.request_job_to_miner = AsyncMock(side_effect=request_job_side_effect)

    # DAH-2380: sync() fetches default docker image digests from Docker Hub at job-cycle
    # start. Stub it here so these unit tests never hit the network (the real fetch would
    # open aiohttp connections to auth.docker.io / registry-1.docker.io per cycle).
    with (
        patch(
            "core.validator.fetch_default_image_digests",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "core.validator.fetch_executor_image_digest",
            new=AsyncMock(return_value=None),
        ),
    ):
        await validator.sync()


# Existing test scenarios

@pytest.mark.asyncio
async def test_scenario_dominant_miner_premium_gpus(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Dominant miner with H100s should get higher weight than miners with lower-tier GPUs."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="dominant_miner"),
        create_neuron_info(uid=3, hotkey="small_miner_1"),
        create_neuron_info(uid=4, hotkey="small_miner_2"),
    ]

    all_job_results = {
        "dominant_miner": [
            create_job_result(gpu_model="H100", gpu_count=8, sysbox_runtime=True, collateral_deposited=True),
        ],
        "small_miner_1": [
            create_job_result(gpu_model="A100", gpu_count=2, sysbox_runtime=True, collateral_deposited=True),
        ],
        "small_miner_2": [
            create_job_result(gpu_model="RTX4090", gpu_count=2, sysbox_runtime=True, collateral_deposited=True),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify incentive logging for successful executors
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:  # Only check if logs were populated
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, [
                    "mining_score",
                    "total_mining_score",
                    "mining_share",
                    "incentive"
                ])

    dominant_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=8,
        total_gpu_count=8,
    )
    small_a100_score = expected_score(
        portion=GPU_PORTION["A100"],
        gpu_count=2,
        total_gpu_count=2,
    )
    small_4090_score = expected_score(
        portion=GPU_PORTION["RTX4090"],
        gpu_count=2,
        total_gpu_count=2,
    )
    total_score = dominant_score + small_a100_score + small_4090_score
    mining_allocation = 1 - TOTAL_BURN_EMISSION
    burn_share_per_burner = TOTAL_BURN_EMISSION / BURNER_COUNT

    expected_dominant_cycle = mining_allocation * dominant_score / total_score
    expected_small_a100_cycle = mining_allocation * small_a100_score / total_score
    expected_small_4090_cycle = mining_allocation * small_4090_score / total_score

    assert validator.miner_scores["dominant_miner"] == pytest.approx(expected_dominant_cycle)
    assert validator.miner_scores["small_miner_1"] == pytest.approx(expected_small_a100_cycle)
    assert validator.miner_scores["small_miner_2"] == pytest.approx(expected_small_4090_cycle)

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[100] == pytest.approx(burn_share_per_burner)
    assert weights_by_uid[101] == pytest.approx(burn_share_per_burner)
    assert weights_by_uid[2] == pytest.approx(expected_dominant_cycle)
    assert weights_by_uid[3] == pytest.approx(expected_small_a100_cycle)
    assert weights_by_uid[4] == pytest.approx(expected_small_4090_cycle)


@pytest.mark.asyncio
async def test_scenario_mixed_collateral_strategies(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Miners with collateral should score slightly better than those relying on uptime."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="collateral_miner"),
        create_neuron_info(uid=3, hotkey="uptime_miner_high"),
        create_neuron_info(uid=4, hotkey="uptime_miner_low"),
    ]

    async def get_uptime_side_effect(executor_info):
        if "uptime_miner_high" in str(executor_info.uuid):
            return 150
        if "uptime_miner_low" in str(executor_info.uuid):
            return 60
        return 100

    validator.redis_service.get_executor_uptime = AsyncMock(side_effect=get_uptime_side_effect)

    all_job_results = {
        "collateral_miner": [
            create_job_result(gpu_model="H100", gpu_count=1, collateral_deposited=True),
        ],
        "uptime_miner_high": [
            create_job_result(
                gpu_model="H100",
                gpu_count=1,
                collateral_deposited=False,
                executor_id="uptime_miner_high",
            ),
        ],
        "uptime_miner_low": [
            create_job_result(
                gpu_model="H100",
                gpu_count=1,
                collateral_deposited=False,
                executor_id="uptime_miner_low",
            ),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify incentive logging
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, [
                    "mining_score",
                    "total_mining_score",
                    "incentive"
                ])

    base_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=1,
        total_gpu_count=3,
    )
    high_uptime_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=1,
        total_gpu_count=3,
        collateral_deposited=False,
        uptime_minutes=150,
    )
    low_uptime_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=1,
        total_gpu_count=3,
        collateral_deposited=False,
        uptime_minutes=60,
    )
    total_score = base_score + high_uptime_score + low_uptime_score
    expected_base_cycle = MINING_ALLOCATION * base_score / total_score
    expected_high_cycle = MINING_ALLOCATION * high_uptime_score / total_score
    expected_low_cycle = MINING_ALLOCATION * low_uptime_score / total_score

    assert validator.miner_scores["collateral_miner"] == pytest.approx(expected_base_cycle)
    assert validator.miner_scores["uptime_miner_high"] == pytest.approx(expected_high_cycle)
    assert validator.miner_scores["uptime_miner_low"] == pytest.approx(expected_low_cycle)
    assert validator.miner_scores["uptime_miner_low"] < validator.miner_scores["collateral_miner"]

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[2] == pytest.approx(expected_base_cycle)
    assert weights_by_uid[3] == pytest.approx(expected_high_cycle)
    assert weights_by_uid[4] == pytest.approx(expected_low_cycle)


@pytest.mark.asyncio
async def test_scenario_new_vs_established_miner(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """New miner with low uptime should score lower than established miner."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="established_miner"),
        create_neuron_info(uid=3, hotkey="new_miner"),
    ]

    async def get_uptime_side_effect(executor_info):
        if "new_miner" in str(executor_info.uuid):
            return 10
        return 100

    validator.redis_service.get_executor_uptime = AsyncMock(side_effect=get_uptime_side_effect)

    all_job_results = {
        "established_miner": [
            create_job_result(gpu_model="H100", gpu_count=4, collateral_deposited=True),
        ],
        "new_miner": [
            create_job_result(
                gpu_model="H100",
                gpu_count=4,
                collateral_deposited=False,
                executor_id="new_miner",
            ),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify incentive logging
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, [
                    "mining_score",
                    "total_mining_score",
                    "incentive"
                ])

    established_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=4,
        total_gpu_count=8,
    )
    new_miner_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=4,
        total_gpu_count=8,
        collateral_deposited=False,
        uptime_minutes=10,
    )
    total_score = established_score + new_miner_score
    expected_established_cycle = MINING_ALLOCATION * established_score / total_score
    expected_new_cycle = MINING_ALLOCATION * new_miner_score / total_score

    assert validator.miner_scores["established_miner"] == pytest.approx(expected_established_cycle)
    assert validator.miner_scores["new_miner"] == pytest.approx(expected_new_cycle)
    assert validator.miner_scores["new_miner"] < validator.miner_scores["established_miner"]

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[2] == pytest.approx(expected_established_cycle)
    assert weights_by_uid[3] == pytest.approx(expected_new_cycle)


@pytest.mark.asyncio
async def test_scenario_mixed_sysbox_usage(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Miners using sysbox should score higher than those not using it."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="sysbox_miner"),
        create_neuron_info(uid=3, hotkey="no_sysbox_miner"),
    ]

    all_job_results = {
        "sysbox_miner": [
            create_job_result(gpu_model="A100", gpu_count=2, sysbox_runtime=True, collateral_deposited=True),
        ],
        "no_sysbox_miner": [
            create_job_result(gpu_model="A100", gpu_count=2, sysbox_runtime=False, collateral_deposited=True),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify incentive logging
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, [
                    "mining_score",
                    "total_mining_score",
                    "incentive"
                ])

    sysbox_score = expected_score(
        portion=GPU_PORTION["A100"],
        gpu_count=2,
        total_gpu_count=4,
    )
    no_sysbox_score = expected_score(
        portion=GPU_PORTION["A100"],
        gpu_count=2,
        total_gpu_count=4,
        sysbox_runtime=False,
    )
    total_score = sysbox_score + no_sysbox_score
    expected_sysbox_cycle = MINING_ALLOCATION * sysbox_score / total_score
    expected_no_sysbox_cycle = MINING_ALLOCATION * no_sysbox_score / total_score

    assert validator.miner_scores["sysbox_miner"] == pytest.approx(expected_sysbox_cycle)
    assert validator.miner_scores["no_sysbox_miner"] == pytest.approx(expected_no_sysbox_cycle)
    assert validator.miner_scores["no_sysbox_miner"] < validator.miner_scores["sysbox_miner"]

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[2] == pytest.approx(expected_sysbox_cycle)
    assert weights_by_uid[3] == pytest.approx(expected_no_sysbox_cycle)


@pytest.mark.asyncio
async def test_scenario_multiple_executors_per_miner(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Miner with multiple executors should accumulate scores correctly."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="multi_executor_miner"),
        create_neuron_info(uid=3, hotkey="single_executor_miner"),
    ]

    all_job_results = {
        "multi_executor_miner": [
            create_job_result(
                executor_id="exec-1",
                gpu_model="H100",
                gpu_count=2,
                sysbox_runtime=True,
                collateral_deposited=True,
            ),
            create_job_result(
                executor_id="exec-2",
                gpu_model="A100",
                gpu_count=1,
                sysbox_runtime=True,
                collateral_deposited=True,
            ),
            create_job_result(
                executor_id="exec-3",
                gpu_model="RTX4090",
                gpu_count=1,
                sysbox_runtime=False,
                collateral_deposited=False,
            ),
        ],
        "single_executor_miner": [
            create_job_result(
                gpu_model="H100",
                gpu_count=2,
                sysbox_runtime=True,
                collateral_deposited=True,
            ),
        ],
    }

    async def get_uptime_side_effect(executor_info):
        if "exec-3" in str(executor_info.uuid):
            return 60
        return 100

    validator.redis_service.get_executor_uptime = AsyncMock(side_effect=get_uptime_side_effect)

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify incentive logging - CRITICAL: Each executor should have independent log entry
    from tests.helpers import (
        assert_incentive_log_present,
        assert_executor_has_log,
        assert_log_contains_keys,
        count_incentive_log_entries,
    )

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, [
                    "mining_score",
                    "total_mining_score",
                    "incentive"
                ])

    # Verify multi-executor miner has 3 independent log entries (one per executor)
    multi_executor_results = all_job_results["multi_executor_miner"]
    assert len(multi_executor_results) == 3, "Should have 3 executors"
    for result in multi_executor_results:
        # Each result should have its own executor_id in logs (if logs were populated)
        if result.incentive_logs:
            assert str(result.executor_info.uuid) in result.full_log_text

    exec1_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=2,
        total_gpu_count=4,
    )
    exec2_score = expected_score(
        portion=GPU_PORTION["A100"],
        gpu_count=1,
        total_gpu_count=1,
    )
    exec3_score = expected_score(
        portion=GPU_PORTION["RTX4090"],
        gpu_count=1,
        total_gpu_count=1,
        sysbox_runtime=False,
        collateral_deposited=False,
        uptime_minutes=60,
    )
    multi_executor_score = exec1_score + exec2_score + exec3_score
    single_executor_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=2,
        total_gpu_count=4,
    )
    total_score = multi_executor_score + single_executor_score
    expected_multi_cycle = MINING_ALLOCATION * multi_executor_score / total_score
    expected_single_cycle = MINING_ALLOCATION * single_executor_score / total_score

    assert validator.miner_scores["multi_executor_miner"] == pytest.approx(expected_multi_cycle)
    assert validator.miner_scores["single_executor_miner"] == pytest.approx(expected_single_cycle)
    assert validator.miner_scores["multi_executor_miner"] > validator.miner_scores["single_executor_miner"]

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[2] == pytest.approx(expected_multi_cycle)
    assert weights_by_uid[3] == pytest.approx(expected_single_cycle)


@pytest.mark.asyncio
async def test_scenario_all_jobs_failed(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """When all jobs fail, only burners should get weights."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner1"),
        create_neuron_info(uid=3, hotkey="miner2"),
    ]

    all_job_results = {
        "miner1": [
            create_job_result(score=0.0, gpu_model="H100", gpu_count=2),
        ],
        "miner2": [
            create_job_result(score=0.0, gpu_model="A100", gpu_count=1),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify no incentive logs for failed jobs (Edge case)
    from tests.helpers import extract_incentive_section

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.mining_score is None:
                # Failed jobs should not have incentive logs or only error logs
                section = extract_incentive_section(result.full_log_text)
                if section:
                    # If logs exist, should be error logs
                    assert "Mining score is not set" in section

    assert validator.miner_scores.get("miner1", 0) == 0
    assert validator.miner_scores.get("miner2", 0) == 0

    if validator.miner_scores:
        captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
        weights_by_uid = _weights_by_uid(captured)

        assert weights_by_uid[100] == pytest.approx(TOTAL_BURN_EMISSION / BURNER_COUNT)
        assert weights_by_uid[101] == pytest.approx(TOTAL_BURN_EMISSION / BURNER_COUNT)
        assert weights_by_uid[2] == 0
        assert weights_by_uid[3] == 0


@pytest.mark.asyncio
async def test_scenario_single_miner_monopoly(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Single successful miner should get full mining allocation."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="monopoly_miner"),
        create_neuron_info(uid=3, hotkey="failed_miner"),
    ]

    all_job_results = {
        "monopoly_miner": [
            create_job_result(gpu_model="H100", gpu_count=4, collateral_deposited=True),
        ],
        "failed_miner": [
            create_job_result(score=0.0, gpu_model="A100", gpu_count=2),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    monopoly_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=4,
        total_gpu_count=4,
    )
    failed_score = 0.0

    assert validator.miner_scores["monopoly_miner"] == pytest.approx(MINING_ALLOCATION)
    assert validator.miner_scores.get("failed_miner", 0) == failed_score

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[100] == pytest.approx(TOTAL_BURN_EMISSION / BURNER_COUNT)
    assert weights_by_uid[101] == pytest.approx(TOTAL_BURN_EMISSION / BURNER_COUNT)
    assert weights_by_uid[2] == pytest.approx(MINING_ALLOCATION)
    assert weights_by_uid[3] == 0


@pytest.mark.asyncio
async def test_scenario_gpu_type_scarcity(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Scarce GPU types should have higher per-GPU contribution to score."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="scarce_gpu_miner"),
        create_neuron_info(uid=3, hotkey="abundant_gpu_miner"),
    ]

    all_job_results = {
        "scarce_gpu_miner": [
            create_job_result(gpu_model="H200", gpu_count=1, collateral_deposited=True),
        ],
        "abundant_gpu_miner": [
            create_job_result(gpu_model="RTX4090", gpu_count=1, collateral_deposited=True),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    scarce_score = expected_score(
        portion=GPU_PORTION["H200"],
        gpu_count=1,
        total_gpu_count=1,
    )
    abundant_score = expected_score(
        portion=GPU_PORTION["RTX4090"],
        gpu_count=1,
        total_gpu_count=1,
    )
    total_score = scarce_score + abundant_score
    expected_scarce_cycle = MINING_ALLOCATION * scarce_score / total_score
    expected_abundant_cycle = MINING_ALLOCATION * abundant_score / total_score

    assert validator.miner_scores["scarce_gpu_miner"] == pytest.approx(expected_scarce_cycle)
    assert validator.miner_scores["abundant_gpu_miner"] == pytest.approx(expected_abundant_cycle)
    assert validator.miner_scores["scarce_gpu_miner"] > validator.miner_scores["abundant_gpu_miner"]

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[2] == pytest.approx(expected_scarce_cycle)
    assert weights_by_uid[3] == pytest.approx(expected_abundant_cycle)


@pytest.mark.asyncio
async def test_scenario_equal_competition(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Miners with identical configurations should receive equal weights."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="equal_miner_1"),
        create_neuron_info(uid=3, hotkey="equal_miner_2"),
        create_neuron_info(uid=4, hotkey="equal_miner_3"),
    ]

    all_job_results = {
        "equal_miner_1": [
            create_job_result(gpu_model="A100", gpu_count=2, sysbox_runtime=True, collateral_deposited=True),
        ],
        "equal_miner_2": [
            create_job_result(gpu_model="A100", gpu_count=2, sysbox_runtime=True, collateral_deposited=True),
        ],
        "equal_miner_3": [
            create_job_result(gpu_model="A100", gpu_count=2, sysbox_runtime=True, collateral_deposited=True),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify incentive logging for identical configurations
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, [
                    "mining_score",
                    "total_mining_score",
                    "incentive"
                ])

    equal_score = expected_score(
        portion=GPU_PORTION["A100"],
        gpu_count=2,
        total_gpu_count=6,
    )
    total_score = equal_score * 3
    expected_equal_cycle = MINING_ALLOCATION * equal_score / total_score

    assert validator.miner_scores["equal_miner_1"] == pytest.approx(expected_equal_cycle, rel=1e-3)
    assert validator.miner_scores["equal_miner_2"] == pytest.approx(expected_equal_cycle, rel=1e-3)
    assert validator.miner_scores["equal_miner_3"] == pytest.approx(expected_equal_cycle, rel=1e-3)

    assert validator.miner_scores["equal_miner_1"] == pytest.approx(validator.miner_scores["equal_miner_2"])
    assert validator.miner_scores["equal_miner_2"] == pytest.approx(validator.miner_scores["equal_miner_3"])

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[2] == pytest.approx(expected_equal_cycle)
    assert weights_by_uid[3] == pytest.approx(expected_equal_cycle)
    assert weights_by_uid[4] == pytest.approx(expected_equal_cycle)


@pytest.mark.asyncio
async def test_scenario_mixed_fleet_diversity(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Complex real-world scenario with diverse miners and configurations."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="enterprise_miner"),
        create_neuron_info(uid=3, hotkey="hobbyist_miner"),
        create_neuron_info(uid=4, hotkey="startup_miner"),
        create_neuron_info(uid=5, hotkey="datacenter_miner"),
    ]

    async def get_uptime_side_effect(executor_info):
        if "startup_miner" in str(executor_info.uuid):
            return 30
        return 120

    validator.redis_service.get_executor_uptime = AsyncMock(side_effect=get_uptime_side_effect)

    all_job_results = {
        "enterprise_miner": [
            create_job_result(gpu_model="H100", gpu_count=8, sysbox_runtime=True, collateral_deposited=True),
        ],
        "hobbyist_miner": [
            create_job_result(gpu_model="RTX4090", gpu_count=4, sysbox_runtime=False, collateral_deposited=True),
        ],
        "startup_miner": [
            create_job_result(
                gpu_model="H200",
                gpu_count=2,
                sysbox_runtime=True,
                collateral_deposited=False,
                executor_id="startup_miner_h200",
            ),
            create_job_result(
                gpu_model="A100",
                gpu_count=2,
                sysbox_runtime=True,
                collateral_deposited=False,
                executor_id="startup_miner_a100",
            ),
        ],
        "datacenter_miner": [
            create_job_result(gpu_model="A100", gpu_count=16, sysbox_runtime=True, collateral_deposited=True),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify incentive logging for mixed fleet with diverse GPU types
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, [
                    "mining_score",
                    "total_mining_score",
                    "incentive"
                ])
                # Verify GPU model appears in logs (mixed GPU types edge case)
                assert result.gpu_model in result.full_log_text

    enterprise_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=8,
        total_gpu_count=8,
    )
    hobbyist_score = expected_score(
        portion=GPU_PORTION["RTX4090"],
        gpu_count=4,
        total_gpu_count=4,
        sysbox_runtime=False,
    )
    startup_h200_score = expected_score(
        portion=GPU_PORTION["H200"],
        gpu_count=2,
        total_gpu_count=2,
        collateral_deposited=False,
        uptime_minutes=30,
    )
    startup_a100_score = expected_score(
        portion=GPU_PORTION["A100"],
        gpu_count=2,
        total_gpu_count=18,
        collateral_deposited=False,
        uptime_minutes=30,
    )
    startup_score = startup_h200_score + startup_a100_score
    datacenter_score = expected_score(
        portion=GPU_PORTION["A100"],
        gpu_count=16,
        total_gpu_count=18,
    )
    total_score = enterprise_score + hobbyist_score + startup_score + datacenter_score
    expected_enterprise_cycle = MINING_ALLOCATION * enterprise_score / total_score
    expected_hobbyist_cycle = MINING_ALLOCATION * hobbyist_score / total_score
    expected_startup_cycle = MINING_ALLOCATION * startup_score / total_score
    expected_datacenter_cycle = MINING_ALLOCATION * datacenter_score / total_score

    assert validator.miner_scores["enterprise_miner"] == pytest.approx(expected_enterprise_cycle)
    assert validator.miner_scores["hobbyist_miner"] == pytest.approx(expected_hobbyist_cycle)
    assert validator.miner_scores["startup_miner"] == pytest.approx(expected_startup_cycle, rel=1e-3)
    assert validator.miner_scores["datacenter_miner"] == pytest.approx(expected_datacenter_cycle, rel=1e-3)

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[2] == pytest.approx(expected_enterprise_cycle)
    assert weights_by_uid[3] == pytest.approx(expected_hobbyist_cycle)
    assert weights_by_uid[4] == pytest.approx(expected_startup_cycle)
    assert weights_by_uid[5] == pytest.approx(expected_datacenter_cycle)


@pytest.mark.asyncio
async def test_scenario_burners_only_no_miner_scores(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """When no miners have scores, burners should get all emission."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner1"),
        create_neuron_info(uid=3, hotkey="miner2"),
    ]

    all_job_results = {
        "miner1": [
            create_job_result(score=0.0, job_score=0.0, gpu_model="H100", gpu_count=1),
        ],
        "miner2": [
            create_job_result(score=0.0, job_score=0.0, gpu_model="A100", gpu_count=1),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    assert validator.miner_scores.get("miner1") == 0
    assert validator.miner_scores.get("miner2") == 0

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[100] == pytest.approx(TOTAL_BURN_EMISSION / BURNER_COUNT)
    assert weights_by_uid[101] == pytest.approx(TOTAL_BURN_EMISSION / BURNER_COUNT)
    assert weights_by_uid[2] == 0
    assert weights_by_uid[3] == 0


@pytest.mark.asyncio
async def test_scenario_process_weights_normalization(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Verify process_weights_for_netuid normalizes weights to sum to 1.0."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner1"),
        create_neuron_info(uid=3, hotkey="miner2"),
    ]

    all_job_results = {
        "miner1": [
            create_job_result(gpu_model="H100", gpu_count=2, collateral_deposited=True),
        ],
        "miner2": [
            create_job_result(gpu_model="A100", gpu_count=2, collateral_deposited=True),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    captured = await _run_set_weights_and_capture(
        mock_subtensor_client,
        miners,
        validator.miner_scores,
        normalize=True,
    )

    processed = captured.get("processed_weights")
    assert processed is not None
    assert processed.sum() == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_scenario_extreme_uptime_penalty(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Miner with zero uptime and no collateral should receive minimum score."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="zero_uptime_miner"),
        create_neuron_info(uid=3, hotkey="normal_miner"),
    ]

    async def get_uptime_side_effect(executor_info):
        if "zero_uptime_miner" in str(executor_info.uuid):
            return 0
        return 120

    validator.redis_service.get_executor_uptime = AsyncMock(side_effect=get_uptime_side_effect)

    all_job_results = {
        "zero_uptime_miner": [
            create_job_result(
                gpu_model="H100",
                gpu_count=2,
                sysbox_runtime=False,
                collateral_deposited=False,
                executor_id="zero_uptime_miner",
            ),
        ],
        "normal_miner": [
            create_job_result(
                gpu_model="H100",
                gpu_count=2,
                sysbox_runtime=True,
                collateral_deposited=True,
            ),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify incentive logging even with extreme uptime penalty
    from tests.helpers import assert_incentive_log_present, assert_executor_has_log, assert_log_contains_keys

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.score > 0 and result.incentive_logs:
                assert_incentive_log_present(result.full_log_text)
                assert_executor_has_log(result.full_log_text, str(result.executor_info.uuid))
                assert_log_contains_keys(result.full_log_text, [
                    "mining_score",
                    "total_mining_score",
                    "incentive"
                ])

    zero_uptime_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=2,
        total_gpu_count=4,
        sysbox_runtime=False,
        collateral_deposited=False,
        uptime_minutes=0,
    )
    normal_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=2,
        total_gpu_count=4,
    )
    total_score = zero_uptime_score + normal_score
    expected_zero_cycle = MINING_ALLOCATION * zero_uptime_score / total_score
    expected_normal_cycle = MINING_ALLOCATION * normal_score / total_score

    assert validator.miner_scores["zero_uptime_miner"] == pytest.approx(expected_zero_cycle)
    assert validator.miner_scores["normal_miner"] == pytest.approx(expected_normal_cycle)
    assert validator.miner_scores["zero_uptime_miner"] < validator.miner_scores["normal_miner"]

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[2] == pytest.approx(expected_zero_cycle)
    assert weights_by_uid[3] == pytest.approx(expected_normal_cycle)


@pytest.mark.asyncio
async def test_scenario_weight_distribution_fairness(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Verify weights are distributed fairly proportional to scores."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner_high"),
        create_neuron_info(uid=3, hotkey="miner_low"),
    ]

    all_job_results = {
        "miner_high": [
            create_job_result(gpu_model="H100", gpu_count=6, collateral_deposited=True),
        ],
        "miner_low": [
            create_job_result(gpu_model="H100", gpu_count=2, collateral_deposited=True),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    high_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=6,
        total_gpu_count=8,
    )
    low_score = expected_score(
        portion=GPU_PORTION["H100"],
        gpu_count=2,
        total_gpu_count=8,
    )
    total_score = high_score + low_score
    ratio = high_score / low_score
    expected_high_cycle = MINING_ALLOCATION * high_score / total_score
    expected_low_cycle = MINING_ALLOCATION * low_score / total_score

    assert validator.miner_scores["miner_high"] == pytest.approx(expected_high_cycle)
    assert validator.miner_scores["miner_low"] == pytest.approx(expected_low_cycle)
    assert validator.miner_scores["miner_high"] / validator.miner_scores["miner_low"] == pytest.approx(ratio)

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[2] == pytest.approx(expected_high_cycle)
    assert weights_by_uid[3] == pytest.approx(expected_low_cycle)
    assert weights_by_uid[2] / weights_by_uid[3] == pytest.approx(ratio)


@pytest.mark.asyncio
async def test_scenario_zero_gpu_count_edge_case(
    validator_with_mocks,
    mock_subtensor_client,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    """Handle edge case where GPU type not in total_gpu_model_count_map."""
    validator = validator_with_mocks
    validator.miner_scores = {}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=101, hotkey="burner2"),
        create_neuron_info(uid=2, hotkey="miner1"),
    ]

    all_job_results = {
        "miner1": [
            create_job_result(gpu_model="UNKNOWN_GPU", gpu_count=2, collateral_deposited=True, score=0.0, job_score=0.0),
        ],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    # Verify no incentive logs for failed edge case (unknown GPU, zero score)
    from tests.helpers import extract_incentive_section

    for hotkey, results in all_job_results.items():
        for result in results:
            if result.mining_score is None:
                # Failed jobs should not have incentive logs or only error logs
                section = extract_incentive_section(result.full_log_text)
                if section:
                    # If logs exist, should be error logs
                    assert "Mining score is not set" in section

    assert validator.miner_scores.get("miner1", 0) == 0

    captured = await _run_set_weights_and_capture(mock_subtensor_client, miners, validator.miner_scores)
    weights_by_uid = _weights_by_uid(captured)

    assert weights_by_uid[100] == pytest.approx(TOTAL_BURN_EMISSION / BURNER_COUNT)
    assert weights_by_uid[101] == pytest.approx(TOTAL_BURN_EMISSION / BURNER_COUNT)
    assert weights_by_uid[2] == 0


# ---------------------------------------------------------------------------
# Quantization floor: positive scores must never silently collapse to u16=0
# ---------------------------------------------------------------------------


def test_convert_weights_with_positive_floor_lifts_tiny_positive():
    import numpy as _np
    from clients.subtensor_client import _convert_weights_with_positive_floor

    uids = _np.array([1, 2, 3], dtype=_np.int64)
    # uid 1: cap, uid 2: tiny positive that rounds to 0, uid 3: SDK-zeroed
    weights = _np.array([1.0, 1e-9, 0.0], dtype=_np.float32)

    out_uids, out_vals, floored = _convert_weights_with_positive_floor(uids, weights)

    assert out_uids == [1, 2]
    assert out_vals == [65535, 1]
    assert len(floored) == 1
    assert floored[0][0] == 2
    assert floored[0][1] == pytest.approx(1e-9, rel=1e-3)


def test_convert_weights_with_positive_floor_passes_through_normal_weights():
    import numpy as _np
    from clients.subtensor_client import _convert_weights_with_positive_floor

    uids = _np.array([10, 20], dtype=_np.int64)
    weights = _np.array([1.0, 0.5], dtype=_np.float32)

    out_uids, out_vals, floored = _convert_weights_with_positive_floor(uids, weights)

    assert out_uids == [10, 20]
    assert out_vals == [65535, round(0.5 * 65535)]
    assert floored == []


def test_convert_weights_with_positive_floor_handles_empty_and_all_zero():
    import numpy as _np
    from clients.subtensor_client import _convert_weights_with_positive_floor

    empty = _convert_weights_with_positive_floor(
        _np.array([], dtype=_np.int64), _np.array([], dtype=_np.float32)
    )
    assert empty == ([], [], [])

    all_zero = _convert_weights_with_positive_floor(
        _np.array([7, 8], dtype=_np.int64), _np.array([0.0, 0.0], dtype=_np.float32)
    )
    assert all_zero == ([], [], [])


# ---------------------------------------------------------------------------
# DAH-2622: a hotkey with a live executor must never fall out of the vector
# ---------------------------------------------------------------------------


def test_select_floor_candidates_live_zero_selected_burn_inactive_positive_excluded(
    create_neuron_info,
):
    from clients.subtensor_client import _select_floor_candidates

    miners = [
        create_neuron_info(uid=5, hotkey="live_absent"),
        # uid 101 is a burn uid that the vector dropped: only the burn check can exclude it
        create_neuron_info(uid=101, hotkey="live_burner_absent"),
        create_neuron_info(uid=6, hotkey="inactive"),
        create_neuron_info(uid=7, hotkey="live_present"),
    ]
    uint_uids = [100, 7]
    active_hotkeys = {"live_absent", "live_burner_absent", "live_present"}

    candidates = _select_floor_candidates(uint_uids, miners, active_hotkeys, {100, 101})

    assert candidates == [(5, "live_absent")]


def test_apply_eligibility_floor_funds_candidate_from_largest_burn_uid(create_neuron_info):
    from clients.subtensor_client import _apply_eligibility_floor

    miners = [
        create_neuron_info(uid=100, hotkey="big_burner"),
        create_neuron_info(uid=101, hotkey="small_burner"),
        create_neuron_info(uid=3, hotkey="earner"),
        create_neuron_info(uid=5, hotkey="live_absent"),
    ]
    uint_uids = [100, 101, 3]
    uint_weights = [65535, 30000, 500]

    out_uids, out_weights, floored_hotkeys = _apply_eligibility_floor(
        uint_uids, uint_weights, miners, {"live_absent"}, {100, 101}
    )

    assert dict(zip(out_uids, out_weights)) == {100: 65534, 101: 30000, 3: 500, 5: 1}
    assert floored_hotkeys == ["live_absent"]
    assert uint_uids == [100, 101, 3]
    assert uint_weights == [65535, 30000, 500]


def test_apply_eligibility_floor_stops_when_burn_exhausted(create_neuron_info):
    from clients.subtensor_client import _apply_eligibility_floor

    miners = [
        create_neuron_info(uid=100, hotkey="burner"),
        create_neuron_info(uid=5, hotkey="live_a"),
        create_neuron_info(uid=6, hotkey="live_b"),
    ]
    # the single burn entry carries exactly one spare unit above the u16 minimum
    uint_uids = [100]
    uint_weights = [2]

    out_uids, out_weights, floored_hotkeys = _apply_eligibility_floor(
        uint_uids, uint_weights, miners, {"live_a", "live_b"}, {100}
    )

    assert floored_hotkeys == ["live_a"]
    assert dict(zip(out_uids, out_weights)) == {100: 1, 5: 1}


def test_apply_eligibility_floor_floors_several_miners_in_one_publish(create_neuron_info):
    from clients.subtensor_client import _apply_eligibility_floor

    miners = [
        create_neuron_info(uid=100, hotkey="big_burner"),
        create_neuron_info(uid=101, hotkey="small_burner"),
        create_neuron_info(uid=5, hotkey="live_a"),
        create_neuron_info(uid=6, hotkey="live_b"),
        create_neuron_info(uid=7, hotkey="live_c"),
    ]
    # the donor is re-picked per candidate, so the two burn entries drain evenly
    uint_uids = [100, 101]
    uint_weights = [10, 9]

    out_uids, out_weights, floored_hotkeys = _apply_eligibility_floor(
        uint_uids, uint_weights, miners, {"live_a", "live_b", "live_c"}, {100, 101}
    )

    assert floored_hotkeys == ["live_a", "live_b", "live_c"]
    assert dict(zip(out_uids, out_weights)) == {100: 8, 101: 8, 5: 1, 6: 1, 7: 1}
    assert sum(out_weights) == sum(uint_weights)


def test_apply_eligibility_floor_without_any_burn_entry_changes_nothing(create_neuron_info):
    from clients.subtensor_client import _apply_eligibility_floor

    miners = [
        create_neuron_info(uid=3, hotkey="earner"),
        create_neuron_info(uid=5, hotkey="live_absent"),
    ]
    uint_uids = [3]
    uint_weights = [500]

    out_uids, out_weights, floored_hotkeys = _apply_eligibility_floor(
        uint_uids, uint_weights, miners, {"live_absent"}, {100}
    )

    assert floored_hotkeys == []
    assert (out_uids, out_weights) == ([3], [500])


@pytest.mark.asyncio
async def test_set_weights_mirror_payload_carries_the_floored_entries(
    mock_subtensor_client,
    create_neuron_info,
):
    miners = [
        create_neuron_info(uid=100, hotkey="burner"),
        create_neuron_info(uid=2, hotkey="earner"),
        create_neuron_info(uid=3, hotkey="idle"),
    ]

    with patch.object(mock_subtensor_client, "send_weights_to_lium", AsyncMock()) as mirror:
        await _run_set_weights_and_capture(
            mock_subtensor_client,
            miners,
            {"burner": 0.8, "earner": 0.2},
            normalize=True,
            active_hotkeys={"idle"},
        )

    payload = mirror.call_args.args[0]
    assert dict(zip(payload["uids"], payload["weights"]))[3] == 1


@pytest.mark.asyncio
async def test_set_weights_floors_positive_scores_to_min_one(
    mock_subtensor_client,
    create_neuron_info,
):
    """A miner with a tiny but positive score must be submitted with weight >= 1,
    while a miner with score 0 (absent from miner_scores) must be excluded.
    """
    miners = [
        create_neuron_info(uid=100, hotkey="burner"),
        create_neuron_info(uid=2, hotkey="tiny"),
        create_neuron_info(uid=3, hotkey="zero"),
    ]
    miner_scores = {"burner": 1.0, "tiny": 1e-9}  # "zero" deliberately absent

    # normalize=True → captured floats sum to ~1, mirroring the SDK's real output.
    # active_hotkeys=set() pins the pre-DAH-2622 baseline: with no live hotkey the
    # eligibility floor never fires, so a zero-score miner stays out of the vector.
    await _run_set_weights_and_capture(
        mock_subtensor_client, miners, miner_scores, normalize=True, active_hotkeys=set()
    )

    call = mock_subtensor_client.subtensor.set_weights.call_args
    submitted_uids = list(call.kwargs["uids"])
    submitted_weights = list(call.kwargs["weights"])
    by_uid = dict(zip(submitted_uids, submitted_weights))

    assert 2 in by_uid, f"tiny-positive uid was dropped: {by_uid}"
    assert by_uid[2] >= 1
    assert by_uid[100] == 65535
    assert 3 not in by_uid


@pytest.mark.asyncio
async def test_set_weights_floors_active_hotkey_with_zero_score(
    mock_subtensor_client,
    create_neuron_info,
):
    miners = [
        create_neuron_info(uid=100, hotkey="burner"),
        create_neuron_info(uid=2, hotkey="earner"),
        create_neuron_info(uid=3, hotkey="idle"),
    ]
    miner_scores = {"burner": 0.8, "earner": 0.2}  # "idle" scored nothing this epoch

    await _run_set_weights_and_capture(
        mock_subtensor_client, miners, miner_scores, normalize=True, active_hotkeys={"idle"}
    )

    call = mock_subtensor_client.subtensor.set_weights.call_args
    by_uid = dict(zip(list(call.kwargs["uids"]), list(call.kwargs["weights"])))

    assert by_uid[3] == 1, f"live idle miner was dropped from the vector: {by_uid}"
    assert by_uid[100] == 65534


@pytest.mark.asyncio
async def test_set_weights_floors_live_miner_on_a_retired_burner_uid(
    mock_subtensor_client,
    create_neuron_info,
):
    # BURNERS holds the pre-DAH-2274 slots; with the new burn logic on they receive nothing
    # and are ordinary miner uids, so a live miner sitting there still deserves the floor.
    miners = [
        create_neuron_info(uid=100, hotkey="burner"),
        create_neuron_info(uid=2, hotkey="earner"),
        create_neuron_info(uid=1, hotkey="idle_on_retired_burner_uid"),
    ]

    await _run_set_weights_and_capture(
        mock_subtensor_client,
        miners,
        {"burner": 0.8, "earner": 0.2},
        normalize=True,
        active_hotkeys={"idle_on_retired_burner_uid"},
    )

    call = mock_subtensor_client.subtensor.set_weights.call_args
    by_uid = dict(zip(list(call.kwargs["uids"]), list(call.kwargs["weights"])))

    assert by_uid[1] == 1, f"live miner on a retired burner uid was dropped: {by_uid}"
    assert by_uid[100] == 65534


@pytest.mark.asyncio
async def test_set_weights_logs_eligibility_floor_applied(
    mock_subtensor_client,
    create_neuron_info,
    caplog,
):
    miners = [
        create_neuron_info(uid=100, hotkey="burner"),
        create_neuron_info(uid=2, hotkey="earner"),
        create_neuron_info(uid=3, hotkey="idle"),
    ]

    with caplog.at_level(logging.INFO):
        await _run_set_weights_and_capture(
            mock_subtensor_client,
            miners,
            {"burner": 0.8, "earner": 0.2},
            normalize=True,
            active_hotkeys={"idle"},
        )

    assert _floor_log_extras(caplog) == [{"floor_hotkeys": ["idle"], "floor_count": 1}]


@pytest.mark.asyncio
async def test_set_weights_logs_eligibility_floor_even_when_empty(
    mock_subtensor_client,
    create_neuron_info,
    caplog,
):
    miners = [
        create_neuron_info(uid=100, hotkey="burner"),
        create_neuron_info(uid=2, hotkey="earner"),
    ]

    with caplog.at_level(logging.INFO):
        await _run_set_weights_and_capture(
            mock_subtensor_client,
            miners,
            {"burner": 0.8, "earner": 0.2},
            normalize=True,
            active_hotkeys=set(),
        )

    assert _floor_log_extras(caplog) == [{"floor_hotkeys": [], "floor_count": 0}]


@pytest.mark.asyncio
async def test_set_weights_floor_does_not_dilute_earning_miners(
    mock_subtensor_client,
    create_neuron_info,
):
    miners = [
        create_neuron_info(uid=100, hotkey="burner"),
        create_neuron_info(uid=2, hotkey="earner_big"),
        create_neuron_info(uid=3, hotkey="earner_small"),
        create_neuron_info(uid=4, hotkey="idle"),
    ]
    miner_scores = {"burner": 0.8, "earner_big": 0.15, "earner_small": 0.05}

    await _run_set_weights_and_capture(
        mock_subtensor_client, miners, miner_scores, normalize=True, active_hotkeys=set()
    )
    without_floor = dict(
        zip(
            list(mock_subtensor_client.subtensor.set_weights.call_args.kwargs["uids"]),
            list(mock_subtensor_client.subtensor.set_weights.call_args.kwargs["weights"]),
        )
    )
    await _run_set_weights_and_capture(
        mock_subtensor_client, miners, miner_scores, normalize=True, active_hotkeys={"idle"}
    )
    with_floor = dict(
        zip(
            list(mock_subtensor_client.subtensor.set_weights.call_args.kwargs["uids"]),
            list(mock_subtensor_client.subtensor.set_weights.call_args.kwargs["weights"]),
        )
    )

    assert with_floor[2] == without_floor[2]
    assert with_floor[3] == without_floor[3]
    assert with_floor[4] == 1
    assert with_floor[100] == without_floor[100] - 1


@pytest.mark.asyncio
async def test_set_weights_floor_leaves_normalized_scores_untouched(
    mock_subtensor_client,
    create_neuron_info,
):
    miners = [
        create_neuron_info(uid=100, hotkey="burner"),
        create_neuron_info(uid=2, hotkey="earner"),
        create_neuron_info(uid=4, hotkey="idle"),
    ]
    miner_scores = {"burner": 0.8, "earner": 0.2}

    await _run_set_weights_and_capture(
        mock_subtensor_client, miners, miner_scores, normalize=True, active_hotkeys=set()
    )
    without_floor = mock_subtensor_client.redis_service.publish.call_args.args
    await _run_set_weights_and_capture(
        mock_subtensor_client, miners, miner_scores, normalize=True, active_hotkeys={"idle"}
    )
    with_floor = mock_subtensor_client.redis_service.publish.call_args.args

    assert with_floor == without_floor


@pytest.mark.asyncio
async def test_active_hotkeys_reflects_successful_executors_only(
    validator_with_mocks,
    mock_settings,
    create_job_result,
    create_neuron_info,
):
    validator = validator_with_mocks
    validator.miner_scores = {}
    validator.active_hotkeys = {"stale_from_previous_cycle"}

    miners = [
        create_neuron_info(uid=100, hotkey="burner1"),
        create_neuron_info(uid=2, hotkey="idle_miner"),
        create_neuron_info(uid=3, hotkey="spot_miner"),
        create_neuron_info(uid=4, hotkey="dead_miner"),
    ]
    spot_result = create_job_result(gpu_model="H100", gpu_count=1)
    spot_result.is_spot = True

    all_job_results = {
        "idle_miner": [create_job_result(gpu_model="H100", gpu_count=1)],
        "spot_miner": [spot_result],
        "dead_miner": [create_job_result(score=0.0, job_score=0.0, gpu_model="A100", gpu_count=2)],
    }

    await _run_sync_with_jobs(validator, miners, all_job_results)

    assert validator.active_hotkeys == {"idle_miner", "spot_miner"}


@pytest.mark.asyncio
async def test_sync_passes_active_hotkeys_to_set_weights(
    validator_with_mocks,
    mock_settings,
    create_neuron_info,
):
    validator = validator_with_mocks
    validator.miner_scores = {"live_miner": 1.0}
    validator.active_hotkeys = {"live_miner"}
    validator.completed_cycles_since_start = 1
    # current_block == 100 in the fixture, so no new job cycle starts in this sync()
    validator.last_job_run_blocks = 100
    validator.subtensor_client.get_miners = AsyncMock(
        return_value=[create_neuron_info(uid=2, hotkey="live_miner")]
    )
    validator.subtensor_client.should_set_weights = AsyncMock(return_value=True)
    validator.subtensor_client.set_weights = AsyncMock()

    await validator.sync()

    call = validator.subtensor_client.set_weights.call_args
    assert call.kwargs["active_hotkeys"] == {"live_miner"}
