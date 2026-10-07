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


# ---------------------------------------------------------------------------
# Quantization floor: positive scores must never silently collapse to u16=0
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DAH-2622: a hotkey with a live executor must never fall out of the vector
# ---------------------------------------------------------------------------


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


