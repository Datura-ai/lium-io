"""DAH-4658: a scrape that finds no GPU while the renter resets them (or just after) is held, not a 0.

8x H200, 23 Oct: `nvidia-smi -r` on all 8 cards ran ~45 s, a cycle's scrape listed 0 GPUs and failed with
SCRAPE_FAILED_NO_GPU, and the node scored 0. The gpu_reset mark was never read: the scrape is fatal and ends the
pipeline before the checks that skip on it. The mark was also gone by the time a scrape that ended just after the
reset was judged.
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from payload_models.payloads import ResetPodGpuRequest
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.docker_service import DockerService
from core.validator import specs_to_publish
from services.executor_rollout import (
    ExecutorRolloutTracker,
    carry_renter_gpu_reset_results,
    remember_rented_scored_results,
)
from services.gpu_reset import RECENT_RENTER_GPU_RESET_SECONDS, GpuResetReport, renter_gpu_reset_recent
from services.redis_service import (
    GPU_RESET_PENDING_KIND,
    LAST_SCORED_RESULT_TTL_SECONDS,
    RESET_VERIFIED_JOB_CHANNEL,
    last_scored_result_key,
)
from protocol.vc_protocol.compute_requests import FillerRevenueByGpuConfig, RentedExecutor, RentedExecutorsResponse
from protocol.vc_protocol.validator_requests import ContainerState, PodContainerState
from services.ssh_service import SSHService
from services.task.models import build_msg
from test_executor_rollout_grace import _failed, _scored, _validator_process
from neurons.validators.src.services.task.messages import GpuFingerprintMessages as FingerprintMsg
from test_rented_host_gpu_fault_reset import (
    ENCRYPT_KEY,
    EXECUTOR,
    NO_GPU_REPORT,
    POD_ID,
    RENTAL_OWNER,
    DummyBackendClient,
    DummyScoreCalculator,
    DummySSHClient,
    MockContainerCleanup,
    _driver_report,
    _redis_service,
    _resets,
    _Runner,
    _scrape_result,
    _Sink,
    build_rented_data,
)

from helpers import FakeRedis, build_context_config, build_services, build_state
from neurons.validators.src.services.task.checks import machine_spec_scrape
from neurons.validators.src.services.task.checks import gpu_fingerprint
from neurons.validators.src.services.task.checks.gpu_fingerprint import GpuFingerprintCheck
from neurons.validators.src.services.task.checks.gpu_loss_hold import GpuLossHoldCheck
from neurons.validators.src.services.task.checks.machine_spec_scrape import MachineSpecScrapeCheck
from neurons.validators.src.services.task.checks.rented_machine import TenantEnforcementCheck
from neurons.validators.src.services.task.messages import MachineSpecMessages as Msg
from neurons.validators.src.services.task.pipeline import Pipeline
from neurons.validators.src.services.task.result_handler import ResultHandler

ZERO_GPUS = _scrape_result(stdout=NO_GPU_REPORT, exit_code=1)
RESET_REQUIRED = _scrape_result(stdout=_driver_report("NVMLError(16)"), exit_code=1)


async def _cycle(context_factory, service, *, scrape, pod_gpu_count: int | None = None):
    """One rented cycle through the real Pipeline and ResultHandler, with the node's Redis marks in `service`."""
    backend = DummyBackendClient(active=True)
    backend.report_rented_gpu_drop = AsyncMock()
    rented_data = build_rented_data(EXECUTOR, {"containers": [{"name": "tenant-123", "pod_id": POD_ID}], "owner_flag": False})
    rented_data.executors[EXECUTOR].pods[0].gpu_count = pod_gpu_count
    ctx = context_factory(
        miner_hotkey=RENTAL_OWNER,
        services=build_services(
            ssh=SSHService(),
            redis=service,
            score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
            container_cleanup=MockContainerCleanup(),
            backend=backend,
        ),
        config=build_context_config(machine_scrape_filename="scrape.sh", obfuscation_keys={}),
        state=build_state(
            remote_dir="/remote",
            rented_data=rented_data,
        ),
        runner=_Runner(scrape),
        ssh=DummySSHClient(pod_running=True),
        encrypt_key=ENCRYPT_KEY,
        verified=await service.get_verified_job_info(RENTAL_OWNER, EXECUTOR),
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )
    with patch.object(machine_spec_scrape.settings, "RENTED_HOST_GPU_FAULT_RESET_ENABLED", True):
        ok, events, ctx = await Pipeline(
            [MachineSpecScrapeCheck(), GpuLossHoldCheck(), GpuFingerprintCheck(), TenantEnforcementCheck()], _Sink()
        ).run(ctx)
    await ResultHandler(redis_service=service, dry_run=False)._persist_verification_data(
        miner_hotkey=RENTAL_OWNER, executor_id=EXECUTOR, verified_job_info=ctx.verified, context=ctx, success=ok
    )
    return ok, events[-1], ctx, backend


H200_UUIDS = [f"GPU-h200-{i}" for i in range(8)]


def _gpus_listed(uuids: list[str]):
    specs = {"gpu": {"count": len(uuids), "details": [{"name": "NVIDIA H200", "uuid": u} for u in uuids]}, "gpu_processes": []}
    return _scrape_result(stdout=SSHService()._encrypt(ENCRYPT_KEY, json.dumps(specs)), exit_code=0)


async def _reset_in_progress(service, gpu_uuids=H200_UUIDS) -> None:
    await service.add_pending_pod(RENTAL_OWNER, EXECUTOR, POD_ID, kind=GPU_RESET_PENDING_KIND, gpu_uuids=gpu_uuids)


async def _reset_just_ended(service, gpu_uuids=H200_UUIDS) -> None:
    await service.mark_gpu_reset_recent(RENTAL_OWNER, EXECUTOR, POD_ID, gpu_uuids, RECENT_RENTER_GPU_RESET_SECONDS)


async def _scored_on_all_eight(context_factory, service) -> None:
    """The anchor the holds compare against: a scored cycle that saw all 8 cards."""
    assert (await _cycle(context_factory, service, scrape=_gpus_listed(H200_UUIDS)))[0] is True


@pytest.mark.parametrize("scrape", [ZERO_GPUS, RESET_REQUIRED], ids=["zero-gpus", "reset-required"])
@pytest.mark.parametrize("mark", [_reset_in_progress, _reset_just_ended], ids=["in-progress", "just-ended"])
@pytest.mark.asyncio
async def test_a_scrape_without_gpus_during_a_renter_reset_is_held(context_factory, mark, scrape):
    service = _redis_service()
    await _scored_on_all_eight(context_factory, service)
    await mark(service)

    ok, event, ctx, backend = await _cycle(context_factory, service, scrape=scrape)

    assert ok is False
    assert event.check_id == MachineSpecScrapeCheck.check_id
    assert event.reason_code == Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason
    assert ctx.clear_verified_job_info is False
    assert _resets(service) == []
    backend.report_rented_gpu_drop.assert_not_awaited()


@pytest.mark.parametrize("mark", [_reset_in_progress, _reset_just_ended], ids=["in-progress", "just-ended"])
@pytest.mark.asyncio
async def test_a_scrape_without_gpus_on_a_split_node_is_not_held_by_a_reset_of_four(context_factory, mark):
    """The other pod's 4 cards are missing too: not the renter's doing, judged as before."""
    service = _redis_service()
    await _scored_on_all_eight(context_factory, service)
    await mark(service, H200_UUIDS[:4])

    ok, event, ctx, _ = await _cycle(context_factory, service, scrape=ZERO_GPUS)

    assert ok is False
    assert event.reason_code == Msg.SCRAPE_FAILED_NO_GPU.reason
    assert ctx.clear_verified_job_info is True


@pytest.mark.asyncio
async def test_a_scrape_without_gpus_and_no_renter_reset_still_fails_as_before(context_factory):
    service = _redis_service()

    ok, event, ctx, _ = await _cycle(context_factory, service, scrape=ZERO_GPUS)

    assert ok is False
    assert event.reason_code == Msg.SCRAPE_FAILED_NO_GPU.reason
    assert ctx.clear_verified_job_info is True
    assert len(_resets(service)) == 1


@pytest.mark.parametrize("mark", [_reset_in_progress, _reset_just_ended], ids=["in-progress", "just-ended"])
@pytest.mark.asyncio
async def test_a_split_node_whose_renter_resets_its_four_gpus_is_held_not_zeroed(context_factory, mark):
    """8x H200 split 4+4: the renter resets their 4; the scrape lists only the other 4, so the zero-GPU hold
    never fires. The loss hold must hold the cycle, or GpuFingerprintCheck fails it with GPU_MISSING."""
    service = _redis_service()
    with patch.object(gpu_fingerprint.settings, "GPU_ANCHOR_HARD_ENABLED", True):
        await _scored_on_all_eight(context_factory, service)
        await mark(service, H200_UUIDS[:4])

        ok, event, ctx, _ = await _cycle(context_factory, service, scrape=_gpus_listed(H200_UUIDS[4:]))

    assert ok is False
    assert event.check_id == GpuLossHoldCheck.check_id
    assert event.reason_code == Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason
    assert ctx.clear_verified_job_info is False
    assert _resets(service) == []


async def _old_recent_mark_without_gpus(service, gpu_uuids) -> None:
    """A recent mark written before it named the reset GPUs: the bare pod id."""
    await service.set(f"gpu_reset_recent:{RENTAL_OWNER}:{EXECUTOR}", POD_ID, ex=RECENT_RENTER_GPU_RESET_SECONDS)


@pytest.mark.parametrize(
    ("mark", "listed"),
    [
        (_reset_in_progress, H200_UUIDS[4:7]),
        (_reset_just_ended, H200_UUIDS[4:7]),
        (_reset_just_ended, H200_UUIDS[:7]),
        (_old_recent_mark_without_gpus, H200_UUIDS[4:]),
    ],
    ids=["in-progress-other-pods-card-too", "just-ended-other-pods-card-too", "just-ended-only-other-pods-card", "old-mark-without-gpus"],
)
@pytest.mark.asyncio
async def test_a_missing_card_outside_the_reset_gpus_still_fails_the_fingerprint(context_factory, mark, listed):
    service = _redis_service()
    with patch.object(gpu_fingerprint.settings, "GPU_ANCHOR_HARD_ENABLED", True):
        await _scored_on_all_eight(context_factory, service)
        await mark(service, H200_UUIDS[:4])

        ok, event, ctx, _ = await _cycle(context_factory, service, scrape=_gpus_listed(listed))

    assert ok is False
    assert event.reason_code == FingerprintMsg.GPU_MISSING.reason
    assert ctx.clear_verified_job_info is True


@pytest.mark.asyncio
async def test_an_old_recent_mark_without_gpus_still_names_its_pod():
    service = _redis_service()
    await _old_recent_mark_without_gpus(service, [])

    assert await renter_gpu_reset_recent(service, RENTAL_OWNER, EXECUTOR, POD_ID) is True
    assert await renter_gpu_reset_recent(service, RENTAL_OWNER, EXECUTOR, "pod-other") is False


async def _job_result(ctx, ok: bool, event):
    """The cycle's JobResult as TaskService builds it (persistence already done by `_cycle`)."""
    miner_info = MagicMock(miner_hotkey=RENTAL_OWNER, job_batch_id="batch")
    result = await ResultHandler(redis_service=MagicMock(), dry_run=True).handle_result(
        context=ctx, miner_info=miner_info, executor_info=ctx.executor, verified_job_info=ctx.verified or {},
        log_text="", success=ok, validation_event=event,
    )
    if not ok:
        result.failure_reason_code = event.reason_code  # as TaskService sets it
    return result


@pytest.mark.asyncio
async def test_a_carried_split_node_keeps_only_its_rented_gpus_in_the_mining_pool(context_factory):
    """8x H200 split 4+4, the renter's pod holds 4: the carried result must still say 4 rented, or the
    split pricing reads the whole node as rented for up to the carry TTL."""
    service = _redis_service()
    with patch.object(gpu_fingerprint.settings, "GPU_ANCHOR_HARD_ENABLED", True):
        ok, event, ctx, _ = await _cycle(context_factory, service, scrape=_gpus_listed(H200_UUIDS), pod_gpu_count=4)
        previous = await _job_result(ctx, ok, event)
        previous.supports_gpu_splitting, previous.gpu_splitting_min_count = True, 4  # the provider's split opt-in
        await remember_rented_scored_results({RENTAL_OWNER: [previous]}, service)
        await _reset_in_progress(service, H200_UUIDS[:4])

        ok, event, ctx, _ = await _cycle(context_factory, service, scrape=_gpus_listed(H200_UUIDS[4:]), pod_gpu_count=4)
        held = await _job_result(ctx, ok, event)
    job_results = {RENTAL_OWNER: [held]}

    await carry_renter_gpu_reset_results(job_results, service, {EXECUTOR}, ctx.state.rented_data, {})

    [carried] = job_results[RENTAL_OWNER]
    assert carried.carried_for_renter_gpu_reset is True
    assert (carried.is_rented, carried.gpu_count, carried.rented_gpu_count) == (True, 8, 4)
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), job_results, {"NVIDIA H200": 8})
    incentive._expand_partially_rented_split_results()
    assert [(result.is_rented, result.gpu_count) for result in job_results[RENTAL_OWNER]] == [(True, 4), (False, 4)]


@pytest.mark.asyncio
async def test_four_gpus_missing_without_a_renter_reset_still_fail_the_fingerprint(context_factory):
    service = _redis_service()
    with patch.object(gpu_fingerprint.settings, "GPU_ANCHOR_HARD_ENABLED", True):
        await _cycle(context_factory, service, scrape=_gpus_listed(H200_UUIDS))

        ok, event, ctx, _ = await _cycle(context_factory, service, scrape=_gpus_listed(H200_UUIDS[4:]))

    assert ok is False
    assert event.reason_code == FingerprintMsg.GPU_MISSING.reason
    assert ctx.clear_verified_job_info is True
    assert len(_resets(service)) == 1


@pytest.mark.asyncio
async def test_the_recent_reset_mark_expires_after_the_window():
    service = _redis_service()
    await _reset_just_ended(service)

    ttl = await service.redis.ttl(f"gpu_reset_recent:{RENTAL_OWNER}:{EXECUTOR}")

    assert 0 < ttl <= RECENT_RENTER_GPU_RESET_SECONDS
    assert await renter_gpu_reset_recent(service, RENTAL_OWNER, EXECUTOR) is True
    assert await renter_gpu_reset_recent(service, RENTAL_OWNER, EXECUTOR, POD_ID) is True
    assert await renter_gpu_reset_recent(service, RENTAL_OWNER, EXECUTOR, "pod-other") is False
    assert await renter_gpu_reset_recent(service, RENTAL_OWNER, "other-executor") is False


@pytest.mark.asyncio
async def test_an_unreadable_recent_reset_mark_is_no_mark():
    assert await renter_gpu_reset_recent(FakeRedis(failing=True), "hk", "ex") is False
    assert await renter_gpu_reset_recent(AsyncMock(), "hk", "ex") is False  # not a real pod id


@pytest.mark.asyncio
async def test_the_held_result_gets_no_verdict_this_cycle():
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.fetch_executor_digest_or_none = AsyncMock(return_value=None)
    validator_process.observe_executor_rollout = AsyncMock(return_value=MagicMock(covers=lambda _: False))
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)
    failed = _failed("node-2", Msg.SCRAPE_FAILED_NO_GPU.reason)
    healthy = _scored("node-3")

    kept, withheld = await validator_process.withhold_verdicts_for_rollout(
        {"5Miner": [held, failed, healthy]}, 7500, "batch-1", MagicMock()
    )

    assert kept == {"5Miner": [failed, healthy]}
    assert [(w.miner_hotkey, w.result.executor_info.uuid) for w in withheld] == [("5Miner", "node-1")]


def _rented_scored(uuid: str):
    """A rented result with the nested models a real run carries, so the Redis round trip is the real one."""
    result = _scored(uuid)
    result.is_rented = True
    result.spec = {"gpu": {"count": 8, "details": [{"name": "H100"}]}}
    result.validation_event = build_msg(event="Validation completed", reason="VALIDATION_COMPLETED", severity="info", impact="")
    result.pod_states = [
        PodContainerState(pod_id=POD_ID, container_state=ContainerState.RUNNING, observed_at=datetime.now(UTC))
    ]
    return result


async def _carry_then_withhold(service, results, rented_executor_ids=frozenset({"node-1"})):
    """The cycle's order: carry before the duplicate pass and the GPU count, withhold after."""
    job_results = {"5Miner": results}
    await carry_renter_gpu_reset_results(job_results, service, rented_executor_ids, None, {})
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.fetch_executor_digest_or_none = AsyncMock(return_value=None)
    validator_process.observe_executor_rollout = AsyncMock(return_value=MagicMock(covers=lambda _: False))
    return await validator_process.withhold_verdicts_for_rollout(job_results, 7500, "batch-2", MagicMock())


@pytest.mark.asyncio
async def test_a_held_scrape_carries_the_previous_score_into_the_cycle():
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service)
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)

    kept, withheld = await _carry_then_withhold(service, [held])

    [carried] = kept["5Miner"]
    assert (carried.score, carried.gpu_model, carried.gpu_count) == (1.0, "H100", 8)
    assert carried.is_successful
    assert carried.validation_event.reason_code == "VALIDATION_COMPLETED"
    assert carried.job_batch_id == held.job_batch_id
    assert carried.carried_for_renter_gpu_reset is True
    assert withheld == []
    assert specs_to_publish(kept["5Miner"]) == []
    assert RESET_VERIFIED_JOB_CHANNEL not in [call.args[0] for call in service.publish.await_args_list]
    assert 0 < await service.redis.ttl(last_scored_result_key("5Miner", "node-1")) <= LAST_SCORED_RESULT_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_carried_score_takes_the_pay_flags_of_this_cycle_and_a_fresh_ban():
    service = _redis_service()
    previous = _rented_scored("node-1")
    previous.spec["gpu"]["details"][0]["uuid"] = "GPU-banned"
    previous.is_spot = previous.is_on_spot_pay_list = previous.provider_discord_connected = True
    previous.gpu_model = "NVIDIA H100 80GB HBM3"
    previous.default_job_owner, previous.has_lium_filler, previous.filler_revenue_per_gpu_hour = "miner", False, 9.0
    await remember_rented_scored_results({"5Miner": [previous]}, service)
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)
    held.is_spot = held.is_on_spot_pay_list = held.provider_discord_connected = False
    held.is_provider_email_held = held.is_new_rentals_paused = True
    held.default_job_owner, held.has_lium_filler = "lium", True
    job_results = {"5Miner": [held]}
    rented_data = _rented_executors(
        banned_provider_guids=["GPU-banned"],
        filler_revenue_by_gpu_config=[
            FillerRevenueByGpuConfig(base_model="H100", gpu_count=8, usd_per_gpu_hour=2.5, gpu_hours=10_000)
        ],
    )

    await carry_renter_gpu_reset_results(job_results, service, {"node-1"}, rented_data, {})

    [carried] = job_results["5Miner"]
    assert carried.carried_for_renter_gpu_reset is True
    assert (carried.is_spot, carried.is_on_spot_pay_list, carried.provider_discord_connected) == (False, False, False)
    assert carried.is_provider_email_held is True
    assert carried.is_provider_banned is True
    assert (carried.is_new_rentals_paused, carried.default_job_owner, carried.has_lium_filler) == (True, "lium", True)
    # this cycle's filler average for the carried configuration (8x H100), not the held scrape's
    assert carried.filler_revenue_per_gpu_hour == 2.5


@pytest.mark.asyncio
async def test_a_held_scrape_without_a_previous_score_is_withheld():
    service = _redis_service()
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)

    kept, withheld = await _carry_then_withhold(service, [held])

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


@pytest.mark.asyncio
async def test_a_failed_scrape_without_the_mark_keeps_its_zero_score():
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service)
    failed = _failed("node-1", Msg.SCRAPE_FAILED_NO_GPU.reason)

    kept, withheld = await _carry_then_withhold(service, [failed])

    assert kept == {"5Miner": [failed]}
    assert failed.score == 0
    assert withheld == []


@pytest.mark.asyncio
async def test_an_unreadable_previous_score_withholds_the_held_scrape():
    broken = MagicMock(get=AsyncMock(side_effect=ConnectionError("down")))
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)

    kept, withheld = await _carry_then_withhold(broken, [held])

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


@pytest.mark.asyncio
async def test_only_scored_rented_results_are_remembered_and_a_redis_error_is_logged_only():
    service = _redis_service()
    idle = _scored("node-idle")
    failed = _failed("node-failed", Msg.SCRAPE_FAILED_NO_GPU.reason)
    failed.is_rented = True

    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1"), idle, failed]}, service)
    await remember_rented_scored_results(
        {"5Miner": [_rented_scored("node-1")]}, MagicMock(write_atomically=AsyncMock(side_effect=ConnectionError))
    )

    assert await service.get(last_scored_result_key("5Miner", "node-1"))
    assert await service.get(last_scored_result_key("5Miner", "node-idle")) is None
    assert await service.get(last_scored_result_key("5Miner", "node-failed")) is None


@pytest.mark.asyncio
async def test_a_failure_of_its_own_after_the_last_score_leaves_nothing_to_carry():
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service)
    await remember_rented_scored_results({"5Miner": [_failed("node-1", Msg.SCRAPE_FAILED_NO_GPU.reason)]}, service)
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)

    kept, withheld = await _carry_then_withhold(service, [held])

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


@pytest.mark.asyncio
async def test_a_rental_that_ended_after_the_reset_carries_nothing():
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service)
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)

    kept, withheld = await _carry_then_withhold(service, [held], rented_executor_ids=frozenset())

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


def _rented_executors(*executor_ids: str, **fields) -> RentedExecutorsResponse:
    rented = RentedExecutor(miner_hotkey="5Miner", executor_ip_address="203.0.113.5", executor_ip_port="22", pods=[])
    return RentedExecutorsResponse(executors={executor_id: rented for executor_id in executor_ids}, **fields)


@pytest.mark.parametrize(
    ("rented_now", "snapshot", "carried"),
    [
        (_rented_executors("NODE-1"), _rented_executors(), True),
        (None, _rented_executors("NODE-1"), True),
        (None, _rented_executors(), False),
    ],
    ids=["listed-now", "no-answer-cycle-snapshot-lists-it", "no-answer-snapshot-does-not"],
)
@pytest.mark.asyncio
async def test_the_cycle_asks_the_backend_once_whether_the_node_is_still_rented(rented_now, snapshot, carried):
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service)
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.redis_service = service
    validator_process.backend_client = MagicMock(get_rented_executors_now=AsyncMock(return_value=rented_now))
    job_results = {"5Miner": [_failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason), _scored("node-2")]}

    await validator_process.carry_scores_over_renter_gpu_resets(job_results, snapshot, {})
    await validator_process.carry_scores_over_renter_gpu_resets({"5Miner": [_scored("node-3")]}, snapshot, {})

    assert job_results["5Miner"][0].carried_for_renter_gpu_reset is carried
    validator_process.backend_client.get_rented_executors_now.assert_awaited_once()


def _reset_service(service) -> DockerService:
    docker_service = DockerService.__new__(DockerService)
    docker_service.redis_service = service
    docker_service.ssh_service = MagicMock(decrypt_payload=MagicMock(return_value="key"))
    docker_service._prepare_known_hosts_policy = AsyncMock(return_value=None)
    return docker_service


async def _run_reset(service, **connect):
    payload = ResetPodGpuRequest(
        miner_hotkey=RENTAL_OWNER, executor_id=EXECUTOR, pod_id=POD_ID, container_name="c", gpu_uuids=["GPU-1"], request_id="r1"
    )
    executor_info = MagicMock(address="203.0.113.5", ssh_port=22, tdx_quote=None)
    with patch("services.docker_service.asyncssh.import_private_key"), \
            patch("services.docker_service.require_rental_docker_ssh_host_key"), \
            patch("services.docker_service.public_host_sock", AsyncMock(return_value={})), \
            patch("services.docker_service.asyncssh.connect", **connect):
        return await _reset_service(service).reset_pod_gpus(payload, executor_info, MagicMock(), "encrypted")


@pytest.mark.asyncio
async def test_the_reset_leaves_a_recent_mark_when_its_pending_mark_goes():
    service = _redis_service()
    connection = MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False))
    stopped = GpuResetReport(ok=True, pod_stopped=True, container_restarted=True)

    with patch("services.docker_service.reset_pod_gpus", AsyncMock(return_value=stopped)):
        result = await _run_reset(service, return_value=connection)

    assert result.ok is True
    assert await service.gpu_reset_marks(RENTAL_OWNER, EXECUTOR) == []
    assert await renter_gpu_reset_recent(service, RENTAL_OWNER, EXECUTOR, POD_ID) is True


@pytest.mark.asyncio
async def test_a_redis_error_on_the_pending_mark_removal_still_answers_the_reset():
    service = _redis_service()
    service.remove_pending_pod = AsyncMock(side_effect=ConnectionError("redis down"))
    connection = MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False))
    stopped = GpuResetReport(ok=True, pod_stopped=True, container_restarted=True)

    with patch("services.docker_service.reset_pod_gpus", AsyncMock(return_value=stopped)):
        result = await _run_reset(service, return_value=connection)

    assert result.ok is True
    service.remove_pending_pod.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_reset_that_never_stopped_the_pod_leaves_no_recent_mark():
    service = _redis_service()

    result = await _run_reset(service, side_effect=OSError("down"))

    assert result.ok is False
    assert await renter_gpu_reset_recent(service, RENTAL_OWNER, EXECUTOR, POD_ID) is False


@pytest.mark.asyncio
async def test_no_verified_job_reset_is_sent_just_after_a_renter_reset():
    import asyncio
    import json

    from clients.compute_client import ComputeClient
    from services.redis_service import RESET_VERIFIED_JOB_CHANNEL

    def message(executor_uuid: str, pod_id: str | None = None) -> dict:
        data = {"miner_hotkey": RENTAL_OWNER, "executor_uuid": executor_uuid, "reason": 0,
                "check_id": "gpu.validate.fingerprint", "reason_code": "GPU_MISSING", "evidence": {"pod_id": pod_id}}
        return {"channel": RESET_VERIFIED_JOB_CHANNEL.encode(), "data": json.dumps(data)}

    class PubSub:
        async def listen(self):
            yield message(EXECUTOR, POD_ID)
            yield message(EXECUTOR, "pod-other")  # another pod on the node: still sent
            yield message("idle")
            raise asyncio.CancelledError  # ends the subscriber's reconnect loop

        async def aclose(self):
            pass

    service = _redis_service()
    await _reset_just_ended(service)
    service.subscribe = AsyncMock(return_value=PubSub())
    client = ComputeClient.__new__(ComputeClient)
    client.lock = asyncio.Lock()
    client.message_queue = []
    client.logging_extra = {}
    client.miner_service = MagicMock(redis_service=service)
    client.my_hotkey = MagicMock(return_value="validator-hotkey")

    with pytest.raises(asyncio.CancelledError):
        await client.subscribe_mesages_from_redis()

    assert [request.executor_uuid for request in client.message_queue] == [EXECUTOR, "idle"]
