"""DAH-4658: a scrape that finds no GPU while the renter resets them (or just after) is held, not a 0.

8x H200, 23 Oct: `nvidia-smi -r` on all 8 cards ran ~45 s, a cycle's scrape listed 0 GPUs and failed with
SCRAPE_FAILED_NO_GPU, and the node scored 0. The gpu_reset mark was never read: the scrape is fatal and ends the
pipeline before the checks that skip on it. The mark was also gone by the time a scrape that ended just after the
reset was judged.
"""

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as aioredis
from payload_models.payloads import GpuResetErrorCode, ResetPodGpuRequest
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.docker_service import DockerService
from core.config import settings
from core.validator import specs_to_publish
from services.executor_rollout import (
    ExecutorRolloutTracker,
    _idle_passes,
    carry_renter_gpu_reset_results,
    has_renter_gpu_reset_holds,
    remember_rented_scored_results,
    rollout_grace_reason,
    session_kept_hotkeys,
)
from services.gpu_reset import RECENT_RENTER_GPU_RESET_SECONDS, GpuResetReport, renter_gpu_reset_recent
from services.redis_service import (
    GPU_RESET_PENDING_KIND,
    LAST_SCORED_RESULT_TTL_SECONDS,
    RESET_VERIFIED_JOB_CHANNEL,
    last_scored_result_key,
)
from protocol.vc_protocol.compute_requests import (
    FillerRevenueByGpuConfig,
    RentedExecutor,
    RentedExecutorsResponse,
    RentedGpuDropResponse,
    RentedPod,
)
from services.task.checks import rented_gpu_drop
from protocol.vc_protocol.validator_requests import ContainerState, PodContainerState
from services.ssh_service import SSHService
from services.task.models import JobResult, build_msg
from test_executor_rollout_grace import _failed, _scored, _validator_process
from neurons.validators.src.services.task.messages import GpuFingerprintMessages as FingerprintMsg
from services.task.messages import ZEROED_BEFORE_THE_BUDGET_STOP, ExecutorImageMessages, VerifyXMessages
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
from neurons.validators.src.services.task.messages import RentalVerificationMessages
from neurons.validators.src.services.task.pipeline import Pipeline
from neurons.validators.src.services.task.result_handler import ResultHandler

ZERO_GPUS = _scrape_result(stdout=NO_GPU_REPORT, exit_code=1)
RESET_REQUIRED = _scrape_result(stdout=_driver_report("NVMLError(16)"), exit_code=1)


@pytest.fixture(autouse=True)
def _lock_release_without_lua(monkeypatch):
    # fakeredis runs no Lua without lupa (not a dependency): release the pod lock with a plain DEL
    monkeypatch.setattr(aioredis.lock.Lock, "do_release", lambda self, expected_token: self.redis.delete(self.name))


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
# the cycle's job block, and the one of the cycle before it, which vouched for the kept result it wrote
JOB_BLOCK = 7500
PREV_BLOCK = JOB_BLOCK - settings.BLOCKS_FOR_JOB


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
        await remember_rented_scored_results({RENTAL_OWNER: [previous]}, service, job_block=PREV_BLOCK)
        await _reset_in_progress(service, H200_UUIDS[:4])

        ok, event, ctx, _ = await _cycle(context_factory, service, scrape=_gpus_listed(H200_UUIDS[4:]), pod_gpu_count=4)
        held = await _job_result(ctx, ok, event)
    job_results = {RENTAL_OWNER: [held]}

    await carry_renter_gpu_reset_results(
        job_results, service, {EXECUTOR}, ctx.state.rented_data, {}, ctx.state.rented_data, job_block=JOB_BLOCK
    )

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
    result.rental_pod_ids = [POD_ID]
    result.rented_gpu_count = 8
    return result


def _idle_scored(uuid: str):
    """An idle pass as ResultHandler records it: scored with no rental."""
    result = _scored(uuid)
    result.rental_pod_ids = []
    return result


async def _carry_then_withhold(service, results, rented_executor_ids=frozenset({"node-1"})):
    """The cycle's order: carry before the duplicate pass and the GPU count, withhold after."""
    job_results = {"5Miner": results}
    # the snapshot and the cycle-end read list the same rental (pod POD_ID) on the same nodes: nothing changed
    at_end = _rented_executors(*rented_executor_ids)
    await carry_renter_gpu_reset_results(job_results, service, rented_executor_ids, at_end, {}, at_end, job_block=JOB_BLOCK)
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.fetch_executor_digest_or_none = AsyncMock(return_value=None)
    validator_process.observe_executor_rollout = AsyncMock(return_value=MagicMock(covers=lambda _: False))
    return await validator_process.withhold_verdicts_for_rollout(job_results, 7500, "batch-2", MagicMock())


@pytest.mark.asyncio
async def test_a_held_scrape_carries_the_previous_score_into_the_cycle():
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service, job_block=PREV_BLOCK)
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
    await remember_rented_scored_results({"5Miner": [previous]}, service, job_block=PREV_BLOCK)
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)
    held.is_spot = held.is_on_spot_pay_list = held.provider_discord_connected = False
    held.is_provider_email_held = held.is_new_rentals_paused = True
    held.default_job_owner, held.has_lium_filler = "lium", True
    job_results = {"5Miner": [held]}
    rented_data = _rented_executors(
        "node-1",
        banned_provider_guids=["GPU-banned"],
        filler_revenue_by_gpu_config=[
            FillerRevenueByGpuConfig(base_model="H100", gpu_count=8, usd_per_gpu_hour=2.5, gpu_hours=10_000)
        ],
    )

    await carry_renter_gpu_reset_results(job_results, service, {"node-1"}, rented_data, {}, rented_data, job_block=JOB_BLOCK)

    [carried] = job_results["5Miner"]
    assert carried.carried_for_renter_gpu_reset is True
    assert (carried.is_spot, carried.is_on_spot_pay_list, carried.provider_discord_connected) == (False, False, False)
    assert carried.is_provider_email_held is True
    assert carried.is_provider_banned is True
    assert (carried.is_new_rentals_paused, carried.default_job_owner, carried.has_lium_filler) == (True, "lium", True)
    # this cycle's filler average for the carried configuration (8x H100), not the held scrape's
    assert carried.filler_revenue_per_gpu_hour == 2.5
    assert carried.filler_revenue_usd_per_hour == 2.5 * 10_000 / 24


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
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service, job_block=PREV_BLOCK)
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
async def test_only_scored_results_are_remembered_and_a_redis_error_is_logged_only():
    service = _redis_service()
    idle = _idle_scored("node-idle")
    failed = _failed("node-failed", Msg.SCRAPE_FAILED_NO_GPU.reason)
    failed.is_rented = True

    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1"), idle, failed]}, service, job_block=PREV_BLOCK)
    assert list(_idle_passes) == [("5Miner", "node-idle")]  # an idle pass, in the validator process only
    await remember_rented_scored_results(
        {"5Miner": [_rented_scored("node-1")]}, MagicMock(write_atomically=AsyncMock(side_effect=ConnectionError)), job_block=PREV_BLOCK
    )

    assert await service.get(last_scored_result_key("5Miner", "node-1"))
    # rented passes only, as on main
    assert await service.get(last_scored_result_key("5Miner", "node-idle")) is None
    assert await service.get(last_scored_result_key("5Miner", "node-failed")) is None


def _no_task_time(uuid: str):
    return _failed(uuid, RentalVerificationMessages.NO_TASK_TIME.reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", [RentalVerificationMessages.NO_TASK_TIME.reason, VerifyXMessages.NO_TASK_TIME.reason]
)
async def test_an_idle_node_the_task_had_no_time_to_probe_keeps_its_last_score_once(reason):
    """DAH-5933: an idle pass, then a no-time cycle both snapshots saw idle: the pass stands, not a zero, nothing
    published over the backend row. Once: the next such cycle with no pass between is withheld."""
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_idle_scored("node-idle")]}, service, job_block=PREV_BLOCK)
    held = _failed("node-idle", reason)

    kept, withheld = await _carry_then_withhold(service, [held], rented_executor_ids=frozenset())

    [carried] = kept["5Miner"]
    assert carried.is_successful and carried.score == 1.0 and not carried.is_rented
    assert carried.carried_for_renter_gpu_reset is True
    assert carried.job_batch_id == held.job_batch_id
    assert withheld == []
    assert specs_to_publish(kept["5Miner"]) == []
    await remember_rented_scored_results(kept, service, job_block=JOB_BLOCK)
    assert await service.get(last_scored_result_key("5Miner", "node-idle")) is None
    second = {"5Miner": [_failed("node-idle", reason)]}
    await carry_renter_gpu_reset_results(
        second, service, set(), _listed(), {}, _listed(), job_block=JOB_BLOCK + settings.BLOCKS_FOR_JOB
    )
    assert second["5Miner"][0].carried_for_renter_gpu_reset is False


@pytest.mark.asyncio
async def test_an_idle_pass_is_kept_as_it_was_before_scoring_set_its_flags():
    """F-69: scoring sets flags on the cycle's own result after it is kept, so the kept pass is a copy taken then,
    as the rented pass's Redis copy is: a later no-time cycle never carries the earlier cycle's Spot eligibility."""
    service = _redis_service()
    idle = _idle_scored("node-idle")
    await remember_rented_scored_results({"5Miner": [idle]}, service, job_block=PREV_BLOCK)
    idle.spot_pay_candidate = True  # what the incentive engine does to the cycle's result after it is kept

    kept, _ = await _carry_then_withhold(service, [_no_task_time("node-idle")], rented_executor_ids=frozenset())

    [carried] = kept["5Miner"]
    assert carried.carried_for_renter_gpu_reset is True
    assert carried.spot_pay_candidate is False


@pytest.mark.asyncio
async def test_a_no_task_time_cycle_without_a_last_score_gets_no_verdict_and_keeps_no_miner_active():
    # review: the node's own checks can spend the task's time, so running out of it keeps no eligibility
    service = _redis_service()
    held = _no_task_time("node-idle")

    kept, withheld = await _carry_then_withhold(service, [held], rented_executor_ids=frozenset())

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-idle"]
    assert session_kept_hotkeys(withheld) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", [RentalVerificationMessages.NO_TASK_TIME.reason, VerifyXMessages.NO_TASK_TIME.reason]
)
async def test_a_second_no_task_time_cycle_without_a_pass_between_is_withheld(reason):
    # review: a node that stalls its checks past the task's time cannot keep its last pass's score
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service, job_block=PREV_BLOCK)

    first, _ = await _carry_then_withhold(service, [_failed("node-1", reason)])
    await remember_rented_scored_results(first, service, job_block=PREV_BLOCK)
    second, withheld = await _carry_then_withhold(
        service, [_failed("node-1", reason)]
    )

    assert [r.carried_for_renter_gpu_reset for r in first["5Miner"]] == [True]
    assert second == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]
    assert session_kept_hotkeys(withheld) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        RentalVerificationMessages.NO_TASK_TIME.reason,
        VerifyXMessages.NO_TASK_TIME.reason,
        RentalVerificationMessages.RENTED_SINCE_THE_SNAPSHOT.reason,
    ],
)
@pytest.mark.parametrize("earlier_gate", [None, "cpu_truth_passed", "cc_vbios_passed", "price_per_gpu"])
async def test_a_budget_stop_after_a_check_zeroed_the_score_keeps_that_failure(context_factory, reason, earlier_gate):
    # review: a check that already ran this cycle and zeroed its score is no budget stop the last pass may replace;
    # 4d7bf12: a price raised above the cap zeroes it too, with no boolean flag behind it
    from services.task.service import budget_stop_reason

    ctx = context_factory(state=build_state(specs={"network": {"ema_verifyx_download_speed": 500.0}}))
    ctx = ctx.model_copy(update={"executor": ctx.executor.model_copy(update={"price_per_gpu": None})})  # a pass
    gate_event = build_msg(event="CPU count mismatch", reason="CPU_TRUTH_MISMATCH", severity="error", impact="")
    events = [build_msg(event="ok", reason="VALIDATION_STEP_OK", severity="info", impact="")]
    if earlier_gate == "price_per_gpu":
        ctx = ctx.model_copy(update={"executor": ctx.executor.model_copy(update={"price_per_gpu": 1e9})})
        events.append(gate_event)
    elif earlier_gate:
        ctx = ctx.model_copy(update={earlier_gate: False})
        events.append(gate_event)
    events.append(build_msg(event="no time", reason=reason, severity="warning", impact=""))
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service, job_block=PREV_BLOCK)
    result = _failed("node-1", budget_stop_reason(reason, events, ctx))

    kept, withheld = await _carry_then_withhold(service, [result])

    if earlier_gate:
        [published] = kept["5Miner"]
        assert published.failure_reason_code == ZEROED_BEFORE_THE_BUDGET_STOP
        assert published.score == 0 and not published.carried_for_renter_gpu_reset
        assert withheld == []
    elif reason == RentalVerificationMessages.RENTED_SINCE_THE_SNAPSHOT.reason:
        assert kept == {"5Miner": []}  # the node's rentals changed during the cycle: no verdict
        assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]
    else:
        assert [r.carried_for_renter_gpu_reset for r in kept["5Miner"]] == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", [RentalVerificationMessages.NO_TASK_TIME.reason, VerifyXMessages.NO_TASK_TIME.reason]
)
async def test_a_warn_only_image_event_after_a_cpu_mismatch_never_lets_a_rollout_hold_the_zero(
    context_factory, reason
):
    # review at ea74f37: CPU truth fails, the image check then logs a warn-only EXECUTOR_IMAGE_OUTDATED, and the
    # rental check runs out of time during a rollout. The zero is published, not withheld, and the cached pass is
    # never carried into the next cycle
    from services.task.service import budget_stop_reason
    from test_executor_rollout_grace import _open_window, J0

    ctx = context_factory(state=build_state(specs={"network": {"ema_verifyx_download_speed": 500.0}}))
    ctx = ctx.model_copy(
        update={"cpu_truth_passed": False, "executor": ctx.executor.model_copy(update={"price_per_gpu": None})}
    )
    events = [
        build_msg(event="CPU count mismatch", reason="CPU_TRUTH_MISMATCH", severity="error", impact=""),
        build_msg(event="image outdated", reason=ExecutorImageMessages.OUTDATED.reason, severity="warning", impact=""),
        build_msg(event="no time", reason=reason, severity="warning", impact=""),
    ]
    result = _failed("node-idle", budget_stop_reason(reason, events, ctx))

    assert result.failure_reason_code == ZEROED_BEFORE_THE_BUDGET_STOP
    assert rollout_grace_reason(result, _open_window(), J0) is None
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_idle_scored("node-idle")]}, service, job_block=PREV_BLOCK)
    kept, withheld = await _carry_then_withhold(service, [result], rented_executor_ids=frozenset())
    [published] = kept["5Miner"]
    assert published.score == 0 and not published.carried_for_renter_gpu_reset and withheld == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", [RentalVerificationMessages.NO_TASK_TIME.reason, VerifyXMessages.NO_TASK_TIME.reason]
)
async def test_a_budget_stop_before_verifyx_measured_a_node_stays_a_neutral_stop(context_factory, reason):
    # review at ea74f37: a new node with no VerifyX speed on record keeps the no-time reason, so it is held (and
    # carries a recent pass) instead of being published as a failed first pass
    from services.task.service import budget_stop_reason

    ctx = context_factory(state=build_state(specs={}))
    ctx = ctx.model_copy(update={"executor": ctx.executor.model_copy(update={"price_per_gpu": None})})
    events = [build_msg(event="no time", reason=reason, severity="warning", impact="")]

    assert budget_stop_reason(reason, events, ctx) == reason


@pytest.mark.asyncio
async def test_an_idle_pass_never_stands_in_for_a_rented_gpu_reset_cycle():
    # review case B: idle pass, then a rental, then the renter's GPU reset: withheld as on main, not paid as idle
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_scored("node-1")]}, service, job_block=PREV_BLOCK)
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)

    kept, withheld = await _carry_then_withhold(service, [held])

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


@pytest.mark.asyncio
async def test_a_rented_pass_never_stands_in_for_an_idle_no_task_time_cycle():
    # review case A: rented pass, the rental ends, then a no-time cycle: not paid as rented emission
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service, job_block=PREV_BLOCK)
    held = _no_task_time("node-1")

    kept, withheld = await _carry_then_withhold(service, [held], rented_executor_ids=frozenset())

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


@pytest.mark.asyncio
async def test_the_gpu_reset_carry_still_needs_the_node_rented():
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_idle_scored("node-idle")]}, service, job_block=PREV_BLOCK)
    held = _failed("node-idle", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)

    kept, withheld = await _carry_then_withhold(service, [held], rented_executor_ids=frozenset())

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-idle"]


@pytest.mark.asyncio
async def test_a_failure_of_its_own_after_the_last_score_leaves_nothing_to_carry():
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service, job_block=PREV_BLOCK)
    await remember_rented_scored_results({"5Miner": [_failed("node-1", Msg.SCRAPE_FAILED_NO_GPU.reason)]}, service, job_block=PREV_BLOCK)
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)

    kept, withheld = await _carry_then_withhold(service, [held])

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


@pytest.mark.asyncio
async def test_a_rental_that_ended_after_the_reset_carries_nothing():
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service, job_block=PREV_BLOCK)
    held = _failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason)

    kept, withheld = await _carry_then_withhold(service, [held], rented_executor_ids=frozenset())

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


def _rented_executors(*executor_ids: str, **fields) -> RentedExecutorsResponse:
    rented = RentedExecutor(
        miner_hotkey="5Miner",
        executor_ip_address="203.0.113.5",
        executor_ip_port="22",
        pods=[RentedPod(pod_id=POD_ID, container_name="tenant-123", gpu_count=8)],
    )
    return RentedExecutorsResponse(executors={executor_id: rented for executor_id in executor_ids}, **fields)


@pytest.mark.parametrize(
    ("rented_now", "snapshot", "carried"),
    [
        (_rented_executors("NODE-1"), _rented_executors("NODE-1"), True),
        (_rented_executors("NODE-1"), _rented_executors(), False),
        (None, _rented_executors("NODE-1"), True),  # review at d4f5e64: a GPU-reset hold falls back to the snapshot
        (None, _rented_executors(), False),
    ],
    ids=["listed-now-and-in-snapshot", "listed-now-not-in-snapshot", "no-answer-snapshot-lists-it", "no-answer"],
)
@pytest.mark.asyncio
async def test_the_cycle_asks_the_backend_once_whether_the_node_is_still_rented(rented_now, snapshot, carried):
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service, job_block=PREV_BLOCK)
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.redis_service = service
    validator_process.backend_client = MagicMock(get_rented_executors_now=AsyncMock(return_value=rented_now))
    job_results = {"5Miner": [_failed("node-1", Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason), _scored("node-2")]}

    await validator_process.carry_scores_over_renter_gpu_resets(
        job_results, snapshot, {}, await validator_process.rented_at_cycle_end(job_results), job_block=JOB_BLOCK
    )
    no_hold = {"5Miner": [_scored("node-3")]}
    await validator_process.carry_scores_over_renter_gpu_resets(
        no_hold, snapshot, {}, await validator_process.rented_at_cycle_end(no_hold), job_block=JOB_BLOCK
    )

    assert job_results["5Miner"][0].carried_for_renter_gpu_reset is carried
    validator_process.backend_client.get_rented_executors_now.assert_awaited_once()


def _reset_service(service, backend=None) -> DockerService:
    docker_service = DockerService.__new__(DockerService)
    docker_service.redis_service = service
    docker_service.backend_client = backend
    docker_service.ssh_service = MagicMock(decrypt_payload=MagicMock(return_value="key"))
    docker_service._prepare_known_hosts_policy = AsyncMock(return_value=None)
    return docker_service


async def _run_reset(service, backend=None, **connect):
    payload = ResetPodGpuRequest(
        miner_hotkey=RENTAL_OWNER, executor_id=EXECUTOR, pod_id=POD_ID, container_name="c", gpu_uuids=["GPU-1"], request_id="r1"
    )
    executor_info = MagicMock(address="203.0.113.5", ssh_port=22, tdx_quote=None)
    with patch("services.docker_service.asyncssh.import_private_key"), \
            patch("services.docker_service.require_rental_docker_ssh_host_key"), \
            patch("services.docker_service.public_host_sock", AsyncMock(return_value={})), \
            patch("services.docker_service.asyncssh.connect", **connect):
        return await _reset_service(service, backend).reset_pod_gpus(payload, executor_info, MagicMock(), "encrypted")


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
async def test_a_reset_slower_than_the_clock_lock_deadline_still_finishes_its_restart(monkeypatch):
    import asyncio

    import services.docker_service as ds

    monkeypatch.setattr(ds, "POD_GPU_OPERATION_DEADLINE_SECONDS", 0.01)
    service = _redis_service()
    connection = MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False))

    async def slow_reset(*_args, **_kwargs):
        await asyncio.sleep(0.05)  # the container restart, past the clock lock's deadline
        return GpuResetReport(ok=True, pod_stopped=True, container_restarted=True)

    with patch("services.docker_service.reset_pod_gpus", slow_reset):
        result = await _run_reset(service, return_value=connection)

    assert (result.ok, result.container_restarted) == (True, True)
    assert not ds.pod_gpu_operation_locks._locks


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


GPUS_LOST_DETAIL = "The GPUs stopped responding on the host. The provider has to reboot it before they can be used again."
GPUS_LOST = GpuResetReport(ok=False, error_code=GpuResetErrorCode.GpusLostOnHost, detail=GPUS_LOST_DETAIL)
SPLIT_NODE = RentedExecutorsResponse(
    executors={
        EXECUTOR: RentedExecutor(
            miner_hotkey=RENTAL_OWNER,
            executor_ip_address="203.0.113.5",
            executor_ip_port="22",
            pods=[
                RentedPod(pod_id=POD_ID, container_name="c", gpu_count=1, status="RUNNING"),
                RentedPod(pod_id="pod-2", container_name="c2", gpu_count=2, status="RUNNING"),
            ],
        )
    }
)


async def _reset_with_gpus_lost(service, backend):
    connection = MagicMock(__aenter__=AsyncMock(), __aexit__=AsyncMock(return_value=False))
    with patch("services.docker_service.reset_pod_gpus", AsyncMock(return_value=GPUS_LOST)):
        result = await _run_reset(service, backend, return_value=connection)
    assert (result.ok, result.error_code, result.detail, result.container_restarted, result.gpus) == (
        False, GpuResetErrorCode.GpusLostOnHost, GPUS_LOST_DETAIL, False, []
    )


@pytest.mark.parametrize(
    "existing_incident, race, states",
    [
        (False, "memory-read", ["fault", "recovered"]),
        (True, "memory-read", ["fault", "recovered"]),
        (True, "recovery-reply", ["recovered", "fault", "recovered"]),
        (True, "busy-lock", None),
        (False, "redis-down", None),
        (False, "redis-read-error", None),
        (False, "redis-write-error", ["fault", "recovered"]),
        (True, "dirty-cycle-local", ["fault"]),
    ],
    ids=[
        "new-incident", "existing-incident", "existing-incident-recovery-reply", "busy-lock", "redis-down",
        "redis-read-error", "redis-write-error", "dirty-cycle-local",
    ],
)
@pytest.mark.asyncio
async def test_gpus_lost_on_the_host_post_one_drop_for_the_pod_and_a_clean_cycle_recovers_it(
    context_factory, monkeypatch, existing_incident, race, states
):
    import asyncio
    from datetime import timedelta

    monkeypatch.setattr(rented_gpu_drop, "_LOCAL_MARKS", {})
    service = _redis_service()
    if existing_incident:
        previous_drop = rented_gpu_drop.judge_rented_gpus(
            rented_gpu_count=3, anchor_uuids=[], nvml_count=3, listed_uuids=["GPU-0", "GPU-1"], listed_count=2,
            scrape_error=None,
        )
        mark = rented_gpu_drop.DropMark(
            first_seen_at=(datetime.now(UTC) - timedelta(minutes=15)).isoformat(),
            reported=True,
            recorded=True,
            evidence=previous_drop.evidence(3, 1),
        )
        await service.set(
            f"rented_gpu_drop:{POD_ID}", mark.dump(), ex=rented_gpu_drop.settings.RENTED_GPU_DROP_STATE_TTL_SECONDS
        )
    backend = build_services().backend
    backend.get_rented_executors_now.return_value = SPLIT_NODE
    backend.report_rented_gpu_drop.return_value = RentedGpuDropResponse(recorded=True, delivery="notified")
    details = [{"uuid": f"GPU-{i}"} for i in range(3)]

    def clean_cycle_ctx():
        return context_factory(
            services=build_services(redis=service, backend=backend),
            state=build_state(specs={"gpu": {"count": 3, "details": details}}, gpu_count=3, gpu_details=details, rented_data=SPLIT_NODE),
        )

    if race == "dirty-cycle-local":
        key = f"rented_gpu_drop:{POD_ID}"
        older_ctx = clean_cycle_ctx()
        check = rented_gpu_drop.RentedGpuDropCheck()
        with patch.object(service, "set", AsyncMock(side_effect=OSError("transient"))):
            await check._save(older_ctx, POD_ID, rented_gpu_drop.DropMark.load(await service.get(key)), True)
        with monkeypatch.context() as connector:  # the connector process does not share the cycle's cache
            connector.setattr(rented_gpu_drop, "_LOCAL_MARKS", {})
            await _reset_with_gpus_lost(service, backend)
        before = await service.get(key)
        await check.run(older_ctx)
        assert [call.kwargs["state"] for call in backend.report_rented_gpu_drop.await_args_list] == states
        assert await service.get(key) == before
        return
    if race == "busy-lock":
        @asynccontextmanager
        async def busy_lock(_):
            yield False

        monkeypatch.setattr(service, "rented_gpu_drop_lock", busy_lock)
        before = await service.get(f"rented_gpu_drop:{POD_ID}")
        now = datetime.now(UTC).isoformat()
        await rented_gpu_drop.RentedGpuDropCheck()._track_pod(
            clean_cycle_ctx(), SPLIT_NODE.executors[EXECUTOR].pods[0], None, 3, 3, now, now,
        )
        backend.report_rented_gpu_drop.assert_not_awaited()
        assert await service.get(f"rented_gpu_drop:{POD_ID}") == before
        return
    if race.startswith("redis"):
        original_set = service.set

        async def set_failing_after_answer(key, value, **kwargs):
            if json.loads(value)["reported"]:
                raise OSError("down")
            return await original_set(key, value, **kwargs)

        with monkeypatch.context() as outage:
            if race == "redis-read-error":
                outage.setattr(service, "get", AsyncMock(side_effect=OSError("down")))
            outage.setattr(service, "set", set_failing_after_answer)
            backend, report = _report_gpus_lost(FakeRedis(failing=True) if race == "redis-down" else service)
            await report
        if states is None:
            backend.report_rented_gpu_drop.assert_not_awaited()
            return
        rented_gpu_drop._LOCAL_MARKS.clear()  # the cycle runs in the validator process

    # a cycle that read its clean scrape before the reset, still running when the reset reports the loss
    entered, resume = asyncio.Event(), asyncio.Event()

    if race == "memory-read":
        async def pause_memory_read(ctx):
            entered.set()
            await resume.wait()
            return rented_gpu_drop.BrokenCardsRead(None, {})

        with monkeypatch.context() as patcher:
            patcher.setattr(rented_gpu_drop, "_read_broken_uuids", pause_memory_read)
            older_cycle = asyncio.create_task(rented_gpu_drop.RentedGpuDropCheck().run(clean_cycle_ctx()))
            await entered.wait()
            await _reset_with_gpus_lost(service, backend)
            await _reset_with_gpus_lost(service, backend)
            resume.set()
            await older_cycle
    elif race == "recovery-reply":
        lookup_entered = asyncio.Event()

        async def delayed_report(*args, state, **kwargs):
            if state == "recovered":
                entered.set()
                await resume.wait()
            return RentedGpuDropResponse(recorded=True, delivery="notified")

        async def rented_now():
            lookup_entered.set()
            return SPLIT_NODE

        backend.report_rented_gpu_drop.side_effect = delayed_report
        backend.get_rented_executors_now.side_effect = rented_now
        older_cycle = asyncio.create_task(rented_gpu_drop.RentedGpuDropCheck().run(clean_cycle_ctx()))
        await entered.wait()
        reset = asyncio.create_task(_reset_with_gpus_lost(service, backend))
        await lookup_entered.wait()
        resume.set()
        await asyncio.gather(older_cycle, reset)

    fault = next(c for c in backend.report_rented_gpu_drop.await_args_list if c.kwargs["state"] == "fault")
    assert fault.args == (POD_ID,)
    assert fault.kwargs | {
        "visible_gpu_count": 0, "expected_gpu_count": 1, "pod_gpu_count": 1,
        "rented_gpu_count": 3, "faults": ["below_rented_count"],
    } == fault.kwargs
    assert await service.get(f"rented_gpu_drop:{POD_ID}") is not None

    await rented_gpu_drop.RentedGpuDropCheck().run(clean_cycle_ctx())

    assert [call.kwargs["state"] for call in backend.report_rented_gpu_drop.await_args_list] == states
    assert await service.get(f"rented_gpu_drop:{POD_ID}") is None


@pytest.mark.parametrize(
    "failure",
    [
        {"get_rented_executors_now.return_value": None},
        {"get_rented_executors_now.side_effect": RuntimeError("down")},
        {"report_rented_gpu_drop.side_effect": RuntimeError("down")},
    ],
)
@pytest.mark.asyncio
async def test_a_failing_backend_leaves_the_gpus_lost_answer_unchanged(failure, monkeypatch):
    monkeypatch.setattr(rented_gpu_drop, "_LOCAL_MARKS", {})
    backend = build_services().backend
    backend.get_rented_executors_now.return_value = SPLIT_NODE
    backend.configure_mock(**failure)

    await _reset_with_gpus_lost(_redis_service(), backend)


def _report_gpus_lost(redis):
    backend = build_services().backend
    backend.get_rented_executors_now.return_value = SPLIT_NODE
    backend.report_rented_gpu_drop.return_value = RentedGpuDropResponse(recorded=True, delivery="notified")
    report = rented_gpu_drop.report_gpus_lost_on_reset(
        redis, backend, miner_hotkey=RENTAL_OWNER, executor_id=EXECUTOR, pod_id=POD_ID, default_extra={}
    )
    return backend, report


@pytest.mark.asyncio
async def test_the_reset_report_waits_for_a_pod_lock_another_process_holds(monkeypatch):
    import asyncio

    monkeypatch.setattr(rented_gpu_drop, "_LOCAL_MARKS", {})
    service = _redis_service()
    backend, report = _report_gpus_lost(service)
    async with service.rented_gpu_drop_lock(POD_ID):
        pending = asyncio.create_task(report)
        await asyncio.sleep(0.3)
        backend.report_rented_gpu_drop.assert_not_awaited()
    await pending
    backend.report_rented_gpu_drop.assert_awaited_once()


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


@pytest.mark.asyncio
async def test_a_no_task_time_carry_needs_the_cycle_end_read():
    # review F-21: the start snapshot lists 8 rented GPUs, the run saw the rental close, the end read failed
    service = _redis_service()
    previous = _rented_scored("node-1")
    previous.rented_gpu_count = 8
    await remember_rented_scored_results({"5Miner": [previous]}, service, job_block=PREV_BLOCK)
    pod = RentedPod(pod_id=POD_ID, container_name="c", gpu_count=8, status="RUNNING")
    at_start = RentedExecutorsResponse(
        executors={
            "node-1": RentedExecutor(
                miner_hotkey="5Miner", executor_ip_address="203.0.113.5", executor_ip_port="22", pods=[pod]
            )
        }
    )
    job_results = {"5Miner": [_no_task_time("node-1")]}

    await carry_renter_gpu_reset_results(job_results, service, {"node-1"}, at_start, {}, None, job_block=JOB_BLOCK)

    [result] = job_results["5Miner"]
    assert not result.carried_for_renter_gpu_reset


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner", "endpoint", "carried"),
    [
        ("5Miner", ("198.51.100.9", "22"), True),
        ("5Other", ("203.0.113.5", "8080"), True),
        ("5Other", ("198.51.100.9", "22"), False),
    ],
    ids=["same-owner", "other-owner-same-endpoint", "other-owner-other-endpoint"],
)
async def test_a_no_task_time_rented_carry_counts_only_this_nodes_own_rental(owner, endpoint, carried):
    # review at a8afce8: the uuid is the miner's word; another miner's eight-GPU rental on another endpoint under the
    # same uuid never pays this miner's cached rented pass, the end read is owned as the checks own it
    service = _redis_service()
    previous = _rented_scored("node-1")
    previous.rented_gpu_count = 8
    await remember_rented_scored_results({"5Miner": [previous]}, service, job_block=PREV_BLOCK)
    pod = RentedPod(pod_id=POD_ID, container_name="c", gpu_count=8, status="RUNNING")
    at_end = RentedExecutorsResponse(
        executors={
            "node-1": RentedExecutor(
                miner_hotkey=owner, executor_ip_address=endpoint[0], executor_ip_port=endpoint[1], pods=[pod]
            )
        }
    )
    job_results = {"5Miner": [_no_task_time("node-1")]}

    await carry_renter_gpu_reset_results(job_results, service, {"node-1"}, at_end, {}, at_end, job_block=JOB_BLOCK)

    [result] = job_results["5Miner"]
    assert result.carried_for_renter_gpu_reset is carried


@pytest.mark.parametrize(
    "reason", [RentalVerificationMessages.NO_TASK_TIME.reason, VerifyXMessages.NO_TASK_TIME.reason]
)
@pytest.mark.parametrize("earlier_gate", ["cpu_truth_passed", "price_per_gpu"])
def test_a_budget_stop_after_a_zeroing_check_shows_the_provider_that_check_not_the_no_time_event(
    context_factory, reason, earlier_gate
):
    # review at a8afce8: the reason was ZEROED_BEFORE_THE_BUDGET_STOP but the published event still said the task had
    # no time and the last score stands; the provider is shown the zeroing check (or the score's own warning)
    from services.task.service import budget_stop_event, budget_stop_reason

    ctx = context_factory(state=build_state(specs={"network": {"ema_verifyx_download_speed": 500.0}}))
    ctx = ctx.model_copy(update={"executor": ctx.executor.model_copy(update={"price_per_gpu": None})})
    events = [build_msg(event="ok", reason="VALIDATION_STEP_OK", severity="info", impact="")]
    if earlier_gate == "price_per_gpu":
        ctx = ctx.model_copy(update={"executor": ctx.executor.model_copy(update={"price_per_gpu": 1e9})})
    else:
        ctx = ctx.model_copy(update={earlier_gate: False})
        events.append(build_msg(event="CPU count mismatch", reason="CPU_TRUTH_MISMATCH", severity="error", impact=""))
    events.append(build_msg(event="image outdated", reason=ExecutorImageMessages.OUTDATED.reason, severity="warning", impact=""))
    events.append(build_msg(event="no time", reason=reason, severity="warning", impact=""))
    assert budget_stop_reason(reason, events, ctx) == ZEROED_BEFORE_THE_BUDGET_STOP

    payload = budget_stop_event(events, ctx).model_dump(mode="json")

    if earlier_gate == "price_per_gpu":
        assert payload["reason_code"] == ZEROED_BEFORE_THE_BUDGET_STOP and payload["severity"] == "error"
        assert payload["impact"] and "WARNING" not in payload["impact"]
    else:
        assert payload["reason_code"] == "CPU_TRUTH_MISMATCH"
    assert reason not in json.dumps(payload)


def _rented_during_check(uuid: str):
    return _failed(uuid, RentalVerificationMessages.RENTED_SINCE_THE_SNAPSHOT.reason)


def _rented_at_end(gpu_count: int) -> RentedExecutorsResponse:
    pod = RentedPod(pod_id=POD_ID, container_name="c", gpu_count=gpu_count, status="RUNNING")
    return RentedExecutorsResponse(
        executors={
            "node-1": RentedExecutor(
                miner_hotkey="5Miner", executor_ip_address="203.0.113.5", executor_ip_port="22", pods=[pod]
            )
        }
    )


@pytest.mark.asyncio
async def test_a_split_node_rented_during_the_check_never_puts_all_its_gpus_in_the_rented_pool():
    """Review at 1601cba: an 8-GPU node idle in the snapshot gets a 2-GPU rental during the rental check. Scored from the
    snapshot it went rented with rented_gpu_count None, all 8 GPUs in the rented pool. Held, it asks for the cycle-end
    read, and the idle last score never stands in for the rented cycle: no verdict, nothing published."""
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_scored("node-1")]}, service, job_block=PREV_BLOCK)
    held = _rented_during_check("node-1")
    job_results = {"5Miner": [held]}
    assert has_renter_gpu_reset_holds(job_results)  # the cycle-end rented read is taken

    await carry_renter_gpu_reset_results(job_results, service, set(), _rented_executors(), {}, _rented_at_end(2), job_block=JOB_BLOCK)
    [result] = job_results["5Miner"]
    assert result.carried_for_renter_gpu_reset is False
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.fetch_executor_digest_or_none = AsyncMock(return_value=None)
    validator_process.observe_executor_rollout = AsyncMock(return_value=MagicMock(covers=lambda _: False))
    kept, withheld = await validator_process.withhold_verdicts_for_rollout(job_results, 7500, "batch-2", MagicMock())

    assert kept == {"5Miner": []}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


def _listed(**pods: int | None) -> RentedExecutorsResponse:
    """node-1 as the backend lists it: no rental when no pod is given, else one rental with these pods' GPU counts."""
    if not pods:
        return RentedExecutorsResponse(executors={})
    return RentedExecutorsResponse(
        executors={
            "node-1": RentedExecutor(
                miner_hotkey="5Miner",
                executor_ip_address="203.0.113.5",
                executor_ip_port="22",
                pods=[RentedPod(pod_id=pod_id, container_name=pod_id, gpu_count=n, status="RUNNING") for pod_id, n in pods.items()],
            )
        }
    )


def _kept(pods: dict[str, int | None] | None) -> JobResult:
    """The node's last pass as ResultHandler kept it: scored idle ({}), rented with these pods, or before the field (None)."""
    result = _scored("node-1")
    if pods is None:
        return result
    result.is_rented = bool(pods)
    result.rental_pod_ids = sorted(pods)
    result.rented_gpu_count = sum(n for n in pods.values() if n) or None if pods else None
    return result


NO_TIME = RentalVerificationMessages.NO_TASK_TIME.reason
SEEN_RENTED = RentalVerificationMessages.RENTED_SINCE_THE_SNAPSHOT.reason
GPU_RESET = Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason

# (kept pass, snapshot, what the held cycle saw, cycle-end read, carried): one row per rental timeline. The rule:
# a kept pass carries only for the exact rental state, no rental or the same pod ids, seen at the cycle's start and
# at its end read, with no change seen during it.
RENTAL_TIMELINES = [
    ({}, {}, NO_TIME, {}, True),  # no rental all along
    ({"pod-a": 8}, {"pod-a": 8}, NO_TIME, {"pod-a": 8}, True),  # rented the whole cycle by the same rental
    ({"pod-a": 2}, {"pod-a": 2}, NO_TIME, {"pod-a": 2}, True),  # split node, the same 2-GPU rental throughout
    ({"pod-a": 8}, {"pod-a": 8}, GPU_RESET, {"pod-a": 8}, True),  # the renter's own GPU reset, same rental
    ({}, {}, SEEN_RENTED, {"pod-a": 2}, False),  # starts after the snapshot, still rented at the end read
    ({}, {}, NO_TIME, {"pod-a": 2}, False),  # starts after the snapshot, the check never saw it
    ({}, {}, SEEN_RENTED, {}, False),  # starts and ends inside the cycle
    ({"pod-a": 8}, {"pod-a": 8}, NO_TIME, {}, False),  # ends before the end read
    ({"pod-a": 8}, {"pod-a": 8}, NO_TIME, {"pod-b": 8}, False),  # a new rental replaces the old one inside the cycle
    ({"pod-a": 8}, {"pod-b": 8}, NO_TIME, {"pod-b": 8}, False),  # the kept pass was scored with the old rental
    ({"pod-a": 2}, {"pod-a": 2, "pod-b": 2}, NO_TIME, {"pod-a": 2, "pod-b": 2}, False),  # split node, a second rental
    ({"pod-a": 2}, {"pod-a": 2}, NO_TIME, {"pod-a": 3}, False),  # same pod, other GPU count: not the same allocation
    ({"pod-a": 8}, {"pod-a": 8}, NO_TIME, {"pod-a": None}, False),  # the end read cannot count the rental's GPUs
    ({"pod-a": None}, {"pod-a": None}, NO_TIME, {"pod-a": None}, False),  # nobody knows the rented GPU count
    (None, {}, NO_TIME, {}, False),  # a pass kept before rental pod ids were recorded
    ({}, {}, GPU_RESET, {}, False),  # the GPU-reset carry stands for a rented node only
    ({}, {}, NO_TIME, None, False),  # no end read
    ({"pod-a": 8}, {"pod-a": 8}, NO_TIME, None, False),  # no end read for a rented node
    ({"pod-a": 2}, {"pod-a": 2}, GPU_RESET, None, True),  # no end read: the reset's own scrape saw the start rental
    ({}, {}, SEEN_RENTED, None, False),  # a rental seen starting, then no end read
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("kept", "start", "reason", "end", "carried"), RENTAL_TIMELINES)
async def test_a_kept_pass_carries_only_for_the_exact_rental_state_seen_at_both_ends_of_the_cycle(
    kept, start, reason, end, carried
):
    """Reviews at 1601cba and 1b092ea: one rule for every hold, over each rental timeline. A carried pass is paid from
    the pool its rental state put it in, so any other state is withheld, never scored or paid, and the kept pass is
    dropped: the state it was scored with is gone."""
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_kept(kept)]}, service, job_block=PREV_BLOCK)
    job_results = {"5Miner": [_failed("node-1", reason)]}

    await carry_renter_gpu_reset_results(
        job_results, service, set(), _listed(**start), {}, None if end is None else _listed(**end), job_block=JOB_BLOCK
    )

    [result] = job_results["5Miner"]
    assert result.carried_for_renter_gpu_reset is carried
    if carried:
        assert (result.is_rented, result.rented_gpu_count) == (bool(kept), _kept(kept).rented_gpu_count)
    else:
        # dropped when the state the pass was scored with is gone, a change seen by the cycle itself included; kept
        # when a read is missing and nothing was seen to change, and for an idle pass a GPU-reset hold cannot use
        dropped = reason == SEEN_RENTED or (end is not None and reason != GPU_RESET) or not _kept(kept).is_rented
        assert (await service.get(last_scored_result_key("5Miner", "node-1")) is None) is dropped
    assert ("5Miner", "node-1") not in _idle_passes  # an idle pass is taken by the cycle that reads it


@pytest.mark.asyncio
async def test_a_rental_that_starts_and_closes_inside_the_cycle_pays_nothing_for_it():
    """Review at 1b092ea: idle in the snapshot, the rental check sees a rental start, the renter closes it before the
    cycle-end read. The idle pass kept from before is not paid for this cycle, and not for the next either."""
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_kept({})]}, service, job_block=PREV_BLOCK)
    job_results = {"5Miner": [_failed("node-1", SEEN_RENTED)]}

    await carry_renter_gpu_reset_results(job_results, service, set(), _listed(), {}, _listed(), job_block=JOB_BLOCK)
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.fetch_executor_digest_or_none = AsyncMock(return_value=None)
    validator_process.observe_executor_rollout = AsyncMock(return_value=MagicMock(covers=lambda _: False))
    kept, withheld = await validator_process.withhold_verdicts_for_rollout(job_results, 7500, "batch-2", MagicMock())

    assert kept == {"5Miner": []}  # nothing scored, so nothing paid
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]
    next_cycle = {"5Miner": [_failed("node-1", NO_TIME)]}
    await carry_renter_gpu_reset_results(next_cycle, service, set(), _listed(), {}, _listed(), job_block=JOB_BLOCK)
    assert next_cycle["5Miner"][0].carried_for_renter_gpu_reset is False


@pytest.mark.asyncio
async def test_a_kept_pass_carries_on_the_vouch_of_the_cycle_before_and_never_across_a_silent_cycle():
    """The vouch chain: a pass, then a GPU-reset carry, vouches for each next cycle; a cycle that wrote nothing (a
    Redis outage) breaks it, so the next held cycle is withheld."""
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_rented_scored("node-1")]}, service, job_block=PREV_BLOCK)
    rented = _rented_executors("node-1")
    for block in (JOB_BLOCK, JOB_BLOCK + settings.BLOCKS_FOR_JOB):
        held = {"5Miner": [_failed("node-1", GPU_RESET)]}
        await carry_renter_gpu_reset_results(held, service, {"node-1"}, rented, {}, rented, job_block=block)
        assert held["5Miner"][0].carried_for_renter_gpu_reset is True
        await remember_rented_scored_results(held, service, job_block=block)
    # the cycle at JOB_BLOCK + 2 * BLOCKS_FOR_JOB wrote nothing
    held = {"5Miner": [_failed("node-1", GPU_RESET)]}
    await carry_renter_gpu_reset_results(
        held, service, {"node-1"}, rented, {}, rented, job_block=JOB_BLOCK + 3 * settings.BLOCKS_FOR_JOB
    )
    assert held["5Miner"][0].carried_for_renter_gpu_reset is False


async def _idle_cycle(service, reason: str, job_block: int) -> list[float]:
    """One idle cycle end to end, as the validator runs it: carry, withhold, keep. The scores left standing."""
    job_results = {"5Miner": [_failed("node-1", reason)]}
    await carry_renter_gpu_reset_results(job_results, service, set(), _listed(), {}, _listed(), job_block=job_block)
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.fetch_executor_digest_or_none = AsyncMock(return_value=None)
    validator_process.observe_executor_rollout = AsyncMock(return_value=MagicMock(covers=lambda _: False))
    standing, _ = await validator_process.withhold_verdicts_for_rollout(job_results, job_block, "batch", MagicMock())
    await remember_rented_scored_results(standing, service, job_block=job_block)
    return [result.score for results in standing.values() for result in results]


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["delete", "delete_then_restart", "read_and_take"])
@pytest.mark.parametrize("next_block", [JOB_BLOCK, JOB_BLOCK + settings.BLOCKS_FOR_JOB])
async def test_an_idle_pass_a_rental_start_and_close_invalidated_never_stands_after_redis_recovers(
    monkeypatch, failing, next_block
):
    """Review F-63: an idle pass, then a cycle that sees a rental start and close. Whatever Redis refuses in between
    (a read, a delete) and whether or not the validator restarts, a later idle cycle with no task time has no score
    standing for the node: the idle pass lives in the validator process only and that cycle took it."""
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_kept({})]}, service, job_block=PREV_BLOCK)
    assert await service.get(last_scored_result_key("5Miner", "node-1")) is None  # never in Redis
    method = "delete" if failing.startswith("delete") else failing
    working = getattr(service, method)
    monkeypatch.setattr(service, method, AsyncMock(side_effect=ConnectionError("redis down")))
    assert await _idle_cycle(service, SEEN_RENTED, JOB_BLOCK) == []
    monkeypatch.setattr(service, method, working)
    if failing == "delete_then_restart":
        restarted = _redis_service()  # a new process on the same store
        restarted.redis = service.redis
        service = restarted
        _idle_passes.clear()

    assert await _idle_cycle(service, NO_TIME, next_block) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [NO_TIME, VerifyXMessages.NO_TASK_TIME.reason])
async def test_an_idle_pass_stands_for_the_next_idle_no_task_time_cycle_end_to_end(reason):
    """The control for the case above: with no rental seen in between, the next idle no-time cycle keeps the pass."""
    service = _redis_service()
    await remember_rented_scored_results({"5Miner": [_kept({})]}, service, job_block=PREV_BLOCK)

    assert await _idle_cycle(service, reason, JOB_BLOCK) == [1.0]
    assert await _idle_cycle(service, reason, JOB_BLOCK + settings.BLOCKS_FOR_JOB) == []  # once
