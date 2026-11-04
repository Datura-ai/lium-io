"""DAH-3405: an executor that fails during a known executor-image rollout gets no verdict, not a 0.

11 Sep 2026 07:50–08:00Z: executor-v1.126 was pushed mid-cycle, watchtower recreated the executor
containers under the validator's SSH sessions, and the cycle read the fleet as 168 transport-unreachable,
51 scrape-failed, 50 insufficient-ports and 70 EXECUTOR_IMAGE_OUTDATED rows (Rustam's cycle log; tsdb:
506 of 514 executors at 0). Each test below names the regression that would bring one of those zeros back.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import asyncssh
import pytest
from core.validator import Validator
from datura.requests.miner_requests import ExecutorSSHInfo
from services.executor_rollout import (
    ROLLOUT_FAILURE_REASONS,
    ExecutorRolloutTracker,
    RolloutWindow,
    WithheldVerdict,
    rollout_grace_reason,
    session_kept_hotkeys,
    withhold_rollout_verdicts,
)
from protocol.vc_protocol.validator_requests import PodSshObservation, PodSshResult
from services.miner_service import MinerService
from services.task.models import JobResult


OLD = "sha256:" + "a" * 64
NEW = "sha256:" + "b" * 64
T0 = datetime(2026, 9, 11, 7, 50, tzinfo=UTC)
BPC = 75  # blocks per cycle, what BLOCKS_FOR_JOB is in production
J0 = 7500  # job block of the cycle in which the change is seen
J1, J2 = J0 + BPC, J0 + 2 * BPC


class _FakeRedis:
    """The two RedisService calls the tracker makes, over a dict that outlives a tracker."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value

    async def publish(self, channel: str, message: dict) -> None:
        pass


def _tracker(redis: _FakeRedis, grace_cycles: int = 2) -> ExecutorRolloutTracker:
    return ExecutorRolloutTracker(redis_service=redis, grace_cycles=grace_cycles)


def _open_window(cycles_seen: int = 1, grace_cycles: int = 2) -> RolloutWindow:
    """The window as the tracker reports it in the cycle the change was seen (cycles_seen=1)."""
    return RolloutWindow(
        digest=NEW,
        previous_digest=OLD,
        started_at=T0,
        opened_job_block=J0,
        cycles_seen=cycles_seen,
        grace_cycles=grace_cycles,
    )


def _executor(uuid: str) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=uuid,
        address="203.0.113.5",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )


def _failed(
    uuid: str,
    reason: str | None,
    *,
    observed_digest: str | None = None,
    image_status: str | None = None,
) -> JobResult:
    """A result the way TaskService builds one for a run that ended without a score."""
    spec = None
    report = None
    if observed_digest is not None or image_status is not None:
        spec = {
            "docker": {
                "container_id": "c1",
                "containers": [{"container_id": "c1", "name": "executor-executor-1", "digest": observed_digest}],
            }
        }
    if image_status is not None:
        report = {
            "status": image_status,
            "observed_digest": observed_digest,
            "expected_ref": "daturaai/compute-subnet-executor:latest",
            "expected_digest": NEW,
        }
    return JobResult(
        spec=spec,
        executor_info=_executor(uuid),
        score=0,
        job_score=0,
        job_batch_id="batch-1",
        log_status="error",
        log_text="x",
        failure_reason_code=reason,
        executor_image_report=report,
    )


def _scored(uuid: str) -> JobResult:
    return JobResult(
        spec={},
        executor_info=_executor(uuid),
        score=1.0,
        job_score=1.0,
        job_batch_id="batch-1",
        log_status="info",
        log_text="ok",
        gpu_model="H100",
        gpu_count=8,
    )


# --- the tracker: when is a rollout "known"? -------------------------------------------------


@pytest.mark.asyncio
async def test_a_digest_change_opens_the_window_and_the_first_digest_does_not() -> None:
    """Regression: a window on the first digest a fresh Redis sees would grace every failure
    after each redeploy; no window on a change would zero the fleet again on the next release."""
    redis = _FakeRedis()
    tracker = _tracker(redis)

    first = await tracker.observe(OLD, J0 - BPC, now=T0)
    assert first.covers(J0 - BPC) is False
    assert first.digest == OLD

    unchanged = await tracker.observe(OLD, J0 - BPC, now=T0 + timedelta(minutes=15))
    assert unchanged.covers(J0 - BPC) is False
    assert unchanged.opened_job_block is None

    opened = await tracker.observe(NEW, J0, now=T0 + timedelta(minutes=20))
    assert opened.digest == NEW
    assert opened.previous_digest == OLD
    assert opened.started_at == T0 + timedelta(minutes=20)
    assert opened.opened_job_block == J0
    assert opened.covers(J0) is True


# --- the classifier: which results does a rollout explain? ------------------------------------


def test_the_same_failure_after_the_window_is_a_zero_as_today() -> None:
    """Regression: a window with no end turns every unreachable node into "no verdict" forever."""
    unreachable = _failed("node-1", "EXECUTOR_SSH_UNREACHABLE")
    third_cycle = _open_window(cycles_seen=3)

    kept, withheld = withhold_rollout_verdicts({"5Miner": [unreachable]}, third_cycle, J2)

    assert kept == {"5Miner": [unreachable]}
    assert withheld == []


def test_a_failure_the_rollout_does_not_explain_stands() -> None:
    """Regression: the window becoming a blanket amnesty — too many GPUs or a result with a score
    has nothing to do with a container restart."""
    window = _open_window()
    too_many_gpus = _failed("node-1", "GPU_COUNT_EXCEEDS_MAX")
    unknown_error = _failed("node-2", None)

    assert rollout_grace_reason(too_many_gpus, window, J0) is None
    assert rollout_grace_reason(unknown_error, window, J0) is None
    assert rollout_grace_reason(_scored("node-3"), window, J0) is None


# --- TaskService hands the classifier the reason a run ended on ---------------------------------


class _ShellThatRefusesToConnect:
    def __init__(self, **_: object) -> None:
        pass

    async def __aenter__(self) -> "_ShellThatRefusesToConnect":
        raise asyncssh.Error(code=1, reason="Connection refused")

    async def __aexit__(self, *_: object) -> bool:
        return False


class _ShellThatOpens:
    async def __aenter__(self) -> "_ShellThatOpens":
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False


# --- the cycle: what happens when the tracker itself cannot be read -----------------------------


def _validator_process(tracker: ExecutorRolloutTracker) -> Validator:
    validator_process = Validator.__new__(Validator)
    validator_process.default_extra = {}
    validator_process.rollout_tracker = tracker
    return validator_process


@pytest.mark.asyncio
async def test_a_redis_error_leaves_every_verdict_standing_instead_of_ending_the_cycle() -> None:
    """Regression: the tracker's exception reaching `sync` — the whole cycle is lost to the
    `[sync] Unexpected error` handler (no weights, no publish for anyone) because the grace
    bookkeeping failed. Without Redis the cycle must run as it did before this change."""
    broken = ExecutorRolloutTracker(
        redis_service=MagicMock(get=AsyncMock(side_effect=ConnectionError("down"))), grace_cycles=2
    )
    validator_process = _validator_process(broken)
    unreachable = _failed("node-1", "EXECUTOR_SSH_UNREACHABLE")

    window = await validator_process.observe_executor_rollout(NEW, J0)
    kept, withheld = withhold_rollout_verdicts({"5Miner": [unreachable]}, window, J0)

    assert window.covers(J0) is False
    assert kept == {"5Miner": [unreachable]}
    assert withheld == []

    # Redis was fine at cycle start and died before the cycle-end observation: the cycle keeps
    # the window it saw at its start instead of dropping every withheld verdict.
    seen_at_start = _open_window()
    assert await validator_process.observe_executor_rollout(NEW, J0, fallback=seen_at_start) is seen_at_start


def test_an_idle_node_the_miner_cannot_reach_during_the_rollout_gets_the_rented_twins_grace() -> None:
    """Regression: EXECUTOR_UNREACHABLE_BY_MINER outside the set, so a rollout scored every
    restarting idle node 0 while the rented ones (RENTED_EXECUTOR_NOT_LISTED) were withheld."""
    still_on_old = _failed("node-2", "EXECUTOR_UNREACHABLE_BY_MINER", observed_digest=OLD)

    assert "EXECUTOR_UNREACHABLE_BY_MINER" in ROLLOUT_FAILURE_REASONS
    assert rollout_grace_reason(still_on_old, _open_window(), J1) == "EXECUTOR_UNREACHABLE_BY_MINER"


# --- DAH-5718: the validator's own session closed during the port check --------------------------


def _session_lost(uuid: str, availability_errors: list[dict] | None) -> JobResult:
    result = _failed(uuid, "EXECUTOR_TRANSPORT_UNREACHABLE", observed_digest=OLD)
    result.session_lost_in_port_check = True
    result.availability_errors = availability_errors
    return result


def _miner_service(published: dict[str, list[dict]], redis: "_FakeRedis | None" = None) -> MinerService:
    service = MinerService.__new__(MinerService)
    service.cycle_published_availability = published
    service.redis_service = redis or _FakeRedis()
    service.publish_machine_specs = AsyncMock()
    return service


GPU_FINDING = {"reason_code": "GPU_MEMORY_HEALTH_FAILED"}
STORED_HOLD = {"reason_code": "GPU_LOST"}


@pytest.mark.asyncio
@pytest.mark.parametrize("in_rollout", [False, True])
@pytest.mark.parametrize("findings", [None, [GPU_FINDING]])
async def test_a_session_lost_in_the_port_check_is_unscored_and_publishes_only_its_findings(in_rollout, findings):
    """Reviews 4469140353 / 4469140371 / 4469140385: outside a rollout the lost session scored 0; inside one the
    rollout withheld an earlier GPU finding with it; a [] cleared the stored hold. Now the result is out of
    scoring either way, and only a finding the finished checks made is published, on top of the stored one."""
    tracker = _tracker(_FakeRedis())
    validator_process = _validator_process(tracker)
    validator_process.fetch_executor_digest_or_none = AsyncMock(return_value=NEW)
    window = _open_window() if in_rollout else MagicMock(covers=lambda _: False)
    validator_process.observe_executor_rollout = AsyncMock(return_value=window)
    validator_process.miner_service = _miner_service({"node-1": [STORED_HOLD]})
    lost = _session_lost("node-1", findings)
    healthy = _scored("node-2")

    kept, withheld = await validator_process.withhold_verdicts_for_rollout(
        {"5Miner": [lost, healthy]}, J0, "batch-1", window
    )
    await validator_process.publish_session_lost_findings(withheld, {"5Miner": "5Cold"})

    assert kept == {"5Miner": [healthy]}  # what IncentiveFactory scores
    assert [w.result for w in withheld] == [lost]
    publish = validator_process.miner_service.publish_machine_specs
    if findings is None:
        publish.assert_not_awaited()
    else:
        publish.assert_awaited_once_with([lost], "5Miner", "5Cold", is_whole_miner_batch=False)
        assert lost.availability_errors == [GPU_FINDING, STORED_HOLD]
        assert lost.scored_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", [PodSshResult.TIMEOUT, PodSshResult.BANNER])
async def test_a_session_lost_with_no_finding_still_publishes_the_pod_ssh_observations(probe):
    """Review 4470167895: withholding the result dropped the rented pod's SSH observations, so billing could not
    tell a silent pod (TIMEOUT) and kept charging. They go out on the unscored result, the stored reasons untouched."""
    validator_process = _validator_process(_tracker(_FakeRedis()))
    validator_process.miner_service = _miner_service({})
    lost = _session_lost("node-1", None)
    observations = [PodSshObservation(pod_id="pod-1", result=probe)]

    await validator_process.publish_session_lost_findings(
        [WithheldVerdict(miner_hotkey="5Miner", result=lost)], {"5Miner": "5Cold"}, {"node-1": observations}
    )

    validator_process.miner_service.publish_machine_specs.assert_awaited_once_with(
        [lost], "5Miner", "5Cold", is_whole_miner_batch=False
    )
    assert lost.pod_ssh == observations
    assert lost.availability_errors is None  # None leaves the backend's stored reasons alone
    assert lost.scored_at is None


@pytest.mark.asyncio
async def test_consecutive_partial_publications_keep_every_earlier_reason():
    """Review 4470167912: a second lost-session publication merged with the last whole-batch list only, so it
    dropped the reason the first one published."""
    validator_process = _validator_process(_tracker(_FakeRedis()))
    validator_process.miner_service = _miner_service({"node-1": [STORED_HOLD]})
    other = {"reason_code": "GPU_COUNT_MISMATCH"}
    first, second = _session_lost("node-1", [GPU_FINDING]), _session_lost("node-1", [other])

    for lost in (first, second):
        await validator_process.publish_session_lost_findings(
            [WithheldVerdict(miner_hotkey="5Miner", result=lost)], {"5Miner": "5Cold"}
        )

    assert second.availability_errors == [other, GPU_FINDING, STORED_HOLD]


@pytest.mark.asyncio
@pytest.mark.parametrize("before_restart", ["whole_batch", "partial"])
async def test_a_partial_publication_after_a_restart_keeps_the_reason_published_before_it(
    monkeypatch, before_restart
):
    """Review 4471425927: the last published list lived only in memory, so after a validator restart a partial
    publication had nothing to merge with and sent GPU_MEMORY_HEALTH_FAILED over a stored GPU_LOST. The list is kept
    in Redis as well, and a new process reads it back."""
    monkeypatch.setattr("services.miner_service.settings.DRY_RUN", False)
    redis = _FakeRedis()

    def process() -> Validator:
        validator_process = _validator_process(_tracker(redis))
        validator_process.miner_service = _miner_service({}, redis)
        return validator_process

    before = process()
    if before_restart == "whole_batch":
        held = _failed("node-1", "GPU_LOST")
        held.availability_errors = [STORED_HOLD]
        del before.miner_service.publish_machine_specs  # the real one
        before.miner_service._pod_states_capped_for_spec = lambda result, extra: None
        await before.miner_service.publish_machine_specs([held], "5Miner", "5Cold", is_whole_miner_batch=True)
    else:
        before.miner_service.cycle_published_availability = {"node-1": []}
        await before.publish_session_lost_findings(
            [WithheldVerdict(miner_hotkey="5Miner", result=_session_lost("node-1", [STORED_HOLD]))], {"5Miner": "5Cold"}
        )

    after = process()  # the restart: nothing in memory
    lost = _session_lost("node-1", [GPU_FINDING])
    await after.publish_session_lost_findings([WithheldVerdict(miner_hotkey="5Miner", result=lost)], {"5Miner": "5Cold"})

    assert lost.availability_errors == [GPU_FINDING, STORED_HOLD]


@pytest.mark.asyncio
@pytest.mark.parametrize("cache", ["missing", "malformed", "read_fails"])
@pytest.mark.parametrize("with_pod_ssh", [False, True])
async def test_a_partial_publication_replaces_no_list_when_the_earlier_one_is_unknown(cache, with_pod_ssh):
    """Review 4472141980 / 4472141972: a missing (pre-upgrade or expired), malformed or unreadable cache was read as
    "nothing published before", so the partial list could replace a stored hold. Now nothing is replaced; the pod
    SSH observations still go out with availability_errors None."""
    redis = _FakeRedis()
    if cache == "malformed":
        redis.store["published_availability:node-1"] = "{not json"
    elif cache == "read_fails":
        redis.get = AsyncMock(side_effect=ConnectionError("redis down"))
    validator_process = _validator_process(_tracker(_FakeRedis()))
    validator_process.miner_service = _miner_service({}, redis)
    lost = _session_lost("node-1", [GPU_FINDING])
    observations = [PodSshObservation(pod_id="pod-1", result=PodSshResult.TIMEOUT)]

    await validator_process.publish_session_lost_findings(
        [WithheldVerdict(miner_hotkey="5Miner", result=lost)],
        {"5Miner": "5Cold"},
        {"node-1": observations} if with_pod_ssh else None,
    )

    assert lost.availability_errors is None
    assert "node-1" not in validator_process.miner_service.cycle_published_availability
    publish = validator_process.miner_service.publish_machine_specs
    if with_pod_ssh:
        publish.assert_awaited_once_with([lost], "5Miner", "5Cold", is_whole_miner_batch=False)
    else:
        publish.assert_not_awaited()


def test_a_miner_whose_only_withheld_result_is_a_lost_session_gets_no_eligibility_floor():
    """Review 4469609776: a miner that closes SSH in every port check must not stay active (eligibility floor)."""
    withheld = [
        WithheldVerdict(miner_hotkey="5Closer", result=_session_lost("node-1", None)),
        WithheldVerdict(miner_hotkey="5Rollout", result=_failed("node-2", "EXECUTOR_TRANSPORT_UNREACHABLE")),
    ]

    assert session_kept_hotkeys(withheld) == {"5Rollout"}
