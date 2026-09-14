"""DAH-3405: an executor that fails during a known executor-image rollout gets no verdict, not a 0.

11 Sep 2026 07:50–08:00Z: executor-v1.126 was pushed mid-cycle, watchtower recreated the executor
containers under the validator's SSH sessions, and the cycle read the fleet as 168 transport-unreachable,
51 scrape-failed, 50 insufficient-ports and 70 EXECUTOR_IMAGE_OUTDATED rows (Rustam's cycle log; tsdb:
506 of 514 executors at 0). Each test below names the regression that would bring one of those zeros back.
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import asyncssh
import pytest
from core.config import settings
from core.validator import Validator
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import MinerJobRequestPayload
from services.attestation_service import HostPolicyResult
from services.executor_image_policy import ExecutorImageReport, ImageVerdict
from services.executor_rollout import (
    ROLLOUT_STATE_KEY,
    ExecutorRolloutTracker,
    RolloutWindow,
    WithheldVerdict,
    rollout_grace_reason,
    withhold_rollout_verdicts,
)
from services.task import service as task_service_module
from services.task.models import JobResult, build_msg
from services.task.pipeline import ContextState
from services.task.service import TaskService

from helpers import make_context

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

    async def set(self, key: str, value: str) -> None:
        self.store[key] = value


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


@pytest.mark.asyncio
async def test_a_push_during_the_cycle_covers_that_same_cycle() -> None:
    """The 11 Sep case: the cycle started on the old digest and the push landed while its jobs
    ran. Regression: a window that opens only at the next cycle start leaves this cycle's
    failures — the ones the push caused — as zeros."""
    redis = _FakeRedis()
    tracker = _tracker(redis)
    await tracker.observe(OLD, J0 - BPC, now=T0)
    at_start = await tracker.observe(OLD, J0, now=T0)
    assert at_start.covers(J0) is False

    at_end = await tracker.observe(NEW, J0, now=T0 + timedelta(minutes=10))

    assert at_end.opened_job_block == J0
    assert at_end.cycles_seen == 1
    assert at_end.covers(J0) is True


@pytest.mark.asyncio
async def test_the_window_covers_the_observing_cycle_and_the_next_whatever_the_clock_says() -> None:
    """Regression: a window measured in seconds at cycle end — a 13-minute cycle after a 2-minute
    one lands outside it — or in job blocks: a cycle that starts late in its block window and runs
    long makes the next one land two job blocks later (`sync` starts a cycle on the first 12-s tick
    at ≥ 75 blocks since the last), and that cycle would fall outside a block-arithmetic window.
    Cycles are counted as the validator observes them; the third one is not covered, whatever the
    clock or the block says, because a third withheld cycle trips the backend's 1-hour sweep."""
    redis = _FakeRedis()
    await _tracker(redis).observe(OLD, J0 - BPC, now=T0)
    opened = await _tracker(redis).observe(NEW, J0, now=T0)
    # the cycle-end observation of the same cycle is the same cycle
    same_cycle = await _tracker(redis).observe(NEW, J0, now=T0 + timedelta(minutes=14))
    assert same_cycle.cycles_seen == 1

    after_restart = _tracker(redis)
    next_cycle = await after_restart.observe(NEW, J2, now=T0 + timedelta(hours=3))  # landed two blocks on
    assert next_cycle.opened_job_block == J0
    assert next_cycle.cycles_seen == 2
    assert next_cycle.covers(J2) is True

    third_cycle = await after_restart.observe(NEW, J2 + BPC, now=T0 + timedelta(minutes=16))
    assert third_cycle.covers(J2 + BPC) is False
    assert opened.covers(J0 - BPC) is False  # a cycle before the change is never covered
    assert json.loads(redis.store[ROLLOUT_STATE_KEY])["closed"] is True


@pytest.mark.asyncio
async def test_a_second_push_inside_the_window_does_not_extend_it() -> None:
    """Regression: a hotfix or rollback pushed while the fleet is still restarting for the first
    image reopens the window at the new cycle — three cycles withheld in a row, and the backend's
    1-hour inactive sweep marks the old-image fleet inactive with EXECUTOR_INACTIVE_MID_RENTAL.
    The new digest becomes "current"; the window keeps its opening cycle and its count."""
    hotfix = "sha256:" + "c" * 64
    redis = _FakeRedis()
    tracker = _tracker(redis)
    await tracker.observe(OLD, J0 - BPC, now=T0)
    await tracker.observe(NEW, J0, now=T0)
    await tracker.record_withheld(200)

    window = await tracker.observe(hotfix, J1, now=T0 + timedelta(minutes=15))
    assert window.digest == hotfix
    assert window.previous_digest == NEW
    assert window.opened_job_block == J0
    assert window.covers(J1) is True
    assert json.loads(redis.store[ROLLOUT_STATE_KEY])["withheld"] == 200

    third_cycle = await tracker.observe(hotfix, J2, now=T0 + timedelta(minutes=30))
    assert third_cycle.covers(J2) is False


@pytest.mark.asyncio
async def test_a_push_right_after_the_window_closes_does_not_chain_a_second_window() -> None:
    """Regression: a change seen in the first cycle after the window closed opens a fresh window —
    an unreachable or OUTDATED executor is withheld four cycles in a row (J0..J3), over an hour
    without a published row, and the backend's inactive sweep fires. One cycle must publish in
    between; a change seen one cycle later does open a new window."""
    hotfix = "sha256:" + "c" * 64
    redis = _FakeRedis()
    tracker = _tracker(redis)
    await tracker.observe(OLD, J0 - BPC, now=T0)
    await tracker.observe(NEW, J0, now=T0)
    await tracker.observe(NEW, J1, now=T0)
    closed = await tracker.observe(NEW, J2, now=T0)
    assert closed.covers(J2) is False

    too_soon = await tracker.observe(hotfix, J2, now=T0)
    assert too_soon.digest == hotfix
    assert too_soon.covers(J2) is False

    late_enough = await tracker.observe("sha256:" + "d" * 64, J2 + BPC, now=T0)
    assert late_enough.opened_job_block == J2 + BPC
    assert late_enough.covers(J2 + BPC) is True


def test_more_than_two_cycles_are_refused_because_of_the_backend_inactive_sweep() -> None:
    """Regression: an operator setting 4 cycles — the withheld executors' rows go unrefreshed for
    more than an hour and lium-io-backend's check_and_update_executors marks them inactive with
    EXECUTOR_INACTIVE_MID_RENTAL, the penalty the ticket counts."""
    tracker = ExecutorRolloutTracker(redis_service=_FakeRedis(), grace_cycles=4)

    assert tracker.grace_cycles == 2


@pytest.mark.asyncio
async def test_a_registry_outage_keeps_the_window_as_it_was() -> None:
    """Regression: reading `None` as "the digest changed" would open a window on every Docker
    Hub hiccup; reading it as "no rollout" would close a real one halfway."""
    redis = _FakeRedis()
    tracker = _tracker(redis)
    await tracker.observe(OLD, J0 - BPC, now=T0)
    await tracker.observe(NEW, J0, now=T0)

    window = await tracker.observe(None, J1, now=T0 + timedelta(minutes=15))

    assert window.digest == NEW
    assert window.opened_job_block == J0
    assert window.covers(J1) is True


@pytest.mark.asyncio
async def test_the_withheld_total_accumulates_across_the_cycles_of_one_window() -> None:
    """Regression: a per-cycle count that overwrites the last one makes the window-end line
    report only the final cycle."""
    redis = _FakeRedis()
    tracker = _tracker(redis)
    await tracker.observe(OLD, J0 - BPC, now=T0)
    await tracker.observe(NEW, J0, now=T0)

    await tracker.record_withheld(200)
    await tracker.record_withheld(0)
    await tracker.record_withheld(39)

    assert json.loads(redis.store[ROLLOUT_STATE_KEY])["withheld"] == 239


@pytest.mark.asyncio
async def test_a_corrupt_redis_value_is_replaced_instead_of_failing_every_cycle() -> None:
    """Regression: one bad write leaves `json.loads` raising on every observe, so the fail-open
    path runs forever and no rollout is ever known again until someone deletes the key."""
    for corrupt in ("{not json", "[1, 2]"):
        redis = _FakeRedis()
        redis.store[ROLLOUT_STATE_KEY] = corrupt
        tracker = _tracker(redis)

        window = await tracker.observe(OLD, J0, now=T0)

        assert window.digest == OLD
        reseeded = json.loads(redis.store[ROLLOUT_STATE_KEY])
        assert reseeded["digest"] == OLD
        assert reseeded["opened_job_block"] is None


@pytest.mark.asyncio
async def test_zero_grace_cycles_never_cover_a_cycle(caplog) -> None:
    """Regression: the operator's off switch (EXECUTOR_ROLLOUT_GRACE_CYCLES=0) must mean today's
    behaviour — no cycle covered, not even the observing one, and no window lines in the log
    that would make an operator look for verdicts that were never withheld."""
    redis = _FakeRedis()
    tracker = _tracker(redis, grace_cycles=0)
    await tracker.observe(OLD, J0 - BPC, now=T0)

    with caplog.at_level("WARNING", logger="services.executor_rollout"):
        window = await tracker.observe(NEW, J0, now=T0)
        again = await tracker.observe("sha256:" + "c" * 64, J1, now=T0)

    assert window.opened_job_block == J0
    assert window.covers(J0) is False
    assert again.covers(J1) is False
    assert "[rollout-grace]" not in caplog.text


# --- the classifier: which results does a rollout explain? ------------------------------------


def test_an_unreachable_executor_inside_the_window_gets_no_verdict() -> None:
    """The transport-unreachable rows of 11 Sep: the connect failed, nothing about the image is
    known, the digest just changed — no score is written."""
    unreachable = _failed("node-1", "EXECUTOR_SSH_UNREACHABLE")
    healthy = _scored("node-2")
    window = _open_window()

    kept, withheld = withhold_rollout_verdicts({"5Miner": [unreachable, healthy]}, window, J0)

    assert kept == {"5Miner": [healthy]}
    assert [(w.miner_hotkey, w.result.executor_info.uuid) for w in withheld] == [("5Miner", "node-1")]


def test_the_same_failure_after_the_window_is_a_zero_as_today() -> None:
    """Regression: a window with no end turns every unreachable node into "no verdict" forever."""
    unreachable = _failed("node-1", "EXECUTOR_SSH_UNREACHABLE")
    third_cycle = _open_window(cycles_seen=3)

    kept, withheld = withhold_rollout_verdicts({"5Miner": [unreachable]}, third_cycle, J2)

    assert kept == {"5Miner": [unreachable]}
    assert withheld == []


@pytest.mark.parametrize(
    "reason",
    ["EXECUTOR_TRANSPORT_UNREACHABLE", "FILLER_TRANSPORT_UNREACHABLE", "SCRAPE_FAILED", "INSUFFICIENT_PORTS"],
)
def test_a_failure_on_an_executor_already_running_the_new_image_is_a_real_failure(reason: str) -> None:
    """Regression: grace hiding a port-table or SSH failure on a node that has finished
    restarting — the rollout cannot explain a failure on the image it rolled out."""
    on_new_image = _failed("node-1", reason, observed_digest=NEW)
    still_on_old = _failed("node-2", reason, observed_digest=OLD)
    window = _open_window()

    assert rollout_grace_reason(on_new_image, window, J1) is None
    assert rollout_grace_reason(still_on_old, window, J1) == reason


def test_an_outdated_image_inside_the_window_gets_no_verdict_rented_or_not(monkeypatch) -> None:
    """The 70 EXECUTOR_IMAGE_OUTDATED rows of 11 Sep, with the image check enforced as it was that
    day. Unrented, the fatal check ends the run with that reason; rented, the run completes with
    score 0 and only the report says OUTDATED. Both are the fleet not having pulled yet (or the
    validator's snapshot predating the push, in which case the observed digest is already the new
    one)."""
    monkeypatch.setattr(settings, "EXECUTOR_IMAGE_CHECK_ENFORCE", True)
    window = _open_window()
    unrented = _failed("node-1", "EXECUTOR_IMAGE_OUTDATED", observed_digest=OLD, image_status="OUTDATED")
    rented = _failed("node-2", "RENTED", observed_digest=OLD, image_status="OUTDATED")
    validator_snapshot_was_stale = _failed(
        "node-3", "EXECUTOR_IMAGE_OUTDATED", observed_digest=NEW, image_status="OUTDATED"
    )

    assert rollout_grace_reason(unrented, window, J0) == "EXECUTOR_IMAGE_OUTDATED"
    assert rollout_grace_reason(rented, window, J0) == "EXECUTOR_IMAGE_OUTDATED"
    assert rollout_grace_reason(validator_snapshot_was_stale, window, J0) == "EXECUTOR_IMAGE_OUTDATED"


def test_a_rented_zero_under_an_unenforced_outdated_report_stands(monkeypatch) -> None:
    """Regression: with EXECUTOR_IMAGE_CHECK_ENFORCE off (the default since DAH-3439) the image
    check passes an OUTDATED node and leaves its score alone, so a rented run at score 0 owes its
    0 to another gate (price cap, TDX, collateral). Reading the OUTDATED report as the cause would
    withhold that verdict for two cycles and keep the executor's previous scored row."""
    monkeypatch.setattr(settings, "EXECUTOR_IMAGE_CHECK_ENFORCE", False)
    window = _open_window()
    rented_zero_from_another_gate = _failed("node-2", "RENTED", observed_digest=OLD, image_status="OUTDATED")

    assert rollout_grace_reason(rented_zero_from_another_gate, window, J0) is None


def test_a_rented_executor_failing_a_later_check_of_its_own_stands_even_when_outdated() -> None:
    """Regression: the OUTDATED report riding along on every rented run makes any failure of that
    run (the pod gone, the filler killed) read as "not pulled yet" and hides it for two cycles."""
    window = _open_window()
    pod_gone = _failed("node-1", "POD_NOT_RUNNING", observed_digest=OLD, image_status="OUTDATED")

    assert rollout_grace_reason(pod_gone, window, J0) is None


def test_a_failure_the_rollout_does_not_explain_stands() -> None:
    """Regression: the window becoming a blanket amnesty — too many GPUs or a result with a score
    has nothing to do with a container restart."""
    window = _open_window()
    too_many_gpus = _failed("node-1", "GPU_COUNT_EXCEEDS_MAX")
    unknown_error = _failed("node-2", None)

    assert rollout_grace_reason(too_many_gpus, window, J0) is None
    assert rollout_grace_reason(unknown_error, window, J0) is None
    assert rollout_grace_reason(_scored("node-3"), window, J0) is None


def test_no_digest_change_means_unchanged_results() -> None:
    """Regression: the classifier acting without a rollout — every cycle without a release must
    hand its results on untouched, the same objects, nothing marked."""
    unreachable = _failed("node-1", "EXECUTOR_SSH_UNREACHABLE")
    results = {"5Miner": [unreachable]}
    no_rollout = RolloutWindow(
        digest=OLD, previous_digest=None, started_at=None, opened_job_block=None, cycles_seen=0, grace_cycles=2
    )

    kept, withheld = withhold_rollout_verdicts(results, no_rollout, J0)

    assert kept is results
    assert withheld == []


# --- TaskService hands the classifier the reason a run ended on ---------------------------------


def _task_service_that_reaches_the_ssh_connect() -> TaskService:
    service = TaskService.__new__(TaskService)
    service.ssh_service = MagicMock()
    service.ssh_service.decrypt_payload = MagicMock(return_value="decrypted-key")
    service.attestation_service = MagicMock()
    service.attestation_service.prepare_host_policy = AsyncMock(return_value=HostPolicyResult())
    return service


def _a_job_for(executor_uuid: str) -> tuple[MinerJobRequestPayload, ExecutorSSHInfo]:
    miner = MinerJobRequestPayload(
        job_batch_id="batch-1",
        miner_hotkey="5Miner",
        miner_coldkey="5Cold",
        miner_address="203.0.113.5",
        miner_port=8080,
        executors=[],
    )
    return miner, _executor(executor_uuid)


async def _run_cycle(service: TaskService, miner: MinerJobRequestPayload, executor: ExecutorSSHInfo) -> JobResult:
    return await service.create_task(
        miner_info=miner,
        executor_info=executor,
        keypair=MagicMock(ss58_address="5Val"),
        private_key="key",
        public_key="pub",
        encrypted_files=MagicMock(),
        rented_data=MagicMock(),
        default_docker_image_digests={},
    )


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


@pytest.mark.asyncio
async def test_a_refused_connect_records_its_reason_code_for_the_classifier(monkeypatch) -> None:
    """Regression: the connect failure reaching the cycle end with no reason code — the classifier
    would let a refused connect stand as a zero inside the window."""
    monkeypatch.setattr(task_service_module, "InteractiveShellService", _ShellThatRefusesToConnect)
    miner, executor = _a_job_for("node-9")

    result = await _run_cycle(_task_service_that_reaches_the_ssh_connect(), miner, executor)

    assert result.score == 0
    assert result.failure_reason_code == "EXECUTOR_SSH_UNREACHABLE"
    assert rollout_grace_reason(result, _open_window(), J0) == "EXECUTOR_SSH_UNREACHABLE"


@pytest.mark.asyncio
async def test_a_shell_that_dies_under_a_check_records_the_transport_code(monkeypatch) -> None:
    """The recreate lands after the connect, inside a check that lets asyncssh raise (upload,
    a `ctx.ssh.run`). Regression: the run ends in the generic error branch with no reason code and
    the classifier lets it stand as 0 inside the window. A non-transport crash after the connect
    must stay unnamed — it is not something a rollout explains."""
    monkeypatch.setattr(task_service_module, "InteractiveShellService", lambda **_: _ShellThatOpens())
    service = _task_service_that_reaches_the_ssh_connect()
    service.pipeline_factory = MagicMock()
    miner, executor = _a_job_for("node-9")

    service.pipeline_factory.build_context = AsyncMock(side_effect=asyncssh.Error(code=1, reason="Connection lost"))
    died = await _run_cycle(service, miner, executor)

    service.pipeline_factory.build_context = AsyncMock(side_effect=ValueError("a bug of our own"))
    crashed = await _run_cycle(service, miner, executor)

    assert died.failure_reason_code == "EXECUTOR_TRANSPORT_UNREACHABLE"
    assert died.availability_errors == []  # the connect succeeded: not an availability error
    assert rollout_grace_reason(died, _open_window(), J0) == "EXECUTOR_TRANSPORT_UNREACHABLE"
    assert crashed.failure_reason_code is None
    assert rollout_grace_reason(crashed, _open_window(), J0) is None


@pytest.mark.asyncio
async def test_a_failed_check_records_its_reason_code_and_a_passing_run_records_none(monkeypatch) -> None:
    """Regression: the reason of the failed fatal check not reaching the result (the 51 scrape
    failures would stand), or a scored run carrying a reason it did not fail on."""
    monkeypatch.setattr(task_service_module, "InteractiveShellService", lambda **_: _ShellThatOpens())
    service = _task_service_that_reaches_the_ssh_connect()
    # ResultHandler writes the verified-job counters here; the pipeline itself is stubbed.
    service.redis_service = AsyncMock()
    service.pipeline_factory = MagicMock()
    service.pipeline_factory.build_checks = MagicMock(return_value=[])
    miner, executor = _a_job_for("node-9")

    scrape_failed = build_msg(
        event="Machine specs scrape failed", reason="SCRAPE_FAILED", severity="error", impact="no score"
    )
    failed_ctx = make_context(executor=executor, miner_hotkey=miner.miner_hotkey, score=0.0, success=False)
    service.pipeline_factory.build_context = AsyncMock(return_value=failed_ctx)
    service.pipeline_factory.build_pipeline = MagicMock(
        return_value=MagicMock(run=AsyncMock(return_value=(False, [scrape_failed], failed_ctx)))
    )
    failed = await _run_cycle(service, miner, executor)

    finished = build_msg(event="Validation finished", reason="VALIDATION_COMPLETED", severity="info", impact="")
    passed_ctx = make_context(executor=executor, miner_hotkey=miner.miner_hotkey, score=1.0, success=True)
    service.pipeline_factory.build_context = AsyncMock(return_value=passed_ctx)
    service.pipeline_factory.build_pipeline = MagicMock(
        return_value=MagicMock(run=AsyncMock(return_value=(True, [finished], passed_ctx)))
    )
    passed = await _run_cycle(service, miner, executor)

    assert failed.failure_reason_code == "SCRAPE_FAILED"
    assert passed.failure_reason_code is None
    assert passed.score == 1.0


@pytest.mark.asyncio
async def test_a_rented_outdated_run_carries_the_rented_halt_as_its_reason(monkeypatch) -> None:
    """A rented executor's image check passes and TenantEnforcementCheck halts the run with
    success=True and score 0. Regression: recording a reason only for failed runs leaves this
    result with none, and the classifier publishes the rented fleet's OUTDATED zeros as today."""
    monkeypatch.setattr(settings, "EXECUTOR_IMAGE_CHECK_ENFORCE", True)
    monkeypatch.setattr(task_service_module, "InteractiveShellService", lambda **_: _ShellThatOpens())
    service = _task_service_that_reaches_the_ssh_connect()
    service.redis_service = AsyncMock()
    service.pipeline_factory = MagicMock()
    service.pipeline_factory.build_checks = MagicMock(return_value=[])
    miner, executor = _a_job_for("node-9")
    rented_halt = build_msg(event="Executor already rented", reason="RENTED", severity="info", impact="")
    report = ExecutorImageReport(
        status=ImageVerdict.OUTDATED, observed_digest=OLD, expected_ref="ref", expected_digest=NEW
    )
    rented_ctx = make_context(
        executor=executor,
        miner_hotkey=miner.miner_hotkey,
        state=ContextState(executor_image_report=report),
        score=0.0,
        job_score=1.0,
        rented=True,
        success=True,
    )
    service.pipeline_factory.build_context = AsyncMock(return_value=rented_ctx)
    service.pipeline_factory.build_pipeline = MagicMock(
        return_value=MagicMock(run=AsyncMock(return_value=(True, [rented_halt], rented_ctx)))
    )

    result = await _run_cycle(service, miner, executor)

    assert result.score == 0
    assert result.failure_reason_code == "RENTED"
    assert rollout_grace_reason(result, _open_window(), J0) == "EXECUTOR_IMAGE_OUTDATED"


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


@pytest.mark.asyncio
async def test_the_cycle_end_read_of_the_registry_withholds_this_cycles_results(monkeypatch) -> None:
    """The 11 Sep case end to end at the cycle level: the cycle started on the old digest, the push
    landed while its jobs ran. Regression: the cycle-end registry read dropped from `sync` — the
    window would open at the next cycle and this cycle's failures would publish as zeros."""
    redis = _FakeRedis()
    tracker = _tracker(redis)
    validator_process = _validator_process(tracker)
    await tracker.observe(OLD, J0 - BPC, now=T0)
    at_start = await validator_process.observe_executor_rollout(OLD, J0)
    assert at_start.covers(J0) is False
    monkeypatch.setattr(validator_process, "fetch_executor_digest_or_none", AsyncMock(return_value=NEW))
    unreachable = _failed("node-1", "EXECUTOR_SSH_UNREACHABLE")
    healthy = _scored("node-2")

    kept, withheld = await validator_process.withhold_verdicts_for_rollout(
        {"5Miner": [unreachable, healthy]}, J0, "batch-1", at_start
    )

    assert kept == {"5Miner": [healthy]}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]
    assert json.loads(redis.store[ROLLOUT_STATE_KEY])["withheld"] == 1


@pytest.mark.asyncio
async def test_the_cycle_adds_its_withheld_count_to_the_window_total_only_inside_the_window() -> None:
    """Regression: a count written on every cycle — the window-end line would carry cycles the
    window never covered — or a count never written, leaving the window-end line at 0."""
    redis = _FakeRedis()
    tracker = _tracker(redis)
    await tracker.observe(OLD, J0 - BPC, now=T0)
    window = await tracker.observe(NEW, J0, now=T0)
    validator_process = _validator_process(tracker)
    withheld = [WithheldVerdict("5Miner", _failed("node-1", "EXECUTOR_SSH_UNREACHABLE"))] * 3

    await validator_process.record_withheld_verdicts(window, withheld, job_batch_id="batch-1", job_block=J1)
    assert json.loads(redis.store[ROLLOUT_STATE_KEY])["withheld"] == 3

    await tracker.observe(NEW, J1, now=T0)
    closed = await tracker.observe(NEW, J2, now=T0)
    assert closed.covers(J2) is False
    await validator_process.record_withheld_verdicts(closed, withheld, job_batch_id="batch-2", job_block=J2)
    assert json.loads(redis.store[ROLLOUT_STATE_KEY])["withheld"] == 3
