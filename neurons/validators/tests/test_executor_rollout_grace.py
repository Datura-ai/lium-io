"""DAH-3405: an executor that fails during a known executor-image rollout gets no verdict, not a 0.

11 Sep 2026 07:50–08:00Z: executor-v1.126 was pushed mid-cycle, watchtower recreated the executor
containers under the validator's SSH sessions, and the cycle read the fleet as 168 transport-unreachable,
51 scrape-failed, 50 insufficient-ports and 70 EXECUTOR_IMAGE_OUTDATED rows (Rustam's cycle log; tsdb:
506 of 514 executors at 0). Each test below names the regression that would bring one of those zeros back.
"""

from datetime import UTC, datetime, timedelta

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from services.executor_rollout import (
    ExecutorRolloutTracker,
    RolloutWindow,
    rollout_grace_reason,
    withhold_rollout_verdicts,
)
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


# --- the cycle: what happens when the tracker itself cannot be read -----------------------------


