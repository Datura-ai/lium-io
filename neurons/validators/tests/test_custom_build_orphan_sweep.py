"""DAH-2211 — Phase 3.4(ii) — orphan sweep behaviour test."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from services.task.checks.custom_build_orphan_sweep import (
    CustomBuildOrphanSweepCheck,
)


def _result(stdout: str = "", exit_status: int = 0):
    r = Mock()
    r.stdout = stdout
    r.stderr = ""
    r.exit_status = exit_status
    return r


def _make_ctx(*, executor_uuid: str = "exec-1", active_pod_ids: set[str] | None = None, ssh=None):
    """Build a minimal ctx-like object the check actually consumes."""
    ssh = ssh or AsyncMock()
    state = Mock()
    if active_pod_ids is None:
        state.rented_data = None
    else:
        executor = Mock()
        executor.pods = [Mock(pod_id=pid) for pid in active_pod_ids]
        rented_data = Mock()
        rented_data.executors = {executor_uuid: executor}
        state.rented_data = rented_data
    executor_info = Mock()
    executor_info.uuid = executor_uuid
    ctx = Mock()
    ctx.executor = executor_info
    ctx.ssh = ssh
    ctx.state = state
    ctx.default_extra = {}
    ctx.pipeline_id = "pipe-1"
    return ctx


@pytest.mark.asyncio
async def test_sweep_removes_orphan_image_and_scratch():
    check = CustomBuildOrphanSweepCheck(interval_seconds=0)  # always sweep

    ssh = AsyncMock()
    calls: list[str] = []

    async def _run(cmd, **kw):
        calls.append(cmd)
        if "docker images" in cmd:
            return _result(stdout="lium-build-POD-DEAD:latest\nlium-build-POD-ALIVE:latest\n")
        if "ls -1d /tmp/lium-build-" in cmd:
            return _result(stdout="/tmp/lium-build-POD-DEAD\n/tmp/lium-build-POD-ALIVE\n")
        return _result()

    ssh.run = _run
    ctx = _make_ctx(executor_uuid="exec-1", active_pod_ids={"POD-ALIVE"}, ssh=ssh)

    result = await check.run(ctx)
    assert result.passed is True

    # Only POD-DEAD removed; POD-ALIVE preserved.
    image_rm_calls = [c for c in calls if "docker image rm" in c]
    rm_dir_calls = [c for c in calls if c.startswith("/usr/bin/rm -rf /tmp/lium-build-")]
    assert any("lium-build-POD-DEAD:latest" in c for c in image_rm_calls)
    assert not any("lium-build-POD-ALIVE" in c for c in image_rm_calls)
    assert any("/tmp/lium-build-POD-DEAD" in c for c in rm_dir_calls)
    assert not any("/tmp/lium-build-POD-ALIVE" in c for c in rm_dir_calls)


@pytest.mark.asyncio
async def test_sweep_respects_cadence():
    """Second invocation within the cadence window must NOT re-list / remove."""
    check = CustomBuildOrphanSweepCheck(interval_seconds=6 * 60 * 60)

    ssh = AsyncMock()
    calls: list[str] = []

    async def _run(cmd, **kw):
        calls.append(cmd)
        if "docker images" in cmd:
            return _result(stdout="lium-build-X:latest\n")
        if "ls -1d /tmp/lium-build-" in cmd:
            return _result(stdout="/tmp/lium-build-X\n")
        return _result()

    ssh.run = _run
    ctx = _make_ctx(executor_uuid="exec-A", active_pod_ids=set(), ssh=ssh)

    await check.run(ctx)
    first_call_count = len(calls)
    await check.run(ctx)
    # The second run should have issued zero new SSH commands.
    assert len(calls) == first_call_count


@pytest.mark.asyncio
async def test_sweep_no_op_when_nothing_orphaned():
    """Happy path: inline cleanup at pod release means listings are empty,
    so no destructive command runs."""
    check = CustomBuildOrphanSweepCheck(interval_seconds=0)

    ssh = AsyncMock()
    calls: list[str] = []

    async def _run(cmd, **kw):
        calls.append(cmd)
        # No orphan output at all
        return _result(stdout="")

    ssh.run = _run
    ctx = _make_ctx(executor_uuid="exec-1", active_pod_ids={"POD-1"}, ssh=ssh)
    result = await check.run(ctx)
    assert result.passed is True
    assert not any("docker image rm" in c for c in calls)
    assert not any(c.startswith("/usr/bin/rm -rf /tmp/lium-build-") for c in calls)


@pytest.mark.asyncio
async def test_sweep_non_fatal_on_ssh_error():
    """An SSH failure during list/rm must NOT mark the check fatal."""
    check = CustomBuildOrphanSweepCheck(interval_seconds=0)

    ssh = AsyncMock()

    async def _raise(cmd, **kw):
        raise RuntimeError("ssh boom")

    ssh.run = _raise
    ctx = _make_ctx(executor_uuid="exec-1", active_pod_ids=set(), ssh=ssh)
    result = await check.run(ctx)
    assert result.passed is True
    assert check.fatal is False
