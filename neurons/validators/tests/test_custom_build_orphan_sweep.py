"""DAH-2211 — Phase 3.4(ii) — orphan sweep behaviour test."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from services.task.checks.custom_build_orphan_sweep import (
    BUILD_DIND_PREFIX,
    CustomBuildOrphanSweepCheck,
    dind_container_pod_id,
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
async def test_sweep_removes_orphan_dind_container():
    """DAH-2211: a leftover `lium-dind-build-*` container (validator crashed
    mid-build) is force-removed unless its pod is still active."""
    check = CustomBuildOrphanSweepCheck(interval_seconds=0)

    ssh = AsyncMock()
    calls: list[str] = []

    async def _run(cmd, **kw):
        calls.append(cmd)
        if "docker ps -a" in cmd:
            return _result(
                stdout="lium-dind-build-POD-DEAD\nlium-dind-build-POD-ALIVE\n"
            )
        return _result()

    ssh.run = _run
    ctx = _make_ctx(executor_uuid="exec-1", active_pod_ids={"POD-ALIVE"}, ssh=ssh)

    result = await check.run(ctx)
    assert result.passed is True

    dind_rm_calls = [c for c in calls if "docker rm -fv lium-dind-build-" in c]
    assert any("lium-dind-build-POD-DEAD" in c for c in dind_rm_calls)
    assert not any("lium-dind-build-POD-ALIVE" in c for c in dind_rm_calls)


@pytest.mark.asyncio
async def test_sweep_keeps_the_firewall_helpers_of_an_active_build():
    """Regression: `docker ps --filter name=lium-dind-build-` lists a build's
    firewall helpers (`<dind>-fw-apply`, `<dind>-fw-remove`) with the DinD, and
    the sweep read `<pod_id>-fw-apply` as the pod id. No pod has that id, so a
    helper of an ACTIVE build was removed while it was inserting the build's
    egress rules. The suffix is stripped before the active-pod check; a helper
    whose pod is gone is still an orphan and goes."""
    check = CustomBuildOrphanSweepCheck(interval_seconds=0)

    ssh = AsyncMock()
    calls: list[str] = []

    async def _run(cmd, **kw):
        calls.append(cmd)
        if "docker ps -a" in cmd:
            return _result(
                stdout=(
                    "lium-dind-build-POD-ALIVE\n"
                    "lium-dind-build-POD-ALIVE-fw-apply\n"
                    "lium-dind-build-POD-DEAD\n"
                    "lium-dind-build-POD-DEAD-fw-remove\n"
                )
            )
        return _result()

    ssh.run = _run
    ctx = _make_ctx(executor_uuid="exec-1", active_pod_ids={"POD-ALIVE"}, ssh=ssh)

    result = await check.run(ctx)
    assert result.passed is True

    removed = [c.split()[3] for c in calls if c.startswith("/usr/bin/docker rm -fv ")]
    assert removed == ["lium-dind-build-POD-DEAD", "lium-dind-build-POD-DEAD-fw-remove"], calls
    assert result.event.what_we_saw["removed_dind_containers"] == removed


def test_dind_container_pod_id_agrees_with_the_names_docker_service_gives():
    """The sweep's suffixes are a copy of `DockerService._dind_firewall_helper_names`;
    this keeps the two from drifting apart."""
    from services.docker_service import DockerService

    dind = DockerService._dind_container_name("POD-1")
    assert dind.startswith(BUILD_DIND_PREFIX)
    assert dind_container_pod_id(dind) == "POD-1"
    for helper in DockerService._dind_firewall_helper_names(dind):
        assert dind_container_pod_id(helper) == "POD-1", helper
    assert dind_container_pod_id("pod_renter") is None
    assert dind_container_pod_id(BUILD_DIND_PREFIX) is None


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
