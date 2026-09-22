"""DAH-2211 — Phase 3.4(ii) — orphan sweep behaviour test."""

from __future__ import annotations

import shlex
from unittest.mock import AsyncMock, Mock

import pytest
from core.config import settings
from services.task.checks.custom_build_orphan_sweep import (
    BUILD_ORPHAN_MIN_GRACE_SECONDS,
    DIND_CONTAINERS_COMMAND,
    CustomBuildOrphanSweepCheck,
    build_images_command,
    default_grace_seconds,
)

MAINNET = 51
STAGING = 37
NOW = 1_000_000
OLD = NOW - 3 * 60 * 60


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
    check = CustomBuildOrphanSweepCheck(interval_seconds=0, netuid=MAINNET)  # always sweep

    ssh = AsyncMock()
    calls: list[str] = []

    async def _run(cmd, **kw):
        calls.append(cmd)
        if "docker images" in cmd:
            return _result(stdout="lium-build-POD-DEAD:latest\nlium-build-POD-ALIVE:latest\n")
        if "ls -1d /tmp/lium-build-" in cmd:
            return _result(stdout="/tmp/lium-build-POD-DEAD\n/tmp/lium-build-POD-ALIVE\n")
        if "json .Created" in cmd:
            return _result(stdout=str(OLD))
        if cmd == "date +%s":
            return _result(stdout=str(NOW))
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
    check = CustomBuildOrphanSweepCheck(interval_seconds=0, netuid=MAINNET)

    ssh = AsyncMock()
    calls: list[str] = []

    async def _run(cmd, **kw):
        calls.append(cmd)
        if "docker ps -a" in cmd:
            return _result(
                stdout="lium-dind-build-POD-DEAD\nlium-dind-build-POD-ALIVE\n"
            )
        if "json .Created" in cmd:
            return _result(stdout=str(OLD))
        if cmd == "date +%s":
            return _result(stdout=str(NOW))
        return _result()

    ssh.run = _run
    ctx = _make_ctx(executor_uuid="exec-1", active_pod_ids={"POD-ALIVE"}, ssh=ssh)

    result = await check.run(ctx)
    assert result.passed is True

    dind_rm_calls = [c for c in calls if "docker rm -fv lium-dind-build-" in c]
    assert any("lium-dind-build-POD-DEAD" in c for c in dind_rm_calls)
    assert not any("lium-dind-build-POD-ALIVE" in c for c in dind_rm_calls)


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


# ------------------------------------------------------------------
# network scope: a testnet executor can share the host's Docker daemon
# ------------------------------------------------------------------


class FakeHost:
    """Build images and DinD containers on one daemon, each with its netuid label and age."""

    def __init__(self, *, images=None, dind=None, created=None, broken=()):
        self.images: dict[str, str | None] = images or {}
        self.dind: dict[str, str | None] = dind or {}
        self.created: dict[str, int] = created or {}
        self.broken = set(broken)
        self.calls: list[str] = []

    async def run(self, cmd, **kw):
        self.calls.append(cmd)
        if cmd in self.broken:
            return _result(exit_status=1)
        if cmd == build_images_command():
            return _result("\n".join(self.images))
        if cmd == build_images_command("io.lium.netuid"):
            return _result("\n".join(ref for ref, label in self.images.items() if label))
        for netuid in (MAINNET, STAGING):
            if cmd == build_images_command(f"io.lium.netuid={netuid}"):
                refs = [ref for ref, label in self.images.items() if label == str(netuid)]
                return _result("\n".join(refs))
        if cmd == DIND_CONTAINERS_COMMAND:
            return _result("\n".join(f"{name} {label or ''}" for name, label in self.dind.items()))
        if "json .Created" in cmd:
            return _result(str(self.created.get(shlex.split(cmd)[2], OLD)))
        if cmd == "date +%s":
            return _result(str(NOW))
        return _result()

    def removed(self) -> list[str]:
        prefixes = ("/usr/bin/docker image rm ", "/usr/bin/docker rm -fv ")
        return [cmd.split()[3] for cmd in self.calls if cmd.startswith(prefixes)]


# a mainnet build from before the label, a labeled mainnet build and a staging build
SHARED_HOST = dict(
    images={
        "lium-build-legacy:latest": None,
        "lium-build-prod:latest": "51",
        "lium-build-stage:latest": "37",
    },
    dind={
        "lium-dind-build-legacy": None,
        "lium-dind-build-prod": "51",
        "lium-dind-build-stage": "37",
        # docker's name filter matches anywhere in the name
        "x-lium-dind-build-stage2": "37",
    },
)


async def _sweep(host: FakeHost, netuid: int, active: set[str] | None = None) -> list[str]:
    check = CustomBuildOrphanSweepCheck(interval_seconds=0, netuid=netuid)
    result = await check.run(_make_ctx(active_pod_ids=active or set(), ssh=host))
    assert result.passed is True
    return sorted(host.removed())


@pytest.mark.asyncio
async def test_mainnet_sweep_removes_legacy_and_own_builds_and_keeps_staging():
    assert await _sweep(FakeHost(**SHARED_HOST), MAINNET) == [
        "lium-build-legacy:latest",
        "lium-build-prod:latest",
        "lium-dind-build-legacy",
        "lium-dind-build-prod",
    ]


@pytest.mark.asyncio
async def test_staging_sweep_keeps_prod_labeled_and_unlabeled_builds():
    assert await _sweep(FakeHost(**SHARED_HOST), STAGING) == [
        "lium-build-stage:latest",
        "lium-dind-build-stage",
    ]


@pytest.mark.asyncio
async def test_sweep_leaves_a_build_younger_than_the_grace():
    young = NOW - 10 * 60
    host = FakeHost(
        images={"lium-build-new:latest": "37", "lium-build-old:latest": "37"},
        dind={"lium-dind-build-new": "37", "lium-dind-build-old": "37"},
        created={"lium-build-new:latest": young, "lium-dind-build-new": young},
    )
    assert await _sweep(host, STAGING) == ["lium-build-old:latest", "lium-dind-build-old"]


@pytest.mark.asyncio
async def test_sweep_leaves_a_build_whose_age_cannot_be_read():
    host = FakeHost(dind={"lium-dind-build-x": "51"}, broken={"date +%s"})
    assert await _sweep(host, MAINNET) == []


@pytest.mark.asyncio
async def test_mainnet_sweep_removes_no_image_when_the_label_listing_fails():
    host = FakeHost(**SHARED_HOST, broken={build_images_command("io.lium.netuid")})
    assert await _sweep(host, MAINNET) == ["lium-dind-build-legacy", "lium-dind-build-prod"]


@pytest.mark.asyncio
async def test_sweep_keeps_the_builds_of_active_pods():
    host = FakeHost(**SHARED_HOST)
    assert await _sweep(host, MAINNET, active={"prod"}) == [
        "lium-build-legacy:latest",
        "lium-dind-build-legacy",
    ]


def test_listings_filter_by_label_and_anchor_the_container_name():
    assert build_images_command("io.lium.netuid=37") == (
        '/usr/bin/docker images --filter "reference=lium-build-*" '
        '--filter "label=io.lium.netuid=37" --format "{{.Repository}}:{{.Tag}}"'
    )
    assert DIND_CONTAINERS_COMMAND == (
        '/usr/bin/docker ps -a --filter "name=^lium-dind-build-" '
        "--format '{{.Names}} {{.Label \"io.lium.netuid\"}}'"
    )


def test_grace_outlasts_the_build_timeout(monkeypatch):
    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_BUILD_TIMEOUT_SECONDS", 1200)
    assert default_grace_seconds() == BUILD_ORPHAN_MIN_GRACE_SECONDS
    monkeypatch.setattr(settings, "CUSTOM_DOCKERFILE_BUILD_TIMEOUT_SECONDS", 3 * 60 * 60)
    assert default_grace_seconds() == 6 * 60 * 60
