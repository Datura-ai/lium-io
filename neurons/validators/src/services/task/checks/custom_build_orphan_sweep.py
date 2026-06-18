"""DAH-2211 — Phase 3.4(ii) — orphan sweep for `lium-build-*` artifacts.

The happy path (Phase 3.4(i)) inlines image+scratch cleanup at pod release.
This sweep handles only the leftover case: validator crashed mid-release, SSH
disconnected mid-cleanup, or a pod row vanished from the rented-data view
before the inline cleanup ran.

Implementation notes
--------------------
- Co-located with ``stale_container_cleanup.py`` per the plan, but the cadence
  is fundamentally different: stale-container runs every pipeline cycle
  (~12 s), while this sweep is throttled to **at most once per 6 h per
  executor** to avoid hammering executor hosts with repeated `docker images`
  scans when the happy path keeps the orphan set empty.
- The throttle uses a per-executor in-memory timestamp on the check instance.
  Because checks are constructed once at pipeline-factory time and reused
  across cycles, this gives us the desired cadence without depending on Redis
  or persistent state. On validator restart we run again immediately, which
  is the desired forgiving behavior.
- The check is non-fatal. Any sweep error is logged and the pipeline
  proceeds: the validator must not block scoring on a janitorial GC.
"""

from __future__ import annotations

import asyncio
import logging
import time

from core.utils import _m, get_extra_info

from ..messages import CustomBuildOrphanSweepMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# 6 hour cadence per the plan / Phase 0.8 outcome (b).
SWEEP_INTERVAL_SECONDS = 6 * 60 * 60

# Tag prefix and scratch-dir prefix invariants from the build subroutine in
# ``docker_service._custom_build_image_tag`` / ``_custom_build_scratch_dir``.
BUILD_IMAGE_PREFIX = "lium-build-"
BUILD_SCRATCH_PREFIX = "/tmp/lium-build-"
# DAH-2211 — internet-enabled builds run in a throwaway sysbox DinD container
# named ``lium-dind-build-{pod_id}`` (see ``docker_service._dind_container_name``).
# Inline teardown removes it; this sweep mops up the validator-crashed-mid-build
# leftover so a stale container does not pin CPU/mem/disk on the executor host.
BUILD_DIND_PREFIX = "lium-dind-build-"


class CustomBuildOrphanSweepCheck:
    """Per-executor orphan sweep for `lium-build-{pod_id}` artifacts.

    Runs at most once per ``SWEEP_INTERVAL_SECONDS`` per executor uuid.
    """

    check_id = "executor.cleanup.custom_build_orphans"
    fatal = False

    def __init__(self, interval_seconds: int = SWEEP_INTERVAL_SECONDS) -> None:
        self._interval = interval_seconds
        # executor_uuid -> last successful sweep monotonic timestamp
        self._last_sweep: dict[str, float] = {}

    def _should_sweep(self, executor_uuid: str, now: float) -> bool:
        last = self._last_sweep.get(executor_uuid)
        if last is None:
            return True
        return (now - last) >= self._interval

    @staticmethod
    def _active_pod_ids(ctx: Context) -> set[str]:
        """Pod ids the validator currently believes are rented on *this* executor."""
        if ctx.state.rented_data is None:
            return set()
        executor = ctx.state.rented_data.executors.get(ctx.executor.uuid)
        if executor is None:
            return set()
        return {pod.pod_id for pod in executor.pods}

    async def _list_orphan_images(self, ssh, active_pod_ids: set[str]) -> list[str]:
        """Return `lium-build-*` image tags that are NOT in active_pod_ids."""
        cmd = (
            f'/usr/bin/docker images --filter "reference={BUILD_IMAGE_PREFIX}*" '
            f'--format "{{{{.Repository}}}}:{{{{.Tag}}}}" 2>/dev/null || true'
        )
        try:
            result = await ssh.run(cmd, check=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Custom build orphan sweep: docker images failed: %s", exc)
            return []
        orphans: list[str] = []
        for raw in (result.stdout or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            # Format is `repo:tag`. Repo is `lium-build-{pod_id}`; tag is usually
            # `latest`. We match on the repo segment.
            repo = line.split(":", 1)[0]
            if not repo.startswith(BUILD_IMAGE_PREFIX):
                continue
            pod_id = repo[len(BUILD_IMAGE_PREFIX):]
            if pod_id in active_pod_ids:
                continue
            orphans.append(line)
        return orphans

    async def _list_orphan_scratch_dirs(self, ssh, active_pod_ids: set[str]) -> list[str]:
        """Return `/tmp/lium-build-*` scratch dirs whose pod_id is not active."""
        cmd = (
            'ls -1d /tmp/lium-build-* 2>/dev/null || true'
        )
        try:
            result = await ssh.run(cmd, check=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Custom build orphan sweep: ls scratch dirs failed: %s", exc)
            return []
        orphans: list[str] = []
        for raw in (result.stdout or "").splitlines():
            path = raw.strip()
            if not path.startswith(BUILD_SCRATCH_PREFIX):
                continue
            pod_id = path[len(BUILD_SCRATCH_PREFIX):]
            if not pod_id or pod_id in active_pod_ids:
                continue
            orphans.append(path)
        return orphans

    async def _list_orphan_dind_containers(self, ssh, active_pod_ids: set[str]) -> list[str]:
        """Return `lium-dind-build-*` container names whose pod_id is not active."""
        cmd = (
            f'/usr/bin/docker ps -a --filter "name={BUILD_DIND_PREFIX}" '
            f'--format "{{{{.Names}}}}" 2>/dev/null || true'
        )
        try:
            result = await ssh.run(cmd, check=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Custom build orphan sweep: docker ps failed: %s", exc)
            return []
        orphans: list[str] = []
        for raw in (result.stdout or "").splitlines():
            name = raw.strip()
            if not name.startswith(BUILD_DIND_PREFIX):
                continue
            pod_id = name[len(BUILD_DIND_PREFIX):]
            if not pod_id or pod_id in active_pod_ids:
                continue
            orphans.append(name)
        return orphans

    async def _remove_dind_container(self, ssh, name: str) -> bool:
        # Defensive — only act on names matching our prefix.
        if not name.startswith(BUILD_DIND_PREFIX):
            return False
        cmd = f'/usr/bin/docker rm -f {name} 2>/dev/null || true'
        try:
            await ssh.run(cmd, check=False)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Custom build orphan sweep: dind rm failed (%s): %s", name, exc
            )
            return False

    async def _remove_image(self, ssh, image_ref: str) -> bool:
        cmd = f'/usr/bin/docker image rm {image_ref} 2>/dev/null || true'
        try:
            await ssh.run(cmd, check=False)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Custom build orphan sweep: image rm failed (%s): %s", image_ref, exc
            )
            return False

    async def _remove_scratch(self, ssh, path: str) -> bool:
        # Defensive — only act on paths matching our prefix.
        if not path.startswith(BUILD_SCRATCH_PREFIX):
            return False
        cmd = f'/usr/bin/rm -rf {path}'
        try:
            await ssh.run(cmd, check=False)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Custom build orphan sweep: rm scratch failed (%s): %s", path, exc
            )
            return False

    async def run(self, ctx: Context) -> CheckResult:
        executor_uuid = ctx.executor.uuid
        now = time.monotonic()

        if not self._should_sweep(executor_uuid, now):
            event = render_message(
                Msg.SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"reason": "within cadence window"},
            )
            return CheckResult(passed=True, event=event)

        active_pod_ids = self._active_pod_ids(ctx)
        try:
            orphan_images, orphan_scratch, orphan_dind = await asyncio.gather(
                self._list_orphan_images(ctx.ssh, active_pod_ids),
                self._list_orphan_scratch_dirs(ctx.ssh, active_pod_ids),
                self._list_orphan_dind_containers(ctx.ssh, active_pod_ids),
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal
            logger.warning(
                _m(
                    "Custom build orphan sweep: list failed",
                    extra=get_extra_info({"executor_uuid": executor_uuid, "error": str(exc)}),
                )
            )
            event = render_message(
                Msg.SWEPT,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "removed_image_count": 0,
                    "removed_scratch_count": 0,
                    "removed_dind_count": 0,
                    "error": str(exc),
                },
            )
            return CheckResult(passed=True, event=event)

        removed_images = 0
        removed_scratch = 0
        removed_dind = 0
        for image_ref in orphan_images:
            if await self._remove_image(ctx.ssh, image_ref):
                removed_images += 1
        for path in orphan_scratch:
            if await self._remove_scratch(ctx.ssh, path):
                removed_scratch += 1
        for name in orphan_dind:
            if await self._remove_dind_container(ctx.ssh, name):
                removed_dind += 1

        # Only update cadence timestamp on a successful run end-to-end; this way
        # a transient SSH failure (which we logged above and turned into an
        # empty list) lets us retry on the next cycle.
        self._last_sweep[executor_uuid] = now

        event = render_message(
            Msg.SWEPT,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "removed_image_count": removed_images,
                "removed_scratch_count": removed_scratch,
                "removed_dind_count": removed_dind,
                "removed_images": orphan_images,
                "removed_scratch_paths": orphan_scratch,
                "removed_dind_containers": orphan_dind,
                "active_pod_ids": sorted(active_pod_ids),
            },
        )
        return CheckResult(passed=True, event=event)
