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
- The build container and the built image live on the executor host's Docker
  daemon, which a testnet executor can share with a mainnet one. Both carry the
  `io.lium.netuid` label (services/rental_container_labels.py); the sweep
  removes only its own network's (unlabeled ones only on mainnet), and only
  once they are older than the grace, so a build still running for a pod the
  backend does not list yet is left alone. A listing that cannot be read
  removes nothing.
"""

from __future__ import annotations

import asyncio
import logging
import time

from core.config import settings
from core.docker_utils import DockerCommand
from core.utils import _m, get_extra_info
from services.rental_container_labels import (
    NETUID_LABEL,
    netuid_owns,
    parse_names_with_netuid,
    ps_filter_names_netuid_command,
)

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

# A build container or image younger than this is never an orphan: the build
# runs up to CUSTOM_DOCKERFILE_BUILD_TIMEOUT_SECONDS before the pod exists.
BUILD_ORPHAN_MIN_GRACE_SECONDS = 2 * 60 * 60


def default_grace_seconds() -> int:
    return max(
        BUILD_ORPHAN_MIN_GRACE_SECONDS,
        2 * int(settings.CUSTOM_DOCKERFILE_BUILD_TIMEOUT_SECONDS),
    )


def build_images_command(label_filter: str | None = None) -> str:
    label = f'--filter "label={label_filter}" ' if label_filter else ""
    return (
        f'/usr/bin/docker images --filter "reference={BUILD_IMAGE_PREFIX}*" {label}'
        f'--format "{{{{.Repository}}}}:{{{{.Tag}}}}"'
    )


DIND_CONTAINERS_COMMAND = ps_filter_names_netuid_command(f"^{BUILD_DIND_PREFIX}")


class CustomBuildOrphanSweepCheck:
    """Per-executor orphan sweep for `lium-build-{pod_id}` artifacts.

    Runs at most once per ``SWEEP_INTERVAL_SECONDS`` per executor uuid.
    """

    check_id = "executor.cleanup.custom_build_orphans"
    fatal = False

    def __init__(
        self,
        interval_seconds: int = SWEEP_INTERVAL_SECONDS,
        netuid: int | None = None,
        grace_seconds: int | None = None,
    ) -> None:
        self._interval = interval_seconds
        self._netuid = settings.BITTENSOR_NETUID if netuid is None else netuid
        self._grace = default_grace_seconds() if grace_seconds is None else grace_seconds
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

    @staticmethod
    async def _list(ssh, cmd: str, what: str) -> list[str] | None:
        """Non-blank output lines, or None when the listing cannot be read."""
        try:
            result = await ssh.run(cmd, check=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Custom build orphan sweep: %s failed: %s", what, exc)
            return None
        if result.exit_status != 0:
            logger.warning(
                "Custom build orphan sweep: %s exited %s", what, result.exit_status
            )
            return None
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    async def _list_orphan_images(self, ssh, active_pod_ids: set[str]) -> list[str]:
        """Return this network's `lium-build-*` image tags that are NOT in active_pod_ids.

        `docker images` has no label column, so three listings (every build
        image, the labeled ones, this network's) tell each image's label apart.
        """
        listings = await asyncio.gather(
            self._list(ssh, build_images_command(), "docker images"),
            self._list(ssh, build_images_command(NETUID_LABEL), "docker images (labeled)"),
            self._list(
                ssh,
                build_images_command(f"{NETUID_LABEL}={self._netuid}"),
                "docker images (own network)",
            ),
        )
        if any(listing is None for listing in listings):
            return []
        every, labeled, own = (set(listing) for listing in listings)
        orphans: list[str] = []
        for line in sorted(every):
            # Format is `repo:tag`. Repo is `lium-build-{pod_id}`; tag is usually
            # `latest`. We match on the repo segment.
            repo = line.split(":", 1)[0]
            if not repo.startswith(BUILD_IMAGE_PREFIX):
                continue
            pod_id = repo[len(BUILD_IMAGE_PREFIX):]
            if pod_id in active_pod_ids:
                continue
            if line in own:
                orphans.append(line)
            elif line not in labeled and netuid_owns(None, self._netuid):
                orphans.append(line)
        return orphans

    async def _list_orphan_scratch_dirs(self, ssh, active_pod_ids: set[str]) -> list[str]:
        """Return `/tmp/lium-build-*` scratch dirs whose pod_id is not active.

        The dirs sit in this executor's own filesystem, never on the Docker
        daemon another network's executor may share, so they need no label.
        """
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
        """Return this network's `lium-dind-build-*` container names whose pod_id is not active."""
        lines = await self._list(ssh, DIND_CONTAINERS_COMMAND, "docker ps")
        if lines is None:
            return []
        orphans: list[str] = []
        for entry in parse_names_with_netuid("\n".join(lines)):
            name = entry.name
            if not name.startswith(BUILD_DIND_PREFIX):
                continue
            pod_id = name[len(BUILD_DIND_PREFIX):]
            if not pod_id or pod_id in active_pod_ids:
                continue
            if not netuid_owns(entry.netuid_label, self._netuid):
                continue
            orphans.append(name)
        return orphans

    async def _older_than_grace(self, ssh, refs: list[str]) -> list[str]:
        """The containers/images among ``refs`` created at least the grace ago; an unreadable age keeps one."""
        if not refs:
            return []
        now = await self._host_seconds(ssh, "date +%s")
        if now is None:
            return []
        old: list[str] = []
        for ref in refs:
            created = await self._host_seconds(ssh, DockerCommand.inspect_created_timestamp(ref))
            if created is not None and now - created >= self._grace:
                old.append(ref)
        return old

    @staticmethod
    async def _host_seconds(ssh, cmd: str) -> int | None:
        try:
            result = await ssh.run(cmd, check=False)
            if result.exit_status != 0:
                return None
            return int((result.stdout or "").strip())
        except Exception:  # noqa: BLE001
            return None

    async def _remove_dind_container(self, ssh, name: str) -> bool:
        # Defensive — only act on names matching our prefix.
        if not name.startswith(BUILD_DIND_PREFIX):
            return False
        cmd = f'/usr/bin/docker rm -fv {name} 2>/dev/null || true'
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
            orphan_images = await self._older_than_grace(ctx.ssh, orphan_images)
            orphan_dind = await self._older_than_grace(ctx.ssh, orphan_dind)
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
