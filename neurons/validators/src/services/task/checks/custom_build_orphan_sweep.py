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
import re
import shlex
import time

from core.utils import _m, get_extra_info

from ..messages import CustomBuildOrphanSweepMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# 6 hour cadence per the plan / Phase 0.8 outcome (b).
SWEEP_INTERVAL_SECONDS = 6 * 60 * 60

# Names before builds were named per create attempt: ``<prefix><pod id>``. Validators of that release still write
# and sweep them, so they stay readable here.
BUILD_IMAGE_PREFIX = "lium-build-"
BUILD_SCRATCH_PREFIX = "/tmp/lium-build-"
# DAH-2211 — internet-enabled builds run in a throwaway sysbox DinD container
# named ``lium-dind-build-{pod_id}.{attempt}`` (see ``docker_service._dind_container_name``).
# Inline teardown removes it; this sweep mops up the validator-crashed-mid-build
# leftover so a stale container does not pin CPU/mem/disk on the executor host.
BUILD_DIND_PREFIX = "lium-dind-build-"
# Per-attempt names, ``<prefix><pod id>.<attempt>`` (``docker_service._custom_build_image_tag``,
# ``_custom_build_scratch_dir``, ``_dind_container_name``). The prefixes are ones an older validator's sweep never
# selects: it reads everything after its own prefix as the pod id, so it would take `<pod id>.<attempt>` for an
# unknown pod and remove a running build.
ATTEMPT_IMAGE_PREFIX = "lium-podbuild-"
ATTEMPT_SCRATCH_PREFIX = "/tmp/lium-podbuild-"
ATTEMPT_DIND_PREFIX = "lium-poddind-"
IMAGE_PREFIXES = (ATTEMPT_IMAGE_PREFIX, BUILD_IMAGE_PREFIX)
SCRATCH_PREFIXES = (ATTEMPT_SCRATCH_PREFIX, BUILD_SCRATCH_PREFIX)
DIND_PREFIXES = (ATTEMPT_DIND_PREFIX, BUILD_DIND_PREFIX)
# A build's two iptables helpers are named after its DinD with these suffixes
# (``docker_service._dind_firewall_helper_names``). ``docker ps --filter name=``
# lists them with the DinD, so the pod id is read after the suffix is stripped:
# a helper of an active build belongs to that build and is not an orphan.
BUILD_DIND_HELPER_SUFFIXES = ("-fw-apply", "-fw-remove")
# The node's build listing covers every pod on it, the local keep set only this registration's: a candidate goes
# only once the backend confirms its rental over. Bounded so a slow backend never stalls the pipeline.
MAX_RENTAL_LOOKUPS = 20
RENTAL_LOOKUP_TIMEOUT_SECONDS = 15


def dind_container_pod_id(name: str) -> str | None:
    """Pod id of a ``lium-dind-build-*`` container or of one of its firewall helpers.

    ``None`` when the name carries another prefix or no pod id at all.
    """
    pod_id = _after_prefix(name, DIND_PREFIXES)
    if pod_id is None:
        return None
    for suffix in BUILD_DIND_HELPER_SUFFIXES:
        if pod_id.endswith(suffix):
            pod_id = pod_id[: -len(suffix)]
            break
    return _pod_id_of(pod_id)


def dind_build_pod_id(name: str) -> str | None:
    """Pod id of a ``lium-dind-build-*`` build container itself, not of its firewall helpers."""
    rest = _after_prefix(name, DIND_PREFIXES)
    if name.endswith(BUILD_DIND_HELPER_SUFFIXES) or rest is None:
        return None
    # only `<pod id>.<attempt>` with the create's 8-hex attempt (secrets.token_hex(4)), or a bare pod id named before
    # attempts: any other suffix is not a build this validator made
    pod_id, dot, attempt = rest.partition(".")
    if dot and not re.fullmatch(r"[0-9a-f]{8}", attempt):
        return None
    return pod_id or None


def _after_prefix(name: str, prefixes: tuple[str, ...]) -> str | None:
    """What follows the one of `prefixes` that `name` starts with; None when it starts with none."""
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def _pod_id_of(name_rest: str) -> str | None:
    # `<pod id>.<create attempt>` (docker_service._custom_build_image_tag), or a bare pod id named before attempts
    return name_rest.split(".", 1)[0] or None


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
        # executor_uuid -> {candidate pod id: the sweep number it was last asked about}, so each sweep asks the
        # candidates asked longest ago first and every candidate is reached over successive sweeps
        self._rental_asked: dict[str, dict[str, int]] = {}
        self._sweeps = 0

    def _should_sweep(self, executor_uuid: str, now: float) -> bool:
        last = self._last_sweep.get(executor_uuid)
        if last is None:
            return True
        return (now - last) >= self._interval

    @staticmethod
    def _active_pod_ids(ctx: Context) -> set[str]:
        """Pod ids the validator currently believes are rented on *this* executor.

        Pods still being created are added from the pending-pods set in ``run``.
        """
        if ctx.state.rented_data is None:
            return set()
        executor = ctx.state.rented_data.executors.get(ctx.executor.uuid)
        if executor is None:
            return set()
        return {pod.pod_id for pod in executor.pods}

    async def _list_orphan_images(self, ssh, active_pod_ids: set[str]) -> list[str]:
        """Return `lium-build-*` image tags that are NOT in active_pod_ids."""
        cmd = (
            f'/usr/bin/docker images --filter "reference={ATTEMPT_IMAGE_PREFIX}*" '
            f'--filter "reference={BUILD_IMAGE_PREFIX}*" '
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
            rest = _after_prefix(repo, IMAGE_PREFIXES)
            if rest is None:
                continue
            pod_id = _pod_id_of(rest)
            if pod_id in active_pod_ids:
                continue
            orphans.append(line)
        return orphans

    async def _list_orphan_scratch_dirs(self, ssh, active_pod_ids: set[str]) -> list[str]:
        """Return `/tmp/lium-build-*` scratch dirs whose pod_id is not active."""
        cmd = (
            'ls -1d /tmp/lium-podbuild-* /tmp/lium-build-* 2>/dev/null || true'
        )
        try:
            result = await ssh.run(cmd, check=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Custom build orphan sweep: ls scratch dirs failed: %s", exc)
            return []
        orphans: list[str] = []
        for raw in (result.stdout or "").splitlines():
            path = raw.strip()
            rest = _after_prefix(path, SCRATCH_PREFIXES)
            if rest is None:
                continue
            pod_id = _pod_id_of(rest)
            if not pod_id or pod_id in active_pod_ids:
                continue
            orphans.append(path)
        return orphans

    async def _list_orphan_dind_containers(self, ssh, active_pod_ids: set[str]) -> list[str]:
        """Return `lium-dind-build-*` container names whose pod_id is not active.

        The listing includes the build's firewall helpers (`<dind>-fw-apply`,
        `<dind>-fw-remove`); they go with their pod's verdict.
        """
        cmd = (
            f'/usr/bin/docker ps -a --filter "name={ATTEMPT_DIND_PREFIX}" --filter "name={BUILD_DIND_PREFIX}" '
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
            pod_id = dind_container_pod_id(name)
            if not pod_id or pod_id in active_pod_ids:
                continue
            orphans.append(name)
        return orphans

    async def _pods_confirmed_over(self, ctx: Context, pod_ids: set[str]) -> set[str]:
        """The ``pod_ids`` the backend says have no live rental. A failed, timed-out or capped lookup keeps the pod:
        a removal cannot be undone. The cap takes the pods never asked, then those asked longest ago."""
        over: set[str] = set()
        self._sweeps += 1
        asked = self._rental_asked.get(ctx.executor.uuid, {})
        chosen = sorted(pod_ids, key=lambda p: (asked.get(p, 0), p))[:MAX_RENTAL_LOOKUPS]
        # only current candidates are remembered: a pod whose artifacts are gone is never asked again
        self._rental_asked[ctx.executor.uuid] = {
            **{p: asked[p] for p in pod_ids if p in asked}, **dict.fromkeys(chosen, self._sweeps)
        }

        async def ask(pod_id: str) -> None:
            try:
                answer = await ctx.services.backend.get_pod_rental_active(pod_id)
            except Exception as exc:  # noqa: BLE001 — unknown: keep
                logger.warning("Custom build orphan sweep: rental lookup failed (%s): %s", pod_id, exc)
                return
            if answer is not None and not answer.active:
                over.add(pod_id)

        try:
            async with asyncio.timeout(RENTAL_LOOKUP_TIMEOUT_SECONDS):
                await asyncio.gather(*(ask(p) for p in chosen))
        except TimeoutError:
            logger.warning("Custom build orphan sweep: rental lookups timed out; unanswered pods kept")
        return over

    async def _remove_dind_container(self, ssh, name: str) -> bool:
        # Defensive — only act on names matching our prefix.
        if _after_prefix(name, DIND_PREFIXES) is None:
            return False
        # name, image ref and path are node output: quoted, so a planted name cannot run as shell
        cmd = f'/usr/bin/docker rm -fv {shlex.quote(name)} 2>/dev/null || true'
        try:
            await ssh.run(cmd, check=False)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Custom build orphan sweep: dind rm failed (%s): %s", name, exc
            )
            return False

    async def _remove_image(self, ssh, image_ref: str) -> bool:
        cmd = f'/usr/bin/docker image rm {shlex.quote(image_ref)} 2>/dev/null || true'
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
        rest = _after_prefix(path, SCRATCH_PREFIXES)
        if rest is None or "/" in rest:
            return False
        cmd = f'/usr/bin/rm -rf -- {shlex.quote(path)}'
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
        # A pod still being created is not in rented_data yet; its in-flight build
        # (DinD, scratch dir, image) is listed in the shared pending-pods set. Every
        # entry counts, not only the last 30 min: setup + build + export can run
        # past that, and a create cut off by a restart has its entry cleared then.
        try:
            pending = await ctx.services.redis.get_pending_pods(
                ctx.miner_hotkey, executor_uuid, max_age_seconds=None
            )
        except Exception as exc:  # noqa: BLE001 — cannot tell what is in flight: sweep nothing
            event = render_message(
                Msg.SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"reason": "pending pods unreadable", "error": str(exc)},
            )
            return CheckResult(passed=True, event=event)
        active_pod_ids |= {item["pod_id"] for item in pending if item.get("pod_id")}
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

        pod_of = {
            **{ref: _pod_id_of(_after_prefix(ref.split(":", 1)[0], IMAGE_PREFIXES) or "") for ref in orphan_images},
            **{path: _pod_id_of(_after_prefix(path, SCRATCH_PREFIXES) or "") for path in orphan_scratch},
            **{name: dind_container_pod_id(name) for name in orphan_dind},
        }
        over = await self._pods_confirmed_over(ctx, {p for p in pod_of.values() if p})
        orphan_images, orphan_scratch, orphan_dind = (
            [item for item in items if pod_of[item] in over] for items in (orphan_images, orphan_scratch, orphan_dind)
        )

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
