from __future__ import annotations

import shlex
import time
from dataclasses import replace

from core.config import settings

from ..messages import PrePullCachedMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

_DOCKER = "/usr/bin/docker"
# A first-missing record outlives the grace by this much, so a node checked a few cycles apart
# keeps its clock; one that stops being checked (rented, gone) is forgotten a day after.
_MISSING_TTL_SLACK_SECONDS = 24 * 3600


class PrePullCachedCheck:
    """Report which of the backend's pre-pull images an idle executor holds (DAH-2977).

    The backend serves each node its default image plus the top-N official templates marked
    ``pre_pull: true`` and digest-pinned (``/executors/default-docker-image?include_pre_pull=true``,
    lium-platform#577). The executor warms those while idle (lium-io#1283, on by default with
    lium-io#1412). This check asks the same endpoint the executor asks and probes
    ``docker image inspect <repo>@<digest>`` for every pre-pull entry: exit 0 means the node
    holds exactly the content a rental of that template would otherwise pull.

    Always ``passed=True`` and never fatal. The count, the missing refs and the refs missing
    past their grace window go into ``executor.specs["pre_pull_images"]`` through pipeline state,
    so the fleet's coverage per node and per template is readable from the backend and Loki.
    Score reads ``missing_past_grace`` in ``incentive.default.get_pre_pull_multiplier``, which is
    1.0 unless ``PRE_PULL_REQUIRED_CUTOFF`` is set and past. The node's default image stays with
    ``CachedTemplateVerificationCheck``.

    Fails open on every uncertainty (flag off, unknown GPU/driver, backend unreachable, no
    pre-pull entries, SSH error, unparseable output): a skip event, nothing published.
    """

    check_id = "executor.validate.pre_pull_cached"
    fatal = False

    def _skip(self, ctx: Context, reason: str, **what) -> CheckResult:
        event = render_message(
            Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"reason": reason, **what}
        )
        return CheckResult(passed=True, event=event)

    async def _missing_past_grace(self, ctx: Context, entries: list, statuses: list[str]) -> list[str]:
        """Missing refs whose grace window is over. The window starts per node and per
        ``repo@digest`` at the first cycle that sees it missing, and ends when the node holds it.
        Any Redis error fails open to an empty list: no image counts against the node."""
        redis = ctx.services.redis
        if redis is None:
            return []
        grace = settings.PRE_PULL_REQUIRED_GRACE_SECONDS
        now = time.time()
        past: list[str] = []
        try:
            for image, status in zip(entries, statuses):
                pinned_ref = f"{image.docker_image}@{image.docker_image_digest}"
                if status == "0":
                    await redis.clear_pre_pull_missing(ctx.executor.uuid, pinned_ref)
                    continue
                since = await redis.pre_pull_missing_since(
                    ctx.executor.uuid, pinned_ref, now, ttl_seconds=grace + _MISSING_TTL_SLACK_SECONDS
                )
                if now - since >= grace:
                    past.append(image.image_ref)
        except Exception:
            return []
        return past

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.PRE_PULL_CACHED_CHECK_ENABLED:
            return self._skip(ctx, "PRE_PULL_CACHED_CHECK_ENABLED is off")

        gpu_model = ctx.state.gpu_model
        driver_version = str((ctx.state.specs or {}).get("gpu", {}).get("driver") or "")
        if not gpu_model or not driver_version:
            return self._skip(
                ctx,
                "missing gpu_model or driver_version",
                gpu_model=gpu_model,
                driver_version=driver_version,
            )

        try:
            images = await ctx.services.backend.get_default_docker_image(
                gpu_model, driver_version, include_pre_pull=True
            )
        except Exception as exc:
            return self._skip(ctx, "backend request failed", error=str(exc))

        entries = [
            image
            for image in images or []
            if getattr(image, "pre_pull", False) and image.docker_image_digest
        ]
        if not entries:
            return self._skip(
                ctx,
                "no pre-pull images served for this GPU and driver",
                gpu_model=gpu_model,
                driver_version=driver_version,
            )

        # One round trip: one exit status per line, in entry order.
        command = "; ".join(
            f"{_DOCKER} image inspect --format '{{{{.Id}}}}' "
            f"{shlex.quote(f'{image.docker_image}@{image.docker_image_digest}')} "
            '>/dev/null 2>&1; echo "$?"'
            for image in entries
        )
        try:
            result = await ctx.ssh.run(command, check=False)
        except Exception as exc:
            return self._skip(ctx, "docker image inspect failed", error=str(exc))

        statuses = (getattr(result, "stdout", None) or "").split()
        if len(statuses) != len(entries) or any(not s.isdigit() for s in statuses):
            return self._skip(
                ctx, "unparseable docker image inspect output", output=(result.stdout or "")[:200]
            )

        cached = [image.image_ref for image, status in zip(entries, statuses) if status == "0"]
        missing = [image.image_ref for image, status in zip(entries, statuses) if status != "0"]
        report = {
            "expected": len(entries),
            "cached": len(cached),
            "missing": missing,
            "missing_past_grace": await self._missing_past_grace(ctx, entries, statuses),
        }

        event = render_message(
            Msg.MISSING if missing else Msg.ALL_CACHED,
            ctx=ctx,
            check_id=self.check_id,
            what={
                **report,
                "cached_refs": cached,
                "digests": {image.image_ref: image.docker_image_digest for image in entries},
                "gpu_model": gpu_model,
                "driver_version": driver_version,
            },
        )
        return CheckResult(
            passed=True,
            event=event,
            updates={"state": replace(ctx.state, pre_pull_images=report)},
        )
