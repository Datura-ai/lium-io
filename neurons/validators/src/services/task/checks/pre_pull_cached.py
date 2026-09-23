from __future__ import annotations

import shlex
from dataclasses import replace

from core.config import settings

from ..messages import PrePullCachedMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

_DOCKER = "/usr/bin/docker"


class PrePullCachedCheck:
    """Report which of the backend's pre-pull images an idle executor holds (DAH-2977).

    The backend serves each node its default image plus the top-N official templates marked
    ``pre_pull: true`` and digest-pinned (``/executors/default-docker-image?include_pre_pull=true``,
    lium-platform#577). The executor warms those while idle (lium-io#1283, on by default with
    lium-io#1412). This check asks the same endpoint the executor asks and probes
    ``docker image inspect <repo>@<digest>`` for every pre-pull entry: exit 0 means the node
    holds exactly the content a rental of that template would otherwise pull.

    Advisory: always ``passed=True``, never fatal, never changes score. The count and the
    missing refs go into ``executor.specs["pre_pull_images"]`` through pipeline state, so the
    fleet's coverage per node and per template is readable from the backend and Loki. The
    node's default image stays with ``CachedTemplateVerificationCheck``.

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
        report = {"expected": len(entries), "cached": len(cached), "missing": missing}

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
