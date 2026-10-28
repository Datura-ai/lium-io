from __future__ import annotations

import base64
import logging
import shlex
import time
from pathlib import Path

from core.config import settings
from core.docker_utils import DockerCommand
from core.utils import _m, get_extra_info
from services.redis_service import (
    IN_CONTAINER_SSH_BOOTSTRAP_PATH,
    SSHD_BOOTSTRAP_FAILED_PREFIX,
    SSHD_BOOTSTRAP_FAILED_TTL_SECONDS,
    SSHD_VERIFY_FAILED_MARKER,
)

from ...const import DEFAULT_JOB_OWNER_LIUM, FILLER_CONTAINER_PREFIX
from ..messages import SshdBootstrapMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# a filler younger than this may still be bootstrapping sshd, so it is not judged
RUNNING_FILLER_GRACE_SECONDS = 15 * 60
_BOOTSTRAP_SCRIPT = Path(__file__).resolve().parents[2] / "assets" / "sshd_bootstrap.sh"
_PROBE_TIMEOUT_SECONDS = 300
# a probe gets no less than this; with less task time left it is skipped
_MIN_PROBE_SECONDS = 60
# kept free before JOB_TIME_OUT, as in rental_probe
_TASK_TIMEOUT_MARGIN_SECONDS = 120


def running_filler_sshd_bootstrap_command(name: str) -> str:
    """One host command: re-run the create's sshd bootstrap in a filler that has run past the grace.

    A restarted filler loses the sshd the create started with `docker exec`; the re-run starts it again
    (as the create does), so only its own "sshd verification failed" marks the node. Otherwise it exits 3.
    """
    script = base64.b64encode(_BOOTSTRAP_SCRIPT.read_bytes()).decode()
    quoted = shlex.quote(name)
    return (
        f"s=$(/usr/bin/docker inspect -f '{{{{.State.Running}}}} {{{{.State.StartedAt}}}}' {quoted}) || exit 3; "
        'set -- $s; [ "$1" = true ] || exit 3; t=$(date -d "$2" +%s) || exit 3; '
        f"[ $(($(date +%s) - t)) -ge {RUNNING_FILLER_GRACE_SECONDS} ] || exit 3; "
        f"printf %s {script} | base64 -d | "
        + DockerCommand.exec_command(
            name, f"cat > {IN_CONTAINER_SSH_BOOTSTRAP_PATH} && sh {IN_CONTAINER_SSH_BOOTSTRAP_PATH} --grace 0"
        )
    )


class SshdBootstrapCheck:
    """Fail an unrented node on which a filler create could not bring sshd up.

    The connector stamps the executor when the sshd bootstrap in Lium's own filler image ends with sshd
    installed but not running or listening, and clears it when a later filler create verifies sshd; the
    stamp lapses after SSHD_BOOTSTRAP_FAILED_TTL_SECONDS. A rented node is left alone: its stamp predates
    the rental and failing it would not help the renter.

    A running Lium filler does not exempt the node: the failed filler create is the only signal, so the
    nodes this check exists for always have one. Instead, while the node is failed, the validator removes
    the Lium fillers the backend lists there and reports each as missing (the backend closes such a run,
    as for any filler gone from its host), so Lium never uses the GPUs of a node it scores 0.

    An unstamped node whose Lium filler predates any new create is probed: the sshd bootstrap is re-run in
    a filler running past RUNNING_FILLER_GRACE_SECONDS, and only its "sshd verification failed" stamps the
    node too, as on create (an apt failure or any probe error passes). A dry run does not probe.
    """

    check_id = "executor.validate.sshd_bootstrap"
    fatal = True

    def __init__(self, remove_fillers: bool = True) -> None:
        self.remove_fillers = remove_fillers

    async def run(self, ctx: Context) -> CheckResult:
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0
        sshd_bootstrap_failed = False if is_rented else await self._sshd_bootstrap_failed(ctx)
        what = {"sshd_bootstrap_failed": sshd_bootstrap_failed, "is_rented": is_rented}
        if not sshd_bootstrap_failed:
            return CheckResult(
                passed=True,
                event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what),
            )
        if not settings.SSHD_BOOTSTRAP_ENFORCEMENT_ENABLED:
            return CheckResult(
                passed=True,
                event=render_message(
                    Msg.FAILED_OBSERVED, ctx=ctx, check_id=self.check_id, what=what
                ),
            )
        if (
            self.remove_fillers
            and rented_data
            and rented_data.get_default_job_owner(ctx.executor.uuid) == DEFAULT_JOB_OWNER_LIUM
        ):
            what["removed_fillers"] = await remove_lium_fillers(
                ctx, rented_data.get_filler_containers(ctx.executor.uuid)
            )
        return CheckResult(
            passed=False,
            event=render_message(Msg.FAILED, ctx=ctx, check_id=self.check_id, what=what),
        )

    async def _sshd_bootstrap_failed(self, ctx: Context) -> bool:
        """The verdict for an unrented node: its stamp, else a probe of its running filler (stamped on failure)."""
        key = f"{SSHD_BOOTSTRAP_FAILED_PREFIX}:{ctx.executor.uuid}"
        try:
            if await ctx.services.redis.get(key):
                return True
        except Exception:
            # an unreadable stamp is no evidence against the node
            logger.warning(
                _m("sshd bootstrap stamp could not be read", extra=get_extra_info(ctx.default_extra)),
                exc_info=True,
            )
        # the dry-run pipeline must not start sshd in a filler
        if not self.remove_fillers or not await self._running_filler_fails(ctx):
            return False
        if not settings.DRY_RUN:
            try:
                await ctx.services.redis.set(key, "1", ex=SSHD_BOOTSTRAP_FAILED_TTL_SECONDS)
            except Exception:
                logger.warning(
                    _m("sshd bootstrap stamp could not be written", extra=get_extra_info(ctx.default_extra)),
                    exc_info=True,
                )
        return True

    async def _running_filler_fails(self, ctx: Context) -> bool:
        """Re-run the sshd bootstrap in each listed Lium filler; True if one ends with sshd not serving."""
        rented_data = ctx.state.rented_data
        if (
            not rented_data
            or rented_data.get_default_job_owner(ctx.executor.uuid) != DEFAULT_JOB_OWNER_LIUM
        ):
            return False
        for name in rented_data.get_filler_containers(ctx.executor.uuid):
            task_seconds_left = (
                settings.JOB_TIME_OUT
                - _TASK_TIMEOUT_MARGIN_SECONDS
                - (time.monotonic() - ctx.started_at_monotonic)
            )
            if task_seconds_left < _MIN_PROBE_SECONDS:
                return False
            try:
                result = await ctx.ssh.run(
                    running_filler_sshd_bootstrap_command(name),
                    timeout=min(_PROBE_TIMEOUT_SECONDS, int(task_seconds_left)),
                )
            except Exception:
                continue  # our own probe failing is no evidence against the node
            if SSHD_VERIFY_FAILED_MARKER in f"{result.stdout}{result.stderr}":
                return True
        return False


async def remove_lium_fillers(ctx: Context, fillers: list[str]) -> list[str]:
    """`docker rm -fv` each filler, then report it missing so the backend closes its run; never raises."""
    removed: list[str] = []
    for name in fillers:
        try:
            result = await ctx.ssh.run(DockerCommand.remove_with_volumes(shlex.quote(name)), timeout=60)
            if result.exit_status != 0:
                continue
            removed.append(name)
            await ctx.services.backend.get_filler_run_active(
                name.removeprefix(FILLER_CONTAINER_PREFIX), container_missing=True
            )
        except Exception:
            logger.warning(
                _m(
                    "Filler could not be removed",
                    extra=get_extra_info({**ctx.default_extra, "container_name": name}),
                ),
                exc_info=True,
            )
    return removed
