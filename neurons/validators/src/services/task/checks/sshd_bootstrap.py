from __future__ import annotations

import logging
import shlex

from core.config import settings
from core.docker_utils import DockerCommand
from core.utils import _m, get_extra_info
from services.redis_service import SSHD_BOOTSTRAP_FAILED_PREFIX, SSHD_BOOTSTRAP_FAILED_TTL_SECONDS

from ...const import DEFAULT_JOB_OWNER_LIUM, FILLER_CONTAINER_PREFIX
from ..messages import SshdBootstrapMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# a filler younger than this may still be bootstrapping sshd, so it is not judged
RUNNING_FILLER_GRACE_SECONDS = 15 * 60
# the probe's exit status for "running past the grace, sshd not listening"; anything else but 0 is inconclusive
_NOT_LISTENING_EXIT = 10
# sshd_bootstrap.sh is_sshd_listening: a LISTEN row on :22, unreadable /proc files count as a pass
_LISTENING_TEST = (
    "command -v grep >/dev/null || exit 0; c=0; for f in /proc/net/tcp /proc/net/tcp6; do "
    '[ -r "$f" ] || continue; c=1; grep -qE \':0016 0+:0+ 0A \' "$f" && exit 0; done; '
    f"[ $c = 1 ] && exit {_NOT_LISTENING_EXIT}; exit 0"
)


def running_filler_probe(name: str) -> str:
    """One host command: exit 10 only if the filler has run past the grace and sshd is not listening in it."""
    quoted = shlex.quote(name)
    return (
        f"s=$(/usr/bin/docker inspect -f '{{{{.State.Running}}}} {{{{.State.StartedAt}}}}' {quoted}) || exit 3; "
        'set -- $s; [ "$1" = true ] || exit 3; t=$(date -d "$2" +%s) || exit 3; '
        f"[ $(($(date +%s) - t)) -ge {RUNNING_FILLER_GRACE_SECONDS} ] || exit 3; "
        f"{DockerCommand.exec_command(name, _LISTENING_TEST)}"
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

    An unstamped node whose Lium filler predates any new create is probed: a filler running past
    RUNNING_FILLER_GRACE_SECONDS without sshd listening stamps the node too (any probe error passes).
    """

    check_id = "executor.validate.sshd_bootstrap"
    fatal = True

    def __init__(self, remove_fillers: bool = True) -> None:
        self.remove_fillers = remove_fillers

    async def run(self, ctx: Context) -> CheckResult:
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0
        try:
            failure_stamp = (
                None
                if is_rented
                else await ctx.services.redis.get(
                    f"{SSHD_BOOTSTRAP_FAILED_PREFIX}:{ctx.executor.uuid}"
                )
            )
        except Exception:
            # an unreadable stamp is no evidence against the node
            logger.warning(
                _m(
                    "sshd bootstrap stamp could not be read",
                    extra=get_extra_info(ctx.default_extra),
                ),
                exc_info=True,
            )
            failure_stamp = None
        if not failure_stamp and not is_rented and await self._running_filler_fails(ctx):
            failure_stamp = "1"
        what = {"sshd_bootstrap_failed": bool(failure_stamp), "is_rented": is_rented}
        if not failure_stamp:
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
            what["removed_fillers"] = await self._remove_lium_fillers(
                ctx, rented_data.get_filler_containers(ctx.executor.uuid)
            )
        return CheckResult(
            passed=False,
            event=render_message(Msg.FAILED, ctx=ctx, check_id=self.check_id, what=what),
        )

    async def _running_filler_fails(self, ctx: Context) -> bool:
        """Probe each listed Lium filler; stamp the node (not on a dry run) if one is past the grace without sshd."""
        rented_data = ctx.state.rented_data
        if (
            not rented_data
            or rented_data.get_default_job_owner(ctx.executor.uuid) != DEFAULT_JOB_OWNER_LIUM
        ):
            return False
        for name in rented_data.get_filler_containers(ctx.executor.uuid):
            try:
                result = await ctx.ssh.run(running_filler_probe(name), timeout=30)
            except Exception:
                continue  # our own probe failing is no evidence against the node
            if result.exit_status != _NOT_LISTENING_EXIT:
                continue
            if self.remove_fillers and not settings.DRY_RUN:
                try:
                    await ctx.services.redis.set(
                        f"{SSHD_BOOTSTRAP_FAILED_PREFIX}:{ctx.executor.uuid}",
                        "1",
                        ex=SSHD_BOOTSTRAP_FAILED_TTL_SECONDS,
                    )
                except Exception:
                    logger.warning(
                        _m(
                            "sshd bootstrap stamp could not be written",
                            extra=get_extra_info(ctx.default_extra),
                        ),
                        exc_info=True,
                    )
            return True
        return False

    async def _remove_lium_fillers(self, ctx: Context, fillers: list[str]) -> list[str]:
        """`docker rm -fv` each filler, then report it missing so the backend closes its run; never raises."""
        removed: list[str] = []
        for name in fillers:
            try:
                result = await ctx.ssh.run(
                    DockerCommand.remove_with_volumes(shlex.quote(name)), timeout=60
                )
                if result.exit_status != 0:
                    continue
                removed.append(name)
                await ctx.services.backend.get_filler_run_active(
                    name.removeprefix(FILLER_CONTAINER_PREFIX), container_missing=True
                )
            except Exception:
                logger.warning(
                    _m(
                        "Filler on a node failing the sshd check could not be removed",
                        extra=get_extra_info({**ctx.default_extra, "container_name": name}),
                    ),
                    exc_info=True,
                )
        return removed
