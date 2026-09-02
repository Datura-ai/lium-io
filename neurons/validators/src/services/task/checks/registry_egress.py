from __future__ import annotations

from dataclasses import replace

from ..messages import RegistryEgressMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

# The registry's own API root. It answers without pulling anything, so the probe costs the
# host no image layers and counts against no Docker Hub pull limit.
REGISTRY_PING_URL = "https://registry-1.docker.io/v2/"
# Unauthenticated /v2/ answers 401; a client holding a token gets 200. Either proves the TCP
# and TLS path a `docker pull` needs. Everything else — 000 from curl when nothing answered,
# or a 5xx — is a host no rental of a non-cached image can start on.
REACHABLE_STATUS_CODES = frozenset({"200", "401"})
# One retry covers a single dropped packet; the whole probe still ends well inside the
# per-executor budget the pipeline gives every other SSH step.
PROBE_TIMEOUT_SECONDS = 5
PROBE_COMMAND = (
    f"curl -sS -o /dev/null -w '%{{http_code}}' --retry 1 --retry-delay 1 "
    f"--max-time {PROBE_TIMEOUT_SECONDS} {REGISTRY_PING_URL}"
)


class RegistryEgressCheck:
    """Ask whether this host can still reach Docker Hub (DAH-2835).

    Nothing else in the pipeline asks. The sysbox probe bundles `hello-world` into its image
    (DAH-1959), `docker run` is issued without `--pull`, and the cached-template check reads a
    local `docker image inspect`. So a host whose egress to the registry died keeps passing
    every check, keeps its score, stays listed — and fails every rental of an image it has not
    already cached, at `failure_step=docker_pull`.

    Advisory by construction: the verdict rides `executor.specs` as `registry_reachable`, and
    the backend drops a `false` host from the available listing (and therefore from its
    fabric). Score is never touched — the host is not cheating, its provider's network is
    broken, and a scored penalty for a transient outage is the wrong lever.

    Fails open on every uncertainty, and publishes nothing when it cannot answer: an absent
    key leaves the previous cycle's verdict standing rather than inventing a new one.
    """

    check_id = "executor.validate.registry_egress"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        # The validator fetches this snapshot from Docker Hub at cycle start. Empty means the
        # VALIDATOR could not reach the registry either, so every host would look unreachable
        # and a Docker Hub outage would hide the whole fleet in one cycle.
        if not ctx.config.default_docker_image_digests:
            return self._skipped(ctx, "validator has no Docker Hub digest snapshot this cycle")

        try:
            probe = await ctx.ssh.run(PROBE_COMMAND, check=False)
        except Exception as exc:
            return self._skipped(ctx, "ssh failed", error=str(exc)[:200])

        http_status = (getattr(probe, "stdout", None) or "").strip()
        if not http_status.isdigit():
            return self._skipped(ctx, "curl printed no status code", output=http_status[:200])

        reachable = http_status in REACHABLE_STATUS_CODES
        event = render_message(
            Msg.REACHABLE if reachable else Msg.UNREACHABLE,
            ctx=ctx,
            check_id=self.check_id,
            what={"http_status": http_status, "url": REGISTRY_PING_URL},
        )
        return CheckResult(
            passed=True,
            event=event,
            updates={"state": replace(ctx.state, registry_reachable=reachable)},
        )

    def _skipped(self, ctx: Context, reason: str, **what) -> CheckResult:
        event = render_message(
            Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"reason": reason, **what}
        )
        return CheckResult(passed=True, event=event)
