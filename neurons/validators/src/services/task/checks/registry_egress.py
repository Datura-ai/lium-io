from __future__ import annotations

from ..messages import RegistryEgressMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

# The registry's own API root. It answers without pulling anything, so the probe costs the
# host no image layers and counts against no Docker Hub pull limit (measured: 200 pings left
# ratelimit-remaining untouched, a real pull straight after moved it).
REGISTRY_PING_URL = "https://registry-1.docker.io/v2/"
# Unauthenticated /v2/ answers 401; a client holding a token gets 200. Either proves the TCP
# and TLS path a `docker pull` needs. Everything else — 000 from curl when nothing answered,
# or a 5xx — is a host no rental of a non-cached image can start on.
REACHABLE_STATUS_CODES = frozenset({"200", "401"})
# curl applies --max-time per attempt, so with the retry below the worst case is about 11s.
PROBE_TIMEOUT_SECONDS = 5
# `--retry 1` so a single dropped packet does not read as a broken host.
PROBE_COMMAND = (
    f"curl -sS -o /dev/null -w '%{{http_code}}' --retry 1 --retry-delay 1 "
    f"--max-time {PROBE_TIMEOUT_SECONDS} {REGISTRY_PING_URL}"
)


class RegistryEgressCheck:
    """Fail a host that cannot reach Docker Hub (DAH-2835).

    Nothing else in the pipeline asks. The sysbox probe bundles `hello-world` into its image
    (DAH-1959), `docker run` is issued without `--pull`, and the cached-template check reads a
    local `docker image inspect`. So a host whose egress to the registry died keeps passing
    every check, keeps its score, stays listed — and fails every rental of an image it has not
    already cached, at `failure_step=docker_pull`.

    A failed probe zeroes the score, like every other fatal check. What stops a network blip
    from costing the provider a rental is DAH-2748 on the backend side: a node is hidden from
    browse and refused for new rentals only after `VALIDATION_FAILURE_STREAK_TO_BLOCK_RENTALS`
    failed cycles in a row, and one good cycle resets that streak. Billing of a renter already
    on the node is never touched — a rented executor short-circuits before this check.

    Fails open on every uncertainty: an SSH error, unreadable curl output, or a validator that
    lost Docker Hub itself all pass the check rather than punish a host for our own outage.
    """

    check_id = "executor.validate.registry_egress"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        # The validator fetches this snapshot from Docker Hub at cycle start. Empty means the
        # VALIDATOR could not reach the registry either, so every host would look unreachable
        # and a Docker Hub outage would zero the whole fleet in one cycle.
        if not ctx.config.default_docker_image_digests:
            return self._skipped(ctx, "validator has no Docker Hub digest snapshot this cycle")

        try:
            probe = await ctx.ssh.run(PROBE_COMMAND, check=False)
        except Exception as exc:
            return self._skipped(ctx, "ssh failed", error=str(exc)[:200])

        http_status: str = (getattr(probe, "stdout", None) or "").strip()
        if not http_status.isdigit():
            return self._skipped(ctx, "curl printed no status code", output=http_status[:200])

        reachable: bool = http_status in REACHABLE_STATUS_CODES
        event = render_message(
            Msg.REACHABLE if reachable else Msg.UNREACHABLE,
            ctx=ctx,
            check_id=self.check_id,
            what={"http_status": http_status, "url": REGISTRY_PING_URL},
        )
        return CheckResult(passed=reachable, event=event)

    def _skipped(self, ctx: Context, reason: str, **what: str) -> CheckResult:
        event = render_message(
            Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"reason": reason, **what}
        )
        return CheckResult(passed=True, event=event)
