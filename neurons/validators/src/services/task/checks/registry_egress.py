from __future__ import annotations

import time
from dataclasses import replace

from pydantic import BaseModel, ValidationError

from ..messages import MessageTemplate
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
# curl applies --max-time per attempt, so the retry below puts the real worst case at
# 2 * PROBE_TIMEOUT_SECONDS + --retry-delay ~= 11s — still small against JOB_TIME_OUT.
PROBE_TIMEOUT_SECONDS = 5
# `--retry 1` so a single dropped packet does not read as a broken host.
PROBE_COMMAND = (
    f"curl -sS -o /dev/null -w '%{{http_code}}' --retry 1 --retry-delay 1 "
    f"--max-time {PROBE_TIMEOUT_SECONDS} {REGISTRY_PING_URL}"
)
COUNTER_KEY_PREFIX = "registry_unreachable:"
# RedisService.set takes no TTL, so the record expires by its own timestamp: a run of bad
# cycles that stopped being observed (executor gone, validator down) must not count forever.
COUNTER_WINDOW_SECONDS = 4 * 3600


class RegistryUnreachableRecord(BaseModel):
    """Cycles in a row this executor failed the registry probe, and when the last one was."""

    cycles: int
    last_seen_at: float  # unix seconds; ages the record out against COUNTER_WINDOW_SECONDS


class RegistryEgressCheck:
    """Count the cycles in a row this host could not reach Docker Hub (DAH-2835).

    Nothing else in the pipeline asks. The sysbox probe bundles `hello-world` into its image
    (DAH-1959), `docker run` is issued without `--pull`, and the cached-template check reads a
    local `docker image inspect`. So a host whose egress to the registry died keeps passing
    every check, keeps its score, stays listed — and fails every rental of an image it has not
    already cached, at `failure_step=docker_pull`.

    A single bad cycle proves nothing but a network blip, so the count — not the verdict —
    rides `executor.specs` as `registry_unreachable_cycles`, and the unrented-incentive gate in
    `incentive/rental_price.py` withholds the idle incentive only after several consecutive
    cycles. One reachable cycle deletes the counter. Score is never touched: the provider is
    not cheating, its network is broken.

    Fails open on every uncertainty — SSH, curl, Redis — and publishes nothing when it cannot
    answer: an absent key means "not measured", not "unreachable".
    """

    check_id = "executor.validate.registry_egress"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        # The validator fetches this snapshot from Docker Hub at cycle start. Empty means the
        # VALIDATOR could not reach the registry either, so every host would look unreachable
        # and a Docker Hub outage would penalise the whole fleet in one cycle.
        if not ctx.config.default_docker_image_digests:
            return self._publish_no_measurement(
                ctx, "validator has no Docker Hub digest snapshot this cycle"
            )

        try:
            probe = await ctx.ssh.run(PROBE_COMMAND, check=False)
        except Exception as exc:
            return self._publish_no_measurement(ctx, "ssh failed", error=str(exc)[:200])

        http_status: str = (getattr(probe, "stdout", None) or "").strip()
        if not http_status.isdigit():
            return self._publish_no_measurement(
                ctx, "curl printed no status code", output=http_status[:200]
            )

        key: str = f"{COUNTER_KEY_PREFIX}{ctx.executor.uuid}"
        if http_status in REACHABLE_STATUS_CODES:
            try:
                await ctx.services.redis.delete(key)
            except Exception as exc:
                return self._publish_no_measurement(ctx, "redis unavailable", error=str(exc)[:200])
            return self._publish_cycle_count(ctx, Msg.REACHABLE, http_status, cycles=0)

        try:
            unreachable_cycles: int = await self._next_cycle_count(ctx, key)
            record = RegistryUnreachableRecord(cycles=unreachable_cycles, last_seen_at=time.time())
            await ctx.services.redis.set(key, record.model_dump_json())
        except Exception as exc:
            return self._publish_no_measurement(ctx, "redis unavailable", error=str(exc)[:200])
        return self._publish_cycle_count(
            ctx, Msg.UNREACHABLE, http_status, cycles=unreachable_cycles
        )

    async def _next_cycle_count(self, ctx: Context, key: str) -> int:
        raw: bytes | str | None = await ctx.services.redis.get(key)
        if not raw:
            return 1
        try:
            record = RegistryUnreachableRecord.model_validate_json(raw)
        except ValidationError:
            return 1
        if time.time() - record.last_seen_at > COUNTER_WINDOW_SECONDS:
            return 1
        return record.cycles + 1

    def _publish_cycle_count(
        self, ctx: Context, template: MessageTemplate, http_status: str, *, cycles: int
    ) -> CheckResult:
        event = render_message(
            template,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "http_status": http_status,
                "url": REGISTRY_PING_URL,
                "unreachable_cycles": cycles,
            },
        )
        return CheckResult(
            passed=True,
            event=event,
            updates={"state": replace(ctx.state, registry_unreachable_cycles=cycles)},
        )

    def _publish_no_measurement(self, ctx: Context, reason: str, **what: str) -> CheckResult:
        event = render_message(
            Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"reason": reason, **what}
        )
        return CheckResult(passed=True, event=event)
