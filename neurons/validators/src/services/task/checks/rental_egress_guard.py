"""Running rental containers get the link-local egress drop that custom builds already have.

Builds drop egress to blocked ranges through host DOCKER-USER rules set by a
``--network=host --cap-add=NET_ADMIN`` helper container (``docker_service._egress_filter_script``).
This check applies the same kind of rule for the rental network: traffic from the
``lium-rentals`` subnet to link-local 169.254/16 is dropped, except DNS (port 53), because a
rental container's embedded resolver forwards from the container's network namespace and
some hosts use a link-local resolver.

- Never before a scored step: on a rented node it runs just before TenantEnforcementCheck, which
  halts the rented node, so only the halt waits for it; on an idle node it runs after ScoreCheck.
  A first helper-image pull can take up to ``HELPER_TIMEOUT_SECONDS``, so it is skipped (and
  retried next cycle) when less than that is left before the executor task is cancelled.
- Periodic, not on the create path: host iptables are gone after a reboot while rental
  containers come back (``restart=unless-stopped``), and a rental create gains no extra step.
- Idempotent: every rule is ``-C || -I``; nothing accumulates across cycles. Rules are scoped
  to the network's subnet, not to a pod, so nothing is left to remove when a pod ends.
- Fail-open: a host without a DOCKER-USER chain or without the helper image logs a warning and
  validation proceeds; a hardening rule must not take a provider's node out of rotation.
- DOCKER-USER sees forwarded traffic only: Docker's embedded DNS, published ports,
  container-to-container traffic on the subnet and the validator's inbound SSH are unaffected.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import shlex
import time

from services.rental_docker_sdk import RENTAL_NETWORK_NAME

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import RentalEgressGuardMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node
from .task_time import task_seconds_left

logger = logging.getLogger(__name__)

APPLY_INTERVAL_SECONDS = 10 * 60
HELPER_TIMEOUT_SECONDS = 90
# the inspect read and FinalizeCheck after the helper
BUDGET_MARGIN_SECONDS = 30
# RFC1918 is not blocked: multi-node clusters and provider LANs use private addresses.
# The IPv4 link-local /16 that holds the cloud metadata address.
LINK_LOCAL_CIDR = str(ipaddress.ip_network("169.254.169.254/16", strict=False))


def rental_subnets(inspect_stdout: str) -> list[str]:
    """IPv4 subnets of the rental network from ``docker network inspect``; anything else is dropped."""
    subnets: list[str] = []
    for raw in (inspect_stdout or "").split():
        try:
            net = ipaddress.ip_network(raw, strict=False)
        except ValueError:
            continue
        # IPv6 (fe80::/10) would need ip6tables; out of scope.
        if net.version == 4 and str(net) not in subnets:
            subnets.append(str(net))
    return subnets


def link_local_drop_script(subnets: list[str]) -> str:
    """Host DOCKER-USER rules: drop rental-subnet egress to link-local, except DNS.

    Two DROP rules (tcp and udp, both ``! --dport 53``) instead of a DROP plus an ACCEPT, so the
    result does not depend on rule order. ``subnets`` come from ``rental_subnets`` and are shell-safe.
    """
    lines = [
        "IPT=iptables-nft",
        "$IPT -L DOCKER-USER -n >/dev/null 2>&1 || IPT=iptables-legacy",
        '$IPT -L DOCKER-USER -n >/dev/null 2>&1 || { echo "DOCKER-USER chain not found" >&2; exit 3; }',
    ]
    for subnet in subnets:
        for proto in ("tcp", "udp"):
            rule = f"DOCKER-USER -s {subnet} -d {LINK_LOCAL_CIDR} -p {proto} ! --dport 53 -j DROP"
            lines.append(f"$IPT -C {rule} 2>/dev/null || $IPT -I {rule} || exit 4")
    return "; ".join(lines)


class RentalEgressGuardCheck:
    """Keeps the rental network's link-local egress drop in place, at most once per interval per executor."""

    check_id = "executor.network.rental_egress_guard"
    fatal = False

    def __init__(
        self,
        interval_seconds: int = APPLY_INTERVAL_SECONDS,
        *,
        rented_only: bool = False,
        last_run: dict[str, float] | None = None,
    ) -> None:
        self._interval = interval_seconds
        # rented_only: the instance before TenantEnforcementCheck; the other runs after ScoreCheck, idle nodes only
        self._rented_only = rented_only
        # shared by both instances so a node is guarded once per interval whichever one it reaches
        self._last_run: dict[str, float] = {} if last_run is None else last_run

    def _skip(self, ctx: Context, reason: str) -> CheckResult:
        return CheckResult(
            passed=True,
            event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"reason": reason}),
        )

    async def run(self, ctx: Context) -> CheckResult:
        executor_uuid = ctx.executor.uuid
        rental = rental_of_this_node(ctx)
        if self._rented_only != bool(rental and rental.pods):
            return self._skip(ctx, "guarded at the other position")
        if task_seconds_left(ctx) < HELPER_TIMEOUT_SECONDS + BUDGET_MARGIN_SECONDS:
            return self._skip(ctx, "not enough task time left")
        now = time.monotonic()
        last = self._last_run.get(executor_uuid)
        if last is not None and now - last < self._interval:
            return self._skip(ctx, "within cadence window")
        # Retry on the next interval whatever happens below, never every cycle.
        self._last_run[executor_uuid] = now

        try:
            # Every remote call shares one deadline that ends BUDGET_MARGIN_SECONDS before the task is
            # cancelled, so a stalled SSH call can never cost the cycle its validation result.
            subnets, error = await asyncio.wait_for(
                self._apply(ctx), timeout=task_seconds_left(ctx) - BUDGET_MARGIN_SECONDS
            )
        except TimeoutError:
            # Not a run: the next cycle tries again.
            if last is None:
                self._last_run.pop(executor_uuid, None)
            else:
                self._last_run[executor_uuid] = last
            return self._skip(ctx, "out of task time")
        except Exception as exc:  # noqa: BLE001 — fail-open, see module docstring
            subnets, error = [], str(exc)
        if subnets is None:
            # No rental network yet: nothing runs on it; the next interval checks again.
            return self._skip(ctx, "no rental network")

        if error is not None:
            logger.warning(
                _m(
                    "Rental egress guard: link-local drop not applied",
                    extra=get_extra_info({"executor_uuid": executor_uuid, "error": error}),
                )
            )
            return CheckResult(
                passed=True,
                event=render_message(Msg.NOT_APPLIED, ctx=ctx, check_id=self.check_id, what={"error": error}),
            )
        return CheckResult(
            passed=True,
            event=render_message(Msg.APPLIED, ctx=ctx, check_id=self.check_id, what={"subnets": subnets}),
        )

    async def _apply(self, ctx: Context) -> tuple[list[str] | None, str | None]:
        """(subnets, error); subnets None when there is no rental network."""
        inspect = await ctx.ssh.run(
            f"/usr/bin/docker network inspect {RENTAL_NETWORK_NAME} "
            "-f '{{range .IPAM.Config}}{{.Subnet}} {{end}}' 2>/dev/null",
            check=False,
        )
        subnets = rental_subnets(inspect.stdout if inspect.exit_status == 0 else "")
        if not subnets:
            return None, None
        helper = (
            f"/usr/bin/docker run --rm --network=host --cap-add=NET_ADMIN --entrypoint /bin/sh "
            f"{shlex.quote(settings.CUSTOM_DOCKERFILE_DIND_IMAGE)} -c {shlex.quote(link_local_drop_script(subnets))}"
        )
        result = await ctx.ssh.run(f"timeout -k 5 {HELPER_TIMEOUT_SECONDS} {helper}", check=False)
        error = None if result.exit_status == 0 else (result.stderr or "").strip()[-300:] or f"exit {result.exit_status}"
        return subnets, error
