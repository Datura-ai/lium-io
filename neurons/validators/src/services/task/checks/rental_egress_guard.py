"""Running rental containers get the link-local egress drop that custom builds already have.

Builds drop egress to blocked ranges through host DOCKER-USER rules set by a
``--network=host --cap-add=NET_ADMIN`` helper container (``docker_service._egress_filter_script``).
This check applies the same kind of rule for the rental network: traffic from the
``lium-rentals`` subnet to 169.254.0.0/16 is dropped, except DNS (port 53), because a
rental container's embedded resolver forwards from the container's network namespace and
some hosts use a link-local resolver.

- Periodic, not on the create path: host iptables are gone after a reboot while rental
  containers come back (``restart=unless-stopped``), and a rental create gains no extra step.
- Idempotent: every rule is ``-C || -I``; nothing accumulates across cycles. Rules are scoped
  to the network's subnet, not to a pod, so nothing is left to remove when a pod ends.
- Fail-open: a host without a DOCKER-USER chain or without the helper image logs a warning and
  validation proceeds; a hardening rule must not take a provider's node out of rotation.
- DOCKER-USER sees forwarded traffic only: Docker DNS (127.0.0.11), published ports,
  container-to-container traffic on the subnet and the validator's inbound SSH are unaffected.
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)

APPLY_INTERVAL_SECONDS = 10 * 60
HELPER_TIMEOUT_SECONDS = 90
# RFC1918 is not blocked: multi-node clusters and provider LANs use private addresses.
LINK_LOCAL_CIDR = "169.254.0.0/16"


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

    def __init__(self, interval_seconds: int = APPLY_INTERVAL_SECONDS) -> None:
        self._interval = interval_seconds
        self._last_run: dict[str, float] = {}

    async def run(self, ctx: Context) -> CheckResult:
        executor_uuid = ctx.executor.uuid
        now = time.monotonic()
        last = self._last_run.get(executor_uuid)
        if last is not None and now - last < self._interval:
            return CheckResult(
                passed=True,
                event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"reason": "within cadence window"}),
            )
        # Retry on the next interval whatever happens below, never every cycle.
        self._last_run[executor_uuid] = now

        try:
            inspect = await ctx.ssh.run(
                f"/usr/bin/docker network inspect {RENTAL_NETWORK_NAME} "
                "-f '{{range .IPAM.Config}}{{.Subnet}} {{end}}' 2>/dev/null",
                check=False,
            )
            subnets = rental_subnets(inspect.stdout if inspect.exit_status == 0 else "")
            if not subnets:
                # No rental network yet: nothing runs on it; the next interval checks again.
                return CheckResult(
                    passed=True,
                    event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"reason": "no rental network"}),
                )
            helper = (
                f"/usr/bin/docker run --rm --network=host --cap-add=NET_ADMIN --entrypoint /bin/sh "
                f"{shlex.quote(settings.CUSTOM_DOCKERFILE_DIND_IMAGE)} -c {shlex.quote(link_local_drop_script(subnets))}"
            )
            result = await ctx.ssh.run(f"timeout -k 5 {HELPER_TIMEOUT_SECONDS} {helper}", check=False)
            error = None if result.exit_status == 0 else (result.stderr or "").strip()[-300:] or f"exit {result.exit_status}"
        except Exception as exc:  # noqa: BLE001 — fail-open, see module docstring
            subnets, error = [], str(exc)

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
