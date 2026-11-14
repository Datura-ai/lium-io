"""Running rental containers get the link-local egress drop that custom builds already have.

Builds drop egress to blocked ranges through host DOCKER-USER rules set by a
``--network=host --cap-add=NET_ADMIN`` helper container (``docker_service._egress_filter_script``).
This check applies the same kind of rule for rentals: traffic entering from the ``lium-rentals``
bridge, and from a running legacy rental still on another network (the default bridge; a running
pod is never moved), to link-local 169.254/16 is dropped, except DNS (port 53), because a rental
container's embedded resolver forwards from the container's network namespace and some hosts use
a link-local resolver.

- Never before a scored step: on a rented node TenantEnforcementCheck runs it in its rented halt
  branch, so only the halt waits for it; on an idle node it runs after ScoreCheck. A first
  helper-image pull can take up to ``HELPER_TIMEOUT_SECONDS``, so it is skipped (and retried next
  cycle) when less than that is left before the executor task is cancelled.
- Periodic, not on the create path: host iptables are gone after a reboot while rental
  containers come back (``restart=unless-stopped``), and a rental create gains no extra step.
- Reconciled, not accumulated: the rules live in our own chain, rebuilt each run from what the
  host shows now and swapped in, so a recreated rental network or an ended legacy pod leaves no
  rule behind. The rental bridge is matched by interface, not subnet: a subnet Docker later
  gives an unrelated network is never caught.
- Fail-open: a host without a DOCKER-USER chain or without the helper image logs a warning and
  validation proceeds; a hardening rule must not take a provider's node out of rotation.
- DOCKER-USER sees forwarded traffic only: Docker's embedded DNS, published ports,
  container-to-container traffic on the subnet and the validator's inbound SSH are unaffected.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
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


# Two chains used in turn: the new set is built in the idle one and swapped in, so there is no unguarded gap.
CHAINS = ("LIUM-RENTAL-EGRESS-A", "LIUM-RENTAL-EGRESS-B")
BRIDGE_NAME_OPTION = "com.docker.network.bridge.name"
_IFNAME = re.compile(r"[A-Za-z0-9_.-]{1,15}")
_NETWORK_ID = re.compile(r"[0-9a-f]{12,64}")


def rental_bridge(inspect_stdout: str) -> str | None:
    """Host interface of the rental network from ``docker network inspect`` ("<id> <bridge-name option>")."""
    parts = (inspect_stdout or "").split()
    if parts and _NETWORK_ID.fullmatch(parts[0]):
        if len(parts) > 1 and _IFNAME.fullmatch(parts[1]):
            return parts[1]
        return f"br-{parts[0][:12]}"
    return None


def legacy_sources(inspect_stdout: str) -> list[str]:
    """IPv4 /32s of running rental containers on a network other than the rental one; anything else is dropped."""
    sources: list[str] = []
    for raw in (inspect_stdout or "").split():
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            continue
        # IPv6 would need ip6tables; out of scope.
        if addr.version == 4 and f"{addr}/32" not in sources:
            sources.append(f"{addr}/32")
    return sources


def link_local_drop_script(bridge: str | None, sources: list[str]) -> str:
    """Host DOCKER-USER rules: drop rental egress to link-local, except DNS, reconciled to exactly this set.

    Two DROP rules (tcp and udp, both ``! --dport 53``) per match instead of a DROP plus an ACCEPT, so the
    result does not depend on rule order. ``bridge`` and ``sources`` come from ``rental_bridge`` and
    ``legacy_sources`` and are shell-safe.
    """
    a, b = CHAINS
    lines = [
        "IPT=iptables-nft",
        "$IPT -L DOCKER-USER -n >/dev/null 2>&1 || IPT=iptables-legacy",
        '$IPT -L DOCKER-USER -n >/dev/null 2>&1 || { echo "DOCKER-USER chain not found" >&2; exit 3; }',
        f"$IPT -N {a} 2>/dev/null; $IPT -N {b} 2>/dev/null",
        f"if $IPT -C DOCKER-USER -j {a} 2>/dev/null; then NEW={b}; OLD={a}; else NEW={a}; OLD={b}; fi",
        "$IPT -F $NEW || exit 4",
    ]
    matches = ([f"-i {bridge}"] if bridge else []) + [f"-s {source}" for source in sources]
    for match in matches:
        for proto in ("tcp", "udp"):
            lines.append(f"$IPT -A $NEW {match} -d {LINK_LOCAL_CIDR} -p {proto} ! --dport 53 -j DROP || exit 4")
    lines += [
        "$IPT -I DOCKER-USER -j $NEW || exit 4",
        "while $IPT -D DOCKER-USER -j $OLD 2>/dev/null; do :; done",
        "$IPT -F $OLD",
    ]
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
        last_run: dict[tuple, float] | None = None,
    ) -> None:
        self._interval = interval_seconds
        # rented_only: the instance before TenantEnforcementCheck; the other runs after ScoreCheck, idle nodes only
        self._rented_only = rented_only
        # shared by both instances so a node is guarded once per interval whichever one it reaches; keyed by
        # the miner and the endpoint too, since two miners can report one UUID for different hosts
        self._last_run: dict[tuple, float] = {} if last_run is None else last_run

    def _skip(self, ctx: Context, reason: str) -> CheckResult:
        return CheckResult(
            passed=True,
            event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what={"reason": reason}),
        )

    async def run(self, ctx: Context) -> CheckResult:
        executor_uuid = ctx.executor.uuid
        key = (ctx.miner_hotkey, executor_uuid, ctx.executor.address, ctx.executor.ssh_port)
        rental = rental_of_this_node(ctx)
        if self._rented_only != bool(rental and rental.pods):
            return self._skip(ctx, "guarded at the other position")
        if task_seconds_left(ctx) < HELPER_TIMEOUT_SECONDS + BUDGET_MARGIN_SECONDS:
            return self._skip(ctx, "not enough task time left")
        now = time.monotonic()
        last = self._last_run.get(key)
        if last is not None and now - last < self._interval:
            return self._skip(ctx, "within cadence window")
        # Retry on the next interval whatever happens below, never every cycle.
        self._last_run[key] = now

        try:
            # Every remote call shares one deadline that ends BUDGET_MARGIN_SECONDS before the task is
            # cancelled, so a stalled SSH call can never cost the cycle its validation result.
            guarded, error = await asyncio.wait_for(
                self._apply(ctx, rental), timeout=task_seconds_left(ctx) - BUDGET_MARGIN_SECONDS
            )
        except TimeoutError:
            # Not a run: the next cycle tries again.
            if last is None:
                self._last_run.pop(key, None)
            else:
                self._last_run[key] = last
            return self._skip(ctx, "out of task time")
        except Exception as exc:  # noqa: BLE001 — fail-open, see module docstring
            guarded, error = {}, str(exc)
        if guarded is None:
            # No rental network and no legacy rental: nothing to guard; the next interval checks again.
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
            event=render_message(Msg.APPLIED, ctx=ctx, check_id=self.check_id, what=guarded),
        )

    async def _apply(self, ctx: Context, rental) -> tuple[dict | None, str | None]:
        """(what was guarded, error); None when there is neither a rental network nor a legacy rental."""
        inspect = await ctx.ssh.run(
            f"/usr/bin/docker network inspect {RENTAL_NETWORK_NAME} "
            f"-f '{{{{.Id}}}} {{{{index .Options \"{BRIDGE_NAME_OPTION}\"}}}}' 2>/dev/null",
            check=False,
        )
        bridge = rental_bridge(inspect.stdout if inspect.exit_status == 0 else "")
        sources: list[str] = []
        names = [pod.docker_container_name for pod in (rental.pods if rental else [])]
        if names:
            # running pods created before the rental network stay on their own network until they stop
            pods = await ctx.ssh.run(
                "/usr/bin/docker inspect -f '{{if .State.Running}}{{range $k, $v := .NetworkSettings.Networks}}"
                f"{{{{if ne $k \"{RENTAL_NETWORK_NAME}\"}}}}{{{{$v.IPAddress}}}} {{{{end}}}}{{{{end}}}}{{{{end}}}}' "
                f"{' '.join(shlex.quote(name) for name in names)} 2>/dev/null",
                check=False,
            )
            sources = legacy_sources(pods.stdout)
        if bridge is None and not sources:
            return None, None
        helper = (
            f"/usr/bin/docker run --rm --network=host --cap-add=NET_ADMIN --entrypoint /bin/sh "
            f"{shlex.quote(settings.CUSTOM_DOCKERFILE_DIND_IMAGE)} -c {shlex.quote(link_local_drop_script(bridge, sources))}"
        )
        result = await ctx.ssh.run(f"timeout -k 5 {HELPER_TIMEOUT_SECONDS} {helper}", check=False)
        error = None if result.exit_status == 0 else (result.stderr or "").strip()[-300:] or f"exit {result.exit_status}"
        return {"bridge": bridge, "legacy_endpoints": len(sources)}, error
