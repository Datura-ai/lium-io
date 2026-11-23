"""Running rental containers get the link-local egress drop that custom builds already have.

Builds drop egress to blocked ranges through host DOCKER-USER rules (``docker_service._egress_filter_script``).
This check applies the same kind of rule for rentals with the host's own iptables in the host's namespaces
(``nsenter -t 1 -m -n -r``; the executor runs privileged with ``pid: host``), so it needs no image and no registry.
The shell, lock and time bound are the executor's: only iptables runs in the host root, which may have no ``/bin/sh``
or ``timeout``. Traffic entering from the ``lium-rentals``
bridge, and from a running legacy rental still on another network (the default bridge; a running
pod is never moved), to link-local 169.254/16 is dropped, except DNS (port 53), because a rental
container's embedded resolver forwards from the container's network namespace and some hosts use
a link-local resolver.

- Never before a scored step: on a rented node TenantEnforcementCheck runs it in its rented halt
  branch, so only the halt waits for it; on an idle node it runs after ScoreCheck. A first
  helper run is bounded by ``HELPER_TIMEOUT_SECONDS``, so it is skipped (and retried next
  cycle) when less than that is left before the executor task is cancelled.
- Periodic, not on the create path: host iptables are gone after a reboot while rental
  containers come back (``restart=unless-stopped``), and a rental create gains no extra step.
- Reconciled, not accumulated: the rules live in our own chain, rebuilt each run from what the
  host shows now and swapped in, so a recreated rental network or an ended legacy pod leaves no
  rule behind. The rental bridge is matched by interface, not subnet: a subnet Docker later
  gives an unrelated network is never caught.
- Fail-open: a host without a DOCKER-USER chain or without iptables logs a warning and
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

from services.const import LEGACY_LIUM_CONTAINER_NAME
from services.rental_docker_sdk import RENTAL_NETWORK_NAME

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
# One host-wide lock around every apply and cleanup: two miners can report the same host, and two interleaved
# chain swaps could otherwise flush the chain the other just made active. Reached through PID 1's root, so this is
# one host file shared by every executor's helper.
HOST_LOCK_DIR = "/proc/1/root/run/lock/lium-rental-egress"
HOST_LOCK = f"{HOST_LOCK_DIR}/apply.lock"
LOCK_WAIT_SECONDS = 30
# The helper's own bound, inside the validator-side one.
HELPER_RUN_SECONDS = HELPER_TIMEOUT_SECONDS - 10
# Prefix of every iptables call: the host's binary in the host's mount and network namespaces.
HOST_NS = "nsenter -t 1 -m -n -r --"
_ABSENT_NETWORK = re.compile(r"no such network|network \S+ not found", re.IGNORECASE)
_ABSENT_CONTAINER = re.compile(r"no such (object|container)", re.IGNORECASE)


class InventoryUnknown(Exception):
    """Docker could not say what the host runs: the current rules stay, and the next cycle reads it again."""


def network_inventory(result) -> str | None:
    """The rental bridge, or None when Docker says the network is absent; anything else is unknown."""
    if result.exit_status == 0:
        bridge = rental_bridge(result.stdout)
        if bridge is None:
            raise InventoryUnknown(f"unreadable network inspect: {(result.stdout or '').strip()[:100]}")
        return bridge
    if _ABSENT_NETWORK.search(result.stderr or ""):
        return None
    raise InventoryUnknown(_error(result) or "network inspect failed")


def rental_containers(result) -> list[str]:
    """Running pod and filler containers on the host from ``docker ps``, labelled or created before the label."""
    if result.exit_status != 0:
        raise InventoryUnknown(_error(result) or "docker ps failed")
    return [name for name in (result.stdout or "").split() if LEGACY_LIUM_CONTAINER_NAME.fullmatch(name)]


def pod_inventory(result) -> list[str]:
    """Networks of the listed running pods other than the rental one; a pod Docker says is gone is absent, any other
    failure is unknown."""
    if result.exit_status != 0:
        lines = [line for line in (result.stderr or "").splitlines() if line.strip()]
        if not lines or not all(_ABSENT_CONTAINER.search(line) for line in lines):
            raise InventoryUnknown(_error(result) or "docker inspect failed")
    return legacy_network_ids(result.stdout)


def legacy_bridge_inventory(result) -> list[str]:
    """Host interfaces of the legacy pods' bridge networks from ``docker network inspect`` ("<id> <driver> <bridge-name
    option>" per line). Other drivers have no bridge to match; a network Docker says is gone is absent; any other
    failure is unknown."""
    if result.exit_status != 0:
        lines = [line for line in (result.stderr or "").splitlines() if line.strip()]
        if not lines or not all(_ABSENT_NETWORK.search(line) for line in lines):
            raise InventoryUnknown(_error(result) or "legacy network inspect failed")
    bridges: list[str] = []
    for line in (result.stdout or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 2 or not _NETWORK_ID.fullmatch(parts[0]):
            raise InventoryUnknown(f"unreadable network inspect: {line.strip()[:100]}")
        if parts[1] == "bridge":
            bridge = rental_bridge(" ".join(parts[:1] + parts[2:]))
            if bridge not in bridges:
                bridges.append(bridge)
    return bridges
_NETWORK_ID = re.compile(r"[0-9a-f]{12,64}")


def rental_bridge(inspect_stdout: str) -> str | None:
    """Host interface of the rental network from ``docker network inspect`` ("<id> <bridge-name option>")."""
    parts = (inspect_stdout or "").split()
    if parts and _NETWORK_ID.fullmatch(parts[0]):
        if len(parts) > 1 and _IFNAME.fullmatch(parts[1]):
            return parts[1]
        return f"br-{parts[0][:12]}"
    return None


def legacy_network_ids(inspect_stdout: str) -> list[str]:
    """IDs of the networks running rental containers are on other than the rental one; anything else is dropped."""
    ids: list[str] = []
    for raw in (inspect_stdout or "").split():
        if _NETWORK_ID.fullmatch(raw) and raw not in ids:
            ids.append(raw)
    return ids


# The host's iptables whose DOCKER-USER reads, tried in order (plain `iptables` for a host without -nft/-legacy).
# A backend is absent only when its binary is missing (127) or iptables says the chain does not exist; any other
# failed probe is kept in ERRS, so a later backend's absence is never taken for the whole host's.
# $HOST_NS is empty when the script runs where iptables is local (tests).
_BACKEND_PROBE = (
    "IPT=; ERRS=; for B in iptables-nft iptables-legacy iptables; do "
    'ERR=$($HOST_NS $B -L DOCKER-USER -n 2>&1 >/dev/null) && { IPT="$HOST_NS $B"; break; }; '
    '[ $? -eq 127 ] || case "$ERR" in *"No chain"*|*"does not exist"*) ;; *) ERRS="$ERRS$B: $ERR; ";; esac; done'
)


def link_local_drop_script(bridge: str | None, legacy_bridges: list[str]) -> str:
    """Host DOCKER-USER rules: drop rental egress to link-local, except DNS, reconciled to exactly this set.

    Matched by the bridge the traffic comes in on, never by source address: a renter with NET_ADMIN can add
    another address, but not move its veth to another bridge. Two DROP rules (tcp and udp, both ``! --dport 53``)
    per bridge instead of a DROP plus an ACCEPT, so the result does not depend on rule order. ``bridge`` and
    ``legacy_bridges`` come from ``rental_bridge`` and are shell-safe.

    The first of our jumps in DOCKER-USER is the protection in force and stays until the other chain is rebuilt
    and swapped in; a swap killed after its insert leaves two jumps, and the next run keeps the first one
    (the newer) and takes the other out before flushing it.
    """
    a, b = CHAINS
    lines = [
        _BACKEND_PROBE,
        '[ -n "$IPT" ] || { [ -n "$ERRS" ] && { printf %s "$ERRS" >&2; exit 4; }; '
        'echo "no DOCKER-USER chain" >&2; exit 3; }',
        f"$IPT -N {a} 2>/dev/null; $IPT -N {b} 2>/dev/null",
        'RULES=$($IPT -S DOCKER-USER) || { echo "DOCKER-USER not readable" >&2; exit 4; }',
        f"ACTIVE=$(printf '%s\\n' \"$RULES\" | awk '$1 == \"-A\" && $3 == \"-j\" && ($4 == \"{a}\" || $4 == \"{b}\") "
        "{ print $4; exit }')",
        f'if [ "$ACTIVE" = {a} ]; then NEW={b}; OLD={a}; else NEW={a}; OLD={b}; fi',
        "while $IPT -D DOCKER-USER -j $NEW 2>/dev/null; do :; done",
        # 1 is "no such rule"; anything else is a failed read, never taken for an unreferenced chain
        '$IPT -C DOCKER-USER -j $NEW 2>/dev/null; [ $? -eq 1 ] || { echo "jump to $NEW not removed" >&2; exit 4; }',
        "$IPT -F $NEW || exit 4",
    ]
    matches = [f"-i {iface}" for iface in dict.fromkeys(([bridge] if bridge else []) + legacy_bridges)]
    for match in matches:
        for proto in ("tcp", "udp"):
            lines.append(f"$IPT -A $NEW {match} -d {LINK_LOCAL_CIDR} -p {proto} ! --dport 53 -j DROP || exit 4")
    lines += [
        "$IPT -I DOCKER-USER -j $NEW || exit 4",
        "while $IPT -D DOCKER-USER -j $OLD 2>/dev/null; do :; done",
        "$IPT -F $OLD",
    ]
    return "; ".join(lines)


def link_local_cleanup_script() -> str:
    """Remove our chains and their DOCKER-USER jumps: run when a host has nothing left to guard."""
    lines = [
        _BACKEND_PROBE,
        # nothing to clean only when every backend proved absent; a probe that failed otherwise may hide our jumps
        '[ -n "$IPT" ] || { [ -z "$ERRS" ] && exit 0; printf %s "$ERRS" >&2; exit 4; }',
    ]
    for chain in CHAINS:
        # a chain iptables says does not exist is done; one that exists must lose its jumps, rules and itself
        lines.append(
            f"if ERR=$($IPT -n -L {chain} 2>&1 >/dev/null); then "
            f"while $IPT -D DOCKER-USER -j {chain} 2>/dev/null; do :; done; "
            f"if $IPT -C DOCKER-USER -j {chain} 2>/dev/null; then echo 'jump to {chain} not removed' >&2; exit 4; fi; "
            f"$IPT -F {chain} || exit 4; $IPT -X {chain} || exit 4; "
            f'else case "$ERR" in *"No chain"*|*"does not exist"*) ;; *) echo "$ERR" >&2; exit 4;; esac; fi'
        )
    return "; ".join(lines) + "; exit 0"


def locked_script(script: str) -> str:
    """``script`` under the host lock. The lock is on fd 9 of the shell running the iptables calls, which inherit it,
    so it is released only when the last of them exits: no timeout can free it while a mutation still runs."""
    return (
        f"mkdir -p {shlex.quote(HOST_LOCK.rsplit('/', 1)[0])} && exec 9>>{HOST_LOCK} || exit 5; "
        f"flock -w {LOCK_WAIT_SECONDS} 9 || {{ echo 'host lock not taken' >&2; exit 5; }}; {script}"
    )


def helper_command(script: str) -> str:
    """What the executor runs: ``script`` under the lock, killed at ``HELPER_RUN_SECONDS``."""
    return f"exec timeout -s KILL {HELPER_RUN_SECONDS} /bin/sh -c {shlex.quote(locked_script(script))}"


def _helper(script: str) -> str:
    # No helper container, so no image pull that a registry (Docker Hub's anonymous rate limit) can refuse. The shell,
    # flock and timeout are the executor image's; only iptables resolves in the host root, the one binary dockerd
    # itself needs there
    return f"HOST_NS={shlex.quote(HOST_NS)} {helper_command(script)}"


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
        # rented_only: the instance before TenantEnforcementCheck, rented nodes only; the other runs when the
        # pipeline ends, whichever way, for every node: neither a running rental after a reboot nor the cleanup
        # after the last legacy rental may wait behind a fatal check that ends every cycle early
        self._rented_only = rented_only
        self.on_completion = not rented_only
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
        if self._rented_only and not (rental and rental.pods):
            return self._skip(ctx, "guarded at the other position")
        if ctx.host_writes_deferred:
            # before the cadence stamp: the rules change on the first cycle that verifies the node's account
            return self._skip(ctx, "node account not verified")
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
                self._apply(ctx), timeout=task_seconds_left(ctx) - BUDGET_MARGIN_SECONDS
            )
        except InventoryUnknown as exc:
            # Rules stay as they are: an unread host is never taken for an empty one. Not a run.
            if last is None:
                self._last_run.pop(key, None)
            else:
                self._last_run[key] = last
            logger.warning(
                _m(
                    "Rental egress guard: host inventory unknown, rules left as they are",
                    extra=get_extra_info({"executor_uuid": executor_uuid, "error": str(exc)}),
                )
            )
            return self._skip(ctx, "inventory unknown")
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
            if error is None:
                return self._skip(ctx, "no rental network")
            # A failed cleanup may have left stale rules: not a run, so the next cycle retries.
            if last is None:
                self._last_run.pop(key, None)
            else:
                self._last_run[key] = last
            logger.warning(
                _m(
                    "Rental egress guard: stale rule cleanup failed",
                    extra=get_extra_info({"executor_uuid": executor_uuid, "error": error}),
                )
            )
            return self._skip(ctx, "cleanup failed")

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

    async def _apply(self, ctx: Context) -> tuple[dict | None, str | None]:
        """(what was guarded, error); None when there is neither a rental network nor a legacy rental."""
        inspect = await ctx.ssh.run(
            f"/usr/bin/docker network inspect {RENTAL_NETWORK_NAME} "
            f"-f '{{{{.Id}}}} {{{{index .Options \"{BRIDGE_NAME_OPTION}\"}}}}'",
            check=False,
        )
        bridge = network_inventory(inspect)
        legacy: list[str] = []
        # host-wide, never this registration's rental: every miner reporting this host must reconcile the same set,
        # or an idle registration would drop the rules of another registration's legacy pod
        names = rental_containers(await ctx.ssh.run("/usr/bin/docker ps --format '{{.Names}}'", check=False))
        if names:
            # running pods created before the rental network stay on their own network until they stop
            pods = await ctx.ssh.run(
                "/usr/bin/docker inspect -f '{{if .State.Running}}{{range $k, $v := .NetworkSettings.Networks}}"
                f"{{{{if ne $k \"{RENTAL_NETWORK_NAME}\"}}}}{{{{$v.NetworkID}}}} {{{{end}}}}{{{{end}}}}{{{{end}}}}' "
                f"{' '.join(shlex.quote(name) for name in names)}",
                check=False,
            )
            network_ids = pod_inventory(pods)
            if network_ids:
                nets = await ctx.ssh.run(
                    "/usr/bin/docker network inspect "
                    f"-f '{{{{.Id}}}} {{{{.Driver}}}} {{{{index .Options \"{BRIDGE_NAME_OPTION}\"}}}}' "
                    f"{' '.join(network_ids)}",
                    check=False,
                )
                legacy = legacy_bridge_inventory(nets)
        if bridge is None and not legacy:
            # rules from a rental that has since ended must not outlive it (Docker reuses its address)
            result = await ctx.ssh.run(_helper(link_local_cleanup_script()), check=False)
            return None, _error(result)
        result = await ctx.ssh.run(_helper(link_local_drop_script(bridge, legacy)), check=False)
        return {"bridge": bridge, "legacy_bridges": legacy}, _error(result)


def _error(result) -> str | None:
    return None if result.exit_status == 0 else (result.stderr or "").strip()[-300:] or f"exit {result.exit_status}"
