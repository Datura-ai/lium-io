"""Running rental containers get the link-local egress drop that custom builds already have.

Builds drop egress to blocked ranges through host DOCKER-USER rules (``docker_service._egress_filter_script``).
This check applies the same kind of rule for rentals with the host's own iptables in the host's namespaces
(``nsenter -t 1 -m -n -r``; the executor runs privileged with ``pid: host``), so it needs no image and no registry.
The shell, lock and time bound are the executor's: only iptables runs in the host root, which may have no ``/bin/sh``
or ``timeout``. An executor that cannot (not root, no host PID 1, a missing tool) runs the same script in the validator's alpine helper container,
privileged with the host's PIDs, through the dockerd it already uses. Traffic entering from the ``lium-rentals``
bridge, and from every running rental on another network (its own bridge since DAH-6737, or the
default bridge of a legacy pod that was never moved), to link-local 169.254/16 is dropped, except DNS (port 53), because a rental
container's embedded resolver forwards from the container's network namespace and some hosts use
a link-local resolver.

- Never before a scored step: on a rented node TenantEnforcementCheck runs it in its rented halt
  branch, so only the halt waits for it; on an idle node it runs after ScoreCheck. A first
  helper run is bounded by ``HELPER_TIMEOUT_SECONDS``, so it is skipped (and retried next
  cycle) when less than that is left before the executor task is cancelled.
- Periodic, and right before a pod starts on the create and start paths (DAH-6737). Host iptables are gone after a
  reboot while dockerd brings rental containers back (``restart=unless-stopped``) before any validator runs, so every
  apply also writes the rule set it just put in force to the host and a ``docker.service`` drop-in that runs it as
  ``ExecStartPre``: the rules are back before dockerd starts a container, and dockerd does not start while they cannot
  be put back. A host whose dockerd is not the systemd
  ``docker.service`` gets no drop-in and the apply says so on stdout.
- Reconciled, not accumulated: the rules live in our own chain, rebuilt each run from what the
  host shows now and swapped in, so a recreated rental network or an ended legacy pod leaves no
  rule behind. The rental bridge is matched by interface, not subnet: a subnet Docker later
  gives an unrelated network is never caught.
- No DOCKER-USER chain (a dockerd that does not manage iptables, or the boot run before dockerd made it): the
  chain is created with the host's ``iptables`` and jumped to from FORWARD, never skipped. Only a host without any
  iptables fails the run; the periodic run then logs a warning, and the pre-start run refuses the start.
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

from core.docker_utils import ALPINE_HELPER_IMAGE
from services.const import LEGACY_LIUM_CONTAINER_NAME
from services.rental_docker_sdk import POD_NETWORK_LABEL, RENTAL_NETWORK_NAME

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
# Written only by the host's own iptables (DAH-6737): the earlier LIUM-RENTAL-EGRESS-A/B were also written by the
# helper image's newer iptables-nft, by validators on releases before lium-io#2308, and a host's older iptables-nft
# refuses to read or flush a chain a newer one wrote ("is incompatible, use 'nft' tool"). Those chains are never read;
# only their DOCKER-USER jumps are detached (OLD_CHAINS).
CHAINS = ("LIUM-EGRESS-HOST-A", "LIUM-EGRESS-HOST-B")
BRIDGE_NAME_OPTION = "com.docker.network.bridge.name"
_IFNAME = re.compile(r"[A-Za-z0-9_.-]{1,15}")
# One host-wide lock around every apply and cleanup: two miners can report the same host, and two interleaved
# chain swaps could otherwise flush the chain the other just made active. Reached through PID 1's root, so this is
# one host file shared by every executor's helper.
HOST_LOCK_DIR = "/proc/1/root/run/lock/lium-rental-egress"
HOST_LOCK = f"{HOST_LOCK_DIR}/apply.lock"
# Bumped under the lock by every apply and cleanup. A run reads it before its Docker inventory and applies only if it is
# unchanged, so a run whose inventory predates another run's rules (a pod's bridge added by its pre-start guard) never
# replaces them with its older view; it reads the host again instead.
HOST_GENERATION = f"{HOST_LOCK_DIR}/generation"
STALE_INVENTORY_EXIT = 6
INVENTORY_ATTEMPTS = 3
# Chains of releases before DAH-6737 (lium-io#2259/#2308/#2341): their jumps are detached once ours are in force, so a
# host never keeps an old docker0 drop after the last rental it guarded is gone
OLD_CHAINS = ("LIUM-RENTAL-EGRESS-A", "LIUM-RENTAL-EGRESS-B")
LOCK_WAIT_SECONDS = 30
# The helper's own bound, inside the validator-side one.
HELPER_RUN_SECONDS = HELPER_TIMEOUT_SECONDS - 10
# Prefix of every iptables call: the host's binary in the host's mount and network namespaces.
HOST_NS = "nsenter -t 1 -m -n -r --"
# The host's root as the executor sees it (``pid: host``), for the boot copy of the rules and the docker.service drop-in
HOST_ROOT = "/proc/1/root"
BOOT_SCRIPT = "/etc/lium/rental-egress-boot.sh"
BOOT_DROPIN = "/etc/systemd/system/docker.service.d/lium-rental-egress.conf"
BOOT_UNIT = "/etc/systemd/system/lium-rental-egress-boot.service"
# Present from a Docker start whose boot copy has not put the rules back until the next apply, which reports it
BOOT_LATE = "/etc/lium/rental-egress-boot-late"
# How long a Docker start waits for the boot copy before it starts without the rules
BOOT_WAIT_SECONDS = 300
BOOT_LATE_LINE = "rules not back when docker started"
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


def bridge_network_ids(result) -> set[str]:
    """IDs of the bridge networks in the ``docker network inspect`` output ``legacy_bridge_inventory`` read."""
    return {
        parts[0] for parts in (line.split() for line in (result.stdout or "").splitlines()) if parts[1:2] == ["bridge"]
    }


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


# Every helper step names itself on stderr before it runs, and a failed step reports its exit status and the host's
# iptables version and backend: the error then names the step even when the shell itself is killed (``_error``).
STEP_MARK = "@step "
_STEP_FUNCS = (
    f'step() {{ echo "{STEP_MARK}$1" >&2; }}; '
    'fail() { RC=$?; V=; [ -z "$IPT" ] || V=" ($($IPT --version 2>/dev/null))"; echo "${1:+$1: }exit $RC$V" >&2; exit 4; }'
)


def _steps(steps: list[tuple[str, str]]) -> list[str]:
    return [_STEP_FUNCS] + [f"step {name}; {command}" for name, command in steps]


# The host's iptables whose DOCKER-USER reads, tried in order (plain `iptables` for a host without -nft/-legacy).
# A backend is absent only when its binary is missing (127) or iptables says the chain does not exist; any other
# failed probe is kept in ERRS, so a later backend's absence is never taken for the whole host's.
# $HOST_NS is empty when the script runs where iptables is local (tests).
_BACKEND_PROBE = (
    "IPT=; ERRS=; FOUND=; for B in iptables-nft iptables-legacy iptables; do "
    'ERR=$($HOST_NS $B -L DOCKER-USER -n 2>&1 >/dev/null) && { IPT="$HOST_NS $B"; break; }; '
    '[ $? -eq 127 ] || { FOUND=1; case "$ERR" in *"No chain"*|*"does not exist"*) ;; *) ERRS="$ERRS$B: $ERR; ";; esac; }; '
    "done"
)

# Detach the old releases' jumps (never read or flush their chains: another iptables build may have written them). One
# that stays is reported on stdout, apart from whether our own rules applied.
_DETACH_OLD_JUMPS = (
    f"for C in {' '.join(OLD_CHAINS)}; do while $IPT -D DOCKER-USER -j $C 2>/dev/null; do :; done; "
    'ERR=$($IPT -C DOCKER-USER -j $C 2>&1); RC=$?; '
    'if [ $RC -eq 1 ]; then case "$ERR" in *ncompatible*) ;; *) continue;; esac; fi; '
    'echo "old jump to $C not detached: ${ERR:-still present}"; done'
)


# No DOCKER-USER in any backend while an iptables answers: create it with the host's own ``iptables`` (the one dockerd
# uses). A failed probe stays an error (4); only a host without any iptables is 3. The FORWARD jump is then ensured on
# every run, whoever made the chain: a chain restored by a host firewall, or kept by a dockerd that does not manage
# iptables, can exist with nothing jumping to it, and its rules would see no forwarded traffic.
_ENSURE_DOCKER_USER = (
    '[ -n "$IPT" ] || { [ -n "$ERRS" ] && { printf %s "$ERRS" >&2; exit 4; }; '
    'for B in iptables iptables-nft iptables-legacy; do $HOST_NS $B -L FORWARD -n >/dev/null 2>&1 && '
    '{ IPT="$HOST_NS $B"; break; }; done; '
    '[ -n "$IPT" ] || { echo "no iptables" >&2; exit 3; }; '
    "$IPT -N DOCKER-USER 2>/dev/null; "
    '$IPT -L DOCKER-USER -n >/dev/null || { echo "DOCKER-USER not created" >&2; exit 4; }; }; '
    # 1 is "no such rule"; any other failed check is an error, never a reason to insert a second jump
    '$IPT -C FORWARD -j DOCKER-USER 2>/dev/null; RC=$?; '
    '[ $RC -eq 0 ] || { [ $RC -eq 1 ] && $IPT -I FORWARD -j DOCKER-USER; } || fail "FORWARD jump to DOCKER-USER not ensured"'
)


def link_local_drop_script(bridge: str | None, legacy_bridges: list[str], host_root: str = HOST_ROOT) -> str:
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
    steps = [
        ("probe", _BACKEND_PROBE),
        ("docker-user", _ENSURE_DOCKER_USER),
        ("chains", f"$IPT -N {a} 2>/dev/null; $IPT -N {b} 2>/dev/null"),
        ("read", 'RULES=$($IPT -S DOCKER-USER) || fail "DOCKER-USER not readable"'),
        (
            "pick",
            f"ACTIVE=$(printf '%s\\n' \"$RULES\" | awk '$1 == \"-A\" && $3 == \"-j\" && ($4 == \"{a}\" || $4 == \"{b}\") "
            "{ print $4; exit }'); "
            f'if [ "$ACTIVE" = {a} ]; then NEW={b}; OLD={a}; else NEW={a}; OLD={b}; fi',
        ),
        ("unhook-new", "while $IPT -D DOCKER-USER -j $NEW 2>/dev/null; do :; done"),
        # 1 is "no such rule"; anything else is a failed read, never taken for an unreferenced chain
        (
            "check-new",
            '$IPT -C DOCKER-USER -j $NEW 2>/dev/null; RC=$?; [ $RC -eq 1 ] || '
            '{ echo "jump to $NEW not removed: exit $RC" >&2; exit 4; }',
        ),
        ("flush-new", "$IPT -F $NEW || fail"),
    ]
    matches = [f"-i {iface}" for iface in dict.fromkeys(([bridge] if bridge else []) + legacy_bridges)]
    for match in matches:
        for proto in ("tcp", "udp"):
            steps.append(("rule", f"$IPT -A $NEW {match} -d {LINK_LOCAL_CIDR} -p {proto} ! --dport 53 -j DROP || fail"))
    steps += [
        ("swap", "$IPT -I DOCKER-USER -j $NEW || fail"),
        ("unhook-old", "while $IPT -D DOCKER-USER -j $OLD 2>/dev/null; do :; done"),
        ("flush-old", "$IPT -F $OLD || fail"),
        ("old-releases", _DETACH_OLD_JUMPS),
    ]
    lines = _steps(steps)
    # the same rules, run by systemd on the host before dockerd starts any container after a reboot. Tried again for
    # BOOT_WAIT_SECONDS: rules that can never apply (no iptables) must not keep Docker, and so the validator, off the
    # host. The mark next to the script stays from the start until they apply, so a late or timed-out copy is reported
    late = '"${0%/*}/' + BOOT_LATE.rsplit("/", 1)[1] + '"'
    boot = (
        f"{{ : > {late}; }} 2>/dev/null; E=$(( $(date +%s) + ${{LIUM_EGRESS_BOOT_WAIT:-{BOOT_WAIT_SECONDS}}} )); "
        "until ( " + "; ".join(["HOST_NS=", *lines, "exit 0"]) + ' ); do [ "$(date +%s)" -lt "$E" ] || exit 1; '
        f"sleep 1; done; rm -f {late}; exit 0"
    )
    mark = shlex.quote(host_root + BOOT_LATE)
    return "; ".join(
        [
            *lines,
            f"step boot-late; if [ -f {mark} ]; then echo '{BOOT_LATE_LINE}'; rm -f {mark}; fi",
            f"step boot-restore; {_boot_restore(boot, host_root)}",
            "exit 0",
        ]
    )


NO_BOOT_UNIT = "boot restore not installed: no systemd docker.service"
BOOT_AFTER = "firewalld.service nftables.service iptables.service netfilter-persistent.service"


def _boot_restore(boot: str, host_root: str) -> str:
    """Install ``boot`` as a unit of its own that ``docker.service`` requires and starts after; daemon-reload when
    either file is new or systemd has not loaded it yet. Nothing of Docker's own start settings is changed."""
    script, dropin, own = (shlex.quote(host_root + p) for p in (BOOT_SCRIPT, BOOT_DROPIN, BOOT_UNIT))
    dirs = " ".join(shlex.quote(host_root + path.rsplit("/", 1)[0]) for path in (BOOT_SCRIPT, BOOT_DROPIN))
    name = BOOT_UNIT.rsplit("/", 1)[1]
    # printf makes the newlines, so the command stays on one line
    # dockerd restarts rental containers itself (unless-stopped), past the validator's pre-start guard: After= keeps
    # dockerd from starting until the boot copy has put the rules back or given up. Wants=, not Requires=: a copy that
    # fails or times out must still let Docker start, or the validator can never reach the host to repair it. The
    # start timeout is on this unit only, so Docker's own and the provider's settings stay.
    # No RemainAfterExit: the copy runs again on every Docker start, as before. After= the host firewall services
    # Docker itself orders after: their start flushes the ruleset, so the copy must write the rules after they load
    unit = shlex.quote(
        f"[Unit]\\nDescription=Lium rental egress rules\\nAfter={BOOT_AFTER}\\n"
        f"[Service]\\nType=oneshot\\nTimeoutStartSec={BOOT_WAIT_SECONDS + 60}\\nExecStart=/bin/sh {BOOT_SCRIPT}\\n"
    )
    hook = shlex.quote(f"[Unit]\\nWants={name}\\nAfter={name}\\n")
    need = "$HOST_NS systemctl show -p NeedDaemonReload --value"
    return (
        f"if [ ! -d {shlex.quote(host_root + '/run/systemd/system')} ]; then echo '{NO_BOOT_UNIT}'; "
        # a state that cannot be read is not taken for "no docker.service": the pod start refuses on it
        "elif ! boot_unit_state=$($HOST_NS systemctl show -p LoadState --value docker.service 2>/dev/null) || "
        '[ -z "$boot_unit_state" ]; then echo \'boot restore not installed: docker.service state not read\'; '
        f'elif [ "$boot_unit_state" != loaded ]; then echo \'{NO_BOOT_UNIT}\'; '
        # reported here; a pod start refuses on it (apply_rental_egress_guard), so no pod Docker would restart at
        # boot is started without the boot copy
        f"elif ! mkdir -p {dirs}; then echo 'boot restore not installed: directories not created'; "
        f"elif ! {{ printf '%s\\n' {shlex.quote(boot)} > {script}.tmp && mv -f {script}.tmp {script}; }}; then "
        "echo 'boot restore not installed: boot rules not written'; "
        f"elif ! {{ printf {unit} > {own}.tmp && printf {hook} > {dropin}.tmp; }}; then "
        "echo 'boot restore not installed: drop-in not written'; "
        # same bytes: still reloaded while systemd has not loaded them (an earlier run's reload failed); skipped only
        # when both queries succeed and answer exactly "no", so a query that failed never passes for loaded
        f"elif cmp {own}.tmp {own} >/dev/null 2>&1 && cmp {dropin}.tmp {dropin} >/dev/null 2>&1 && "
        f"rm -f {own}.tmp {dropin}.tmp && R=$({need} {name} 2>/dev/null) && [ \"$R\" = no ] && "
        f'R=$({need} docker.service 2>/dev/null) && [ "$R" = no ]; then :; '
        f"elif ! {{ {{ [ ! -e {own}.tmp ] || mv -f {own}.tmp {own}; }} && {{ [ ! -e {dropin}.tmp ] || "
        f"mv -f {dropin}.tmp {dropin}; }} && $HOST_NS systemctl daemon-reload; }}; then "
        "echo 'boot restore not installed: docker.service drop-in not loaded'; fi"
    )


def _boot_clear(host_root: str) -> str:
    """Replace an installed boot copy with a no-op: the drop-in keeps naming it, and a Docker start never brings back
    the rules of a rental that has ended."""
    script = shlex.quote(host_root + BOOT_SCRIPT)
    return (
        f"rm -f {shlex.quote(host_root + BOOT_LATE)}; if [ -f {script} ]; then printf 'exit 0\\n' > {script}.tmp && mv -f {script}.tmp {script} || "
        "{ echo 'boot rules not cleared' >&2; exit 4; }; fi"
    )


# a reservation its holder stopped renewing (its validator process restarted) expires after this, so its bridge is
# reaped once no container references it; a live holder renews it every BRIDGE_RESERVATION_RENEW_SECONDS
BRIDGE_RESERVATION_TTL_MINUTES = 10
BRIDGE_RESERVATION_RENEW_SECONDS = 60


def _reservations_dir() -> str:
    return HOST_LOCK.rsplit("/", 1)[0] + "/bridges"


def _reservation(network: str, token: str) -> str:
    return f"{shlex.quote(_reservations_dir())}/{shlex.quote(f'{network}.{token}')}"


def reserve_pod_bridge_script(network: str, token: str) -> str:
    """Under the host lock (DAH-6737): reserve the pod bridge ``network`` for this holder, then remove every other pod
    bridge that no holder reserves and no container references, checked right before its removal. Every validator
    process reserves through this one host lock, so a bridge is never removed between a holder's reservation (taken
    before it is ensured) and its container's reference to it, whichever process holds it."""
    d = shlex.quote(_reservations_dir())
    return (
        f"mkdir -p {d} && : > {_reservation(network, token)} || {{ echo 'bridge not reserved' >&2; exit 4; }}; "
        # a failed cleanup never blocks the create: unread expired reservations only keep their bridges longer
        f"find {d} -type f -mmin +{BRIDGE_RESERVATION_TTL_MINUTES} -exec rm -f {{}} + || "
        "echo 'bridge not removed: expired reservations not read'; "
        f"for ID in $($HOST_NS docker network ls -q --no-trunc --filter label={POD_NETWORK_LABEL}=pod); do "
        "N=$($HOST_NS docker network inspect -f '{{.Name}}' \"$ID\") || continue; "
        f'case "$N" in {RENTAL_NETWORK_NAME}-*) ;; *) continue ;; esac; '
        f'ls {d}/"$N".* >/dev/null 2>&1 && continue; '
        # a created container holds its network by ID (our creates) or by name, started or not
        'U=$($HOST_NS docker ps -aq --filter network="$ID" --filter network="$N") || continue; [ -z "$U" ] || continue; '
        '$HOST_NS docker network rm "$ID" >/dev/null 2>&1 || echo "bridge not removed: $N"; '
        "done; exit 0"
    )


def release_pod_bridge_script(network: str, token: str) -> str:
    return f"rm -f {_reservation(network, token)}"


def renew_pod_bridge_script(network: str, token: str) -> str:
    return f"touch {_reservation(network, token)}"


async def reserve_pod_bridge(ssh, network: str, token: str) -> None:
    result = await ssh.run(_helper(reserve_pod_bridge_script(network, token)), check=False)
    if result.exit_status != 0:
        raise InventoryUnknown(_error(result) or "pod bridge not reserved")
    if left := [line for line in (result.stdout or "").splitlines() if line.startswith("bridge not removed")]:
        logger.warning(_m("Rental egress guard: an ended rental's bridge was not removed", extra={"bridges": left}))


async def release_pod_bridge(ssh, network: str, token: str) -> None:
    result = await ssh.run(_helper(release_pod_bridge_script(network, token)), check=False)
    if result.exit_status != 0:
        raise InventoryUnknown(_error(result) or "pod bridge not released")


async def renew_pod_bridge(ssh, network: str, token: str) -> None:
    result = await ssh.run(_helper(renew_pod_bridge_script(network, token)), check=False)
    if result.exit_status != 0:
        raise InventoryUnknown(_error(result) or "pod bridge reservation not renewed")


def link_local_cleanup_script(host_root: str = HOST_ROOT) -> str:
    """Remove our chains, their DOCKER-USER jumps and the boot copy of the rules: run when a host has nothing left to
    guard."""
    clear = _boot_clear(host_root)
    steps = [
        ("probe", _BACKEND_PROBE),
        # nothing to clean only when an iptables said so on every backend; a probe that failed otherwise may hide our
        # jumps, and no iptables at all (a PID 1 that is not the host's) proves nothing: 3, so the helper reads the host
        (
            "backend",
            '[ -n "$IPT" ] || { [ -n "$ERRS" ] && { printf %s "$ERRS" >&2; exit 4; }; '
            f'[ -n "$FOUND" ] && {{ {clear}; exit 0; }}; '
            'echo "no iptables" >&2; exit 3; }',
        ),
    ]
    for chain in CHAINS:
        # a chain iptables says does not exist is done; one that exists must lose its jumps, rules and itself
        steps.append((
            f"remove-{chain}",
            f"if ERR=$($IPT -n -L {chain} 2>&1 >/dev/null); then "
            f"while $IPT -D DOCKER-USER -j {chain} 2>/dev/null; do :; done; "
            f"if $IPT -C DOCKER-USER -j {chain} 2>/dev/null; then echo 'jump to {chain} not removed' >&2; exit 4; fi; "
            f"$IPT -F {chain} || fail; $IPT -X {chain} || fail; "
            f'else case "$ERR" in *"No chain"*|*"does not exist"*) ;; *) echo "$ERR" >&2; exit 4;; esac; fi'
        ))
    steps += [("old-releases", _DETACH_OLD_JUMPS), ("boot-clear", clear)]
    return "; ".join(_steps(steps)) + "; exit 0"


def locked_script(script: str, generation: str | None = None) -> str:
    """``script`` under the host lock. The lock is on fd 9 of the shell running the iptables calls, which inherit it,
    so it is released only when the last of them exits: no timeout can free it while a mutation still runs.

    With ``generation`` (what ``HOST_GENERATION`` held before the caller's inventory) the script runs only if no other
    run has changed the rules since (else ``STALE_INVENTORY_EXIT``), and the generation is bumped after it. The wait is
    ``timeout`` around a plain ``flock``: busybox's flock (the alpine helper) has no ``-w``."""
    lock_dir = shlex.quote(HOST_LOCK.rsplit("/", 1)[0])
    locked = (
        f"{_STEP_FUNCS}; step lock; "
        f"mkdir -p {lock_dir} && exec 9>>{HOST_LOCK} || exit 5; "
        f"timeout -s KILL {LOCK_WAIT_SECONDS} flock 9 || {{ echo 'host lock not taken' >&2; exit 5; }}; "
    )
    if generation is None:
        return locked + script
    gen, tmp = shlex.quote(HOST_GENERATION), shlex.quote(f"{HOST_GENERATION}.tmp")
    return locked + (
        f"step generation; G=$(cat {gen} 2>/dev/null); "
        f'[ "${{G:-0}}" = {shlex.quote(generation)} ] || '
        f"{{ echo 'host rules changed since this inventory' >&2; exit {STALE_INVENTORY_EXIT}; }}; "
        # bumped before any rule changes, atomically: a run that cannot record it (a full /run) changes nothing and
        # fails, so an older inventory can never pass the check above after newer rules went in
        f"echo $((${{G:-0}} + 1)) > {tmp} && mv -f {tmp} {gen} || "
        f"{{ echo 'rule generation not recorded' >&2; exit 4; }}; "
        f"( {script} )"
    )


def _reporting_exit(command: str, what: str) -> str:
    """``command``, then its status as the exit; a signal exit (a crash or a kill) is reported on stderr first."""
    return f'{command}; RC=$?; [ $RC -le 128 ] || echo "{what} ended by signal $((RC - 128))" >&2; exit $RC'


def helper_command(script: str, generation: str | None = None) -> str:
    """What the executor runs: ``script`` under the lock, killed at ``HELPER_RUN_SECONDS``."""
    return _reporting_exit(
        f"timeout -s KILL {HELPER_RUN_SECONDS} /bin/sh -c {shlex.quote(locked_script(script, generation))}",
        "helper shell",
    )


def _helper(script: str, generation: str | None = None) -> str:
    # No helper container on the shipped executor, so no image pull that a registry (Docker Hub's anonymous rate limit)
    # can refuse. The shell, flock and timeout are the executor image's; only iptables resolves in the host root, the one
    # binary dockerd itself needs there
    inner = f"HOST_NS={shlex.quote(HOST_NS)} {helper_command(script, generation)}"
    in_helper = _reporting_exit(f"{NON_ROOT_HELPER} /bin/sh -c {shlex.quote(inner)}", "helper container")
    # the lock directory in the host's /run through PID 1
    lock_dir = shlex.quote(HOST_LOCK.rsplit("/", 1)[0])
    tools = " && ".join(f"command -v {tool} >/dev/null" for tool in ("nsenter", "flock", "timeout"))
    # PID 1 must be the host's: in another mount namespace than the executor (``pid: host``), or no container at all. A
    # private PID 1 would give this route its own lock and rule generation, never the host's (DAH-6737)
    host_pid1 = f'{{ [ "$(readlink /proc/1/ns/mnt)" != "$(readlink /proc/self/ns/mnt)" ] || [ ! -e {CONTAINER_MARK} ]; }}'
    # and the host's DOCKER-USER must answer from here: the route is chosen once, before any read or change, so the
    # rule generation is always read and bumped where the rules are written (DAH-6737)
    # (or a host iptables without the chain yet, which the script creates)
    reach = f"{{ HOST_NS={shlex.quote(HOST_NS)}; {_BACKEND_PROBE}; [ -n \"$IPT\" ] || [ -z \"$ERRS\" -a -n \"$FOUND\" ]; }}"
    direct = f'[ "$(id -u)" = 0 ] && {host_pid1} && mkdir -p {lock_dir} 2>/dev/null && {tools} && {reach}'
    return f"if ! {{ {direct}; }}; then {in_helper}; fi; {inner}"


# An executor that cannot run the script on the host itself (another user than root, a PID 1 that is not the host's,
# or an image without nsenter, flock or timeout) still reaches dockerd, which runs as root: the same script runs in the
# alpine helper the validator already uses on every node, privileged with the host's PIDs, so it reaches the same host
# lock and the host's own iptables through PID 1 (DAH-6737). busybox there has sh, flock, timeout and nsenter.
# Docker's mark in every container's root: an executor without it runs on the host itself, so its PID 1 is the host's
CONTAINER_MARK = "/.dockerenv"
NON_ROOT_HELPER = f"/usr/bin/docker run --rm --privileged --pid=host --network host {ALPINE_HELPER_IMAGE}"


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
        return await apply_rental_egress_guard(ctx.ssh)


async def apply_rental_egress_guard(
    ssh, attempts: int | None = INVENTORY_ATTEMPTS, required_network_id: str | None = None
) -> tuple[dict | None, str | None]:
    """(what was guarded, error); None when there is neither a rental network nor a legacy rental.

    Also run on the create and start paths right before a pod starts (DAH-6737), so a pod's own bridge is guarded
    before its first packet, not at the next periodic run. The rules are applied only if no other run changed them
    since this inventory began; otherwise the host is read again (``attempts`` times, then unknown). The pre-start
    path passes ``attempts=None``: concurrent guards on one host always settle, and its caller's timeout is the bound.
    It also passes the pod's bridge as ``required_network_id``: inventoried whatever its labels, and an error unless it
    resolved to a bridge in the applied rules, so a pod never starts on a bridge the drop was not applied to.
    """
    tries = 0
    while True:
        generation = await _host_generation(ssh)
        guarded, result, required_missing = await _inventory_and_apply(ssh, generation, required_network_id)
        if result.exit_status != STALE_INVENTORY_EXIT:
            break
        tries += 1
        if attempts is not None and tries >= attempts:
            raise InventoryUnknown("the host's rules kept changing during the inventory")
    if boot := [line for line in (result.stdout or "").splitlines() if line.startswith("boot restore")]:
        logger.warning(_m("Rental egress guard: rules not restored before dockerd after a reboot", extra={"why": boot}))
        if guarded is not None:
            guarded["boot_restore"] = boot
    if BOOT_LATE_LINE in (result.stdout or "").splitlines():
        logger.warning(_m("Rental egress guard: Docker started before the boot copy put the rules back"))
        if guarded is not None:
            guarded["boot_late"] = True
    if left := [line for line in (result.stdout or "").splitlines() if line.startswith("old jump")]:
        logger.warning(_m("Rental egress guard: an old release's jump was not detached", extra={"jumps": left}))
        if guarded is not None:
            guarded["old_jumps_left"] = left
    error = _error(result)
    if error is None and required_missing:
        error = f"the pod's bridge {required_network_id[:12]} is not in the applied rules"
    # Docker restarts a started pod at boot on its own (unless-stopped): never start one where the boot copy is missing
    if error is None and required_network_id is not None and (not_installed := [b for b in boot if b != NO_BOOT_UNIT]):
        error = not_installed[0]
    return guarded, error


async def _host_generation(ssh) -> str:
    result = await ssh.run(_helper(f"cat {shlex.quote(HOST_GENERATION)} 2>/dev/null; :"), check=False)
    generation = (result.stdout or "").strip() or "0"
    if result.exit_status != 0 or not generation.isdigit():
        raise InventoryUnknown(_error(result) or f"unreadable rule generation: {generation[:50]}")
    return generation


async def _inventory_and_apply(ssh, generation: str, required_network_id: str | None = None):
    inspect = await ssh.run(
        f"/usr/bin/docker network inspect {RENTAL_NETWORK_NAME} "
        f"-f '{{{{.Id}}}} {{{{index .Options \"{BRIDGE_NAME_OPTION}\"}}}}'",
        check=False,
    )
    bridge = network_inventory(inspect)
    legacy: list[str] = []
    # host-wide, never this registration's rental: every miner reporting this host must reconcile the same set,
    # or an idle registration would drop the rules of another registration's legacy pod
    names = rental_containers(await ssh.run("/usr/bin/docker ps --format '{{.Names}}'", check=False))
    network_ids: list[str] = []
    if names:
        # running pods created before the rental network stay on their own network until they stop
        pods = await ssh.run(
            "/usr/bin/docker inspect -f '{{if .State.Running}}{{range $k, $v := .NetworkSettings.Networks}}"
            f"{{{{if ne $k \"{RENTAL_NETWORK_NAME}\"}}}}{{{{$v.NetworkID}}}} {{{{end}}}}{{{{end}}}}{{{{end}}}}' "
            f"{' '.join(shlex.quote(name) for name in names)}",
            check=False,
        )
        network_ids = pod_inventory(pods)
    # every pod's own bridge, its pod stopped or not started yet, so a start never runs on an unguarded bridge
    labelled = await ssh.run(
        f"/usr/bin/docker network ls -q --no-trunc --filter label={POD_NETWORK_LABEL}=pod", check=False
    )
    if labelled.exit_status != 0:
        raise InventoryUnknown(_error(labelled) or "pod network list failed")
    network_ids += [nid for nid in legacy_network_ids(labelled.stdout) if nid not in network_ids]
    if required_network_id and required_network_id not in network_ids:
        network_ids.append(required_network_id)
    required_missing = bool(required_network_id)
    if network_ids:
        nets = await ssh.run(
            "/usr/bin/docker network inspect "
            f"-f '{{{{.Id}}}} {{{{.Driver}}}} {{{{index .Options \"{BRIDGE_NAME_OPTION}\"}}}}' "
            f"{' '.join(network_ids)}",
            check=False,
        )
        legacy = legacy_bridge_inventory(nets)
        required_missing = bool(required_network_id) and required_network_id not in bridge_network_ids(nets)
    if bridge is None and not legacy:
        # rules from a rental that has since ended must not outlive it (Docker reuses its address)
        return None, await ssh.run(_helper(link_local_cleanup_script(), generation), check=False), required_missing
    result = await ssh.run(_helper(link_local_drop_script(bridge, legacy), generation), check=False)
    return {"bridge": bridge, "legacy_bridges": legacy}, result, required_missing


def _error(result) -> str | None:
    """A failed run's stderr; for a helper, led by the step it was in and its exit status (and signal, if killed)."""
    if result.exit_status == 0:
        return None
    lines = (result.stderr or "").splitlines()
    steps = [line[len(STEP_MARK) :] for line in lines if line.startswith(STEP_MARK)]
    text = "\n".join(line for line in lines if not line.startswith(STEP_MARK)).strip()
    if not steps:
        return text[-300:] or f"exit {result.exit_status}"
    status = f"exit {result.exit_status}"
    if result.exit_status > 128:
        status += f", signal {result.exit_status - 128}"
    return f"step {steps[-1]} ({status})" + (f": {text[-250:]}" if text else "")
