"""The rental egress rules: running rental containers cannot reach link-local 169.254/16, except DNS (port 53).

One source for both sides. The executor applies the rules itself (at start, when the rental network is created
and when a rental container starts, then every ``APPLY_INTERVAL_SECONDS``), and the validator reads them back; for
an executor that does not advertise ``RENTAL_EGRESS_CAPABILITY`` the validator still applies them over SSH. Every
command runs inside the executor container either way (privileged, ``pid: host``), so both sides run the same
strings, and ``run`` is whatever executes one there: the validator's SSH session or the executor's own shell. It
returns an object with ``exit_status``, ``stdout`` and ``stderr``.

Builds drop egress to blocked ranges through host DOCKER-USER rules. These rules use the host's own iptables in the
host's namespaces (``nsenter -t 1 -m -n -r``), so they need no image and no registry. The shell, lock and time bound
are the executor's: only iptables runs in the host root, which may have no ``/bin/sh`` or ``timeout``. Traffic
entering from the ``lium-rentals`` bridge, and from a running legacy rental still on another network (the default
bridge; a running pod is never moved), to link-local 169.254/16 is dropped, except DNS (port 53), because a rental
container's embedded resolver forwards from the container's network namespace and some hosts use a link-local
resolver.

- Reconciled, not accumulated: the rules live in our own chain, rebuilt each run from what the host shows now and
  swapped in, so a recreated rental network or an ended legacy pod leaves no rule behind. The rental bridge is
  matched by interface, not subnet: a subnet Docker later gives an unrelated network is never caught.
- Fail-open: a host without a DOCKER-USER chain or without iptables reports exit 3 and keeps running; a hardening
  rule must not take a provider's node out of rotation.
- DOCKER-USER sees forwarded traffic only: Docker's embedded DNS, published ports, container-to-container traffic
  on the subnet and the validator's inbound SSH are unaffected.
"""

from __future__ import annotations

import ipaddress
import re
import shlex
from collections.abc import Awaitable, Callable
from typing import Any

# What `/version` lists when the executor applies these rules itself; the validator then only reads them.
RENTAL_EGRESS_CAPABILITY = "rental_egress/1"
RENTAL_NETWORK_NAME = "lium-rentals"
# Pod and filler containers, labelled or created before the label (__prev/__stopped: a pod kept across a move).
RENTAL_CONTAINER_NAME = re.compile(
    r"^(?:pod|filler)_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?:__prev|__stopped)?$"
)
# What the helper says (exit 3) on a host without a DOCKER-USER chain: fail-open, nothing to retry.
NO_DOCKER_USER = "no DOCKER-USER chain"

Runner = Callable[[str], Awaitable[Any]]

APPLY_INTERVAL_SECONDS = 10 * 60
HELPER_TIMEOUT_SECONDS = 90
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
# The helper's own bound, inside the caller's one.
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
    raise InventoryUnknown(error_of(result) or "network inspect failed")


def rental_containers(result) -> list[str]:
    """Running pod and filler containers on the host from ``docker ps``, labelled or created before the label."""
    if result.exit_status != 0:
        raise InventoryUnknown(error_of(result) or "docker ps failed")
    return [name for name in (result.stdout or "").split() if RENTAL_CONTAINER_NAME.fullmatch(name)]


def pod_inventory(result) -> list[str]:
    """Networks of the listed running pods other than the rental one; a pod Docker says is gone is absent, any other
    failure is unknown."""
    if result.exit_status != 0:
        lines = [line for line in (result.stderr or "").splitlines() if line.strip()]
        if not lines or not all(_ABSENT_CONTAINER.search(line) for line in lines):
            raise InventoryUnknown(error_of(result) or "docker inspect failed")
    return legacy_network_ids(result.stdout)


def legacy_bridge_inventory(result) -> list[str]:
    """Host interfaces of the legacy pods' bridge networks from ``docker network inspect`` ("<id> <driver> <bridge-name
    option>" per line). Other drivers have no bridge to match; a network Docker says is gone is absent; any other
    failure is unknown."""
    if result.exit_status != 0:
        lines = [line for line in (result.stderr or "").splitlines() if line.strip()]
        if not lines or not all(_ABSENT_NETWORK.search(line) for line in lines):
            raise InventoryUnknown(error_of(result) or "legacy network inspect failed")
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


_NEED_DOCKER_USER = (
    '[ -n "$IPT" ] || { [ -n "$ERRS" ] && { printf %s "$ERRS" >&2; exit 4; }; '
    f'echo "{NO_DOCKER_USER}" >&2; exit 3; }}'
)
# The chain in force: the first of our two that DOCKER-USER jumps to (empty when there is none).
_ACTIVE_CHAIN = (
    'RULES=$($IPT -S DOCKER-USER) || { echo "DOCKER-USER not readable" >&2; exit 4; }; '
    f"ACTIVE=$(printf '%s\\n' \"$RULES\" | awk '$1 == \"-A\" && $3 == \"-j\" && "
    f"($4 == \"{CHAINS[0]}\" || $4 == \"{CHAINS[1]}\") {{ print $4; exit }}')"
)
PROTOCOLS = ("tcp", "udp")


def drop_rule(iface: str, proto: str) -> str:
    """One rule of the set: ``proto`` from bridge ``iface`` to link-local, other than DNS, is dropped."""
    return f"-i {iface} -d {LINK_LOCAL_CIDR} -p {proto} ! --dport 53 -j DROP"


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
        _NEED_DOCKER_USER,
        f"$IPT -N {a} 2>/dev/null; $IPT -N {b} 2>/dev/null",
        _ACTIVE_CHAIN,
        f'if [ "$ACTIVE" = {a} ]; then NEW={b}; OLD={a}; else NEW={a}; OLD={b}; fi',
        "while $IPT -D DOCKER-USER -j $NEW 2>/dev/null; do :; done",
        # 1 is "no such rule"; anything else is a failed read, never taken for an unreferenced chain
        '$IPT -C DOCKER-USER -j $NEW 2>/dev/null; [ $? -eq 1 ] || { echo "jump to $NEW not removed" >&2; exit 4; }',
        "$IPT -F $NEW || exit 4",
    ]
    matches = list(dict.fromkeys(([bridge] if bridge else []) + legacy_bridges))
    for match in matches:
        for proto in PROTOCOLS:
            lines.append(f"$IPT -A $NEW {drop_rule(match, proto)} || exit 4")
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


def helper_command(script: str, *, locked: bool = True) -> str:
    """What the executor runs: ``script`` (under the lock unless it only reads), killed at ``HELPER_RUN_SECONDS``."""
    body = locked_script(script) if locked else script
    return f"exec timeout -s KILL {HELPER_RUN_SECONDS} /bin/sh -c {shlex.quote(body)}"


def _helper(script: str, *, locked: bool = True) -> str:
    # No helper container, so no image pull that a registry (Docker Hub's anonymous rate limit) can refuse. The shell,
    # flock and timeout are the executor image's; only iptables resolves in the host root, the one binary dockerd
    # itself needs there
    return f"HOST_NS={shlex.quote(HOST_NS)} {helper_command(script, locked=locked)}"


def link_local_read_script() -> str:
    """Print the rules of the chain in force (nothing when DOCKER-USER jumps to neither of ours). Reads only."""
    return "; ".join([_BACKEND_PROBE, _NEED_DOCKER_USER, _ACTIVE_CHAIN, '[ -n "$ACTIVE" ] || exit 0', "$IPT -S $ACTIVE || exit 4"])


def _chunks(tokens: list[str]) -> frozenset[str] | None:
    """A rule as its set of "[!]option value" parts, ``-m <module>`` left out; None for anything unparsable."""
    chunks, negate, i = set(), "", 0
    while i < len(tokens):
        token = tokens[i]
        if token == "!":
            negate = "! "
        elif token.startswith("-") and i + 1 < len(tokens):
            if token != "-m":
                chunks.add(f"{negate}{token} {tokens[i + 1]}")
            negate = ""
            i += 1
        else:
            return None
        i += 1
    return frozenset(chunks)


def missing_rules(read_stdout: str, bridges: list[str]) -> list[str]:
    """The bridges of ``bridges`` whose drop the chain in force lacks (``iptables -S`` output, either rule order).
    A rule with any extra condition is not counted: it would drop less than the set's."""
    present = set()
    for line in (read_stdout or "").splitlines():
        tokens = line.split()
        if len(tokens) > 2 and tokens[0] == "-A":
            chunks = _chunks(tokens[2:])
            if chunks is not None:
                present.add(chunks)
    return [
        bridge
        for bridge in bridges
        if any(_chunks(drop_rule(bridge, proto).split()) not in present for proto in PROTOCOLS)
    ]


async def inventory(run: Runner) -> tuple[str | None, list[str]]:
    """(rental bridge, legacy rentals' bridges) as Docker shows them host-wide; InventoryUnknown when it cannot say."""
    bridge = network_inventory(
        await run(
            f"/usr/bin/docker network inspect {RENTAL_NETWORK_NAME} "
            f"-f '{{{{.Id}}}} {{{{index .Options \"{BRIDGE_NAME_OPTION}\"}}}}'"
        )
    )
    legacy: list[str] = []
    # host-wide, never one registration's rental: every miner reporting this host must reconcile the same set,
    # or an idle registration would drop the rules of another registration's legacy pod
    names = rental_containers(await run("/usr/bin/docker ps --format '{{.Names}}'"))
    if names:
        # running pods created before the rental network stay on their own network until they stop
        network_ids = pod_inventory(
            await run(
                "/usr/bin/docker inspect -f '{{if .State.Running}}{{range $k, $v := .NetworkSettings.Networks}}"
                f"{{{{if ne $k \"{RENTAL_NETWORK_NAME}\"}}}}{{{{$v.NetworkID}}}} {{{{end}}}}{{{{end}}}}{{{{end}}}}' "
                f"{' '.join(shlex.quote(name) for name in names)}"
            )
        )
        if network_ids:
            legacy = legacy_bridge_inventory(
                await run(
                    "/usr/bin/docker network inspect "
                    f"-f '{{{{.Id}}}} {{{{.Driver}}}} {{{{index .Options \"{BRIDGE_NAME_OPTION}\"}}}}' "
                    f"{' '.join(network_ids)}"
                )
            )
    return bridge, legacy


async def apply_rules(run: Runner) -> tuple[dict | None, str | None]:
    """Reconcile the rules to what the host runs: (what was guarded, error); None when there is neither a rental
    network nor a legacy rental, after removing any rules left from one."""
    bridge, legacy = await inventory(run)
    if bridge is None and not legacy:
        # rules from a rental that has since ended must not outlive it (Docker reuses its address)
        return None, error_of(await run(_helper(link_local_cleanup_script())))
    result = await run(_helper(link_local_drop_script(bridge, legacy)))
    return {"bridge": bridge, "legacy_bridges": legacy}, error_of(result)


async def verify_rules(run: Runner) -> tuple[dict | None, str | None]:
    """Read the rules back without changing anything: ({bridge, legacy_bridges, missing}, None) once read, (None, None)
    when there is nothing to guard, (None, error) when the rules could not be read (no DOCKER-USER chain included)."""
    bridge, legacy = await inventory(run)
    if bridge is None and not legacy:
        return None, None
    result = await run(_helper(link_local_read_script(), locked=False))
    error = error_of(result)
    if error is not None:
        return None, error
    bridges = list(dict.fromkeys(([bridge] if bridge else []) + legacy))
    return {"bridge": bridge, "legacy_bridges": legacy, "missing": missing_rules(result.stdout, bridges)}, None


def error_of(result) -> str | None:
    return None if result.exit_status == 0 else (result.stderr or "").strip()[-300:] or f"exit {result.exit_status}"


