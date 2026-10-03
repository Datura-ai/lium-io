"""Docker-in-Docker defaults for a rented sysbox pod.

Three things every renter who runs Docker inside a pod hits:

- The inner dockerd carves networks out of Docker's built-in pools (172.17-31/16, 192.168/20): 31
  networks, less its own docker0 and the /16 of the pod's address on `lium-rentals`, so the 30th
  `docker network create` fails with "all predefined address pools have been fully subnetted". The
  validator seeds `default-address-pools` into the pod's /etc/docker/daemon.json before its first
  start.
- The inner image store lives on sysbox's per-container mount, so a Lium reboot or an edit (both
  re-create the container) comes back with no images, no inner containers and no inner volumes.
  A per-pod named volume at /var/lib/docker keeps them until the pod is deleted.
- An encrypted pod's /root is a gocryptfs FUSE mount, which sysbox cannot propagate into an inner
  container: `docker run -v /root/x:/x` fails with "error mounting ... change mount propagation
  through procfd". /workspace is a plain path that bind-mounts fine; a per-pod volume there keeps it
  across reboots and edits, so a compose project can live somewhere its bind mounts work.

Both per-pod volumes are plain local volumes: on an encrypted pod their contents are plaintext on
the host disk until the pod is deleted (docs/lium-io/rental-dind.md).
"""

from __future__ import annotations

import ipaddress
import json
import re
import shlex
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

INNER_DAEMON_CONFIG_PATH = "/etc/docker/daemon.json"
# The image's daemon.json is renter-controlled and read into validator memory; real ones are
# a few hundred bytes.
INNER_DAEMON_CONFIG_MAX_BYTES = 64 * 1024
DEFAULT_ADDRESS_POOLS_KEY = "default-address-pools"

DIND_STORE_TARGET = "/var/lib/docker"
DIND_STORE_SUFFIX = "_docker"
DIND_WORKSPACE_TARGET = "/workspace"
DIND_WORKSPACE_SUFFIX = "_workspace"
# In the store volume: the `dockerd --version` line of the last pod dockerd that used the store.
DIND_STORE_VERSION_MARKER = ".lium-dockerd-version"
_POD_VOLUME_PREFIX = "volume_"
_COMPANION_VOLUME_RE = re.compile(
    rf"({_POD_VOLUME_PREFIX}[0-9a-f]{{8}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{4}}-[0-9a-f]{{12}})"
    rf"(?:{DIND_STORE_SUFFIX}|{DIND_WORKSPACE_SUFFIX})"
)

# Ranges an inner network must never shadow inside the pod: the host daemon's own pools and bridge
# (Docker's defaults, the sysbox installer's 172.20/172.25, neurons/executor/daemon.json's 172.24
# and 172.31, all inside 172.16/12), Docker's 192.168 tail, and a cluster pod's WireGuard overlay.
# A fixed list, checked when the setting is parsed. At create, the pools are also checked against
# the subnets of the pod's own network as the host daemon reports them (pool_network_conflicts),
# because a provider can give the host daemon other pools.
RESERVED_POD_RANGES: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("10.42.0.0/24"),
)
# A /29 still leaves 5 container addresses after the network, broadcast and gateway.
_MAX_POOL_SUBNET_PREFIX = 29


@dataclass(frozen=True, slots=True)
class AddressPool:
    base: ipaddress.IPv4Network
    size: int

    def as_daemon_json(self) -> dict[str, object]:
        return {"base": str(self.base), "size": self.size}

    @property
    def network_count(self) -> int:
        return 2 ** (self.size - self.base.prefixlen)


def parse_address_pools(raw: str) -> tuple[AddressPool, ...]:
    """The pools from their JSON form (dockerd's own `default-address-pools` shape).

    Raises ValueError on anything dockerd would refuse or that would shadow a range the pod needs.
    """
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"address pools are not JSON: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError("address pools must be a non-empty JSON list")
    pools: list[AddressPool] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"base", "size"}:
            raise ValueError(f"address pool {entry!r} must have exactly 'base' and 'size'")
        base, size = entry["base"], entry["size"]
        if not isinstance(base, str) or not isinstance(size, int) or isinstance(size, bool):
            raise ValueError(f"address pool {entry!r}: base must be a string, size an integer")
        try:
            network = ipaddress.IPv4Network(base)
        except ValueError as exc:
            raise ValueError(f"address pool {entry!r}: {exc}") from exc
        if not network.prefixlen <= size <= _MAX_POOL_SUBNET_PREFIX:
            raise ValueError(
                f"address pool {entry!r}: size must be between /{network.prefixlen}"
                f" and /{_MAX_POOL_SUBNET_PREFIX}"
            )
        for reserved in RESERVED_POD_RANGES:
            if network.overlaps(reserved):
                raise ValueError(
                    f"address pool {entry!r} overlaps {reserved}, which the pod itself uses"
                )
        for other in pools:
            if network.overlaps(other.base):
                raise ValueError(f"address pool {entry!r} overlaps pool {other.base}")
        pools.append(AddressPool(base=network, size=size))
    return tuple(pools)


def merge_inner_daemon_config(existing: bytes | None, pools: Sequence[AddressPool]) -> bytes | None:
    """The pod's daemon.json with the pools added, or None when it must be left as it is.

    Left alone: a file that already names its own pools (the image or the renter chose), and one
    that is not a JSON object (dockerd would refuse it anyway; rewriting it would hide that). Every
    other key the image ships — the nvidia runtime, a cluster pod's default runtime — is kept.
    """
    if not pools:
        return None
    if existing is None or not existing.strip():
        config: dict = {}
    else:
        try:
            config = json.loads(existing)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(config, dict):
            return None
    if DEFAULT_ADDRESS_POOLS_KEY in config:
        return None
    config[DEFAULT_ADDRESS_POOLS_KEY] = [pool.as_daemon_json() for pool in pools]
    return (json.dumps(config, indent=4) + "\n").encode()


def dind_store_volume_name(local_volume: str) -> str:
    return f"{local_volume}{DIND_STORE_SUFFIX}"


def dind_workspace_volume_name(local_volume: str) -> str:
    return f"{local_volume}{DIND_WORKSPACE_SUFFIX}"


def dind_companion_volume_names(local_volume: str) -> tuple[str, ...]:
    """Every per-pod volume this module may have attached next to `local_volume`.

    Every removal names all of them whatever the flags are now: a pod created while a flag was on
    still owns its volumes after the flag goes off.
    """
    return (dind_store_volume_name(local_volume), dind_workspace_volume_name(local_volume))


def has_dind_companion_volumes(volume_names: Iterable[str]) -> bool:
    """Whether any of `volume_names` is a pod's store or /workspace volume."""
    return any(dind_base_volume_name(name) is not None for name in volume_names)


def with_dind_companion_volumes(volume_names: Iterable[str]) -> list[str]:
    """`volume_names` followed by their companions, for a teardown that removes a pod's volumes."""
    names = [name for name in volume_names if name]
    return names + [companion for name in names for companion in dind_companion_volume_names(name)]


def dind_base_volume_name(volume_name: str) -> str | None:
    """The pod volume a companion volume belongs to; None for any other volume.

    Only `volume_<pod uuid>_docker` / `volume_<pod uuid>_workspace` count, so no pod volume (and no
    other volume) is ever read as a companion. A companion of a non-uuid pod volume is removed with
    its pod but never swept as an orphan.
    """
    match = _COMPANION_VOLUME_RE.fullmatch(volume_name)
    return None if match is None else match.group(1)


def orphaned_dind_companion_volumes(
    volume_names: Iterable[str],
    *,
    unreferenced: Iterable[str],
    protected: Iterable[str] = (),
    removing: Iterable[str] = (),
) -> list[str]:
    """Companion volumes whose pod is gone, sorted.

    A companion is orphaned when no container (running or stopped) references it, neither it nor
    its pod volume is protected (the backend's active volumes, the pod being created), and its pod
    volume is no longer on the host or is being removed in the same pass. A companion whose pod
    volume is still there is left with it: whatever keeps that volume keeps its companions.
    """
    names = set(volume_names)
    unreferenced_set = set(unreferenced)
    protected_set = set(protected)
    removing_set = set(removing)
    orphans = []
    for name in sorted(names):
        base = dind_base_volume_name(name)
        if base is None or name not in unreferenced_set:
            continue
        if name in protected_set or base in protected_set:
            continue
        if base in names and base not in removing_set:
            continue
        orphans.append(name)
    return orphans


def pool_network_conflicts(pools: Sequence[AddressPool], subnets: Iterable[str]) -> list[str]:
    """Each pool/subnet overlap between the pools and the pod network's IPv4 subnets."""
    conflicts = []
    for raw in subnets:
        try:
            subnet = ipaddress.ip_network(raw, strict=False)
        except ValueError:
            continue
        if subnet.version != 4:
            continue
        conflicts.extend(
            f"{pool.base} overlaps {subnet}" for pool in pools if pool.base.overlaps(subnet)
        )
    return conflicts


# The marker and the image's `dockerd --version` are both renter-controlled text: every read is cut
# to this many bytes on the host, and only a strict version line is ever acted on.
DIND_VERSION_MAX_BYTES = 256
_DOCKERD_VERSION_RE = re.compile(
    r"Docker version (\d{1,4})\.(\d{1,4})\.(\d{1,4})(?:[-+~][0-9A-Za-z.+~_-]{1,64})?"
    r"(?:, build [0-9A-Za-z.+~_-]{1,64})?"
)

# Helper containers of the store check: named per pod, labeled for the stale sweep, limited.
DIND_PROBE_CONTAINER_PREFIX = "lium-dind-probe-"
DIND_PROBE_LABEL = "io.lium.purpose=dind-store-probe"
DIND_PROBE_RESOURCE_FLAGS = "--memory 128m --memory-swap 128m --cpus 0.5 --pids-limit 32"
# The image's dockerd gets this long to print its version before it is killed and removed.
DIND_PROBE_DOCKERD_DEADLINE_SEC = 20
# The marker read (our helper image, a regular file, 256 bytes) is killed inside the helper after this.
DIND_PROBE_MARKER_DEADLINE_SEC = 10
DIND_STORE_RESET_DEADLINE_SEC = 300
# A probe container older than this is left over from a validator that lost its SSH session.
DIND_PROBE_STALE_AFTER_SEC = 600


def parse_dockerd_version(text: str | None) -> tuple[int, int, int] | None:
    """(major, minor, patch) from a `dockerd --version` line ("Docker version 27.3.1, build …").

    Anything but that exact line, in at most DIND_VERSION_MAX_BYTES, is unknown (None).
    """
    if not text or len(text) > DIND_VERSION_MAX_BYTES:
        return None
    match = _DOCKERD_VERSION_RE.fullmatch(text.strip())
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def format_dockerd_version(version: tuple[int, int, int] | None) -> str | None:
    return None if version is None else ".".join(str(part) for part in version)


def is_dind_store_downgrade(recorded: str | None, current: str | None) -> bool:
    """The pod's dockerd is older than the one that last wrote its store; unknown is not older."""
    recorded_version = parse_dockerd_version(recorded)
    current_version = parse_dockerd_version(current)
    return (
        recorded_version is not None
        and current_version is not None
        and current_version < recorded_version
    )


def dind_probe_container_names(container_name: str) -> tuple[str, str, str]:
    """(marker read, image dockerd, store reset) helper names for one pod."""
    base = f"{DIND_PROBE_CONTAINER_PREFIX}{container_name}"
    return f"{base}-marker", f"{base}-dockerd", f"{base}-reset"


def dind_store_version_probe_command(
    *, store_volume: str, image: str, runtime: str | None, helper_image: str, container_name: str
) -> str:
    """Prints `recorded=<marker>` and, when a marker exists, `current=<the image's dockerd --version>`.

    A store volume that does not exist yet prints nothing. Each value is at most
    DIND_VERSION_MAX_BYTES printable bytes. The marker is read only if it is a regular file, by our
    helper, with a deadline inside it. The image's dockerd runs detached under the pod's own runtime,
    named, labeled and limited, with no network and no mounts; it gets
    DIND_PROBE_DOCKERD_DEADLINE_SEC to exit, then its first bytes of log are read and it is removed
    whether it exited or not.
    """
    store = shlex.quote(store_volume)
    marker_name, dockerd_name, _ = (
        shlex.quote(name) for name in dind_probe_container_names(container_name)
    )
    runtime_flag = f" --runtime {shlex.quote(runtime)}" if runtime else ""
    common = f"--network none --label {DIND_PROBE_LABEL} {DIND_PROBE_RESOURCE_FLAGS}"
    marker = f"/store/{DIND_STORE_VERSION_MARKER}"
    read_marker = (
        f'm={marker}; [ -f "$m" ] && [ ! -h "$m" ] || exit 0; '
        f'timeout {DIND_PROBE_MARKER_DEADLINE_SEC} head -c {DIND_VERSION_MAX_BYTES} "$m"'
    )
    cut = f"head -c {DIND_VERSION_MAX_BYTES} | head -n 1 | tr -cd '[:print:]'"
    return (
        f"/usr/bin/docker volume inspect {store} >/dev/null 2>&1 || exit 0; "
        f"/usr/bin/docker rm -f {marker_name} {dockerd_name} >/dev/null 2>&1; "
        f"recorded=$(/usr/bin/docker run --rm --name {marker_name} {common} -v {store}:/store:ro "
        f"{helper_image} sh -c {shlex.quote(read_marker)} 2>/dev/null | {cut}); "
        'printf "recorded=%s\\n" "$recorded"; [ -n "$recorded" ] || exit 0; '
        f"current=; if /usr/bin/docker run -d --name {dockerd_name} {common}"
        " --log-driver json-file --log-opt max-size=64k --log-opt max-file=1"
        f"{runtime_flag} --entrypoint dockerd {shlex.quote(image)} --version >/dev/null 2>&1; then "
        f"i=0; while [ $i -lt {DIND_PROBE_DOCKERD_DEADLINE_SEC} ] && "
        f"[ \"$(/usr/bin/docker inspect -f '{{{{.State.Running}}}}' {dockerd_name} 2>/dev/null)\" = true ]; "
        "do sleep 1; i=$((i+1)); done; "
        f"current=$(/usr/bin/docker logs {dockerd_name} 2>&1 | {cut}); fi; "
        f"/usr/bin/docker rm -f {dockerd_name} >/dev/null 2>&1; "
        'printf "current=%s\\n" "$current"'
    )


def dind_probe_cleanup_command(container_name: str) -> str:
    """Remove every helper container of one pod's store check, whatever state it is in."""
    names = " ".join(shlex.quote(name) for name in dind_probe_container_names(container_name))
    return f"/usr/bin/docker rm -f {names} >/dev/null 2>&1 || true"


def parse_dind_store_version_probe(stdout: str | None) -> tuple[str | None, str | None]:
    """(recorded, current) from dind_store_version_probe_command's output; a missing line is None.

    Reads at most the first 4 * DIND_VERSION_MAX_BYTES characters, and each value is cut to
    DIND_VERSION_MAX_BYTES, whatever the host sent.
    """
    values: dict[str, str] = {}
    for line in (stdout or "")[: 4 * DIND_VERSION_MAX_BYTES].splitlines():
        key, sep, value = line.partition("=")
        value = value[:DIND_VERSION_MAX_BYTES].strip()
        if sep and key in ("recorded", "current") and value:
            values[key] = value
    return values.get("recorded"), values.get("current")


_DOCKER_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")


def dind_store_reset_command(*, store_volume: str, helper_image: str, container_name: str) -> str:
    """Create, stopped, the helper that empties the store volume, keeping the volume itself (a parked
    container may still name it); prints the helper's container id. Nothing is deleted until
    dind_store_reset_start_command starts that id, so this step is safe to lose or repeat.

    The version marker goes last, and only once everything else is gone: a reset that fails keeps the
    newer version recorded, so the next create sees the downgrade and resets again. A reset whose status
    is lost may or may not have got that far; dind_store_reset_settle_command finds out which.
    """
    _, _, reset_name = dind_probe_container_names(container_name)
    marker = shlex.quote(DIND_STORE_VERSION_MARKER)
    script = (
        f"timeout {DIND_STORE_RESET_DEADLINE_SEC} find /store -mindepth 1 -maxdepth 1 ! -name {marker} "
        f"-exec rm -rf {{}} + && rm -f /store/{marker}"
    )
    return (
        f"/usr/bin/docker rm -f {shlex.quote(reset_name)} >/dev/null 2>&1; "
        f"/usr/bin/docker create --rm --name {shlex.quote(reset_name)} --network none "
        f"--label {DIND_PROBE_LABEL} {DIND_PROBE_RESOURCE_FLAGS} "
        f"-v {shlex.quote(store_volume)}:/store {helper_image} sh -c {shlex.quote(script)} 2>/dev/null"
    )


def parse_dind_reset_helper_id(stdout: str | None) -> str | None:
    """The container id dind_store_reset_command printed; None for anything but one full id."""
    helper_id = (stdout or "")[:256].strip()
    return helper_id if _DOCKER_CONTAINER_ID_RE.fullmatch(helper_id) else None


def dind_store_reset_start_command(helper_id: str) -> str:
    """Run the created reset helper by its id and wait for it; the exit status is the reset's.

    Started by id, so a start that reaches the host after settlement removed the helper finds no
    container and deletes nothing, even when a later create reused the helper's name.
    """
    if not _DOCKER_CONTAINER_ID_RE.fullmatch(helper_id):
        raise ValueError(f"not a container id: {helper_id[:80]!r}")
    return f"/usr/bin/docker start -a {helper_id} >/dev/null 2>&1"


def dind_store_reset_settle_command(*, store_volume: str, helper_image: str, container_name: str) -> str:
    """After a reset whose status never came back: stop its helper, then read the marker again.

    Prints `stopped=1` only once a successful container listing shows no reset helper, then
    `recorded=<marker>` (empty when the reset got as far as deleting it, which it does last). A failed
    listing, like any other failure, prints no `stopped=1`: the reset may still be deleting the store.
    """
    _, _, raw_reset_name = dind_probe_container_names(container_name)
    reset_filter = shlex.quote(f"name=^/?{re.escape(raw_reset_name)}$")
    marker_name, _, reset_name = (shlex.quote(name) for name in dind_probe_container_names(container_name))
    marker = f"/store/{DIND_STORE_VERSION_MARKER}"
    # the helper's first line proves it ran, so a failed read is never taken for a deleted marker
    read_marker = (
        f'echo read; m={marker}; [ -f "$m" ] && [ ! -h "$m" ] || exit 0; '
        f'timeout {DIND_PROBE_MARKER_DEADLINE_SEC} head -c {DIND_VERSION_MAX_BYTES} "$m"'
    )
    cut = f"head -c {DIND_VERSION_MAX_BYTES} | head -n 1 | tr -cd '[:print:]'"
    return (
        f"/usr/bin/docker rm -f {reset_name} {marker_name} >/dev/null 2>&1; "
        f"left=$(/usr/bin/docker ps -aq --filter {reset_filter} 2>/dev/null) || exit 1; "
        '[ -z "$left" ] || exit 1; '
        f"out=$(/usr/bin/docker run --rm --name {marker_name} --network none --label {DIND_PROBE_LABEL} "
        f"{DIND_PROBE_RESOURCE_FLAGS} -v {shlex.quote(store_volume)}:/store:ro {helper_image} "
        f"sh -c {shlex.quote(read_marker)} 2>/dev/null) || exit 1; "
        '[ "$(printf "%s\\n" "$out" | head -n 1)" = read ] || exit 1; '
        f'recorded=$(printf "%s\\n" "$out" | sed -n 2p | {cut}); '
        'printf "stopped=1\\nrecorded=%s\\n" "$recorded"'
    )


def dind_store_version_record_command(container_name: str) -> str:
    """Record the pod's dockerd version in its store, from inside the pod; an image without dockerd
    records nothing and succeeds. The pod's output is discarded; only docker exec's status counts. The host's
    `timeout` bounds it, so a dockerd that never answers costs DIND_PROBE_DOCKERD_DEADLINE_SEC, not the SSH timeout."""
    marker = f"{DIND_STORE_TARGET}/{DIND_STORE_VERSION_MARKER}"
    script = (
        "command -v dockerd >/dev/null 2>&1 || exit 0; "
        f"v=$(dockerd --version 2>/dev/null | head -c {DIND_VERSION_MAX_BYTES} | head -n 1) && "
        f'[ -n "$v" ] && printf "%s\\n" "$v" > {marker}'
    )
    return (
        f"timeout {DIND_PROBE_DOCKERD_DEADLINE_SEC} /usr/bin/docker exec {shlex.quote(container_name)} "
        f"sh -c {shlex.quote(script)} >/dev/null 2>&1"
    )


_DOCKER_CREATED_RE = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.\d+)?Z")


def stale_dind_probe_containers(listing: str | None) -> list[str]:
    """Probe container ids created more than DIND_PROBE_STALE_AFTER_SEC before the host's now.

    `listing` is stale_dind_probe_list_command's output: the host's epoch, then
    `<id> <name> <created RFC 3339 UTC>` per container. A line that does not parse, or a name
    without the probe prefix, is left alone; without the host's epoch nothing is stale.
    """
    lines = (listing or "").splitlines()
    try:
        now = int(lines[0].strip())
    except (IndexError, ValueError):
        return []
    stale = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 3 or not parts[1].lstrip("/").startswith(DIND_PROBE_CONTAINER_PREFIX):
            continue
        match = _DOCKER_CREATED_RE.fullmatch(parts[2])
        if match is None:
            continue
        created = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        if now - created.timestamp() > DIND_PROBE_STALE_AFTER_SEC:
            stale.append(parts[0])
    return stale


def stale_dind_probe_list_command() -> str:
    """The host's epoch on the first line, then `<id> <name> <created>` per probe container."""
    return (
        f"date +%s; /usr/bin/docker ps -aq --filter label={DIND_PROBE_LABEL} "
        "| xargs -r /usr/bin/docker inspect -f '{{.Id}} {{.Name}} {{.Created}}' 2>/dev/null"
    )
