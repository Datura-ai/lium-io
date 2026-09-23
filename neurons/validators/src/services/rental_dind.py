"""Docker-in-Docker defaults for a rented sysbox pod (DAH-3796).

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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

INNER_DAEMON_CONFIG_PATH = "/etc/docker/daemon.json"
DEFAULT_ADDRESS_POOLS_KEY = "default-address-pools"

DIND_STORE_TARGET = "/var/lib/docker"
DIND_STORE_SUFFIX = "_docker"
DIND_WORKSPACE_TARGET = "/workspace"
DIND_WORKSPACE_SUFFIX = "_workspace"
_POD_VOLUME_PREFIX = "volume_"

# Ranges an inner network must never shadow inside the pod: the host daemon's own pools and bridge
# (Docker's defaults, the sysbox installer's 172.20/172.25, neurons/executor/daemon.json's 172.24
# and 172.31, all inside 172.16/12), Docker's 192.168 tail, and a cluster pod's WireGuard overlay.
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

    Teardown removes all of them whatever the flags are now: a pod created while a flag was on
    still owns its volumes after the flag goes off.
    """
    return (dind_store_volume_name(local_volume), dind_workspace_volume_name(local_volume))


def with_dind_companion_volumes(volume_names: Iterable[str]) -> list[str]:
    """`volume_names` followed by their companions, for a teardown that removes a pod's volumes."""
    names = [name for name in volume_names if name]
    return names + [companion for name in names for companion in dind_companion_volume_names(name)]


def dind_base_volume_name(volume_name: str) -> str | None:
    """The pod volume a companion volume belongs to; None for any other volume."""
    if not volume_name.startswith(_POD_VOLUME_PREFIX):
        return None
    for suffix in (DIND_STORE_SUFFIX, DIND_WORKSPACE_SUFFIX):
        base = volume_name.removesuffix(suffix)
        if base != volume_name and len(base) > len(_POD_VOLUME_PREFIX):
            return base
    return None


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
