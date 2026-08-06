"""DAH-2602: put a rental container on the RDMA fabric without trusting anything inside it.

Forwarding the verbs devices (DAH-2571) lets a container talk to the card, but not across hosts:
the container's only address is the NAT bridge's, while the card's GID carries the HOST's address,
so RoCE sends from an address that does not exist in the container's namespace. Measured on prod
on 2026-08-06 — handshake completes over the mapped TCP port, zero RDMA bytes move.

The fix is to give the container its own function on the fabric, and to make every property of it
enforced somewhere the tenant cannot reach. The tenant is root in its container and holds
NET_ADMIN, so nothing enforced by the container's own kernel namespace counts:

- an SR-IOV **virtual function** carries a MAC, a VLAN and a rate limit programmed from the
  physical function. `spoofchk on` makes the NIC drop frames that do not match, and `trust off`
  denies promiscuous mode and MAC changes. Enforcement is in hardware, so root in the container
  cannot spoof a neighbour on the segment.
- **exclusive RDMA namespace mode** scopes RDMA devices to network namespaces. The VF's RDMA
  device is moved into the container's namespace, where it is the only device visible and its GID
  table holds only addresses that exist there. The source-address problem disappears rather than
  being patched, and a stolen verbs handle shows nothing outside the namespace.

Neither is configured by us at rental time: exclusive mode is host-wide and disruptive to flip
while RDMA is in use, and only whoever owns the fabric knows which addresses on it are free. Both
are read as preconditions, and the attachment is skipped when they are absent.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import shlex
from dataclasses import dataclass

import asyncssh

logger = logging.getLogger(__name__)

# Written by whoever provisions the box: they own the fabric and know which addresses on it are
# free. A validator-wide setting cannot know that — the segment is shared between hosts.
HOST_FABRIC_CONFIG_PATH = "/etc/lium/rdma-fabric.json"

# What the fabric interface is called inside every container. VF netdev names vary per host and
# driver (enp1s0f0v3, eth4, ...), and NCCL/gloo do not pick a second interface on their own —
# customers point them at it with NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME, so the name has to be
# one documented constant, not something to discover per pod.
CONTAINER_FABRIC_NETDEV_NAME = "fabric0"


@dataclass(frozen=True, slots=True)
class HostFabricConfig:
    """The slice of the fabric segment this host may hand to containers."""

    container_ip_range: str
    vlan_id: int | None


@dataclass(frozen=True, slots=True)
class VirtualFunction:
    """One SR-IOV function of an RDMA card, currently unclaimed."""

    physical_function_netdev: str
    index: int
    netdev: str
    rdma_device: str


@dataclass(frozen=True, slots=True)
class RdmaFabricAttachment:
    """A virtual function prepared for one container, with the identity the NIC will enforce."""

    virtual_function: VirtualFunction
    mac_address: str
    ipv4_cidr: str
    vlan_id: int | None


# A VF whose netdev has been moved into a container's namespace no longer appears under the PF's
# sysfs tree when read from the host, so this lists exactly the unclaimed functions.
_LIST_FREE_VIRTUAL_FUNCTIONS = (
    "for rdma_dir in /sys/class/infiniband/*; do "
    '[ -d "$rdma_dir" ] || continue; '
    'for pf_dir in "$rdma_dir"/device/net/*; do '
    '[ -d "$pf_dir" ] || continue; '
    'pf=$(basename "$pf_dir"); '
    'for vf_dir in /sys/class/net/"$pf"/device/virtfn*; do '
    '[ -d "$vf_dir" ] || continue; '
    'index=${vf_dir##*/virtfn}; '
    'vf_netdev=$(ls "$vf_dir"/net 2>/dev/null | head -1); '
    'vf_rdma=$(ls "$vf_dir"/infiniband 2>/dev/null | head -1); '
    '[ -n "$vf_netdev" ] && [ -n "$vf_rdma" ] && '
    'printf "%s %s %s %s\\n" "$pf" "$index" "$vf_netdev" "$vf_rdma"; '
    "done; done; done"
)


async def read_host_fabric_config(
    ssh_client: asyncssh.SSHClientConnection,
) -> HostFabricConfig | None:
    """The address range this host may allocate from, or None when the host is not configured."""
    result = await ssh_client.run(f"cat {HOST_FABRIC_CONFIG_PATH} 2>/dev/null")
    if not result.stdout.strip():
        return None
    try:
        document = json.loads(result.stdout)
        container_ip_range: str = document["container_ip_range"]
        ipaddress.ip_network(container_ip_range)
    except (ValueError, KeyError, TypeError) as error:
        logger.warning(
            "rdma_fabric: unreadable host fabric config, leaving the container on the NAT bridge",
            extra={"path": HOST_FABRIC_CONFIG_PATH, "error": str(error)},
        )
        return None
    vlan_id: int | None = document.get("vlan_id")
    return HostFabricConfig(container_ip_range=container_ip_range, vlan_id=vlan_id)


async def is_rdma_namespace_mode_exclusive(ssh_client: asyncssh.SSHClientConnection) -> bool:
    """Whether the host scopes RDMA devices per network namespace.

    Shared mode — the default — lets any namespace holding a verbs node see every device on the
    box and every GID the host owns. That is the mode the current passthrough runs in, and it is
    not a namespace a tenant can be given a function inside of. Flipping it requires no RDMA
    resources to be in use, so it belongs to host provisioning, not to a rental.
    """
    result = await ssh_client.run("rdma system show 2>/dev/null")
    return "netns exclusive" in result.stdout


async def find_free_virtual_function(
    ssh_client: asyncssh.SSHClientConnection,
) -> VirtualFunction | None:
    result = await ssh_client.run(_LIST_FREE_VIRTUAL_FUNCTIONS)
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4 or not fields[1].isdigit():
            continue
        return VirtualFunction(
            physical_function_netdev=fields[0],
            index=int(fields[1]),
            netdev=fields[2],
            rdma_device=fields[3],
        )
    return None


def build_attachment(
    virtual_function: VirtualFunction,
    host_config: HostFabricConfig,
    container_name: str,
) -> RdmaFabricAttachment | None:
    """Assign the function's identity, or None when the host's range cannot cover it.

    The address is derived from the VF index rather than allocated: indexes are unique on a host,
    the range belongs to that host alone, and a derived address needs no state to survive a
    validator restart or a second container starting concurrently.
    """
    addresses = list(ipaddress.ip_network(host_config.container_ip_range).hosts())
    if virtual_function.index >= len(addresses):
        logger.warning(
            "rdma_fabric: host range is smaller than the card's function count, "
            "leaving the container on the NAT bridge",
            extra={
                "container_ip_range": host_config.container_ip_range,
                "virtual_function_index": virtual_function.index,
                "usable_addresses": len(addresses),
            },
        )
        return None
    prefix_length = ipaddress.ip_network(host_config.container_ip_range).prefixlen
    return RdmaFabricAttachment(
        virtual_function=virtual_function,
        mac_address=_locally_administered_mac_for(container_name),
        ipv4_cidr=f"{addresses[virtual_function.index]}/{prefix_length}",
        vlan_id=host_config.vlan_id,
    )


def _locally_administered_mac_for(container_name: str) -> str:
    """A stable per-container MAC in the locally-administered unicast space.

    Derived, not random, so a rebuilt container keeps the address a neighbour's ARP cache and the
    switch's MAC table already learned.
    """
    digest = hashlib.sha256(container_name.encode()).digest()
    return ":".join(["02", *(f"{byte:02x}" for byte in digest[:5])])


async def reserve_virtual_function(
    ssh_client: asyncssh.SSHClientConnection,
    attachment: RdmaFabricAttachment,
) -> None:
    """Program the function's identity from the physical function, where the tenant cannot reach.

    Every property set here is enforced by the NIC. This is what makes the container's NET_ADMIN
    harmless against the fabric: root inside can rename or reconfigure the interface all it likes,
    frames that do not match are dropped before they reach the wire.
    """
    virtual_function = attachment.virtual_function
    command = (
        f"ip link set {shlex.quote(virtual_function.physical_function_netdev)} "
        f"vf {virtual_function.index} mac {attachment.mac_address}"
    )
    if attachment.vlan_id is not None:
        command += f" vlan {attachment.vlan_id}"
    command += " spoofchk on trust off"
    result = await ssh_client.run(command)
    if result.exit_status != 0:
        raise RuntimeError(
            f"failed to program virtual function identity: {command!r} "
            f"exited {result.exit_status}, stderr={result.stderr!r}"
        )


async def move_into_container_namespace(
    ssh_client: asyncssh.SSHClientConnection,
    attachment: RdmaFabricAttachment,
    container_name: str,
) -> None:
    """Hand the function to the container: netdev, stable name, address, then the RDMA device.

    Order matters twice. The rename happens inside the namespace, where the name cannot collide
    with another rental's function. The RDMA device moves last, because its GID table is built
    from the addresses present in the namespace it lands in — move it before the address exists
    and the GIDs it publishes are empty.
    """
    quoted_container_name = shlex.quote(container_name)
    pid_result = await ssh_client.run(
        f"/usr/bin/docker inspect -f '{{{{.State.Pid}}}}' {quoted_container_name}"
    )
    container_pid = pid_result.stdout.strip()
    if not container_pid.isdigit() or container_pid == "0":
        raise RuntimeError(
            f"container {container_name} has no namespace to attach to (pid={container_pid!r})"
        )

    virtual_function = attachment.virtual_function
    host_netdev = shlex.quote(virtual_function.netdev)
    fabric_netdev = CONTAINER_FABRIC_NETDEV_NAME
    steps = (
        f"ip link set {host_netdev} netns {container_pid}",
        f"nsenter -t {container_pid} -n ip link set {host_netdev} name {fabric_netdev}",
        f"nsenter -t {container_pid} -n ip addr add {attachment.ipv4_cidr} dev {fabric_netdev}",
        f"nsenter -t {container_pid} -n ip link set {fabric_netdev} up",
        f"rdma dev set {shlex.quote(virtual_function.rdma_device)} netns {container_pid}",
    )
    for step in steps:
        result = await ssh_client.run(step)
        if result.exit_status != 0:
            raise RuntimeError(
                f"failed to attach the RDMA fabric function: {step!r} "
                f"exited {result.exit_status}, stderr={result.stderr!r}"
            )


async def attach_container_to_rdma_fabric(
    ssh_client: asyncssh.SSHClientConnection,
    container_name: str,
) -> RdmaFabricAttachment | None:
    """Give the container a function on the fabric, or None when the host is not set up for it.

    Every precondition is read, never created: the host must be in exclusive RDMA namespace mode,
    must declare an address range, and must have a free virtual function. A host missing any of
    them keeps today's behaviour — the container reaches its own cards and nothing across the wire.
    """
    host_config = await read_host_fabric_config(ssh_client)
    if not host_config:
        return None
    if not await is_rdma_namespace_mode_exclusive(ssh_client):
        logger.warning(
            "rdma_fabric: host is in shared RDMA namespace mode, refusing to attach a function "
            "(run `rdma system set netns exclusive` at provisioning time)",
            extra={"container_name": container_name},
        )
        return None
    virtual_function = await find_free_virtual_function(ssh_client)
    if not virtual_function:
        logger.warning(
            "rdma_fabric: no free SR-IOV function on this host's RDMA cards",
            extra={"container_name": container_name},
        )
        return None
    attachment = build_attachment(virtual_function, host_config, container_name)
    if not attachment:
        return None
    await reserve_virtual_function(ssh_client, attachment)
    await move_into_container_namespace(ssh_client, attachment, container_name)
    return attachment
