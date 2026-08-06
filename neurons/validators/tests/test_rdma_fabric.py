"""DAH-2602: the container gets a function on the fabric whose identity the NIC enforces."""
from __future__ import annotations

import pytest

from services import rdma_fabric
from services.docker_service import DockerService
from services.rental_docker_sdk import DeviceMount
from services.rdma_fabric import (
    HostFabricConfig,
    VirtualFunction,
    attach_container_to_rdma_fabric,
    build_attachment,
)


class _StubSshResult:
    def __init__(self, stdout: str, exit_status: int = 0) -> None:
        self.stdout: str = stdout
        self.stderr: str = ""
        self.exit_status: int = exit_status


class _StubSsh:
    """Answers by substring match on the command, and records everything it was asked to run."""

    def __init__(self, responses: dict[str, _StubSshResult]) -> None:
        self._responses: dict[str, _StubSshResult] = responses
        self.commands: list[str] = []

    async def run(self, command: str) -> _StubSshResult:
        self.commands.append(command)
        for fragment, response in self._responses.items():
            if fragment in command:
                return response
        return _StubSshResult("")


_HOST_CONFIG_JSON = '{"container_ip_range": "38.255.28.192/27", "vlan_id": 4021}'
_ONE_FREE_FUNCTION = "enp1s0f0 3 enp1s0f0v3 mlx5_7\n"


def _prepared_host(**overrides: _StubSshResult) -> _StubSsh:
    responses: dict[str, _StubSshResult] = {
        rdma_fabric.HOST_FABRIC_CONFIG_PATH: _StubSshResult(_HOST_CONFIG_JSON),
        "rdma system show": _StubSshResult("netns exclusive\n"),
        "/sys/class/infiniband/*": _StubSshResult(_ONE_FREE_FUNCTION),
        "docker inspect": _StubSshResult("48211\n"),
    }
    responses.update(overrides)
    return _StubSsh(responses)


@pytest.mark.asyncio
async def test_shared_namespace_mode_refuses_to_attach() -> None:
    ssh = _prepared_host(**{"rdma system show": _StubSshResult("netns shared\n")})

    assert await attach_container_to_rdma_fabric(ssh, "pod_test") is None
    assert not any("ip link set" in command for command in ssh.commands)


@pytest.mark.asyncio
async def test_unconfigured_host_attaches_nothing_and_probes_nothing_further() -> None:
    ssh = _prepared_host(**{rdma_fabric.HOST_FABRIC_CONFIG_PATH: _StubSshResult("")})

    assert await attach_container_to_rdma_fabric(ssh, "pod_test") is None
    assert len(ssh.commands) == 1


@pytest.mark.asyncio
async def test_no_free_virtual_function_leaves_the_container_alone() -> None:
    ssh = _prepared_host(**{"/sys/class/infiniband/*": _StubSshResult("")})

    assert await attach_container_to_rdma_fabric(ssh, "pod_test") is None


@pytest.mark.asyncio
async def test_identity_is_programmed_from_the_physical_function_with_spoofchk_and_no_trust() -> None:
    ssh = _prepared_host()

    attachment = await attach_container_to_rdma_fabric(ssh, "pod_test")

    assert attachment is not None
    reserve = next(c for c in ssh.commands if "vf 3 mac" in c)
    assert reserve.startswith("ip link set enp1s0f0 vf 3 mac 02:")
    # Hardware-enforced identity is the whole point: without these the tenant's NET_ADMIN wins.
    assert "spoofchk on" in reserve
    assert "trust off" in reserve
    assert "vlan 4021" in reserve


@pytest.mark.asyncio
async def test_rdma_device_moves_after_the_address_exists_in_the_namespace() -> None:
    ssh = _prepared_host()

    await attach_container_to_rdma_fabric(ssh, "pod_test")

    address_step = next(i for i, c in enumerate(ssh.commands) if "ip addr add" in c)
    rdma_step = next(i for i, c in enumerate(ssh.commands) if c.startswith("rdma dev set"))
    # The GID table is built from the addresses present when the device lands.
    assert address_step < rdma_step
    assert ssh.commands[rdma_step] == "rdma dev set mlx5_7 netns 48211"


@pytest.mark.asyncio
async def test_the_function_lands_in_the_containers_namespace() -> None:
    ssh = _prepared_host()

    await attach_container_to_rdma_fabric(ssh, "pod_test")

    assert "ip link set enp1s0f0v3 netns 48211" in ssh.commands
    assert "nsenter -t 48211 -n ip addr add 38.255.28.196/27 dev fabric0" in ssh.commands


@pytest.mark.asyncio
async def test_the_interface_is_renamed_to_the_documented_constant_before_it_is_addressed() -> None:
    ssh = _prepared_host()

    await attach_container_to_rdma_fabric(ssh, "pod_test")

    rename = "nsenter -t 48211 -n ip link set enp1s0f0v3 name fabric0"
    assert rename in ssh.commands
    # Customers reach it as NCCL_SOCKET_IFNAME=fabric0; the host-side driver name never leaks out.
    assert ssh.commands.index(rename) < next(
        i for i, c in enumerate(ssh.commands) if "ip addr add" in c
    )


@pytest.mark.asyncio
async def test_a_container_without_a_namespace_is_an_error_not_a_silent_skip() -> None:
    ssh = _prepared_host(**{"docker inspect": _StubSshResult("0\n")})

    with pytest.raises(RuntimeError, match="no namespace to attach to"):
        await attach_container_to_rdma_fabric(ssh, "pod_test")


@pytest.mark.asyncio
async def test_a_failed_step_raises_rather_than_leaving_a_half_attached_function() -> None:
    ssh = _prepared_host(**{"rdma dev set": _StubSshResult("", exit_status=1)})

    with pytest.raises(RuntimeError, match="failed to attach"):
        await attach_container_to_rdma_fabric(ssh, "pod_test")


def test_address_is_derived_from_the_function_index_so_two_containers_cannot_collide() -> None:
    host_config = HostFabricConfig(container_ip_range="38.255.28.192/27", vlan_id=None)

    first = build_attachment(
        VirtualFunction("enp1s0f0", 0, "enp1s0f0v0", "mlx5_4"), host_config, "pod_a"
    )
    second = build_attachment(
        VirtualFunction("enp1s0f0", 1, "enp1s0f0v1", "mlx5_5"), host_config, "pod_b"
    )

    assert first is not None and second is not None
    assert first.ipv4_cidr == "38.255.28.193/27"
    assert second.ipv4_cidr == "38.255.28.194/27"
    assert first.mac_address != second.mac_address


def test_mac_is_stable_across_rebuilds_of_the_same_container() -> None:
    host_config = HostFabricConfig(container_ip_range="38.255.28.192/27", vlan_id=None)
    virtual_function = VirtualFunction("enp1s0f0", 0, "enp1s0f0v0", "mlx5_4")

    first = build_attachment(virtual_function, host_config, "pod_test")
    rebuilt = build_attachment(virtual_function, host_config, "pod_test")

    assert first is not None and rebuilt is not None
    assert first.mac_address == rebuilt.mac_address
    # Locally administered unicast, so it cannot collide with a vendor-assigned address.
    assert first.mac_address.startswith("02:")


def test_a_range_too_small_for_the_function_index_refuses_rather_than_wrapping() -> None:
    host_config = HostFabricConfig(container_ip_range="38.255.28.192/30", vlan_id=None)

    attachment = build_attachment(
        VirtualFunction("enp1s0f0", 7, "enp1s0f0v7", "mlx5_9"), host_config, "pod_test"
    )

    assert attachment is None


@pytest.mark.asyncio
async def test_unparsable_host_config_is_treated_as_unconfigured() -> None:
    ssh = _prepared_host(
        **{rdma_fabric.HOST_FABRIC_CONFIG_PATH: _StubSshResult("{not json")}
    )

    assert await attach_container_to_rdma_fabric(ssh, "pod_test") is None


# --- The guard: an ordinary rental must not notice this feature exists at all ---


def _device(path_on_host: str) -> DeviceMount:
    return DeviceMount(path_on_host=path_on_host, path_in_container=path_on_host)


_ORDINARY_RENTAL_DEVICES = (
    _device("/dev/net/tun"),
    _device("/dev/nvidia0"),
    _device("/dev/nvidiactl"),
    _device("/dev/nvidia-uvm"),
)


@pytest.mark.asyncio
async def test_a_rental_without_rdma_devices_never_touches_the_host() -> None:
    ssh = _prepared_host()

    await DockerService._attach_rdma_fabric_if_available(
        ssh_client=ssh,
        container_name="pod_ordinary",
        forwarded_devices=_ORDINARY_RENTAL_DEVICES,
        default_extra={},
    )

    # Not "did nothing harmful" — did nothing at all, on the box of every rental that has no card.
    assert ssh.commands == []


@pytest.mark.asyncio
async def test_an_unprepared_host_with_rdma_devices_is_left_exactly_as_it_was() -> None:
    ssh = _prepared_host(**{rdma_fabric.HOST_FABRIC_CONFIG_PATH: _StubSshResult("")})

    await DockerService._attach_rdma_fabric_if_available(
        ssh_client=ssh,
        container_name="pod_rdma",
        forwarded_devices=(*_ORDINARY_RENTAL_DEVICES, _device("/dev/infiniband/uverbs0")),
        default_extra={},
    )

    assert not any("ip link set" in command for command in ssh.commands)


@pytest.mark.asyncio
async def test_an_attach_failure_does_not_fail_the_rental() -> None:
    ssh = _prepared_host(**{"rdma dev set": _StubSshResult("", exit_status=1)})

    # The pod is already created and running by this point; a fabric it cannot reach is a
    # degraded rental, not a failed one.
    await DockerService._attach_rdma_fabric_if_available(
        ssh_client=ssh,
        container_name="pod_rdma",
        forwarded_devices=(*_ORDINARY_RENTAL_DEVICES, _device("/dev/infiniband/uverbs0")),
        default_extra={},
    )


@pytest.mark.asyncio
async def test_a_non_integer_vlan_tag_is_refused_rather_than_reaching_the_shell() -> None:
    ssh = _prepared_host(
        **{
            rdma_fabric.HOST_FABRIC_CONFIG_PATH: _StubSshResult(
                '{"container_ip_range": "38.255.28.192/27", "vlan_id": "4021 ; reboot"}'
            )
        }
    )

    assert await attach_container_to_rdma_fabric(ssh, "pod_test") is None


def test_a_huge_host_range_costs_nothing_to_index() -> None:
    host_config = HostFabricConfig(container_ip_range="10.0.0.0/8", vlan_id=None)

    attachment = build_attachment(
        VirtualFunction("enp1s0f0", 2, "enp1s0f0v2", "mlx5_6"), host_config, "pod_test"
    )

    # Enumerating a /8 would allocate ~16.7M addresses inside the validator, per pod.
    assert attachment is not None
    assert attachment.ipv4_cidr == "10.0.0.3/8"
