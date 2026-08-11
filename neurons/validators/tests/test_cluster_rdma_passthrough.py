"""DAH-2620 — a node of a cluster rental gets the verbs devices without waiting for the RoCE flag.

The overlay only carries NCCL's bootstrap; the tensors are supposed to ride InfiniBand. Without
`/dev/infiniband/uverbs*` in the container NCCL silently falls back to its socket transport and the
whole job runs over WireGuard, which is the failure the sysfs-only detection hides.
"""

import pytest

from core.config import settings
from services.docker_service import DockerService
from services.nvidia_devices import _query_shared_nodes
from services.rental_docker_sdk import DeviceMount
from tests.test_nvidia_devices import FakeRun, fake_ssh


@pytest.mark.asyncio
async def test_a_cluster_node_gets_verbs_devices_with_the_roce_flag_off(monkeypatch) -> None:
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_RDMA_DEVICE_PASSTHROUGH", False)
    ssh = fake_ssh(FakeRun(""))

    # Act
    await _query_shared_nodes(ssh, is_whole_host_rental=True, rdma_required=True)

    # Assert
    probe_command: str = ssh.run.call_args.args[0]
    assert "/dev/infiniband/uverbs[0-9]*" in probe_command
    assert "/dev/infiniband/rdma_cm" in probe_command


@pytest.mark.asyncio
async def test_a_cluster_node_still_never_gets_the_subnet_manager_devices(monkeypatch) -> None:
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_RDMA_DEVICE_PASSTHROUGH", False)
    ssh = fake_ssh(FakeRun(""))

    # Act
    await _query_shared_nodes(ssh, is_whole_host_rental=True, rdma_required=True)

    # Assert
    probe_command: str = ssh.run.call_args.args[0]
    assert "/dev/infiniband/issm" not in probe_command
    assert "/dev/infiniband/umad" not in probe_command
    assert " /dev/infiniband " not in probe_command


@pytest.mark.asyncio
async def test_a_partial_rental_gets_nothing_even_when_the_cluster_asks(monkeypatch) -> None:
    # A cluster is whole nodes by construction, so this can only be a caller mistake — and the
    # verbs devices belong to cards the other tenant on the box may be renting.
    monkeypatch.setattr(settings, "ENABLE_RDMA_DEVICE_PASSTHROUGH", False)
    ssh = fake_ssh(FakeRun(""))

    # Act
    await _query_shared_nodes(ssh, is_whole_host_rental=False, rdma_required=True)

    # Assert
    probe_command: str = ssh.run.call_args.args[0]
    assert "/dev/infiniband" not in probe_command


def test_ipc_lock_is_added_exactly_when_the_container_holds_verbs_devices() -> None:
    # Arrange
    plain: tuple[DeviceMount, ...] = (
        DeviceMount(path_on_host="/dev/net/tun", path_in_container="/dev/net/tun"),
    )
    with_rdma: tuple[DeviceMount, ...] = (
        *plain,
        DeviceMount(
            path_on_host="/dev/infiniband/uverbs0", path_in_container="/dev/infiniband/uverbs0"
        ),
    )

    # Act
    plain_caps: tuple[str, ...] = DockerService._capabilities_for(plain)
    rdma_caps: tuple[str, ...] = DockerService._capabilities_for(with_rdma)

    # Assert
    assert plain_caps == ("NET_ADMIN",)
    assert rdma_caps == ("NET_ADMIN", "IPC_LOCK")
