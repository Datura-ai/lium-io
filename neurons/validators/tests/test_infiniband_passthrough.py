"""DAH-2571 — RDMA devices must reach the rental container, and only the safe ones.

A card on the host is worth nothing to a renter who cannot see it: `/dev/infiniband/*` is not
forwarded by default and RDMA cannot register memory under the stock 64 KB memlock. Proven on a
prod B300 on 2026-08-06 — under `sysbox-runc` with the devices and `memlock=-1` a container
enumerates all 24 devices and the four ACTIVE ports, with the same LIDs the host reports.
"""

import inspect

import pytest

from services.nvidia_devices import _query_gpu_nodes_for_uuids, _query_shared_nodes
from services.rental_docker_sdk import ContainerRunSpec, ContainerUlimit, _build_host_config_kwargs
from tests.test_nvidia_devices import FakeRun, fake_ssh


def test_only_the_verbs_nodes_are_forwarded_never_the_whole_directory() -> None:
    """`/dev/infiniband` as a directory also hands over `issm*` (the subnet-manager interface) and
    `umad*` (raw MAD). A renter holding issm can interfere with the fabric everyone else shares."""
    # Arrange / Act
    probe_source = inspect.getsource(_query_shared_nodes)

    # Assert
    assert "/dev/infiniband/uverbs[0-9]*" in probe_source
    assert "/dev/infiniband/rdma_cm" in probe_source
    assert "/dev/infiniband/issm" not in probe_source
    assert "/dev/infiniband/umad" not in probe_source
    assert '"/dev/infiniband"' not in probe_source


def test_rdma_devices_are_skipped_on_a_partial_host_rental() -> None:
    """The cards belong to the whole host. On a split node forwarding all of them would hand one
    tenant the verbs devices of a card the other tenant is renting."""
    # Arrange / Act
    probe_source = inspect.getsource(_query_shared_nodes)
    rdma_block_offset = probe_source.index("/dev/infiniband/uverbs[0-9]*")
    whole_host_guard_offset = probe_source.index("if is_whole_host_rental:")

    # Assert — the RDMA listing sits inside the whole-host branch, not before it
    assert whole_host_guard_offset < rdma_block_offset


def test_memlock_is_unlimited_on_the_rental_container() -> None:
    """The default 64 KB memlock is far below one queue pair; without this the forwarded devices
    are present but unusable."""
    # Arrange
    spec = ContainerRunSpec(
        image="ubuntu:24.04",
        name="pod_test",
        ulimits=(ContainerUlimit(name="memlock", soft=-1, hard=-1),),
    )

    # Act
    host_config = _build_host_config_kwargs(spec)

    # Assert
    assert [(u["Name"], u["Soft"], u["Hard"]) for u in host_config["ulimits"]] == [("memlock", -1, -1)]


def test_a_spec_without_ulimits_sends_none() -> None:
    """Every other container keeps the daemon default — this must not become a global change."""
    # Arrange / Act
    host_config = _build_host_config_kwargs(ContainerRunSpec(image="ubuntu:24.04", name="pod_test"))

    # Assert
    assert "ulimits" not in host_config


@pytest.mark.asyncio
async def test_a_repeated_uuid_does_not_pass_a_single_gpu_rental_off_as_whole_host() -> None:
    """The whole-host decision compares the resolved node count against the host GPU count. Without
    dedup, one UUID sent eight times on an eight-GPU host counts as eight and the tenant gets every
    host-wide device — caps and RDMA both."""
    # Arrange — one GPU requested, repeated, on a host that has four
    ssh = fake_ssh(
        FakeRun("GPU-aaa,0\nGPU-bbb,1\nGPU-ccc,2\nGPU-ddd,3\n"),
    )

    # Act
    per_gpu, host_total = await _query_gpu_nodes_for_uuids(ssh, ["GPU-aaa"] * 4)

    # Assert
    assert per_gpu == ("/dev/nvidia0",)
    assert len(per_gpu) < host_total
