"""DAH-2571 — RDMA devices must reach the rental container, and only the safe ones.

A card on the host is worth nothing to a renter who cannot see it: `/dev/infiniband/*` is not
forwarded by default and RDMA cannot register memory under the stock 64 KB memlock. Proven on a
prod B300 on 2026-08-06 — under `sysbox-runc` with the devices and `memlock=-1` a container
enumerates all 24 devices and the four ACTIVE ports, with the same LIDs the host reports.
"""

import pytest

from services.nvidia_devices import (
    _query_gpu_nodes_for_uuids,
    _query_shared_nodes,
    build_gpu_docker_config_for_executor,
)
from services.rental_docker_sdk import ContainerRunSpec, ContainerUlimit, _build_host_config_kwargs
from services.docker_service import DockerService
from services.rental_docker_sdk import DeviceMount
from tests.test_nvidia_devices import FakeRun, fake_ssh


@pytest.mark.asyncio
@pytest.mark.parametrize("is_whole_host_rental", [True, False])
async def test_rdma_is_forwarded_only_on_a_whole_host_rental(is_whole_host_rental: bool) -> None:
    """The whole-host rule is the only gate left (DAH-2620). On a partial rental the verbs devices
    belong to cards the other tenant on the box may be renting."""
    # Arrange
    ssh = fake_ssh(FakeRun(""))

    # Act
    await _query_shared_nodes(ssh, is_whole_host_rental=is_whole_host_rental)

    # Assert
    probe_command = ssh.run.call_args.args[0]
    assert ("/dev/infiniband/uverbs[0-9]*" in probe_command) is is_whole_host_rental
    assert ("/dev/infiniband/rdma_cm" in probe_command) is is_whole_host_rental


@pytest.mark.asyncio
async def test_the_directory_form_is_never_used() -> None:
    """`/dev/infiniband` as a directory also hands over `issm*` (the subnet-manager interface) and
    `umad*` (raw MAD). A renter holding issm can interfere with the fabric everyone else shares."""
    # Arrange
    ssh = fake_ssh(FakeRun(""))

    # Act
    await _query_shared_nodes(ssh, is_whole_host_rental=True)

    # Assert
    probe_command = ssh.run.call_args.args[0]
    assert "/dev/infiniband/issm" not in probe_command
    assert "/dev/infiniband/umad" not in probe_command
    assert " /dev/infiniband " not in probe_command


def test_memlock_is_raised_only_for_a_container_that_got_rdma_devices() -> None:
    """A pod whose ram_total is 0 carries no memory cgroup limit either, and unlimited memlock on
    top of that would let a tenant pin the host's RAM. Raise it only where it buys something."""
    # Arrange
    rdma = (DeviceMount(path_on_host="/dev/infiniband/uverbs0", path_in_container="/dev/infiniband/uverbs0"),)
    gpu_only = (DeviceMount(path_on_host="/dev/nvidia0", path_in_container="/dev/nvidia0"),)

    # Act / Assert
    assert DockerService._memlock_ulimit_for(rdma, 64) == (ContainerUlimit(name="memlock", soft=-1, hard=-1),)
    assert DockerService._memlock_ulimit_for(gpu_only, 64) == ()
    assert DockerService._memlock_ulimit_for((), 64) == ()
    # no memory cgroup limit -> pinning would come out of the host, not the tenant's own allocation
    assert DockerService._memlock_ulimit_for(rdma, 0) == ()
    assert DockerService._memlock_ulimit_for(rdma, None) == ()


def test_memlock_reaches_the_host_config() -> None:
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


@pytest.mark.asyncio
async def test_a_whole_host_rental_carries_the_card_all_the_way_into_the_host_config() -> None:
    """The seam this feature lives or dies on: what the probe finds on the executor has to survive
    into the Docker host config, or the renter still cannot see the card.

    Everything between is plain plumbing, but it spans three modules — the probe, the run spec and
    the host-config mapping — and each was verified separately until now.
    """
    # Arrange — an executor with one GPU and one RDMA card, rented whole
    ssh = fake_ssh(
        FakeRun("/dev/nvidia0\n"),
        FakeRun("/dev/nvidiactl\n/dev/infiniband/uverbs0\n/dev/infiniband/rdma_cm\n"),
    )

    # Act
    gpu_config = await build_gpu_docker_config_for_executor(ssh, gpu_uuids=None)
    devices = tuple(gpu_config.device_mounts)
    host_config = _build_host_config_kwargs(
        ContainerRunSpec(
            image="ubuntu:24.04",
            name="pod_test",
            devices=devices,
            ulimits=DockerService._memlock_ulimit_for(devices, memory_gb=64),
            memory_gb=64,
        )
    )

    # Assert — the card reaches the container, and it can pin the memory RDMA needs
    forwarded = [entry.split(":")[0] for entry in host_config["devices"]]
    assert "/dev/infiniband/uverbs0" in forwarded
    assert "/dev/infiniband/rdma_cm" in forwarded
    assert [(u["Name"], u["Soft"]) for u in host_config["ulimits"]] == [("memlock", -1)]
    assert host_config["mem_limit"] == "64g"
