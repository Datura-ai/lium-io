"""A filler holding two host ports of the range no longer fails the rental probe.

The probe's pod hit one filler port, moved, then hit the second one and failed as `port_collision`
(only one remap was allowed). The filler's ports were published through docker's iptables, so the
host's listening sockets never showed them as taken.
"""

import time
from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import PayloadPortMapping
from services.docker_service import DockerService, _parse_listening_ports
from services.rental_docker_sdk import ContainerRunSpec, PortBinding, RentalDockerOperationError
from test_docker_service_rental_security import RecordingRentalDockerFactory, RecordingSSHClient

# what `ss` prints (sshd only) followed by what `docker ps --format '{{.Ports}}'` prints for the filler
FILLER_PUBLISHED = (
    "LISTEN 0 128  0.0.0.0:22     0.0.0.0:*\n"
    "0.0.0.0:10083->22/tcp, :::10083->22/tcp, 0.0.0.0:10071->9101/tcp, 0.0.0.0:10063->8888/tcp\n"
)


def _bind_refused(port: int) -> RentalDockerOperationError:
    return RentalDockerOperationError(
        "Docker SDK run container failed: 500 Server Error: driver failed programming external "
        f"connectivity on endpoint pod_x: Bind for 0.0.0.0:{port} failed: port is already allocated"
    )


class _RunRecorder:
    """A docker client whose `run_container` answers from a script of errors (None = success)."""

    def __init__(self, *errors: Exception | None):
        self.errors = list(errors)
        self.specs: list[ContainerRunSpec] = []

    async def run_container(self, spec: ContainerRunSpec) -> None:
        self.specs.append(spec)
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error

    async def remove_container(self, *, container_name: str, force: bool, remove_volumes: bool) -> None:
        return None


def test_container_published_host_ports_count_as_taken():
    assert _parse_listening_ports(FILLER_PUBLISHED) == {22, 10083, 10071, 10063}


def test_listening_ports_read_a_published_range():
    block = "0.0.0.0:20040-20049->20040-20049/tcp, :::20040-20049->20040-20049/tcp\n"
    assert _parse_listening_ports(block) == set(range(20040, 20050))


def test_many_wide_published_ranges_are_merged_not_expanded_each_time():
    wide = "0.0.0.0:1-65535->1-65535/tcp, :::1-65535->1-65535/tcp\n" * 2000
    started = time.monotonic()
    assert _parse_listening_ports(wide) == set(range(1, 65536))
    assert time.monotonic() - started < 2
    overlapping = "0.0.0.0:100-200->1/tcp\n0.0.0.0:150-300->1/tcp\n0.0.0.0:0-0->1/tcp\n0.0.0.0:400->1/tcp\n"
    assert _parse_listening_ports(overlapping) == set(range(100, 301)) | {400}


@pytest.mark.asyncio
async def test_two_filler_ports_move_in_one_remap_and_skip_published_ports(monkeypatch):
    monkeypatch.setattr("services.docker_service.asyncio.sleep", AsyncMock())
    service = DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=RecordingRentalDockerFactory(),
    )
    service.stream_log = AsyncMock()
    port_maps = [(22, 10083, 20083), (8888, 10071, 20071)]
    spare = [
        PayloadPortMapping(internal_port=port, external_port=port + 10000)
        for port in range(10063, 10083)
        if port != 10071
    ]
    client = _RunRecorder(_bind_refused(10083), None)

    await service._run_rental_docker_create_with_port_retry(
        docker_client=client,
        ssh_client=RecordingSSHClient(stdout=FILLER_PUBLISHED),
        run_spec=ContainerRunSpec(
            image="lium/pod:1",
            name="pod_test",
            ports=(PortBinding(container_port=22, host_port=10083), PortBinding(container_port=8888, host_port=10071)),
        ),
        container_name="pod_test",
        default_extra={},
        port_maps=port_maps,
        spare_port_pairs=spare,
    )

    # 22 wraps past 10063 (published by the filler) to 10064; 8888, published too, moves off 10071 to 10072
    assert port_maps == [(22, 10064, 20064), (8888, 10072, 20072)]
    assert len(client.specs) == 2
    assert {(b.container_port, b.host_port) for b in client.specs[-1].ports} == {(22, 10064), (8888, 10072)}


@pytest.mark.asyncio
async def test_a_block_a_departing_pod_still_holds_moves_in_one_remap(monkeypatch):
    """DAH-5797 (4 Nov 13:18, 5 Nov 03:11): a renter deleted a pod and rented again; the backend offered the
    deleted pod's 10-port block while its container was still being removed. One port per remap, 5 remaps."""
    monkeypatch.setattr("services.docker_service.asyncio.sleep", AsyncMock())
    service = DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=RecordingRentalDockerFactory(),
    )
    service.stream_log = AsyncMock()
    held = list(range(20040, 20050))
    departing = ", ".join(f"0.0.0.0:{port}->{port}/tcp" for port in held)
    port_maps = [(22, 20099, 20099)] + [(port, port, port) for port in held]
    spare = [PayloadPortMapping(internal_port=port, external_port=port) for port in range(20050, 20090)]
    client = _RunRecorder(_bind_refused(20040), None)

    await service._run_rental_docker_create_with_port_retry(
        docker_client=client,
        ssh_client=RecordingSSHClient(stdout=f"LISTEN 0 128  0.0.0.0:22     0.0.0.0:*\n{departing}\n"),
        run_spec=ContainerRunSpec(
            image="lium/pod:1",
            name="pod_test",
            ports=tuple(PortBinding(container_port=d, host_port=h) for d, h, _ in port_maps),
        ),
        container_name="pod_test",
        default_extra={},
        port_maps=port_maps,
        spare_port_pairs=spare,
    )

    assert len(client.specs) == 2
    assert port_maps[0] == (22, 20099, 20099)
    new_hosts = {h for _, h, _ in port_maps[1:]}
    assert len(new_hosts) == 10 and not new_hosts & set(held)
    assert {b.host_port for b in client.specs[-1].ports} == {20099} | new_hosts
