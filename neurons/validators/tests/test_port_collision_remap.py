"""A host port already bound moves the pod to the next free mapped port (PORT_COLLISION_RETRY_ENABLED).

Five failed rents on four nodes (16–21 Sep): dockerd refused the bind of a host port the backend
handed the pod — `failed to bind host port <ip>:<port>: address already in use` (3) and `Bind for
<ip>:<port> failed: port is already allocated` (2) — held by a stale container or a provider
process the 90 s same-mapping wait never outlives.
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import FailedContainerRequest, PayloadPortMapping
from services.docker_service import (
    PORT_COLLISION_ERROR_CLASS,
    DockerService,
    RentalPortCollisionError,
    _bound_host_port_from_error,
    _parse_listening_ports,
    _port_collision_candidates,
    _spare_port_pairs,
    port_collision_error_class,
)
from services.rental_docker_sdk import ContainerRunSpec, PortBinding, RentalDockerOperationError
from test_docker_service_rental_security import (
    RecordingRentalDockerFactory,
    RecordingSSHClient,
    _base_create_payload,
    _patch_create_harness,
)

# The cluster's two refusal texts, as dockerd (and the validator's log) wrote them.
ADDRESS_IN_USE_TEXT = (
    "Docker SDK run container failed: 500 Server Error: failed to set up container networking: "
    "driver failed programming external connectivity on endpoint pod_x: failed to bind host port "
    "for 0.0.0.0:9101:172.17.0.2:22/tcp: address already in use"
)
PORT_ALLOCATED_TEXT = (
    "Docker SDK run container failed: 500 Server Error: driver failed programming external "
    "connectivity on endpoint pod_x: Bind for 0.0.0.0:9101 failed: port is already allocated"
)
EADDRINUSE_SHORT_TEXT = "failed to bind host port 0.0.0.0:9101/tcp: address already in use"
NO_SUCH_IMAGE_TEXT = "Docker SDK run container failed: 404 Client Error: No such image: lium/pod:1"

SS_9101_AND_9102_LISTEN = (
    "LISTEN 0 128  0.0.0.0:22     0.0.0.0:*\n"
    "LISTEN 0 4096 0.0.0.0:9101   0.0.0.0:*\n"
    "LISTEN 0 4096 [::]:9102      [::]:*\n"
)


def _pair(port: int) -> PayloadPortMapping:
    return PayloadPortMapping(internal_port=port, external_port=port + 20000)


def _run_spec(host_port: int = 9101) -> ContainerRunSpec:
    return ContainerRunSpec(
        image="lium/pod:1",
        name="pod_test",
        ports=(
            PortBinding(container_port=22, host_port=host_port),
            PortBinding(container_port=51820, host_port=51820, protocol="udp"),
        ),
    )


class _RunRecorder:
    """A docker client whose `run_container` answers from a script of errors (None = success)."""

    def __init__(self, *errors: Exception | None):
        self.errors = list(errors)
        self.specs: list[ContainerRunSpec] = []
        self.removed: list[str] = []

    async def run_container(self, spec: ContainerRunSpec) -> None:
        self.specs.append(spec)
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error

    async def remove_container(
        self, *, container_name: str, force: bool, remove_volumes: bool
    ) -> None:
        self.removed.append(container_name)


@pytest.fixture
def service() -> DockerService:
    redis_service = Mock()
    lock = Mock()
    lock.__aenter__ = Mock(return_value=lock)
    lock.__aexit__ = Mock(return_value=None)
    redis_service.acquire_executor_lock = Mock(return_value=lock)
    return DockerService(
        ssh_service=Mock(),
        redis_service=redis_service,
        attestation_service=Mock(),
        rental_docker_client_factory=RecordingRentalDockerFactory(),
    )


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr("services.docker_service.settings.PORT_COLLISION_RETRY_ENABLED", True)


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setattr("services.docker_service.settings.PORT_COLLISION_RETRY_ENABLED", False)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr("services.docker_service.asyncio.sleep", AsyncMock())


async def _run(service, client, *, port_maps, spare, ssh_client=None):
    await service._run_rental_docker_create_with_port_retry(
        docker_client=client,
        ssh_client=ssh_client or RecordingSSHClient(stdout=SS_9101_AND_9102_LISTEN),
        run_spec=_run_spec(),
        container_name="pod_test",
        default_extra={},
        port_maps=port_maps,
        spare_port_pairs=spare,
    )


# ---------------------------------------------------------------------------
# the exact texts of the cluster
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [ADDRESS_IN_USE_TEXT, PORT_ALLOCATED_TEXT, EADDRINUSE_SHORT_TEXT])
def test_cluster_texts_name_the_bound_host_port(text):
    assert _bound_host_port_from_error(RentalDockerOperationError(text)) == 9101


def test_an_ipv6_bind_address_names_its_port_too():
    exc = Exception(
        "failed to bind host port for [::]:9040:172.17.0.2:22/tcp: address already in use"
    )
    assert _bound_host_port_from_error(exc) == 9040


@pytest.mark.parametrize("text", [NO_SUCH_IMAGE_TEXT, "port 9101 is fine", ""])
def test_other_texts_name_no_port(text):
    assert _bound_host_port_from_error(Exception(text)) is None


@pytest.mark.parametrize("text", [ADDRESS_IN_USE_TEXT, PORT_ALLOCATED_TEXT])
def test_cluster_texts_are_the_port_collision_class(text):
    assert (
        port_collision_error_class(RentalDockerOperationError(text)) == PORT_COLLISION_ERROR_CLASS
    )


def test_typed_error_is_the_class_and_a_daemon_answer_is_not():
    assert (
        port_collision_error_class(RentalPortCollisionError("all taken"))
        == PORT_COLLISION_ERROR_CLASS
    )
    assert port_collision_error_class(RentalDockerOperationError(NO_SUCH_IMAGE_TEXT)) is None


def test_class_is_found_through_the_cause_chain():
    try:
        try:
            raise RentalDockerOperationError(PORT_ALLOCATED_TEXT)
        except RentalDockerOperationError as inner:
            raise RuntimeError("create failed") from inner
    except RuntimeError as outer:
        assert port_collision_error_class(outer) == PORT_COLLISION_ERROR_CLASS


# ---------------------------------------------------------------------------
# candidates and the probe
# ---------------------------------------------------------------------------


def test_spare_pairs_are_the_advertised_pairs_the_pod_is_not_using():
    advertised = [_pair(9100), _pair(9101), _pair(9103), _pair(9102)]
    spare = _spare_port_pairs(advertised, [(22, 9101, 29101)])
    assert [pair.internal_port for pair in spare] == [9100, 9102, 9103]


def test_candidates_are_the_next_three_after_the_colliding_port_wrapping_around():
    spare = [_pair(p) for p in (9100, 9102, 9103, 9104, 9105)]
    chosen = _port_collision_candidates(spare, after_host_port=9103)
    assert [pair.internal_port for pair in chosen] == [9104, 9105, 9100]


def test_listening_ports_are_read_from_ss_and_netstat_lines():
    output = SS_9101_AND_9102_LISTEN + "tcp 0 0 127.0.0.53%lo:53 0.0.0.0:* LISTEN\n"
    assert _parse_listening_ports(output) == {22, 9101, 9102, 53}


# ---------------------------------------------------------------------------
# the retry: second candidate free → one retry with the new mapping, mapping reported
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_candidate_takes_the_pod_and_the_mapping_reports_it(service, flag_on):
    client = _RunRecorder(RentalDockerOperationError(ADDRESS_IN_USE_TEXT), None)
    port_maps = [(22, 9101, 29101)]
    spare = [_pair(9102), _pair(9103), _pair(9104)]
    ssh_client = RecordingSSHClient(stdout=SS_9101_AND_9102_LISTEN)

    await _run(service, client, port_maps=port_maps, spare=spare, ssh_client=ssh_client)

    # 9102 is listening on the host, 9103 is the first free candidate
    assert port_maps == [(22, 9103, 29103)]
    assert [pair.internal_port for pair in spare] == [9102, 9104]
    assert len(client.specs) == 2
    assert client.specs[0].ports[0].host_port == 9101
    assert client.specs[1].ports[0].host_port == 9103
    # the UDP binding is untouched by the move
    assert client.specs[1].ports[1] == PortBinding(
        container_port=51820, host_port=51820, protocol="udp"
    )
    # the Created-state container of the refused run is removed before the retry
    assert client.removed == ["pod_test"]
    # one probe over the create's own SSH session
    assert len(ssh_client.commands) == 1 and "ss -Hltn" in ssh_client.commands[0]


@pytest.mark.asyncio
async def test_port_allocated_text_takes_the_same_path(service, flag_on):
    client = _RunRecorder(RentalDockerOperationError(PORT_ALLOCATED_TEXT), None)
    port_maps = [(22, 9101, 29101), (8888, 9105, 29105)]

    await _run(service, client, port_maps=port_maps, spare=[_pair(9103)])

    assert port_maps == [(22, 9103, 29103), (8888, 9105, 29105)]
    assert len(client.specs) == 2


@pytest.mark.asyncio
async def test_all_candidates_taken_fails_as_the_port_collision_class(service, flag_on):
    client = _RunRecorder(RentalDockerOperationError(ADDRESS_IN_USE_TEXT))
    port_maps = [(22, 9101, 29101)]
    ssh_client = RecordingSSHClient(
        stdout="LISTEN 0 1 0.0.0.0:9102 0.0.0.0:*\nLISTEN 0 1 0.0.0.0:9103 0.0.0.0:*\nLISTEN 0 1 0.0.0.0:9104 0.0.0.0:*\n"
    )

    with pytest.raises(RentalPortCollisionError, match=r"\[9102, 9103, 9104\] are listening too"):
        await _run(
            service,
            client,
            port_maps=port_maps,
            spare=[_pair(p) for p in (9102, 9103, 9104, 9105)],
            ssh_client=ssh_client,
        )

    # no second run, the mapping stands, the fourth spare pair was never a candidate
    assert len(client.specs) == 1
    assert port_maps == [(22, 9101, 29101)]
    assert port_collision_error_class(RentalPortCollisionError("x")) == PORT_COLLISION_ERROR_CLASS


@pytest.mark.asyncio
async def test_a_second_refusal_on_the_new_mapping_fails_without_a_third_candidate(
    service, flag_on
):
    client = _RunRecorder(
        RentalDockerOperationError(ADDRESS_IN_USE_TEXT),
        RentalDockerOperationError(EADDRINUSE_SHORT_TEXT.replace("9101", "9103")),
    )
    port_maps = [(22, 9101, 29101)]

    with pytest.raises(RentalPortCollisionError, match="remapped port either"):
        await _run(service, client, port_maps=port_maps, spare=[_pair(9103), _pair(9104)])

    assert len(client.specs) == 2
    assert port_maps == [(22, 9103, 29103)]


@pytest.mark.asyncio
async def test_an_unreadable_probe_lets_docker_run_try_the_first_candidate(service, flag_on):
    client = _RunRecorder(RentalDockerOperationError(ADDRESS_IN_USE_TEXT), None)
    port_maps = [(22, 9101, 29101)]
    ssh_client = RecordingSSHClient(stdout="", exit_status=127)

    await _run(
        service,
        client,
        port_maps=port_maps,
        spare=[_pair(9102), _pair(9103)],
        ssh_client=ssh_client,
    )

    assert port_maps == [(22, 9102, 29102)]
    assert client.specs[1].ports[0].host_port == 9102


@pytest.mark.asyncio
async def test_a_refusal_naming_no_port_of_this_pod_keeps_the_same_mapping_wait(
    service, flag_on, no_sleep
):
    # the bound port is not one of the pod's: nothing to move, the DAH-1991 wait runs as before
    client = _RunRecorder(
        RentalDockerOperationError(EADDRINUSE_SHORT_TEXT.replace("9101", "7777")), None
    )
    port_maps = [(22, 9101, 29101)]

    await _run(service, client, port_maps=port_maps, spare=[_pair(9102)])

    assert port_maps == [(22, 9101, 29101)]
    assert client.specs[1].ports[0].host_port == 9101


@pytest.mark.asyncio
async def test_no_spare_pair_keeps_the_same_mapping_wait(service, flag_on, no_sleep):
    client = _RunRecorder(RentalDockerOperationError(ADDRESS_IN_USE_TEXT), None)
    port_maps = [(22, 9101, 29101)]

    await _run(service, client, port_maps=port_maps, spare=[])

    assert port_maps == [(22, 9101, 29101)]
    assert len(client.specs) == 2


@pytest.mark.asyncio
async def test_flag_off_retries_the_same_mapping_and_never_probes(service, flag_off, no_sleep):
    client = _RunRecorder(RentalDockerOperationError(ADDRESS_IN_USE_TEXT), None)
    port_maps = [(22, 9101, 29101)]
    ssh_client = RecordingSSHClient(stdout=SS_9101_AND_9102_LISTEN)

    await _run(service, client, port_maps=port_maps, spare=[_pair(9103)], ssh_client=ssh_client)

    assert port_maps == [(22, 9101, 29101)]
    assert [spec.ports[0].host_port for spec in client.specs] == [9101, 9101]
    assert ssh_client.commands == []


@pytest.mark.asyncio
async def test_a_daemon_error_is_not_a_collision_and_is_not_retried(service, flag_on):
    client = _RunRecorder(RentalDockerOperationError(NO_SUCH_IMAGE_TEXT), None)
    port_maps = [(22, 9101, 29101)]
    ssh_client = RecordingSSHClient(stdout=SS_9101_AND_9102_LISTEN)

    with pytest.raises(RentalDockerOperationError, match="No such image"):
        await _run(service, client, port_maps=port_maps, spare=[_pair(9103)], ssh_client=ssh_client)

    assert len(client.specs) == 1
    assert port_maps == [(22, 9101, 29101)]
    assert ssh_client.commands == []


# ---------------------------------------------------------------------------
# through create_container: the answer carries the real port; the failure event the class
# ---------------------------------------------------------------------------


@pytest.fixture
def executor() -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=str(uuid4()),
        address="203.0.113.10",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
        ssh_host_key="ssh-ed25519 AAAATESTKEY",
    )


def _create_payload_with_spare_pairs():
    return _base_create_payload(
        available_ports=[
            PayloadPortMapping(internal_port=22, external_port=30022),
            PayloadPortMapping(internal_port=20000, external_port=30000),
            PayloadPortMapping(internal_port=20001, external_port=30001),
            PayloadPortMapping(internal_port=20002, external_port=30002),
        ],
    )


@pytest.mark.asyncio
async def test_create_container_answers_with_the_port_the_pod_really_got(
    service, executor, monkeypatch, flag_on
):
    _patch_create_harness(monkeypatch, service, RecordingSSHClient())
    client = service.rental_docker_client_factory.client
    errors = [
        RentalDockerOperationError(
            "Docker SDK run container failed: failed to bind host port for 0.0.0.0:20000:172.17.0.2:20000/tcp: "
            "address already in use"
        )
    ]

    async def run_container(spec):
        client.run_specs.append(spec)
        if errors:
            raise errors.pop(0)

    monkeypatch.setattr(client, "run_container", run_container)
    monkeypatch.setattr(service, "_listening_host_ports", AsyncMock(return_value={22, 20000}))

    result = await service.create_container(
        payload=_create_payload_with_spare_pairs(),
        executor_info=executor,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted-private-key",
    )

    assert not isinstance(result, FailedContainerRequest), getattr(result, "detail", result)
    # the backend's pod record and the renter see 30001, not the refused 30000
    assert result.port_maps == [(22, 30022), (20000, 30001)]
    assert [spec.ports[1].host_port for spec in client.run_specs] == [20000, 20001]


@pytest.mark.asyncio
async def test_create_failure_event_carries_the_stage_and_the_port_collision_class(
    service, executor, monkeypatch, flag_on
):
    _patch_create_harness(monkeypatch, service, RecordingSSHClient())
    client = service.rental_docker_client_factory.client
    client.run_container_error = RentalDockerOperationError(
        "Docker SDK run container failed: Bind for 0.0.0.0:20000 failed: port is already allocated"
    )
    monkeypatch.setattr(
        service, "_listening_host_ports", AsyncMock(return_value={22, 20000, 20001, 20002})
    )

    result = await service.create_container(
        payload=_create_payload_with_spare_pairs(),
        executor_info=executor,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted-private-key",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_run"
    assert '"error_class": "port_collision"' in result.detail
    assert "[20001, 20002] are listening too" in result.detail
    assert result.msg == "Failed create_container"


@pytest.mark.asyncio
async def test_create_failure_event_has_no_class_for_a_daemon_answer(
    service, executor, monkeypatch, flag_on
):
    _patch_create_harness(monkeypatch, service, RecordingSSHClient())
    service.rental_docker_client_factory.client.run_container_error = RentalDockerOperationError(
        NO_SUCH_IMAGE_TEXT
    )

    result = await service.create_container(
        payload=_create_payload_with_spare_pairs(),
        executor_info=executor,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted-private-key",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_run"
    assert "error_class" not in result.detail


@pytest.mark.asyncio
async def test_flag_off_create_failure_still_carries_the_class(
    service, executor, monkeypatch, flag_off, no_sleep
):
    _patch_create_harness(monkeypatch, service, RecordingSSHClient())
    service.rental_docker_client_factory.client.run_container_error = RentalDockerOperationError(
        PORT_ALLOCATED_TEXT
    )
    # the 90 s same-mapping budget: already spent
    monkeypatch.setattr("services.docker_service._PORT_ALLOCATED_RETRY_BUDGET_SEC", -1)

    result = await service.create_container(
        payload=_create_payload_with_spare_pairs(),
        executor_info=executor,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted-private-key",
    )

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_run"
    assert '"error_class": "port_collision"' in result.detail
