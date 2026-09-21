"""One retry of a Docker SDK call on a dropped SSH transport; no NoneType crash on a lost channel.

Six failed rents on six nodes (16–21 Sep) shared one fault class: the Docker-over-SSH channel
dropped under a Docker SDK call — `run container failed: EOFError`, `SSHConnectionPool … Read
timed out (60 s)`, `create volume failed: 'NoneType' object has no attribute 'settimeout'`. The
last one is docker-py/urllib3 dereferencing a channel that is `None`.
"""
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from docker.errors import APIError, NotFound
from payload_models.payloads import FailedContainerRequest
from services.docker_service import DockerService
from services.rental_docker_sdk import (
    TRANSPORT_ERROR_CLASS,
    ContainerExecSpec,
    ContainerRunSpec,
    RentalDockerOperationError,
    RentalDockerSdkClient,
    RentalDockerSdkClientFactory,
    RentalDockerTransportDropped,
    RentalDockerTransportError,
    _build_rental_ssh_http_adapter_class,
    is_rental_docker_transport_error,
    rental_docker_error_class,
)
from test_docker_service_rental_security import (
    RecordingRentalDockerFactory,
    RecordingSSHClient,
    _base_create_payload,
    _patch_create_harness,
)

# The three error texts of the cluster, as the validator logged them.
RUN_EOF_TEXT = "Docker SDK run container failed: EOFError"
VOLUME_NONE_CHANNEL_TEXT = (
    "Docker SDK create volume failed: 'NoneType' object has no attribute 'settimeout'"
)
READ_TIMEOUT_TEXT = (
    "Docker SDK run container failed: SSHConnectionPool(host='localhost', port=None): "
    "Read timed out. (read timeout=60)"
)
# Answers from the daemon are not transport errors.
NAME_CONFLICT_TEXT = (
    'Docker SDK run container failed: 409 Client Error for http+docker://ssh/v1.45/containers/create'
    '?name=pod_x: Conflict ("Conflict. The container name "/pod_x" is already in use by container '
    '"abc123". You have to remove (or rename) that container to be able to reuse that name.")'
)
NO_SUCH_IMAGE_TEXT = "Docker SDK run container failed: 404 Client Error: No such image: bogus:latest"
PORT_ALLOCATED_TEXT = (
    "Docker SDK run container failed: 500 Server Error: driver failed programming external "
    "connectivity on endpoint pod_x: Bind for 0.0.0.0:9101 failed: port is already allocated"
)


def _not_found(what: str = "No such object") -> NotFound:
    return NotFound(what, response=Mock(status_code=404))


def _server_error(text: str) -> APIError:
    return APIError(text, response=Mock(status_code=500), explanation=text)


class _FakeAdapter:
    def __init__(self, *, reopen_error: Exception | None = None):
        self.reopen_calls = 0
        self.reopen_error = reopen_error

    def reopen_transport(self):
        self.reopen_calls += 1
        if self.reopen_error is not None:
            raise self.reopen_error


class _FakeApiClient:
    """The docker-py APIClient surface the rental client touches, scripted per call."""

    def __init__(self, *, adapter: _FakeAdapter | None = None):
        self._custom_adapter = adapter
        self.timeout = 60
        self.calls: list[str] = []
        self.create_container_errors: list[Exception] = []
        self.start_errors: list[Exception] = []
        self.create_volume_errors: list[Exception] = []
        self.exec_create_errors: list[Exception] = []
        self.existing_containers: dict[str, dict] = {}
        self.existing_volumes: dict[str, dict] = {}

    def _pop(self, errors: list[Exception]) -> None:
        if errors:
            raise errors.pop(0)

    def inspect_network(self, name):
        return {"Name": name, "Driver": "bridge", "Options": {"com.docker.network.bridge.enable_icc": "false"}}

    def create_host_config(self, **kwargs):
        return {"HostConfig": kwargs}

    def create_container(self, **kwargs):
        self.calls.append("create_container")
        self._pop(self.create_container_errors)
        self.existing_containers[kwargs["name"]] = {"Config": {"Image": kwargs["image"]}}
        return {"Id": "cid"}

    def start(self, name):
        self.calls.append("start")
        self._pop(self.start_errors)

    def inspect_container(self, name):
        self.calls.append("inspect_container")
        if name not in self.existing_containers:
            raise _not_found(f"No such container: {name}")
        return self.existing_containers[name]

    def create_volume(self, *, name, driver, driver_opts):
        self.calls.append("create_volume")
        self._pop(self.create_volume_errors)
        self.existing_volumes[name] = {"Name": name, "Driver": driver or "local"}
        return self.existing_volumes[name]

    def inspect_volume(self, name):
        self.calls.append("inspect_volume")
        if name not in self.existing_volumes:
            raise _not_found(f"No such volume: {name}")
        return self.existing_volumes[name]

    def exec_create(self, **kwargs):
        self.calls.append("exec_create")
        self._pop(self.exec_create_errors)
        return {"Id": "exec-1"}

    def exec_start(self, exec_id, **kwargs):
        self.calls.append("exec_start")
        return (b"ok\n", b"")

    def exec_inspect(self, exec_id):
        return {"ExitCode": 0}


def _client(api_client: _FakeApiClient, *, enabled: bool = True) -> RentalDockerSdkClient:
    return RentalDockerSdkClient(api_client, transport_retry_enabled=enabled)


def _run_spec(name: str = "pod_test") -> ContainerRunSpec:
    return ContainerRunSpec(image="lium/pod:1", name=name, network="lium-rentals")


# ---------------------------------------------------------------------------
# the class: which errors are a dropped transport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [RUN_EOF_TEXT, VOLUME_NONE_CHANNEL_TEXT, READ_TIMEOUT_TEXT])
def test_cluster_error_texts_are_the_transport_class(text):
    exc = RentalDockerOperationError(text)
    assert is_rental_docker_transport_error(exc)
    assert rental_docker_error_class(exc) == TRANSPORT_ERROR_CLASS


@pytest.mark.parametrize("text", [NAME_CONFLICT_TEXT, NO_SUCH_IMAGE_TEXT, PORT_ALLOCATED_TEXT])
def test_daemon_answers_are_not_the_transport_class(text):
    exc = RentalDockerOperationError(text)
    assert not is_rental_docker_transport_error(exc)
    assert rental_docker_error_class(exc) is None


def test_transport_class_is_found_through_the_cause_chain():
    inner = EOFError()
    try:
        raise RentalDockerOperationError("Docker SDK start failed: something else") from inner
    except RentalDockerOperationError as wrapped:
        assert is_rental_docker_transport_error(wrapped)


def test_eof_text_inside_container_output_is_not_the_class():
    # a Python traceback quoted from a container's log tail is a daemon answer, not our channel
    exc = RentalDockerOperationError(
        "Docker SDK exec failed: exit_status=1; stderr=Traceback: EOFError: Ran out of input; stdout="
    )
    assert not is_rental_docker_transport_error(exc)


# ---------------------------------------------------------------------------
# create volume: the NoneType text → one retry after re-open; adopt by name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_volume_retries_once_after_reopening_on_none_channel():
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)
    api.create_volume_errors = [AttributeError("'NoneType' object has no attribute 'settimeout'")]

    await _client(api).create_volume(volume_name="volume_p1", driver="vloopback", driver_opts={"size": "10g"})

    assert adapter.reopen_calls == 1
    # retry checks by name first (nothing there), then creates once more
    assert api.calls == ["create_volume", "inspect_volume", "create_volume"]


@pytest.mark.asyncio
async def test_create_volume_retry_adopts_the_volume_the_lost_first_attempt_made():
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)

    def create_then_drop(*, name, driver, driver_opts):
        api.calls.append("create_volume")
        api.existing_volumes[name] = {"Name": name, "Driver": "vloopback:latest"}
        raise EOFError()

    api.create_volume = create_then_drop

    await _client(api).create_volume(volume_name="volume_p1", driver="vloopback", driver_opts={"size": "10g"})

    assert adapter.reopen_calls == 1
    assert api.calls == ["create_volume", "inspect_volume"]


@pytest.mark.asyncio
async def test_create_volume_retry_refuses_a_same_name_volume_on_another_driver():
    api = _FakeApiClient(adapter=_FakeAdapter())
    api.create_volume_errors = [EOFError()]
    api.existing_volumes["volume_p1"] = {"Name": "volume_p1", "Driver": "local"}

    with pytest.raises(RentalDockerOperationError, match="already exists on driver 'local'"):
        await _client(api).create_volume(volume_name="volume_p1", driver="vloopback", driver_opts=None)

    assert api.calls == ["create_volume", "inspect_volume"]


@pytest.mark.asyncio
async def test_create_volume_flag_off_fails_the_first_time_and_never_reopens():
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)
    api.create_volume_errors = [AttributeError("'NoneType' object has no attribute 'settimeout'")]

    with pytest.raises(RentalDockerOperationError) as exc:
        await _client(api, enabled=False).create_volume(volume_name="volume_p1", driver=None, driver_opts=None)

    assert str(exc.value) == VOLUME_NONE_CHANNEL_TEXT
    assert adapter.reopen_calls == 0
    assert api.calls == ["create_volume"]
    # the class still reaches the failure event with the flag off
    assert rental_docker_error_class(exc.value) == TRANSPORT_ERROR_CLASS


@pytest.mark.asyncio
async def test_create_volume_daemon_error_is_not_retried():
    """Negative control: a daemon answer propagates at once, no re-open, no second call."""
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)
    api.create_volume_errors = [_server_error("VolumeDriver.Create: no space left on device")]

    with pytest.raises(RentalDockerOperationError, match="no space left on device"):
        await _client(api).create_volume(volume_name="volume_p1", driver=None, driver_opts=None)

    assert adapter.reopen_calls == 0
    assert api.calls == ["create_volume"]


# ---------------------------------------------------------------------------
# run container: create is never run twice without the name check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_container_retry_adopts_the_container_the_lost_create_made():
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)

    def create_then_drop(**kwargs):
        api.calls.append("create_container")
        api.existing_containers[kwargs["name"]] = {"Config": {"Image": kwargs["image"]}}
        raise EOFError()

    api.create_container = create_then_drop

    await _client(api).run_container(_run_spec())

    assert adapter.reopen_calls == 1
    # one create, then the retry finds it by name and only starts it
    assert api.calls == ["create_container", "inspect_container", "start"]


@pytest.mark.asyncio
async def test_run_container_retry_creates_when_the_first_create_never_arrived():
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)
    api.create_container_errors = [EOFError()]

    await _client(api).run_container(_run_spec())

    assert adapter.reopen_calls == 1
    assert api.calls == ["create_container", "inspect_container", "create_container", "start"]


@pytest.mark.asyncio
async def test_run_container_retry_after_start_dropped_starts_again_without_a_second_create():
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)
    api.start_errors = [AttributeError("'NoneType' object has no attribute 'settimeout'")]

    await _client(api).run_container(_run_spec())

    assert adapter.reopen_calls == 1
    assert api.calls == ["create_container", "start", "inspect_container", "start"]


@pytest.mark.asyncio
async def test_run_container_retry_refuses_a_same_name_container_on_another_image():
    """A stale `pod_<id>` of another image is not ours: named, not adopted, not removed."""
    api = _FakeApiClient(adapter=_FakeAdapter())
    api.create_container_errors = [EOFError()]
    api.existing_containers["pod_test"] = {"Config": {"Image": "someone/else:latest"}}

    with pytest.raises(RentalDockerOperationError, match="already exists with image 'someone/else:latest'"):
        await _client(api).run_container(_run_spec())

    assert api.calls == ["create_container", "inspect_container"]


@pytest.mark.asyncio
async def test_run_container_daemon_error_is_not_retried():
    """Negative control: the port-collision text is a daemon answer — no re-open, one create."""
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)
    api.create_container_errors = [_server_error("Bind for 0.0.0.0:9101 failed: port is already allocated")]

    with pytest.raises(RentalDockerOperationError, match="port is already allocated"):
        await _client(api).run_container(_run_spec())

    assert adapter.reopen_calls == 0
    assert api.calls == ["create_container"]


@pytest.mark.asyncio
async def test_second_drop_fails_the_call_as_the_transport_class_with_both_errors():
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)
    api.create_container_errors = [EOFError(), EOFError()]

    with pytest.raises(RentalDockerTransportError) as exc:
        await _client(api).run_container(_run_spec())

    text = str(exc.value)
    assert "once more after re-opening" in text and text.count("EOFError") == 2
    assert adapter.reopen_calls == 1
    assert rental_docker_error_class(exc.value) == TRANSPORT_ERROR_CLASS
    # never a third attempt
    assert api.calls.count("create_container") == 2


@pytest.mark.asyncio
async def test_reopen_failure_ends_the_call_without_a_second_attempt():
    adapter = _FakeAdapter(reopen_error=OSError("connect: connection refused"))
    api = _FakeApiClient(adapter=adapter)
    api.create_container_errors = [EOFError()]

    with pytest.raises(RentalDockerTransportError, match="could not be re-opened.*connection refused"):
        await _client(api).run_container(_run_spec())

    assert adapter.reopen_calls == 1
    assert api.calls == ["create_container"]


# ---------------------------------------------------------------------------
# exec: only an exec declared idempotent is retried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_that_appends_is_not_retried_on_a_dropped_transport():
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)
    api.existing_containers["pod_test"] = {"Config": {"Image": "lium/pod:1"}, "State": {"Running": True}}
    api.exec_create_errors = [EOFError()]
    spec = ContainerExecSpec(container_name="pod_test", argv=("sh", "-c", "cat >> /etc/environment"), stdin="A=1\n")
    assert spec.idempotent is False

    with pytest.raises(RentalDockerOperationError, match="Docker SDK exec failed: EOFError"):
        await _client(api).exec_in_container(spec)

    assert adapter.reopen_calls == 0
    assert api.calls == ["inspect_container", "exec_create"]


@pytest.mark.asyncio
async def test_idempotent_exec_is_retried_once_after_reopening():
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)
    api.existing_containers["pod_test"] = {"Config": {"Image": "lium/pod:1"}, "State": {"Running": True}}
    api.exec_create_errors = [EOFError()]
    spec = ContainerExecSpec(container_name="pod_test", argv=("sh", "/tmp/bootstrap.sh"), idempotent=True)

    result = await _client(api).exec_in_container(spec)

    assert result.exit_status == 0
    assert adapter.reopen_calls == 1
    assert api.calls.count("exec_create") == 2


# ---------------------------------------------------------------------------
# the adapter: a None channel is a typed error, never an AttributeError
# ---------------------------------------------------------------------------


def _adapter_class(tmp_path):
    key_path = tmp_path / "id_executor"
    known_hosts_path = tmp_path / "known_hosts"
    key_path.write_text("PRIVATE KEY")
    known_hosts_path.write_text("[203.0.113.10]:2222 ssh-ed25519 AAAATESTKEY\n")
    return _build_rental_ssh_http_adapter_class(key_path=key_path, known_hosts_path=known_hosts_path)


def _adapter(tmp_path, *, transport, pool_size: int = 4):
    """A rental adapter with a fake paramiko client whose transport is `transport`; `_connect` is a Mock."""
    from urllib3._collections import RecentlyUsedContainer

    adapter_class = _adapter_class(tmp_path)
    adapter = adapter_class.__new__(adapter_class)
    adapter.timeout = 60
    adapter.max_pool_size = 10
    adapter.ssh_host = "root@203.0.113.10:2222"
    adapter.ssh_client = Mock()
    adapter.ssh_client.get_transport.return_value = transport
    adapter._connect = Mock()
    adapter.pools = RecentlyUsedContainer(pool_size, dispose_func=lambda p: p.close())
    return adapter


def _rental_connection(tmp_path, *, transport):
    """The connection class production uses, built through the adapter's own pool on `transport`."""
    adapter = _adapter(tmp_path, transport=transport)
    pool = adapter.get_connection("http+docker://ssh/v1.45/containers/create")
    # get_connection reconnects on a dead transport (a Mock here); the tests want the dead one
    pool.ssh_transport = transport
    return pool._new_conn()


def test_connection_on_no_transport_raises_typed_error(tmp_path):
    connection = _rental_connection(tmp_path, transport=None)

    with pytest.raises(RentalDockerTransportDropped, match="SSH session to the executor is closed"):
        connection.connect()


def test_connection_on_inactive_transport_raises_typed_error(tmp_path):
    transport = Mock()
    transport.is_active.return_value = False
    connection = _rental_connection(tmp_path, transport=transport)

    with pytest.raises(RentalDockerTransportDropped):
        connection.connect()
    transport.open_session.assert_not_called()


def test_connection_on_none_channel_raises_typed_error_not_attribute_error(tmp_path):
    transport = Mock()
    transport.is_active.return_value = True
    transport.open_session.return_value = None
    connection = _rental_connection(tmp_path, transport=transport)

    with pytest.raises(RentalDockerTransportDropped, match="returned no channel"):
        connection.connect()


def test_connection_open_session_eof_raises_typed_error(tmp_path):
    transport = Mock()
    transport.is_active.return_value = True
    transport.open_session.side_effect = EOFError()
    connection = _rental_connection(tmp_path, transport=transport)

    with pytest.raises(RentalDockerTransportDropped, match="ended while a Docker API channel"):
        connection.connect()


def test_connection_live_channel_gets_timeout_and_dial_stdio(tmp_path):
    channel = Mock()
    transport = Mock()
    transport.is_active.return_value = True
    transport.open_session.return_value = channel
    connection = _rental_connection(tmp_path, transport=transport)

    connection.connect()

    channel.settimeout.assert_called_once_with(60)
    channel.exec_command.assert_called_once_with("docker system dial-stdio")
    assert connection.sock is channel


def test_getresponse_on_lost_channel_raises_typed_error_not_attribute_error(tmp_path):
    connection = _rental_connection(tmp_path, transport=Mock())
    connection.sock = None

    with pytest.raises(RentalDockerTransportDropped, match="closed before the daemon's answer"):
        connection.getresponse()


def test_reopen_transport_drops_pools_and_reconnects(tmp_path):
    from urllib3._collections import RecentlyUsedContainer

    live_transport = Mock()
    live_transport.is_active.return_value = True
    adapter = _adapter(tmp_path, transport=live_transport)
    disposed = []
    adapter.pools = RecentlyUsedContainer(4, dispose_func=lambda p: disposed.append(p))
    first_pool = adapter.get_connection("http+docker://ssh/v1.45/volumes/create")
    assert adapter._connect.call_count == 0

    adapter.reopen_transport()

    assert disposed == [first_pool]
    adapter.ssh_client.close.assert_called_once_with()
    adapter._connect.assert_called_once_with()
    assert adapter.get_connection("http+docker://ssh/v1.45/volumes/create") is not first_pool


def test_get_connection_reconnects_when_the_transport_is_dead(tmp_path):
    dead_transport = Mock()
    dead_transport.is_active.return_value = False
    adapter = _adapter(tmp_path, transport=dead_transport)

    adapter.get_connection("http+docker://ssh/v1.45/containers/json")

    # upstream reconnects only when get_transport() is None; a dead transport is reconnected too
    adapter._connect.assert_called_once_with()


# ---------------------------------------------------------------------------
# the factory reads the flag per connect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_factory_reads_the_flag_callable_per_connect(monkeypatch):
    monkeypatch.setattr("services.rental_docker_sdk._validate_paramiko_known_hosts", lambda path: None)
    flag = {"on": False}
    api_client_factory = Mock(return_value=_FakeApiClient())
    factory = RentalDockerSdkClientFactory(
        api_client_factory=api_client_factory,
        transport_retry_enabled=lambda: flag["on"],
    )
    info = SimpleNamespace(
        address="203.0.113.10", ssh_port=2222, ssh_username="root", ssh_host_key="ssh-ed25519 AAAATESTKEY"
    )

    async with factory.connect(executor_info=info, private_key="PRIVATE KEY") as client:
        assert client._transport_retry_enabled is False
    flag["on"] = True
    async with factory.connect(executor_info=info, private_key="PRIVATE KEY") as client:
        assert client._transport_retry_enabled is True


# ---------------------------------------------------------------------------
# the failure event names the stage and the class
# ---------------------------------------------------------------------------


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


async def _failed_create(service, executor, monkeypatch, run_error: Exception) -> FailedContainerRequest:
    _patch_create_harness(monkeypatch, service, RecordingSSHClient())
    service.rental_docker_client_factory.client.run_container_error = run_error
    result = await service.create_container(
        payload=_base_create_payload(),
        executor_info=executor,
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted-private-key",
    )
    assert isinstance(result, FailedContainerRequest)
    return result


@pytest.mark.asyncio
async def test_create_failure_event_carries_the_stage_and_the_transport_class(service, executor, monkeypatch):
    result = await _failed_create(service, executor, monkeypatch, RentalDockerOperationError(RUN_EOF_TEXT))

    assert result.failure_step == "docker_run"
    assert '"error_class": "transport"' in result.detail
    assert RUN_EOF_TEXT in result.detail
    # the renter-safe headline is unchanged
    assert result.msg == "Failed create_container"


@pytest.mark.asyncio
async def test_create_failure_event_has_no_class_for_a_daemon_answer(service, executor, monkeypatch):
    result = await _failed_create(service, executor, monkeypatch, RentalDockerOperationError(NO_SUCH_IMAGE_TEXT))

    assert result.failure_step == "docker_run"
    assert "error_class" not in result.detail
