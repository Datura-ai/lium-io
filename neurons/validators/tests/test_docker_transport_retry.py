"""Retry of a Docker SDK call on a dropped SSH transport: the safety cases.

A retry never adopts a same-name volume or container that is not ours, never re-runs an exec
that appends, and one miner's stalled DNS cannot block another miner's rental connect. The
plain retry/classification cases were cut as low-signal in DAH-4027.
"""

import asyncio
import socket
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from docker.errors import APIError, NotFound
from services.node_host import node_owner
from services.rental_docker_sdk import (
    ContainerExecSpec,
    ContainerNameTakenOnRetry,
    ContainerRunSpec,
    RentalDockerConnectionError,
    RentalDockerOperationError,
    RentalDockerSdkClient,
    RentalDockerSdkClientFactory,
    RentalDockerTransportNotReopened,
)

# The three error texts of the cluster, as the validator logged them.
# Answers from the daemon are not transport errors.


def _not_found(what: str = "No such object") -> NotFound:
    return NotFound(what, response=Mock(status_code=404))


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
        return {
            "Name": name,
            "Driver": "bridge",
            "Options": {"com.docker.network.bridge.enable_icc": "false"},
        }

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

    def create_volume(self, *, name, driver, driver_opts, labels=None):
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


def _client(api_client: _FakeApiClient) -> RentalDockerSdkClient:
    return RentalDockerSdkClient(api_client)


def _run_spec(name: str = "pod_test") -> ContainerRunSpec:
    return ContainerRunSpec(image="lium/pod:1", name=name, network="lium-rentals")


# ---------------------------------------------------------------------------
# the class: which errors are a dropped transport
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# create volume: the NoneType text → one retry after re-open; adopt by name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_volume_retry_refuses_a_same_name_volume_on_another_driver():
    api = _FakeApiClient(adapter=_FakeAdapter())
    api.create_volume_errors = [EOFError()]
    api.existing_volumes["volume_p1"] = {"Name": "volume_p1", "Driver": "local"}

    with pytest.raises(RentalDockerOperationError, match="already exists on driver 'local'"):
        await _client(api).create_volume(
            volume_name="volume_p1", driver="vloopback", driver_opts=None
        )

    assert api.calls == ["create_volume", "inspect_volume"]


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_driver, adopted", [("vloopback:v2", False), ("vloopback", True)])
async def test_adopt_takes_a_same_name_volume_only_from_the_requested_plugin_tag(requested_driver, adopted):
    # `vloopback` is `vloopback:latest`, the old plugin; `vloopback:v2` is another plugin
    api = _FakeApiClient(adapter=_FakeAdapter())
    api.create_volume_errors = [EOFError()]
    api.existing_volumes["volume_p1"] = {"Name": "volume_p1", "Driver": "vloopback:latest"}
    created = _client(api).create_volume(volume_name="volume_p1", driver=requested_driver, driver_opts={"size": "10g"})

    if adopted:
        await created
    else:
        with pytest.raises(RentalDockerOperationError, match="already exists on driver 'vloopback:latest'"):
            await created

    assert api.calls == ["create_volume", "inspect_volume"]


# ---------------------------------------------------------------------------
# run container: create is never run twice without the name check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_container_retry_refuses_a_same_name_container_on_another_image():
    """A stale `pod_<id>` of another image is not ours: named, not adopted, not removed."""
    api = _FakeApiClient(adapter=_FakeAdapter())
    api.create_container_errors = [EOFError()]
    api.existing_containers["pod_test"] = {"Config": {"Image": "someone/else:latest"}}

    with pytest.raises(
        ContainerNameTakenOnRetry, match="already exists with image 'someone/else:latest'"
    ):
        await _client(api).run_container(_run_spec())

    assert api.calls == ["create_container", "inspect_container"]


def _name_conflict() -> APIError:
    return APIError(
        "Conflict",
        response=Mock(status_code=409, url="containers/create", reason="Conflict"),
        explanation='Conflict. The container name "/pod_test" is already in use by container "abc".',
    )


@pytest.mark.asyncio
async def test_run_container_retry_adopts_the_first_create_however_long_dockerd_runs_it(monkeypatch):
    """22 Nov (pod d98d08ae): the first create outlived the 60 s read timeout and dockerd kept running it
    past the retry's 60 s wait (409 on the name, inspect 404), so the rent failed on its own name. The
    retry waits while dockerd holds the name and adopts the container once it registers; no third create."""
    import requests
    import services.rental_docker_sdk as sdk

    monkeypatch.setattr(sdk, "_INFLIGHT_CREATE_POLL_SECONDS", 0)
    # the first create outlasts any fixed wait (the removed 60 s one included)
    monkeypatch.setattr(sdk, "_INFLIGHT_CREATE_WAIT_SECONDS", 0, raising=False)
    api = _FakeApiClient(adapter=_FakeAdapter())
    api.create_container_errors = [
        requests.exceptions.ReadTimeout("Read timed out. (read timeout=60)"),
        _name_conflict(),
        _name_conflict(),
    ]
    inspects = iter([_not_found(), _not_found()])

    def inspect_container(name):
        api.calls.append("inspect_container")
        error = next(inspects, None)
        if error is not None:
            raise error
        return {"Id": "first", "Config": {"Image": "lium/pod:1"}}

    api.inspect_container = inspect_container

    assert await _client(api).run_container(_run_spec()) == "first"
    assert api.calls == [
        "create_container",
        "inspect_container", "create_container",
        "inspect_container", "create_container",
        "inspect_container", "start",
    ]


@pytest.mark.asyncio
async def test_run_container_retry_creates_once_dockerd_released_the_name(monkeypatch):
    """The first create failed in dockerd after the read timeout: the name is free again and the retry's
    create is the only container."""
    import requests
    import services.rental_docker_sdk as sdk

    monkeypatch.setattr(sdk, "_INFLIGHT_CREATE_POLL_SECONDS", 0)
    api = _FakeApiClient(adapter=_FakeAdapter())
    api.create_container_errors = [
        requests.exceptions.ReadTimeout("Read timed out. (read timeout=60)"),
        _name_conflict(),
    ]

    assert await _client(api).run_container(_run_spec()) == "cid"
    assert api.calls == [
        "create_container",
        "inspect_container", "create_container",
        "inspect_container", "create_container", "start",
    ]


# ---------------------------------------------------------------------------
# pull: the stream cut mid-pull (27 Oct alert ece89) → one re-open and one more pull
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pull_cut_mid_stream_is_retried_once_after_reopen():
    import http.client

    adapter = _FakeAdapter()
    client = _client(_FakeApiClient(adapter=adapter))
    errors = [http.client.IncompleteRead(b"")]
    pulls = []

    def pull_sync(image):
        pulls.append(image)
        if errors:
            raise errors.pop(0)

    client._pull_sync = pull_sync
    await client.pull(image="daturaai/pytorch:x")

    assert pulls == ["daturaai/pytorch:x", "daturaai/pytorch:x"]
    assert adapter.reopen_calls == 1


@pytest.mark.asyncio
async def test_pull_refused_by_the_registry_is_not_retried():
    adapter = _FakeAdapter()
    client = _client(_FakeApiClient(adapter=adapter))
    pulls = []

    def pull_sync(image):
        pulls.append(image)
        raise RuntimeError("manifest unknown")

    client._pull_sync = pull_sync
    with pytest.raises(RentalDockerOperationError, match="Docker SDK pull failed: manifest unknown"):
        await client.pull(image="nope/nope:1")

    assert len(pulls) == 1 and adapter.reopen_calls == 0


# ---------------------------------------------------------------------------
# exec: only an exec declared idempotent is retried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_that_appends_is_not_retried_on_a_dropped_transport():
    adapter = _FakeAdapter()
    api = _FakeApiClient(adapter=adapter)
    api.existing_containers["pod_test"] = {
        "Config": {"Image": "lium/pod:1"},
        "State": {"Running": True},
    }
    api.exec_create_errors = [EOFError()]
    spec = ContainerExecSpec(
        container_name="pod_test", argv=("sh", "-c", "cat >> /etc/environment"), stdin="A=1\n"
    )
    assert spec.idempotent is False

    with pytest.raises(RentalDockerOperationError, match="Docker SDK exec failed: EOFError"):
        await _client(api).exec_in_container(spec)

    assert adapter.reopen_calls == 0
    assert api.calls == ["inspect_container", "exec_create"]


# ---------------------------------------------------------------------------
# the adapter: a None channel is a typed error, never an AttributeError
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the factory connect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_miner_whose_dns_stalls_cannot_stop_another_miners_rental_connect(monkeypatch):
    # review of #1565: hostname executors behind a zone that drops queries held every thread of the rental
    # lookup pool, and honest miners' hostname rentals then failed at the deadline. The rental connect now
    # resolves under the authenticated miner's quota (node_owner), like every other node lookup.
    import services.node_host as node_host
    import services.rental_docker_sdk as sdk

    release = threading.Event()

    def lookup(host, port, *args, **kwargs):
        if "evil" in host:
            release.wait(5)
            raise socket.gaierror(-3, "Temporary failure in name resolution")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port))]

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    monkeypatch.setattr(sdk, "_validate_paramiko_known_hosts", lambda path: None)
    monkeypatch.setattr(sdk, "_DOCKER_SSH_REOPEN_TIMEOUT_SECONDS", 0.5)
    api_client_factory = Mock(return_value=_FakeApiClient())
    factory = RentalDockerSdkClientFactory(api_client_factory=api_client_factory)

    def info(address):
        return SimpleNamespace(
            address=address, ssh_port=2222, ssh_username="root", ssh_host_key="ssh-ed25519 AAAATESTKEY",
            expected_ssh_host_key=None,
        )

    async def connect_as(miner, address):
        with node_owner(miner):
            async with factory.connect(executor_info=info(address), private_key="PRIVATE KEY"):
                pass

    try:
        hostile = [
            asyncio.create_task(connect_as("hostile-hotkey", f"gpu.evil{i}.example"))
            for i in range(node_host.NODE_DNS_MAX_WORKERS + 8)
        ]
        await asyncio.sleep(0.05)

        await connect_as("honest-hotkey", "gpu-node1.good.example")

        api_client_factory.assert_called_once()
        results = await asyncio.gather(*hostile, return_exceptions=True)
        assert all(isinstance(r, RentalDockerConnectionError) for r in results)
        assert node_host._dns_busy.get("miner:hostile-hotkey", 0) <= node_host.NODE_DNS_PER_MINER
    finally:
        release.set()


# ---------------------------------------------------------------------------
# the failure event names the stage and the class
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# a real dropped link: a loopback socket stands in for the SSH channel and its far end goes away
# ---------------------------------------------------------------------------




@pytest.mark.asyncio
async def test_pull_whose_reply_ends_prematurely_is_retried_once_after_reopen():
    import requests
    import urllib3

    adapter = _FakeAdapter()
    client = _client(_FakeApiClient(adapter=adapter))
    errors = [requests.exceptions.ChunkedEncodingError(urllib3.exceptions.ProtocolError("Response ended prematurely"))]
    pulls = []

    def pull_sync(image):
        pulls.append(image)
        if errors:
            raise errors.pop(0)

    client._pull_sync = pull_sync
    await client.pull(image="daturaai/pytorch:x")

    assert pulls == ["daturaai/pytorch:x", "daturaai/pytorch:x"]
    assert adapter.reopen_calls == 1


@pytest.mark.asyncio
async def test_a_host_gone_silent_fails_the_inspect_as_a_dropped_transport_not_a_bare_timeout(
    tmp_path, monkeypatch
):
    # a host stopped answering mid-create: the session under the Docker SDK died and every
    # re-connect timed out at TCP. The create failure read "inspect container failed: timed out",
    # so the backend kept the renter on that host.
    import services.rental_docker_sdk as sdk

    adapter_class = sdk._build_rental_ssh_http_adapter_class(
        key_path=tmp_path / "key", known_hosts_path=tmp_path / "known_hosts"
    )
    adapter = adapter_class.__new__(adapter_class)
    adapter.pools = SimpleNamespace(lock=threading.Lock(), get=lambda host: None)
    adapter.ssh_host = "node"
    dead = Mock(is_active=Mock(return_value=False))
    adapter.ssh_client = Mock(get_transport=Mock(return_value=dead))
    monkeypatch.setattr(adapter_class, "_connect", Mock(side_effect=TimeoutError("timed out")))

    class _SilentHostApiClient(_FakeApiClient):
        def inspect_container(self, name):
            adapter.get_connection("http+docker://ssh/v1/containers/json")

    api_client = _SilentHostApiClient(adapter=_FakeAdapter(reopen_error=TimeoutError("timed out")))

    with pytest.raises(RentalDockerOperationError) as raised:
        await _client(api_client).inspect_container_state(container_name="pod_test")

    assert "Docker SDK inspect container failed on a dropped SSH transport" in str(raised.value)
    # and it holds the node: the re-open's own timeout must not hide the drop from the strict classifier
    from services.docker_service import is_host_fault_create_failure

    assert is_host_fault_create_failure("container_health_check", raised.value)


@pytest.mark.asyncio
async def test_a_read_timeout_then_a_reset_while_reopening_still_holds_the_node():
    # the first error is a read timeout (a transport error, not a strict drop); the re-open then hits a reset
    import requests

    from services.docker_service import is_host_fault_create_failure

    class _TimeoutApiClient(_FakeApiClient):
        def inspect_container(self, name):
            raise requests.exceptions.ReadTimeout("read timed out")

    api_client = _TimeoutApiClient(adapter=_FakeAdapter(reopen_error=ConnectionResetError("reset by peer")))

    with pytest.raises(RentalDockerOperationError) as raised:
        await _client(api_client).inspect_container_state(container_name="pod_test")

    assert is_host_fault_create_failure("container_health_check", raised.value)
    # the create reports it at ssh_connect: the node refused the new connection
    assert isinstance(raised.value, RentalDockerTransportNotReopened)
