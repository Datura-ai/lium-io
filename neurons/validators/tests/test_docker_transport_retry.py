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
from docker.errors import NotFound
from services.node_host import node_owner
from services.rental_docker_sdk import (
    ContainerExecSpec,
    ContainerRunSpec,
    RentalDockerConnectionError,
    RentalDockerOperationError,
    RentalDockerSdkClient,
    RentalDockerSdkClientFactory,
)


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


def _client(api_client: _FakeApiClient, *, enabled: bool = True) -> RentalDockerSdkClient:
    return RentalDockerSdkClient(api_client, transport_retry_enabled=enabled)


def _run_spec(name: str = "pod_test") -> ContainerRunSpec:
    return ContainerRunSpec(image="lium/pod:1", name=name, network="lium-rentals")


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
        RentalDockerOperationError, match="already exists with image 'someone/else:latest'"
    ):
        await _client(api).run_container(_run_spec())

    assert api.calls == ["create_container", "inspect_container"]


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
# rental connect: a stalled DNS lookup is held per miner
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
            address=address, ssh_port=2222, ssh_username="root", ssh_host_key="ssh-ed25519 AAAATESTKEY"
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
