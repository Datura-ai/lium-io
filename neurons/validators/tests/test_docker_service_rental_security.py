from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    AddSshPublicKeyRequest,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    ContainerStartRequest,
    ContainerStopRequest,
    CustomOptions,
    PayloadPortMapping,
    WorkloadKind,
)
from services.docker_service import DockerService
from services.rental_docker_sdk import ContainerExecResult, build_gpu_docker_config


HOSTILE_PASSWORD = "pw'; echo CREDENTIAL_MARKER; echo '"
HOSTILE_IMAGE = "registry.example/image;echo IMAGE_MARKER\n$(echo IMAGE_SUBSHELL)"
HOSTILE_ENV_VALUE = "value'; echo ENV_MARKER; $(echo env)"
HOSTILE_PUBLIC_KEY = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEfakeKeyForTestsOnly "
    "user@example; echo KEY_MARKER; $(echo key)"
)
HOSTILE_STARTUP_COMMAND = "\n\nsh /tmp/startup-marker.sh"
HOSTILE_CONTAINER_NAME = "pod_name; echo CONTAINER_MARKER; $(echo name)"


class DummySSHConnectionManager:
    def __init__(self, ssh_client):
        self.ssh_client = ssh_client

    async def __aenter__(self):
        return self.ssh_client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class RecordingSSHClient:
    def __init__(self):
        self.commands = []

    async def run(self, command, *args, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(stdout="", stderr="", exit_status=0)


class RecordingRentalDockerClient:
    def __init__(self):
        self.login_calls = []
        self.pulled_images = []
        self.run_specs = []
        self.exec_specs = []
        self.started_containers = []
        self.stopped_containers = []
        self.removed_containers = []

    async def login(self, *, username: str, password: str) -> None:
        self.login_calls.append({"username": username, "password": password})

    async def pull(self, *, image: str) -> None:
        self.pulled_images.append(image)

    async def run_container(self, spec) -> None:
        self.run_specs.append(spec)

    async def exec_in_container(self, spec) -> ContainerExecResult:
        self.exec_specs.append(spec)
        return ContainerExecResult(exit_status=0)

    async def start(self, *, container_name: str) -> None:
        self.started_containers.append(container_name)

    async def stop(self, *, container_name: str) -> None:
        self.stopped_containers.append(container_name)

    async def remove_container(
        self,
        *,
        container_name: str,
        force: bool = True,
        remove_volumes: bool = True,
    ) -> None:
        self.removed_containers.append(
            {
                "container_name": container_name,
                "force": force,
                "remove_volumes": remove_volumes,
            }
        )


class RecordingRentalDockerFactory:
    def __init__(self):
        self.client = RecordingRentalDockerClient()
        self.connect_calls = []

    def connect(self, *, executor_info: ExecutorSSHInfo, private_key: str):
        self.connect_calls.append(
            {"executor_info": executor_info, "private_key": private_key}
        )
        return self

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.fixture
def docker_service():
    ssh_service = Mock()
    redis_service = Mock()
    attestation_service = Mock()
    lock = AsyncMock()
    lock.__aenter__ = AsyncMock(return_value=lock)
    lock.__aexit__ = AsyncMock(return_value=None)
    redis_service.acquire_executor_lock = Mock(return_value=lock)
    return DockerService(
        ssh_service=ssh_service,
        redis_service=redis_service,
        attestation_service=attestation_service,
        rental_docker_client_factory=RecordingRentalDockerFactory(),
    )


@pytest.fixture
def executor_info():
    return ExecutorSSHInfo(
        uuid=str(uuid4()),
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app",
    )


@pytest.fixture
def keypair():
    return Mock(ss58_address="validator-hotkey")


def _base_create_payload(**overrides) -> ContainerCreateRequest:
    values = {
        "miner_hotkey": "miner-hotkey",
        "executor_id": str(uuid4()),
        "pod_id": "00000000-0000-0000-0000-0000000000aa",
        "docker_image": "daturaai/pytorch:security-test",
        "docker_username": "registry-user",
        "docker_password": HOSTILE_PASSWORD,
        "user_public_keys": [HOSTILE_PUBLIC_KEY],
        "gpu_uuids": ["GPU-test"],
        "cpu_count": 2,
        "memory_gb": 8,
        "custom_options": CustomOptions(
            volumes=["/data/rental:/workspace"],
            environment={
                "APP_MODE": "prod",
                "HOSTILE_ENV": HOSTILE_ENV_VALUE,
                "MULTILINE_ENV": "line1\nENV_NEWLINE_MARKER",
            },
            startup_commands=HOSTILE_STARTUP_COMMAND,
            shm_size="1g",
        ),
        "volume_limit_gb": 100,
        "storage_limit_gb": 50,
        "available_ports": [
            PayloadPortMapping(internal_port=22, external_port=30022),
            PayloadPortMapping(internal_port=20000, external_port=30000),
        ],
        "pod_mapping": [],
        "active_container_names": [],
        "active_volume_names": [],
    }
    values.update(overrides)
    return ContainerCreateRequest(**values)


def _lifecycle_payload(payload_cls, *, container_name: str):
    return payload_cls(
        miner_hotkey="miner-hotkey",
        executor_id=str(uuid4()),
        pod_id="pod-id",
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        container_name=container_name,
    )


def _patch_common(monkeypatch, docker_service, ssh_client):
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr(
        "services.docker_service.asyncssh.import_private_key",
        Mock(return_value="pkey"),
    )
    monkeypatch.setattr(
        docker_service,
        "_prepare_known_hosts_policy",
        AsyncMock(return_value=None),
    )
    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")


def _patch_create_harness(monkeypatch, docker_service, ssh_client):
    captured_commands = []

    async def execute_and_capture(*, command, **kwargs):
        captured_commands.append(command)
        return True, ""

    _patch_common(monkeypatch, docker_service, ssh_client)
    monkeypatch.setattr(
        "services.docker_service.build_gpu_docker_config_for_executor",
        AsyncMock(
            return_value=build_gpu_docker_config(
                ["GPU-test"],
                device_nodes=["/dev/nvidia0", "/dev/nvidiactl"],
            )
        ),
    )
    monkeypatch.setattr(docker_service, "execute_and_stream_logs", execute_and_capture)
    monkeypatch.setattr(docker_service, "clean_existing_containers", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_stale_vloopback_volumes", AsyncMock())
    monkeypatch.setattr(docker_service, "create_local_volume", AsyncMock())
    monkeypatch.setattr(
        docker_service,
        "resolve_volume_sizing",
        AsyncMock(
            return_value=SimpleNamespace(volume_limit_gb=44, storage_limit_gb=22)
        ),
    )
    monkeypatch.setattr(
        docker_service,
        "generate_portMappings",
        AsyncMock(return_value=([(22, 22, 30022), (20000, 20000, 30000)], None)),
    )
    monkeypatch.setattr(
        docker_service,
        "wait_for_port_check_containers",
        AsyncMock(return_value=(True, "ok")),
    )
    monkeypatch.setattr(
        docker_service,
        "check_container_running",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(docker_service, "stream_log", AsyncMock())
    monkeypatch.setattr(docker_service, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "handle_stream_logs", AsyncMock())
    docker_service.redis_service.add_pending_pod = AsyncMock()
    docker_service.redis_service.remove_pending_pod = AsyncMock()
    docker_service.redis_service.add_rented_pod = AsyncMock()
    return captured_commands


def _all_host_commands(captured_commands, ssh_client):
    return [*captured_commands, *ssh_client.commands]


def _assert_markers_not_in_host_shell(commands, markers):
    hits = [
        (marker, command)
        for command in commands
        for marker in markers
        if marker in command
    ]
    assert not hits, "user-controlled marker appeared in host shell command text"


@pytest.mark.asyncio
async def test_create_container_keeps_hostile_fields_out_of_host_shell_commands(
    docker_service,
    executor_info,
    keypair,
    monkeypatch,
):
    ssh_client = RecordingSSHClient()
    captured_commands = _patch_create_harness(monkeypatch, docker_service, ssh_client)
    payload = _base_create_payload(docker_image=HOSTILE_IMAGE)

    await docker_service.create_container(
        payload=payload,
        executor_info=executor_info,
        keypair=keypair,
        private_key="encrypted-private-key",
    )

    docker_client = docker_service.rental_docker_client_factory.client
    run_spec = docker_client.run_specs[0]
    key_specs = [
        spec for spec in docker_client.exec_specs if "authorized_keys" in " ".join(spec.argv)
    ]
    env_specs = [
        spec for spec in docker_client.exec_specs if "/etc/environment" in " ".join(spec.argv)
    ]

    _assert_markers_not_in_host_shell(
        _all_host_commands(captured_commands, ssh_client),
        [
            "CREDENTIAL_MARKER",
            "IMAGE_MARKER",
            "IMAGE_SUBSHELL",
            "ENV_MARKER",
            "ENV_NEWLINE_MARKER",
            "KEY_MARKER",
            "startup-marker.sh",
        ],
    )
    assert docker_client.login_calls == [
        {"username": "registry-user", "password": HOSTILE_PASSWORD}
    ]
    assert docker_client.pulled_images == [HOSTILE_IMAGE]
    assert run_spec.image == HOSTILE_IMAGE
    assert run_spec.command == ("sh", "/tmp/startup-marker.sh")
    assert run_spec.environment["HOSTILE_ENV"] == HOSTILE_ENV_VALUE
    assert len(key_specs) == 1
    assert HOSTILE_PUBLIC_KEY not in " ".join(key_specs[0].argv)
    assert key_specs[0].stdin == f"{HOSTILE_PUBLIC_KEY}\n"
    assert len(env_specs) == 1
    assert env_specs[0].argv == ("sh", "-c", "cat >> /etc/environment")
    assert "ENV_NEWLINE_MARKER" in env_specs[0].stdin


@pytest.mark.asyncio
async def test_add_ssh_key_writes_public_keys_as_stdin_data(
    docker_service,
    executor_info,
    keypair,
    monkeypatch,
):
    ssh_client = RecordingSSHClient()
    _patch_common(monkeypatch, docker_service, ssh_client)
    payload = AddSshPublicKeyRequest(
        miner_hotkey="miner-hotkey",
        executor_id=str(uuid4()),
        pod_id="pod-id",
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        container_name=HOSTILE_CONTAINER_NAME,
        user_public_keys=[HOSTILE_PUBLIC_KEY],
    )

    await docker_service.add_ssh_key(
        payload,
        executor_info,
        keypair,
        "encrypted-private-key",
    )

    docker_client = docker_service.rental_docker_client_factory.client
    assert len(docker_client.exec_specs) == 1
    spec = docker_client.exec_specs[0]
    _assert_markers_not_in_host_shell(ssh_client.commands, ["KEY_MARKER"])
    assert spec.container_name == HOSTILE_CONTAINER_NAME
    assert HOSTILE_PUBLIC_KEY not in " ".join(spec.argv)
    assert spec.stdin == f"{HOSTILE_PUBLIC_KEY}\n"


@pytest.mark.asyncio
async def test_lifecycle_operations_pass_container_names_as_sdk_data(
    docker_service,
    executor_info,
    keypair,
    monkeypatch,
):
    ssh_client = RecordingSSHClient()
    retried_commands = []

    async def retry_capture(ssh_client, command, *args, **kwargs):
        retried_commands.append(command)

    _patch_common(monkeypatch, docker_service, ssh_client)
    monkeypatch.setattr("services.docker_service.retry_ssh_command", retry_capture)
    monkeypatch.setattr(docker_service, "_cleanup_custom_build_artifacts", AsyncMock())
    monkeypatch.setattr(
        docker_service,
        "install_open_ssh_server_and_start_ssh_service_with_rental_docker",
        AsyncMock(return_value=True),
    )
    docker_service.redis_service.remove_rented_machine = AsyncMock()

    await docker_service.start_container(
        _lifecycle_payload(ContainerStartRequest, container_name=HOSTILE_CONTAINER_NAME),
        executor_info,
        keypair,
        "encrypted-private-key",
    )
    await docker_service.stop_container(
        _lifecycle_payload(ContainerStopRequest, container_name=HOSTILE_CONTAINER_NAME),
        executor_info,
        keypair,
        "encrypted-private-key",
    )
    await docker_service.delete_container(
        ContainerDeleteRequest(
            miner_hotkey="miner-hotkey",
            executor_id=str(uuid4()),
            pod_id="pod-id",
            workload_kind=WorkloadKind.CUSTOMER_RENTAL,
            container_name=HOSTILE_CONTAINER_NAME,
            local_volume="volume_lifecycle",
        ),
        executor_info,
        keypair,
        "encrypted-private-key",
    )

    docker_client = docker_service.rental_docker_client_factory.client
    _assert_markers_not_in_host_shell(
        _all_host_commands(retried_commands, ssh_client),
        ["CONTAINER_MARKER"],
    )
    assert docker_client.started_containers == [HOSTILE_CONTAINER_NAME]
    assert docker_client.stopped_containers == [HOSTILE_CONTAINER_NAME]
    assert docker_client.removed_containers == [
        {
            "container_name": HOSTILE_CONTAINER_NAME,
            "force": True,
            "remove_volumes": True,
        }
    ]
