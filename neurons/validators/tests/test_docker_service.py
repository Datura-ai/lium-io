from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

import services.docker_service as docker_service_module
from services.docker_service import (
    DockerService,
)
from services.rental_docker_sdk import (
    ContainerExecResult,
    ContainerRunSpec,
    ContainerStateSnapshot,
    build_gpu_docker_config,
)
from payload_models.payloads import (
    ContainerCreateRequest,
)
from datura.requests.miner_requests import ExecutorSSHInfo


FAKE_SSH_HOST_KEY = "ssh-ed25519 AAAATESTKEY"


class _FakeRentalDockerClient:
    def __init__(self):
        self.login_calls = []
        self.inspected_images = []
        self.existing_images = set()
        self.pulled_images = []
        self.run_specs = []
        self.exec_specs = []
        self.started_containers = []
        self.stopped_containers = []
        self.stop_grace_seconds_calls = []
        self.removed_containers = []
        # (operation, container_name) tuples shared by stop/remove, so tests can assert ordering
        self.container_call_order = []
        self.created_volumes = []
        self.removed_volumes = []
        self.pruned_images = 0
        self.login_error = None
        self.pull_error = None
        self.run_error = None
        self.start_error = None
        self.stop_error = None
        self.remove_error = None
        self.remove_volume_error = None
        # per-call answers for remove_volume, consumed in order (None = success); once empty,
        # remove_volume_error applies
        self.remove_volume_errors: list[Exception | None] = []
        self.prune_images_error = None
        # DAH-3467: answers for container_status, consumed in order; the last one repeats.
        # None = 404 (gone), a str = State.Status, an Exception = the inspect raised it.
        self.container_statuses: list[str | Exception | None] = []
        self.inspected_containers = []

    async def container_status(self, *, container_name: str) -> str | None:
        self.inspected_containers.append(container_name)
        if not self.container_statuses:
            raise AssertionError("container_status called without a scripted answer")
        answer = (
            self.container_statuses.pop(0)
            if len(self.container_statuses) > 1
            else self.container_statuses[0]
        )
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def login(self, *, username: str, password: str, image: str) -> None:
        self.login_calls.append({"username": username, "password": password, "image": image})
        if self.login_error is not None:
            raise self.login_error

    async def image_exists(self, *, image: str) -> bool:
        self.inspected_images.append(image)
        return image in self.existing_images

    async def local_image_repo_digests(self, *, image: str) -> tuple[str, ...] | None:
        return () if await self.image_exists(image=image) else None

    async def local_image_is_current(self, *, image: str, auth_config: dict[str, str] | None = None) -> bool:
        return True

    async def pull(self, *, image: str) -> None:
        self.pulled_images.append(image)
        if self.pull_error is not None:
            raise self.pull_error

    async def run_container(self, spec) -> None:
        self.run_specs.append(spec)
        if self.run_error is not None:
            raise self.run_error

    async def exec_in_container(self, spec) -> ContainerExecResult:
        self.exec_specs.append(spec)
        return ContainerExecResult(exit_status=0)

    async def inspect_container_state(self, *, container_name: str) -> ContainerStateSnapshot:
        return ContainerStateSnapshot(
            status="running", running=True, restarting=False, exit_code=0, restart_count=0, error=None,
            oom_killed=False,
        )

    async def start(self, *, container_name: str) -> None:
        self.started_containers.append(container_name)
        if self.start_error is not None:
            raise self.start_error

    async def stop(self, *, container_name: str, stop_grace_seconds: int | None = None) -> None:
        self.stopped_containers.append(container_name)
        self.stop_grace_seconds_calls.append(stop_grace_seconds)
        self.container_call_order.append(("stop", container_name))
        if self.stop_error is not None:
            raise self.stop_error

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
        self.container_call_order.append(("remove", container_name))
        if self.remove_error is not None:
            raise self.remove_error

    async def create_volume(
        self,
        *,
        volume_name: str,
        driver: str | None = None,
        driver_opts: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> None:
        self.created_volumes.append(
            {
                "volume_name": volume_name,
                "driver": driver,
                "driver_opts": driver_opts,
                "timeout": timeout,
            }
        )

    async def remove_volume(self, *, volume_name: str, force: bool = False) -> None:
        self.removed_volumes.append(
            {"volume_name": volume_name, "force": force}
        )
        if self.remove_volume_errors:
            error = self.remove_volume_errors.pop(0)
            if error is not None:
                raise error
            return
        if self.remove_volume_error is not None:
            raise self.remove_volume_error

    async def mount_source_for_destination(
        self, *, container_name: str, destination: str
    ) -> str | None:
        return None

    async def prune_images(self) -> None:
        self.pruned_images += 1
        if self.prune_images_error is not None:
            raise self.prune_images_error


class _FakeRentalDockerFactory:
    def __init__(self):
        self.client = _FakeRentalDockerClient()
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
def mock_dependencies():
    """Mock all DockerService dependencies."""
    ssh_service = Mock()
    redis_service = Mock()
    attestation_service = Mock()

    # Mock the async context manager for Redis lock
    lock_mock = AsyncMock()
    lock_mock.__aenter__ = AsyncMock(return_value=lock_mock)
    lock_mock.__aexit__ = AsyncMock(return_value=None)
    redis_service.acquire_executor_lock = Mock(return_value=lock_mock)

    return ssh_service, redis_service, attestation_service


@pytest_asyncio.fixture
async def docker_service(mock_dependencies):
    """Create DockerService instance with mocked dependencies."""
    ssh_service, redis_service, attestation_service = mock_dependencies
    service = DockerService(
        ssh_service=ssh_service,
        redis_service=redis_service,
        attestation_service=attestation_service,
        rental_docker_client_factory=_FakeRentalDockerFactory(),
    )
    # the pre-run kernel.pid_max read is mandatory; the SSH doubles here do not model /proc
    service._read_host_pid_max = AsyncMock(return_value=4_194_304)
    return service


# =============================================================================
# Tests for _convert_payload_ports and backend data flow
# =============================================================================


# =============================================================================
# Tests for clean_existing_containers
# =============================================================================


class DummySSHConnectionManager:
    def __init__(self, ssh_client):
        self.ssh_client = ssh_client

    async def __aenter__(self):
        return self.ssh_client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.fixture
def retry_ssh_mock(monkeypatch):
    """Mock retry_ssh_command to capture SSH commands without executing."""
    import services.docker_service as ds_module
    mock = AsyncMock()
    monkeypatch.setattr(ds_module, "retry_ssh_command", mock)
    return mock


def _make_ssh_command_result(exit_status: int = 0, stdout: str = "", stderr: str = ""):
    result = Mock()
    result.exit_status = exit_status
    result.stdout = stdout
    result.stderr = stderr
    return result


def _patch_create_container_happy_path(docker_service, monkeypatch):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result())
    monkeypatch.setattr(
        docker_service_module.asyncssh,
        "connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr(docker_service_module.asyncssh, "import_private_key", Mock())
    monkeypatch.setattr(docker_service_module, "build_gpu_flags", AsyncMock(return_value=""))

    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.add_pending_pod = AsyncMock()
    docker_service.redis_service.remove_pending_pod = AsyncMock()
    docker_service.redis_service.add_rented_pod = AsyncMock()
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(
        docker_service,
        "generate_portMappings",
        AsyncMock(return_value=([(22, 20001, 20001)], None)),
    )
    monkeypatch.setattr(docker_service, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(docker_service, "clean_existing_containers", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_stale_vloopback_volumes", AsyncMock())
    monkeypatch.setattr(docker_service, "create_local_volume", AsyncMock())
    monkeypatch.setattr(
        docker_service,
        "wait_for_port_check_containers",
        AsyncMock(return_value=(True, "ok")),
    )
    monkeypatch.setattr(docker_service, "_run_docker_create_with_port_retry", AsyncMock())
    monkeypatch.setattr(docker_service, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(
        docker_service,
        "install_open_ssh_server_and_start_ssh_service",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(docker_service, "stream_log", AsyncMock())
    monkeypatch.setattr(docker_service, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "handle_stream_logs", AsyncMock())

    return ssh_client


def _patch_delete_container_connect(docker_service, monkeypatch, retry_ssh_mock):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result())
    monkeypatch.setattr(
        docker_service_module.asyncssh,
        "connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr(docker_service_module.asyncssh, "import_private_key", Mock())
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))

    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.remove_rented_machine = AsyncMock()
    retry_ssh_mock.return_value = None

    return ssh_client


@pytest.mark.asyncio
async def test_inspector_lifecycle_command_quotes_executor_paths(docker_service):
    executor_info = ExecutorSSHInfo(
        uuid="exec-1",
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python3",
        root_dir="/root/app dir",
    )

    start_command = docker_service._build_inspector_collector_command(
        executor_info,
        "start",
    )
    stop_command = docker_service._build_inspector_collector_command(
        executor_info,
        "stop",
    )

    assert start_command == (
        "nohup /usr/bin/python3 '/root/app dir/src/inspector_executor.py'"
        " --start-collector >/dev/null 2>&1 &"
    )
    assert stop_command == (
        "/usr/bin/python3 '/root/app dir/src/inspector_executor.py' --stop-collector"
    )


def _patch_create_container_happy_path(docker_service, monkeypatch):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result())
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr(
        "services.docker_service.build_gpu_docker_config_for_executor",
        AsyncMock(return_value=build_gpu_docker_config(["GPU-test"])),
    )

    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.add_pending_pod = AsyncMock()
    docker_service.redis_service.remove_pending_pod = AsyncMock()
    docker_service.redis_service.add_rented_pod = AsyncMock()
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(
        docker_service,
        "generate_portMappings",
        AsyncMock(return_value=([(20000, 20020, 20020)], None)),
    )
    monkeypatch.setattr(docker_service, "execute_and_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_existing_containers", AsyncMock())
    monkeypatch.setattr(docker_service, "clean_stale_vloopback_volumes", AsyncMock())
    monkeypatch.setattr(docker_service, "create_local_volume", AsyncMock())
    monkeypatch.setattr(
        docker_service,
        "wait_for_port_check_containers",
        AsyncMock(return_value=(True, "ok")),
    )
    monkeypatch.setattr(docker_service, "_run_docker_create_with_port_retry", AsyncMock())
    monkeypatch.setattr(docker_service, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(docker_service, "install_open_ssh_server_and_start_ssh_service", AsyncMock())
    monkeypatch.setattr(docker_service, "stream_log", AsyncMock())
    monkeypatch.setattr(docker_service, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(docker_service, "handle_stream_logs", AsyncMock())
    return ssh_client


def _patch_delete_container_connect(docker_service, monkeypatch, retry_ssh_mock):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result())
    monkeypatch.setattr(
        "services.docker_service.asyncssh.connect",
        Mock(return_value=DummySSHConnectionManager(ssh_client)),
    )
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    docker_service.ssh_service.decrypt_payload = Mock(return_value="private-key")
    docker_service.redis_service.remove_rented_machine = AsyncMock()
    retry_ssh_mock.return_value = None
    return ssh_client


def _executor_info_for(payload: ContainerCreateRequest, *, tdx_quote: str | None) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        ssh_host_key=FAKE_SSH_HOST_KEY,
        tdx_quote=tdx_quote,
    )


# ---------------------------------------------------------------------------
# DAH-2265 Plan 3: sparse vloopback volume creation, gated to full-node rentals.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DAH-1991: port-9101 race — same-command retry on "port is already allocated"
# + wait_for_port_check_containers extended to include `health_check_*`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_stale_vloopback_mountpoint_refuses_unsafe_volume_name(docker_service):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock()

    for local_volume in ("../volume_test", ".", "volume test"):
        repaired = await docker_service.repair_stale_vloopback_mountpoint(
            ssh_client=ssh_client,
            local_volume=local_volume,
            default_extra={},
        )
        assert repaired is False

    ssh_client.run.assert_not_awaited()


# ---------------------------------------------------------------------------
# DAH-2018: container-name conflict — between port-allocated retries we
# `docker rm -fv <container_name>` to release the name Docker reserved during
# the prior `docker run` parse. Cleanup runs AFTER the backoff sleep so the
# rm→run window stays tight; cleanup failures warn-log but never abort.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DAH-2018: late re-check of port_check containers right before `docker run`
# (after the image pull) reuses the open ssh_client instead of dialing again.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DAH-2272: rentals never wait — a lingering probe is force-removed on sight.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DAH-2183: validator-side fresh vloopback sizing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_encrypted_local_volume_does_not_log_key(docker_service, caplog):
    from services.volume_keys import derive_volume_passphrase

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=_make_ssh_command_result())
    master_secret = "test-master-secret-32-chars-long!!"
    volume_name = "volume_test"
    pod_id = "pod-id"
    passphrase = derive_volume_passphrase(master_secret, pod_id)

    with caplog.at_level("INFO"):
        with patch.object(docker_service_module.settings, "VOLUME_MASTER_SECRET", master_secret):
            await docker_service.setup_encrypted_local_volume(
                ssh_client=ssh_client,
                container_name="pod_test",
                plaintext_path="/root",
                volume_name=volume_name,
                pod_id=pod_id,
                log_tag="test",
                log_extra={},
                authorized_keys=["ssh-ed25519 AAAA'$(id)' renter"],
            )

    logged = " ".join(rec.getMessage() for rec in caplog.records)
    assert master_secret not in logged
    assert passphrase not in logged
    assert all(master_secret not in str(rec.msg or "") for rec in caplog.records)
    commands = [
        call.args[0]
        for call in ssh_client.run.await_args_list
        if call.args
    ]
    assert all("docker cp" not in command for command in commands)
    assert all(master_secret not in command for command in commands)
    assert all(passphrase not in command for command in commands)
    # The script that carries the (wrapped) passphrase travels on the SSH channel's stdin, never
    # in a command string: the command string is the remote shell's argv, which every process on
    # the host can read. Nothing else may use stdin here.
    stdin_calls = [
        call for call in ssh_client.run.await_args_list if call.kwargs.get("input") is not None
    ]
    assert len(stdin_calls) == 1
    upload_call = stdin_calls[0]
    upload_cmd = upload_call.args[0]
    # 0600 from the first byte: the script holds the same material as the passfile it writes
    assert upload_cmd.startswith("/usr/bin/docker exec -u 0 -i pod_test sh -c ")
    assert "(umask 077 && dd bs=1 " in upload_cmd
    assert f"{docker_service_module._VOLUME_SETUP_TMPFS}/.x" in upload_cmd
    assert "<<" not in upload_cmd
    # the renter's key is data on stdin behind the script, never shell text
    assert "AAAA" not in upload_cmd
    stdin_data = upload_call.kwargs["input"]
    setup_script, renter_keys = stdin_data.split("ssh-ed25519", 1)
    assert renter_keys == " AAAA'$(id)' renter\n"
    assert f"dd bs=1 count={len(setup_script)} of=" in upload_cmd
    assert "gocryptfs" in setup_script
    assert passphrase not in setup_script
    assert passphrase.encode("ascii").hex() not in setup_script
    assert f'_pf={docker_service_module._VOLUME_SETUP_TMPFS}/.x' in setup_script
    # nothing about the key ever lands on the container's writable layer
    assert all("/tmp/.x" not in command for command in commands)
    assert "/tmp/" not in setup_script


# DAH-2703: a create that fails because the container was removed from the host mid-creation is
# reported with its own error code, so the backend can count host kills separately from ordinary
# create failures.


# DAH-3678: the backend turns an `add_public_keys` failure into the DAH-2624 renter text ("the image's
# default command exits right after start … restarting while the SSH keys were being installed") only
# when the validator's failure text carries one of its markers (`is not running`, `is restarting`,
# `status='exited'`, `status='restarting'`). Whatever form the exec failure took, the validator now
# looks at the container and names the exiting image when it has exited or restarted.


def _state(**overrides) -> ContainerStateSnapshot:
    base = dict(
        status="running", running=True, restarting=False, exit_code=0, restart_count=0, error=None, oom_killed=False
    )
    return ContainerStateSnapshot(**{**base, **overrides})


# DAH-3980: on the encrypted path the keys ride at the end of the one volume setup exec; its exit
# status says which part failed (90 upload, 91 init/mount, 92 mount check, 93 keys).


_VLOOPBACK_STALE_MOUNT_ERR = (
    "docker: Error response from daemon: failed to populate volume: "
    "error while mounting volume '/mnt/volume_test': "
    "VolumeDriver.Mount: error while mounting volume: "
    "cannot create mount point dir '/mnt/volume_test': "
    "mkdir /mnt/volume_test: file exists"
)


@pytest.mark.asyncio
async def test_rental_create_removes_the_failed_container_before_the_stale_mount_retry(
    docker_service, monkeypatch,
):
    events: list[str] = []

    class FakeRentalClient:
        async def run_container(self, spec):
            events.append("run")
            if events.count("run") == 1:
                raise Exception(_VLOOPBACK_STALE_MOUNT_ERR)
            return "cid"

        async def remove_container(self, *, container_name, force, remove_volumes):
            events.append(f"rm {container_name}")

    monkeypatch.setattr(
        docker_service, "repair_stale_vloopback_mountpoint", AsyncMock(return_value=True)
    )

    container_id = await docker_service._run_rental_docker_create_with_port_retry(
        docker_client=FakeRentalClient(),
        ssh_client=Mock(),
        run_spec=ContainerRunSpec(image="img", name="pod_test"),
        container_name="pod_test",
        default_extra={},
        local_volume="volume_test",
    )

    assert container_id == "cid"
    assert events == ["run", "rm pod_test", "run"]

