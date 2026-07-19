"""Shared harness for the docker-service characterization oracle (DAH-2382, PR-1a).

Consolidates the four near-copies of the fake rental docker client
(test_docker_service.py, test_deploy_optimizations.py, test_custom_dockerfile_build.py,
test_docker_service_rental_security.py) and their happy-path patchers
(`_patch_happy`, `_patch_create_harness`) into the ONE helper every oracle suite
builds on. Do not add a fifth ad-hoc copy — extend this module instead.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    ContainerCreateRequest,
    ContainerDeleteRequest,
    PayloadPortMapping,
    WorkloadKind,
)
from services.docker_service import DockerService
from services.rental_docker_sdk import (
    ContainerExecResult,
    ContainerExecSpec,
    ContainerRunSpec,
    build_gpu_docker_config,
)

from core.config import settings

RecordedCall = tuple[str, dict[str, Any]]

FAKE_SSH_HOST_KEY = "ssh-ed25519 AAAATESTKEY"
DEFAULT_PORT_MAPS: list[tuple[int, int, int]] = [(22, 20001, 20001)]


class FakeRentalDockerClient:
    """Recording fake of RentalDockerSdkClient.

    Every method appends ``(method_name, kwargs)`` to ``.calls`` in invocation
    order (and, prefixed ``docker.``, to the shared cross-layer ``journal``)
    before applying its configured outcome. Outcome knobs:

    - ``image_exists`` is truthy when the image is in ``existing_images``.
    - ``run_container``: ``run_errors`` is a FIFO queue raised one per call and
      then success (retry-loop tests); ``run_error`` raises on every call.
    - ``stop_error`` / ``remove_error`` raise on every call of their method.
    """

    def __init__(self, journal: list[RecordedCall] | None = None) -> None:
        self.calls: list[RecordedCall] = []
        self._journal = journal
        self.existing_images: set[str] = set()
        self.run_errors: list[Exception] = []
        self.run_error: Exception | None = None
        self.stop_error: Exception | None = None
        self.remove_error: Exception | None = None

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))
        if self._journal is not None:
            self._journal.append((f"docker.{name}", kwargs))

    def _named(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for called, kwargs in self.calls if called == name]

    # -- recorded views (parity with the assertion idioms of the four legacy fakes) --

    @property
    def pulled_images(self) -> list[str]:
        return [kwargs["image"] for kwargs in self._named("pull")]

    @property
    def run_specs(self) -> list[ContainerRunSpec]:
        return [kwargs["spec"] for kwargs in self._named("run_container")]

    @property
    def started_containers(self) -> list[str]:
        return [kwargs["container_name"] for kwargs in self._named("start")]

    @property
    def stopped_containers(self) -> list[str]:
        return [kwargs["container_name"] for kwargs in self._named("stop")]

    @property
    def removed_containers(self) -> list[dict[str, Any]]:
        return self._named("remove_container")

    @property
    def created_volumes(self) -> list[dict[str, Any]]:
        return self._named("create_volume")

    @property
    def removed_volumes(self) -> list[dict[str, Any]]:
        return self._named("remove_volume")

    @property
    def pruned_images(self) -> int:
        return len(self._named("prune_images"))

    # -- RentalDockerSdkClient surface --

    async def login(self, *, username: str, password: str) -> None:
        self._record("login", username=username, password=password)

    async def image_exists(self, *, image: str) -> bool:
        self._record("image_exists", image=image)
        return image in self.existing_images

    async def pull(self, *, image: str) -> None:
        self._record("pull", image=image)

    async def run_container(self, spec: ContainerRunSpec) -> None:
        self._record("run_container", spec=spec)
        if self.run_errors:
            raise self.run_errors.pop(0)
        if self.run_error is not None:
            raise self.run_error

    async def exec_in_container(self, spec: ContainerExecSpec) -> ContainerExecResult:
        self._record("exec_in_container", spec=spec)
        return ContainerExecResult(exit_status=0)

    async def start(self, *, container_name: str) -> None:
        self._record("start", container_name=container_name)

    async def stop(self, *, container_name: str, stop_grace_seconds: int | None = None) -> None:
        self._record("stop", container_name=container_name, stop_grace_seconds=stop_grace_seconds)
        if self.stop_error is not None:
            raise self.stop_error

    async def remove_container(
        self,
        *,
        container_name: str,
        force: bool = True,
        remove_volumes: bool = True,
    ) -> None:
        self._record(
            "remove_container",
            container_name=container_name,
            force=force,
            remove_volumes=remove_volumes,
        )
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
        self._record(
            "create_volume",
            volume_name=volume_name,
            driver=driver,
            driver_opts=driver_opts,
            timeout=timeout,
        )

    async def remove_volume(self, *, volume_name: str, force: bool = False) -> None:
        self._record("remove_volume", volume_name=volume_name, force=force)

    async def prune_images(self) -> None:
        self._record("prune_images")


class FakeRentalDockerFactory:
    def __init__(self, client: FakeRentalDockerClient) -> None:
        self.client = client
        self.connect_calls: list[dict[str, Any]] = []

    def connect(
        self, *, executor_info: ExecutorSSHInfo, private_key: str
    ) -> FakeRentalDockerFactory:
        self.connect_calls.append(
            {"executor_info": executor_info, "private_key": private_key}
        )
        return self

    async def __aenter__(self) -> FakeRentalDockerClient:
        return self.client

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


class RecordingRedisService:
    """Recording fake of RedisService for the create/delete paths.

    ``.calls`` keeps the ordered cross-op sequence (the B7/E4 sketches assert
    relative order across redis operations), ``.published`` captures pub/sub
    messages, and get/set/delete are backed by the ``values`` dict so the
    best-effort gpu-power reads see honest empty state instead of mock noise.
    """

    def __init__(self, journal: list[RecordedCall] | None = None) -> None:
        self.calls: list[RecordedCall] = []
        self._journal = journal
        self.values: dict[str, str] = {}
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.get_rented_machine_error: Exception | None = None

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))
        if self._journal is not None:
            self._journal.append((f"redis.{name}", kwargs))

    @contextlib.asynccontextmanager
    async def acquire_executor_lock(self, executor_id: str, **_: Any) -> AsyncIterator[None]:
        self._record("acquire_executor_lock", executor_id=executor_id)
        yield

    async def add_pending_pod(self, miner_hotkey: str, executor_id: str, pod_id: str) -> None:
        self._record(
            "add_pending_pod", miner_hotkey=miner_hotkey, executor_id=executor_id, pod_id=pod_id
        )

    async def remove_pending_pod(self, miner_hotkey: str, executor_id: str, pod_id: str) -> None:
        self._record(
            "remove_pending_pod", miner_hotkey=miner_hotkey, executor_id=executor_id, pod_id=pod_id
        )

    async def add_rented_pod(
        self, executor: ExecutorSSHInfo, pod_id: str, container_name: str
    ) -> None:
        self._record(
            "add_rented_pod", executor=executor, pod_id=pod_id, container_name=container_name
        )

    async def remove_rented_machine(
        self, executor: ExecutorSSHInfo, container_name: str | None = None
    ) -> None:
        # optional container_name mirrors the real RedisService signature
        self._record(
            "remove_rented_machine", executor=executor, container_name=container_name
        )

    async def get_rented_machine(self, executor: ExecutorSSHInfo) -> dict[str, Any] | None:
        # honest empty state (no rented machine) unless a test arms the error knob
        self._record("get_rented_machine", executor=executor)
        if self.get_rented_machine_error is not None:
            raise self.get_rented_machine_error
        return None

    async def publish(self, channel: str, message: dict[str, Any]) -> None:
        self._record("publish", channel=channel, message=message)
        self.published.append((channel, message))

    async def get(self, key: str) -> str | None:
        self._record("get", key=key)
        return self.values.get(key)

    async def set(self, key: str, value: str) -> None:
        self._record("set", key=key, value=value)
        self.values[key] = value

    async def delete(self, key: str) -> None:
        self._record("delete", key=key)
        self.values.pop(key, None)


class RecordingSSHClient:
    """Records every host-shell command and returns a fixed result."""

    def __init__(self, *, stdout: str = "", stderr: str = "", exit_status: int = 0) -> None:
        self.commands: list[str] = []
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status

    async def run(self, command: str, *args: Any, **kwargs: Any) -> SimpleNamespace:
        self.commands.append(command)
        return SimpleNamespace(
            stdout=self.stdout,
            stderr=self.stderr,
            exit_status=self.exit_status,
        )


class _SSHConnCtx:
    def __init__(self, ssh_client: RecordingSSHClient) -> None:
        self.ssh_client = ssh_client

    async def __aenter__(self) -> RecordingSSHClient:
        return self.ssh_client

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


def patch_rental_ssh_connect(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ssh_client: RecordingSSHClient | None = None,
) -> tuple[RecordingSSHClient, Mock]:
    # route services.docker_service.asyncssh (connect + import_private_key) to a
    # recording client; returns (client, connect_mock) so tests can assert both
    # the host commands and whether an SSH connection was opened at all
    client = ssh_client if ssh_client is not None else RecordingSSHClient()
    connect_mock = Mock(return_value=_SSHConnCtx(client))
    monkeypatch.setattr("services.docker_service.asyncssh.connect", connect_mock)
    monkeypatch.setattr(
        "services.docker_service.asyncssh.import_private_key", Mock(return_value="pkey")
    )
    return client, connect_mock


def make_create_request(**overrides: Any) -> ContainerCreateRequest:
    # Small payload with a SHORT port list — the conftest sample_executor_info
    # (1005 port_mappings) is deliberately NOT the base here.
    base: dict[str, Any] = dict(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:1.0.0",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        cpu_count=1,
        memory_gb=1,
        volume_limit_gb=2,
        storage_limit_gb=1,
        available_ports=[PayloadPortMapping(internal_port=20001, external_port=20001)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
    )
    base.update(overrides)
    return ContainerCreateRequest(**base)


def make_executor_info(payload: ContainerCreateRequest, **overrides: Any) -> ExecutorSSHInfo:
    base: dict[str, Any] = dict(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        ssh_host_key=FAKE_SSH_HOST_KEY,
    )
    base.update(overrides)
    return ExecutorSSHInfo(**base)


def make_docker_service() -> DockerService:
    # Bare-Mock dependencies; patch_create_happy_path swaps in the recording fakes.
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
    )


@dataclass
class CreateContainerHarness:
    """Seams into a patched create_container run.

    ``mocks`` maps every patched name to its mock so tests can override
    outcomes (e.g. ``harness.mocks["check_container_running"].return_value =
    False``); ``journal`` interleaves redis and docker calls in invocation
    order for cross-layer ordering asserts.
    """

    docker_client: FakeRentalDockerClient
    docker_factory: FakeRentalDockerFactory
    redis: RecordingRedisService
    ssh_client: RecordingSSHClient
    journal: list[RecordedCall]
    mocks: dict[str, Any]
    host_commands: list[str]


def patch_create_happy_path(
    svc: DockerService,
    monkeypatch: pytest.MonkeyPatch,
    *,
    port_maps: list[tuple[int, int, int]] | None = None,
    jupyter_port_map: tuple[int, int] | None = None,
    volume_limit_gb: int = 10,
    storage_limit_gb: int = 20,
    enable_inspector: bool = False,
) -> CreateContainerHarness:
    """Patch instance seams so create_container reaches a real ContainerCreated
    with no SSH and no docker daemon.

    Consolidates `_patch_happy` (test_deploy_optimizations.py) and
    `_patch_create_harness` (test_docker_service_rental_security.py). Decision
    D3: delegation is always ``self.<method>``, so seams are patched on the
    DockerService INSTANCE; only asyncssh + the gpu-config module function are
    patched in the services.docker_service namespace. The rental SDK layer
    stays REAL down to the fake client — `_run_rental_docker_create_with_port_retry`,
    run-spec building and key/environment exec injection all execute, so their
    calls land in the fake's journal.
    """
    journal: list[RecordedCall] = []
    docker_client = FakeRentalDockerClient(journal)
    docker_factory = FakeRentalDockerFactory(docker_client)
    redis = RecordingRedisService(journal)
    ssh_client = RecordingSSHClient()
    host_commands: list[str] = []

    svc.rental_docker_client_factory = docker_factory
    svc.redis_service = redis
    svc.ssh_service.decrypt_payload = Mock(return_value="private-key")

    async def _capture_streamed_command(*, command: str, **kwargs: Any) -> tuple[bool, str]:
        host_commands.append(command)
        return True, ""

    mocks: dict[str, Any] = {
        "generate_portMappings": AsyncMock(
            return_value=(list(port_maps) if port_maps is not None else list(DEFAULT_PORT_MAPS),
                          jupyter_port_map)
        ),
        "_prepare_known_hosts_policy": AsyncMock(return_value=None),
        "clean_existing_containers": AsyncMock(),
        "clean_stale_vloopback_volumes": AsyncMock(),
        "resolve_volume_sizing": AsyncMock(
            return_value=SimpleNamespace(
                volume_limit_gb=volume_limit_gb, storage_limit_gb=storage_limit_gb
            )
        ),
        "create_local_volume": AsyncMock(),
        "wait_for_port_check_containers": AsyncMock(return_value=(True, "ok")),
        "check_container_running": AsyncMock(return_value=True),
        "install_open_ssh_server_and_start_ssh_service_with_rental_docker": AsyncMock(
            return_value=True
        ),
        "run_jupyter": AsyncMock(),
        "execute_and_stream_logs": AsyncMock(side_effect=_capture_streamed_command),
        "stream_log": AsyncMock(),
        "finish_stream_logs": AsyncMock(),
        "handle_stream_logs": AsyncMock(),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(svc, name, mock)

    connect_mock = Mock(return_value=_SSHConnCtx(ssh_client))
    monkeypatch.setattr("services.docker_service.asyncssh.connect", connect_mock)
    import_private_key_mock = Mock(return_value="pkey")
    monkeypatch.setattr(
        "services.docker_service.asyncssh.import_private_key", import_private_key_mock
    )
    gpu_config_mock = AsyncMock(return_value=build_gpu_docker_config(["GPU-test"]))
    monkeypatch.setattr(
        "services.docker_service.build_gpu_docker_config_for_executor", gpu_config_mock
    )
    mocks["asyncssh.connect"] = connect_mock
    mocks["asyncssh.import_private_key"] = import_private_key_mock
    mocks["build_gpu_docker_config_for_executor"] = gpu_config_mock

    # Pin the inspector flag: goldens must not depend on the developer's env
    # (settings default is True; the B5/B6 sketches assume the step is skipped).
    monkeypatch.setattr(settings, "ENABLE_INSPECTOR", enable_inspector)

    return CreateContainerHarness(
        docker_client=docker_client,
        docker_factory=docker_factory,
        redis=redis,
        ssh_client=ssh_client,
        journal=journal,
        mocks=mocks,
        host_commands=host_commands,
    )


def make_delete_request(**overrides: Any) -> ContainerDeleteRequest:
    base: dict[str, Any] = dict(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        container_name="pod_oracle_delete",
    )
    base.update(overrides)
    return ContainerDeleteRequest(**base)


@dataclass
class DeleteContainerHarness:
    """Seams into a patched delete_container run.

    ``mocks`` maps each patched seam to its mock (set ``.side_effect`` to fail
    a step); ``journal`` interleaves docker calls and the sweep in invocation
    order for the D8 sweep-before-raise ordering assert.
    """

    docker_client: FakeRentalDockerClient
    docker_factory: FakeRentalDockerFactory
    redis: RecordingRedisService
    ssh_client: RecordingSSHClient
    journal: list[RecordedCall]
    mocks: dict[str, Any]


def patch_delete_happy_path(
    svc: DockerService,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enable_inspector: bool = False,
) -> DeleteContainerHarness:
    """Patch seams so delete_container reaches a real ContainerDeleted with no
    SSH and no docker daemon.

    The attestation chokepoint (``_prepare_known_hosts_policy``) runs for REAL —
    only ``attestation_service.prepare_host_policy`` is faked, so D11 exercises
    the actual AttestationError pass-through. ``_sweep_wedged_gpus_after_teardown``
    and ``restore_filler_pod_gpu_power_limits`` are module-level functions on
    baseline (NOT instance methods) and are patched in the
    ``services.docker_service`` namespace; the sweep journals its invocation so
    ordering against docker calls stays observable.
    """
    journal: list[RecordedCall] = []
    docker_client = FakeRentalDockerClient(journal)
    docker_factory = FakeRentalDockerFactory(docker_client)
    redis = RecordingRedisService(journal)
    ssh_client = RecordingSSHClient()

    svc.rental_docker_client_factory = docker_factory
    svc.redis_service = redis
    svc.ssh_service.decrypt_payload = Mock(return_value="private-key")
    svc.attestation_service.prepare_host_policy = AsyncMock(
        return_value=SimpleNamespace(known_hosts=None)
    )

    connect_mock = Mock(return_value=_SSHConnCtx(ssh_client))
    monkeypatch.setattr("services.docker_service.asyncssh.connect", connect_mock)
    import_key_mock = Mock(return_value="pkey")
    monkeypatch.setattr(
        "services.docker_service.asyncssh.import_private_key", import_key_mock
    )

    restore_power_mock = AsyncMock()
    monkeypatch.setattr(
        "services.docker_service.restore_filler_pod_gpu_power_limits",
        restore_power_mock,
    )

    def _journal_sweep(*args: Any, **kwargs: Any) -> None:
        journal.append(("sweep.teardown_sweep", {}))

    sweep_mock = AsyncMock(side_effect=_journal_sweep)
    monkeypatch.setattr(
        "services.docker_service._sweep_wedged_gpus_after_teardown", sweep_mock
    )

    monkeypatch.setattr(settings, "ENABLE_INSPECTOR", enable_inspector)

    mocks: dict[str, Any] = {
        "asyncssh.connect": connect_mock,
        "asyncssh.import_private_key": import_key_mock,
        "prepare_host_policy": svc.attestation_service.prepare_host_policy,
        "restore_filler_pod_gpu_power_limits": restore_power_mock,
        "_sweep_wedged_gpus_after_teardown": sweep_mock,
    }
    return DeleteContainerHarness(
        docker_client=docker_client,
        docker_factory=docker_factory,
        redis=redis,
        ssh_client=ssh_client,
        journal=journal,
        mocks=mocks,
    )
