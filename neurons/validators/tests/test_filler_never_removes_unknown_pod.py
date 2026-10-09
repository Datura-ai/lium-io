"""A filler create never removes, nor starts beside, a `pod_*` its active_container_names do not name.

The backend builds that list when it sends the filler, before a customer's pod exists. A filler create
that reaches its cleanup after a customer's `docker run` (a retried send landing late, a lapsed create
lock, a second connector, a delete that failed before its cancel) would otherwise `docker rm -fv` the
customer's container and run on the rented node. It refuses instead, the way #1518 refuses (no strike):
at the cleanup, on the listing it already has, and in the Docker thread between `create` and `start`.
"""

from __future__ import annotations

import contextlib
import logging
import shlex
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    FailedContainerErrorCodes,
    PayloadPortMapping,
    WorkloadKind,
)

from services.docker_service import (
    DOCKER_FILLER_VOLUME_LS_NAME_DRIVER_CMD,
    DockerService,
    customer_creates,
)
from services.prerun_host_probe import DOCKER_PS_ALL_NAMES_IDS_CMD, DOCKER_VOLUME_LS_NAME_DRIVER_CMD
from services.rental_docker_sdk import (
    FILLER_VOLUME_LABELS,
    RENTAL_NETWORK_OPTIONS,
    ContainerExecResult,
    ContainerStateSnapshot,
    RentalDockerSdkClient,
    build_gpu_docker_config,
)

_CUSTOMER_POD = "pod_6a1f2a52-6f0e-4a3e-9c55-0d3e1b7a9c11"
_CUSTOMER_VOLUME = "volume_6a1f2a52-6f0e-4a3e-9c55-0d3e1b7a9c11"


class _ConnCtx:
    def __init__(self, ssh):
        self.ssh = ssh

    async def __aenter__(self):
        return self.ssh

    async def __aexit__(self, *exc):
        return None


def _ssh_result(exit_status: int = 0, stdout: str = "", stderr: str = ""):
    return Mock(exit_status=exit_status, stdout=stdout, stderr=stderr)


def _ssh_client():
    client = AsyncMock()

    def answer_like_a_clean_host(cmd, *args, **kwargs):
        return _ssh_result(exit_status=0, stdout="4194304\n" if cmd == "cat /proc/sys/kernel/pid_max" else "")

    client.run = AsyncMock(side_effect=answer_like_a_clean_host)
    return client


class _FakeRentalDockerClient:
    """The rental Docker client with the image already on the host; `run_container` is replaced per test."""

    async def login(self, *, username: str, password: str, image: str) -> None:
        return None

    async def image_exists(self, *, image: str) -> bool:
        return True

    async def local_image_repo_digests(self, *, image: str) -> tuple[str, ...]:
        return ()

    async def local_image_is_current(self, *, image: str, auth_config: dict[str, str] | None = None) -> bool:
        return True

    async def pull(self, *, image: str) -> None:
        return None

    async def run_container(self, spec) -> None:
        return None

    async def exec_in_container(self, spec) -> ContainerExecResult:
        return ContainerExecResult(exit_status=0)

    async def inspect_container_state(self, *, container_name: str) -> ContainerStateSnapshot:
        return ContainerStateSnapshot(
            status="running", running=True, restarting=False, exit_code=0, restart_count=0, error=None,
            oom_killed=False,
        )


class _FakeRentalDockerFactory:
    def __init__(self, client):
        self.client = client

    def connect(self, *, executor_info: ExecutorSSHInfo, private_key: str):
        return self

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *exc):
        return None


def _payload(**over) -> ContainerCreateRequest:
    base = dict(
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
    base.update(over)
    return ContainerCreateRequest(**base)


def _executor_info(payload: ContainerCreateRequest) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=payload.executor_id,
        address="127.0.0.1",
        port=8080,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        ssh_host_key="ssh-ed25519 AAAATESTKEY",
    )


def _patch_happy(svc, monkeypatch, ssh_client):
    """Stub the deploy flow around the create so it reaches a real ContainerCreated."""
    svc.rental_docker_client_factory = _FakeRentalDockerFactory(_FakeRentalDockerClient())
    monkeypatch.setattr("services.docker_service.asyncssh.connect", Mock(return_value=_ConnCtx(ssh_client)))
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    monkeypatch.setattr(
        "services.docker_service.build_gpu_docker_config_for_executor",
        AsyncMock(return_value=build_gpu_docker_config(["GPU-test"])),
    )
    svc.ssh_service.decrypt_payload = Mock(return_value="private-key")
    svc.redis_service.add_pending_pod = AsyncMock()
    svc.redis_service.remove_pending_pod = AsyncMock()
    svc.redis_service.add_rented_pod = AsyncMock()
    lock = AsyncMock()
    lock.__aenter__ = AsyncMock(return_value=lock)
    lock.__aexit__ = AsyncMock(return_value=None)
    svc.redis_service.acquire_executor_lock = Mock(return_value=lock)
    monkeypatch.setattr(svc, "_prepare_known_hosts_policy", AsyncMock(return_value=None))
    monkeypatch.setattr(svc, "generate_portMappings", AsyncMock(return_value=([(22, 20001, 20001)], None)))
    monkeypatch.setattr(svc, "clean_existing_containers", AsyncMock(return_value=[]))
    monkeypatch.setattr(svc, "clean_stale_vloopback_volumes", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        svc, "resolve_volume_sizing", AsyncMock(return_value=Mock(volume_limit_gb=10, storage_limit_gb=20))
    )
    monkeypatch.setattr(svc, "create_local_volume", AsyncMock())
    monkeypatch.setattr(svc, "wait_for_port_check_containers", AsyncMock(return_value=(True, "ok")))
    monkeypatch.setattr(svc, "_run_rental_docker_create_with_port_retry", AsyncMock())
    monkeypatch.setattr(svc, "check_container_running", AsyncMock(return_value=True))
    monkeypatch.setattr(
        svc, "install_open_ssh_server_and_start_ssh_service_with_rental_docker", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(svc, "run_jupyter", AsyncMock())
    monkeypatch.setattr(svc, "execute_and_stream_logs", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(svc, "stream_log", AsyncMock())
    monkeypatch.setattr(svc, "finish_stream_logs", AsyncMock())
    monkeypatch.setattr(svc, "handle_stream_logs", AsyncMock())


async def _run(svc, payload):
    return await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


def _docker_client(svc):
    return svc.rental_docker_client_factory.client


@pytest.fixture
def svc() -> DockerService:
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


def _host_listing(
    svc,
    monkeypatch,
    container_names: list[str],
    unmounted_vloopback_volumes: tuple[str, ...] = (),
    unmounted_filler_volumes: tuple[str, ...] = (),
):
    """The create path with its real cleanups on a host whose `docker ps -a` lists ``container_names``
    and whose `docker volume ls` lists ``unmounted_vloopback_volumes`` and ``unmounted_filler_volumes``
    (no container mounts them); only the latter carry the filler label."""
    ssh_client = _ssh_client()
    answer_like_a_clean_host = ssh_client.run.side_effect

    def run(cmd, *args, **kwargs):
        if cmd == DOCKER_PS_ALL_NAMES_IDS_CMD:
            return _ssh_result(exit_status=0, stdout="".join(f"{name}\n" for name in container_names))
        if cmd == DOCKER_VOLUME_LS_NAME_DRIVER_CMD:
            volumes = (*unmounted_vloopback_volumes, *unmounted_filler_volumes)
            return _ssh_result(exit_status=0, stdout="".join(f"{name} vloopback\n" for name in volumes))
        if cmd == DOCKER_FILLER_VOLUME_LS_NAME_DRIVER_CMD:
            return _ssh_result(exit_status=0, stdout="".join(f"{name} vloopback\n" for name in unmounted_filler_volumes))
        return answer_like_a_clean_host(cmd, *args, **kwargs)

    ssh_client.run.side_effect = run
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.delattr(svc, "clean_existing_containers")
    monkeypatch.delattr(svc, "clean_stale_vloopback_volumes")
    return ssh_client


def _docker_daemon(svc, monkeypatch) -> Mock:
    """The real SDK run (create, then start) against a fake daemon that lists no container."""
    monkeypatch.delattr(svc, "_run_rental_docker_create_with_port_retry")
    docker_api = Mock(inspect_network=Mock(return_value={"Driver": "bridge", "Options": RENTAL_NETWORK_OPTIONS}))
    docker_api.create_container.return_value = {"Id": "filler-container"}
    docker_api.containers.return_value = []
    _docker_client(svc).run_container = RentalDockerSdkClient(docker_api).run_container
    return docker_api


def _removals(ssh_client) -> list[list[str]]:
    # the targets of every `docker rm` / `docker volume rm` the create sent
    commands = [call.args[0] for call in ssh_client.run.await_args_list if call.args]
    return [shlex.split(cmd)[2:] for cmd in commands if cmd.startswith(("/usr/bin/docker rm", "/usr/bin/docker volume rm"))]


@pytest.mark.asyncio
async def test_late_filler_create_refuses_and_leaves_the_customer_pod_on_the_host(svc, monkeypatch) -> None:
    ssh_client = _host_listing(svc, monkeypatch, [_CUSTOMER_POD, "filler_old-run"])
    filler = _payload(workload_kind=WorkloadKind.FILLER, active_container_names=[])

    result = await _run(svc, filler)

    assert result.error_code == FailedContainerErrorCodes.RentingInProgress
    assert result.failure_step == "customer_create_in_flight"
    assert _removals(ssh_client) == []
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_pod_appearing_between_create_and_start_refuses_and_removes_only_the_filler(svc, monkeypatch) -> None:
    ssh_client = _host_listing(svc, monkeypatch, [])
    restore_power = AsyncMock()
    monkeypatch.setattr("services.docker_service.restore_filler_pod_gpu_power_limits", restore_power)
    docker_api = _docker_daemon(svc, monkeypatch)
    filler = _payload(workload_kind=WorkloadKind.FILLER)

    def customer_pod_created_meanwhile(**_kwargs) -> dict:
        docker_api.containers.return_value = [{"Names": [f"/{_CUSTOMER_POD}"]}, {"Names": [f"/filler_{filler.pod_id}"]}]
        return {"Id": "filler-container"}

    docker_api.create_container.side_effect = customer_pod_created_meanwhile

    result = await _run(svc, filler)

    assert result.error_code == FailedContainerErrorCodes.RentingInProgress
    assert result.failure_step == "customer_create_in_flight"
    docker_api.start.assert_not_called()
    docker_api.containers.assert_called_once_with(all=True)
    removals = _removals(ssh_client)
    assert any(f"filler_{filler.pod_id}" in targets for targets in removals)
    assert any(f"volume_{filler.pod_id}" in targets for targets in removals)
    assert not any(_CUSTOMER_POD in targets for targets in removals)
    restore_power.assert_awaited_once()


@pytest.mark.asyncio
async def test_pod_appearing_beside_a_running_customer_create_still_logs_the_pod(svc, monkeypatch, caplog) -> None:
    # both guards hold at once: the customer create registered, and its pod is on the host at the start check
    _host_listing(svc, monkeypatch, [])
    docker_api = _docker_daemon(svc, monkeypatch)
    monkeypatch.setattr("services.docker_service.restore_filler_pod_gpu_power_limits", AsyncMock())
    filler = _payload(workload_kind=WorkloadKind.FILLER)
    customer = _payload(executor_id=filler.executor_id)
    customer_registration = contextlib.ExitStack()

    def customer_create_registered_and_pod_created_meanwhile(**_kwargs) -> dict:
        customer_registration.enter_context(customer_creates.track(customer))
        customer_creates.reached_create_container(customer)
        docker_api.containers.return_value = [{"Names": [f"/{_CUSTOMER_POD}"]}]
        return {"Id": "filler-container"}

    docker_api.create_container.side_effect = customer_create_registered_and_pod_created_meanwhile

    with customer_registration, caplog.at_level(logging.INFO, logger="services.docker_service"):
        result = await _run(svc, filler)

    assert result.error_code == FailedContainerErrorCodes.RentingInProgress
    docker_api.start.assert_not_called()
    refusals = [r.msg.to_full_string() for r in caplog.records if "filler create refused" in r.getMessage()]
    assert len(refusals) == 1
    assert refusals[0].startswith("filler create refused: unlisted pod_* on the node >>> ")
    assert _CUSTOMER_POD in refusals[0]


@pytest.mark.parametrize(
    ("host_containers", "listed"),
    [(["filler_old-run"], []), (["filler_old-run", "pod_listed", "pod_listed__prev"], ["pod_listed"])],
    ids=["no_pod", "only_pods_the_backend_listed"],
)
@pytest.mark.asyncio
async def test_filler_create_without_an_unlisted_pod_starts_as_on_main(
    svc, monkeypatch, host_containers: list[str], listed: list[str]
) -> None:
    ssh_client = _host_listing(svc, monkeypatch, host_containers)
    docker_api = _docker_daemon(svc, monkeypatch)
    filler = _payload(workload_kind=WorkloadKind.FILLER, active_container_names=listed)

    result = await _run(svc, filler)

    assert isinstance(result, ContainerCreated)
    assert _removals(ssh_client)[0] == ["-fv", "filler_old-run"]
    docker_api.start.assert_called_once_with(f"filler_{filler.pod_id}")
    # the one extra cost on a clean host: one listing between create and start
    docker_api.containers.assert_called_once_with(all=True)


@pytest.mark.asyncio
async def test_exited_pod_of_an_ended_rental_is_left_to_the_reaper_and_refuses_the_filler(svc, monkeypatch) -> None:
    # `docker ps -a` lists it like a running one: the filler cannot tell an ended rental's pod from a
    # customer's stopped one, so it removes neither the container nor its volume
    ssh_client = _host_listing(svc, monkeypatch, ["pod_ended-rental"])
    filler = _payload(workload_kind=WorkloadKind.FILLER, active_volume_names=[])

    result = await _run(svc, filler)

    assert result.error_code == FailedContainerErrorCodes.RentingInProgress
    assert _removals(ssh_client) == []


@pytest.mark.asyncio
async def test_filler_create_never_removes_an_unmounted_customer_volume(svc, monkeypatch) -> None:
    # a customer create made its encrypted volume and has not run its pod yet (the create lock lapsed or
    # another connector runs it): no `pod_*` exists, so only the volume sweep could touch it
    ssh_client = _host_listing(svc, monkeypatch, [], unmounted_vloopback_volumes=(_CUSTOMER_VOLUME,))
    filler = _payload(workload_kind=WorkloadKind.FILLER)

    result = await _run(svc, filler)

    assert isinstance(result, ContainerCreated)
    assert not any(_CUSTOMER_VOLUME in targets for targets in _removals(ssh_client))


@pytest.mark.asyncio
async def test_filler_create_sweeps_an_unmounted_filler_volume_but_never_a_customers(svc, monkeypatch) -> None:
    # a filler volume left without its container (a failed `docker volume rm`, a validator restart mid-create)
    ssh_client = _host_listing(
        svc,
        monkeypatch,
        [],
        unmounted_vloopback_volumes=(_CUSTOMER_VOLUME,),
        unmounted_filler_volumes=("volume_ended-filler-run",),
    )
    filler = _payload(workload_kind=WorkloadKind.FILLER)

    result = await _run(svc, filler)

    assert isinstance(result, ContainerCreated)
    assert any("volume_ended-filler-run" in targets for targets in _removals(ssh_client))
    assert not any(_CUSTOMER_VOLUME in targets for targets in _removals(ssh_client))


@pytest.mark.parametrize(
    ("workload_kind", "labels"),
    [(WorkloadKind.FILLER, FILLER_VOLUME_LABELS), (WorkloadKind.CUSTOMER_RENTAL, None)],
    ids=["filler", "customer"],
)
@pytest.mark.asyncio
async def test_only_a_filler_volume_is_created_with_the_filler_label(
    svc, monkeypatch, workload_kind: WorkloadKind, labels: dict[str, str] | None
) -> None:
    _host_listing(svc, monkeypatch, [])

    result = await _run(svc, _payload(workload_kind=workload_kind))

    assert isinstance(result, ContainerCreated)
    assert svc.create_local_volume.await_args.kwargs["labels"] == labels


@pytest.mark.asyncio
async def test_customer_create_still_sweeps_unmounted_vloopback_volumes(svc, monkeypatch) -> None:
    ssh_client = _host_listing(svc, monkeypatch, [], unmounted_vloopback_volumes=("volume_ended-rental",))
    customer = _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL)

    result = await _run(svc, customer)

    assert isinstance(result, ContainerCreated)
    assert any("volume_ended-rental" in targets for targets in _removals(ssh_client))


@pytest.mark.asyncio
async def test_refusal_for_an_unlisted_pod_logs_the_pod_not_a_customer_create(svc, monkeypatch, caplog) -> None:
    _host_listing(svc, monkeypatch, [_CUSTOMER_POD])
    filler = _payload(workload_kind=WorkloadKind.FILLER)

    with caplog.at_level(logging.INFO, logger="services.docker_service"):
        await _run(svc, filler)

    refusals = [r.msg.to_full_string() for r in caplog.records if "filler create refused" in r.getMessage()]
    assert len(refusals) == 1
    assert refusals[0].startswith("filler create refused: unlisted pod_* on the node >>> ")
    assert _CUSTOMER_POD in refusals[0]
