"""A filler create refuses beside a `pod_*` its active_container_names do not name, and starts only its own container."""

import logging
import shlex
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest
import services.docker_service as ds_module
from payload_models.payloads import ContainerCreateRequest, FailedContainerErrorCodes, WorkloadKind
from services.prerun_host_probe import DOCKER_PS_ALL_NAMES_IDS_CMD, DOCKER_VOLUME_LS_NAME_DRIVER_CMD
from services.rental_docker_sdk import (
    FILLER_VOLUME_LABELS,
    RENTAL_NETWORK_OPTIONS,
    ContainerExecResult,
    RentalDockerSdkClient,
)

_CUSTOMER_POD = "pod_6a1f2a52-6f0e-4a3e-9c55-0d3e1b7a9c11"
_CUSTOMER_VOLUME = "volume_6a1f2a52-6f0e-4a3e-9c55-0d3e1b7a9c11"


def _payload(workload_kind: WorkloadKind, active_container_names: list[str] | None = None) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id=str(uuid4()),
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:1.0.0",
        user_public_keys=["ssh-ed25519 test-key"],
        gpu_uuids=["GPU-test"],
        memory_gb=1,
        active_container_names=active_container_names or [],
        workload_kind=workload_kind,
    )


def _create_over_a_host(svc, monkeypatch, container_names, unmounted_vloopback=(), unmounted_filler=()):
    """The create with its real cleanups, sweeps and SDK run over a host whose `docker ps -a` lists
    ``container_names`` and whose `docker volume ls` lists the unmounted volumes (the filler label filter
    keeps ``unmounted_filler``); returns the host's SSH client and the Docker API the SDK talks to."""

    def volumes(names) -> str:
        return "".join(f"{name} vloopback\n" for name in names)

    answers = {
        "cat /proc/sys/kernel/pid_max": "4194304\n",
        DOCKER_PS_ALL_NAMES_IDS_CMD: "".join(f"{name}\n" for name in container_names),
        DOCKER_VOLUME_LS_NAME_DRIVER_CMD: volumes((*unmounted_vloopback, *unmounted_filler)),
        ds_module.DOCKER_FILLER_VOLUME_LS_NAME_DRIVER_CMD: volumes(unmounted_filler),
    }
    ssh_client = AsyncMock()
    ssh_client.run.side_effect = lambda cmd, *args, **kwargs: Mock(exit_status=0, stdout=answers.get(cmd, ""))
    docker_api = Mock(inspect_network=Mock(return_value={"Driver": "bridge", "Options": RENTAL_NETWORK_OPTIONS}))
    docker_api.create_container.return_value = {"Id": "filler-container"}
    docker_api.containers.return_value = []
    docker_client = AsyncMock(run_container=RentalDockerSdkClient(docker_api).run_container)
    docker_client.exec_in_container.return_value = ContainerExecResult(exit_status=0)
    docker_client.inspect_container_state.return_value = Mock(running=True, restarting=False, oom_killed=False, status="running")
    svc.rental_docker_client_factory = MagicMock()
    svc.rental_docker_client_factory.connect.return_value.__aenter__.return_value = docker_client
    connection = MagicMock(**{"__aenter__.return_value": ssh_client})
    monkeypatch.setattr("services.docker_service.asyncssh.connect", Mock(return_value=connection))
    monkeypatch.setattr("services.docker_service.asyncssh.import_private_key", Mock())
    for name in ("add_pending_pod", "remove_pending_pod", "add_rented_pod"):
        setattr(svc.redis_service, name, AsyncMock())
    monkeypatch.setattr(svc, "create_local_volume", AsyncMock())
    monkeypatch.setattr(svc, "generate_portMappings", AsyncMock(return_value=([(22, 20001, 20001)], None)))
    monkeypatch.setattr(svc, "check_container_running", AsyncMock(return_value=True))
    return ssh_client, docker_api


async def _run(svc, payload):
    executor_info = Mock(uuid=payload.executor_id, address="127.0.0.1", port=8080, ssh_username="root", ssh_port=2200)
    executor_info.ssh_host_key = "ssh-ed25519 AAAATESTKEY"
    return await svc.create_container(payload, executor_info, Mock(ss58_address="validator-hotkey"), "private-key")


def _removed_names(ssh_client) -> list[str]:
    # the containers and volumes every `docker rm` / `docker volume rm` the create sent targeted, in order
    commands = [c.args[0] for c in ssh_client.run.await_args_list if c.args]
    removals = [cmd for cmd in commands if cmd.startswith(("/usr/bin/docker rm", "/usr/bin/docker volume rm"))]
    return [word for cmd in removals for word in shlex.split(cmd) if word.startswith(("filler_", "pod_", "volume_"))]


@pytest.fixture
def svc() -> ds_module.DockerService:
    return ds_module.DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


@pytest.mark.parametrize(
    ("pod_listed_before_the_create", "customer_create_running"),
    [(True, False), (False, False), (False, True)],
    ids=["pod_at_the_cleanup", "pod_between_create_and_start", "pod_beside_a_running_customer_create"],
)
@pytest.mark.asyncio
async def test_filler_create_beside_an_unlisted_pod_refuses_and_removes_only_the_filler(
    svc, monkeypatch, caplog, pod_listed_before_the_create: bool, customer_create_running: bool
) -> None:
    ssh_client, docker_api = _create_over_a_host(svc, monkeypatch, [_CUSTOMER_POD] if pod_listed_before_the_create else [])
    restore_power = AsyncMock()
    monkeypatch.setattr("services.docker_service.restore_filler_pod_gpu_power_limits", restore_power)
    filler = _payload(WorkloadKind.FILLER)

    def customer_pod_created_meanwhile(**_kwargs) -> dict:
        docker_api.containers.return_value = [{"Names": [f"/{_CUSTOMER_POD}"]}, {"Names": [f"/filler_{filler.pod_id}"]}]
        # a customer create that registered after the filler's `before_create` check and still runs at the handler
        monkeypatch.setattr(ds_module.customer_creates, "ran_since_filler_started", lambda _payload: customer_create_running)
        return {"Id": "filler-container"}

    docker_api.create_container.side_effect = customer_pod_created_meanwhile

    with caplog.at_level(logging.INFO, logger="services.docker_service"):
        result = await _run(svc, filler)

    assert result.error_code == FailedContainerErrorCodes.RentingInProgress
    assert result.failure_step == "customer_create_in_flight"
    docker_api.start.assert_not_called()
    # once created, the filler's container and volume are removed and its power cap lifted; the customer's pod never
    assert _removed_names(ssh_client) == ([] if pod_listed_before_the_create else [f"filler_{filler.pod_id}", f"volume_{filler.pod_id}"])
    assert restore_power.await_count == (0 if pod_listed_before_the_create else 1)
    refusals = [r.msg.to_full_string() for r in caplog.records if "filler create refused" in r.getMessage()]
    assert [r.startswith("filler create refused: unlisted pod_* on the node >>> ") and _CUSTOMER_POD in r for r in refusals] == [True]


@pytest.mark.parametrize(
    ("host_containers", "listed"),
    [(["filler_old-run"], []), (["filler_old-run", "pod_listed", "pod_listed__prev"], ["pod_listed"])],
    ids=["no_pod", "only_pods_the_backend_listed"],
)
@pytest.mark.asyncio
async def test_filler_create_without_an_unlisted_pod_starts_its_own_container_by_id(
    svc, monkeypatch, host_containers: list[str], listed: list[str]
) -> None:
    ssh_client, docker_api = _create_over_a_host(svc, monkeypatch, host_containers)

    result = await _run(svc, _payload(WorkloadKind.FILLER, listed))

    assert type(result).__name__ == "ContainerCreated"
    assert _removed_names(ssh_client)[:1] == ["filler_old-run"]
    # one listing between create and start; the start names the container that was created, not a name
    docker_api.containers.assert_called_once_with(all=True)
    docker_api.start.assert_called_once_with("filler-container")


@pytest.mark.parametrize(
    ("workload_kind", "labels", "swept_volumes"),
    [
        (WorkloadKind.FILLER, FILLER_VOLUME_LABELS, ["volume_ended-filler-run"]),
        (WorkloadKind.CUSTOMER_RENTAL, None, [_CUSTOMER_VOLUME, "volume_ended-filler-run"]),
    ],
    ids=["filler", "customer"],
)
@pytest.mark.asyncio
async def test_only_a_filler_volume_is_labelled_and_a_filler_create_sweeps_only_labelled_volumes(
    svc, monkeypatch, workload_kind: WorkloadKind, labels: dict[str, str] | None, swept_volumes: list[str]
) -> None:
    # a customer create made its encrypted volume and has not run its pod yet (its create lock lapsed or
    # another connector runs it): no `pod_*` exists, so only the volume sweep could touch it
    ssh_client, _ = _create_over_a_host(svc, monkeypatch, [], [_CUSTOMER_VOLUME], ["volume_ended-filler-run"])

    result = await _run(svc, _payload(workload_kind))

    assert type(result).__name__ == "ContainerCreated"
    assert svc.create_local_volume.await_args.kwargs["labels"] == labels
    assert _removed_names(ssh_client) == swept_volumes
