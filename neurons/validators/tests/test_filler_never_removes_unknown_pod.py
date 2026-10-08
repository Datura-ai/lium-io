"""A filler create never removes, nor starts beside, a `pod_*` its active_container_names do not name.

The backend builds that list when it sends the filler, before a customer's pod exists. A filler create
that reaches its cleanup after a customer's `docker run` (a retried send landing late, a lapsed create
lock, a second connector, a delete that failed before its cancel) would otherwise `docker rm -fv` the
customer's container and run on the rented node. It refuses instead, the way #1518 refuses (no strike):
at the cleanup, on the listing it already has, and in the Docker thread between `create` and `start`.
"""

from __future__ import annotations

import shlex
from unittest.mock import Mock

import pytest
from payload_models.payloads import ContainerCreated, FailedContainerErrorCodes, WorkloadKind
from services.docker_service import DockerService
from services.prerun_host_probe import DOCKER_PS_ALL_NAMES_IDS_CMD
from services.rental_docker_sdk import RENTAL_NETWORK_OPTIONS, RentalDockerSdkClient
from test_deploy_optimizations import (
    _docker_client,
    _patch_happy,
    _payload,
    _run,
    _ssh_client,
    _ssh_result,
)

_CUSTOMER_POD = "pod_6a1f2a52-6f0e-4a3e-9c55-0d3e1b7a9c11"


@pytest.fixture
def svc() -> DockerService:
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


def _host_listing(svc, monkeypatch, container_names: list[str]):
    """The create path with its real cleanup on a host whose `docker ps -a` lists ``container_names``."""
    ssh_client = _ssh_client()
    answer_like_a_clean_host = ssh_client.run.side_effect

    def run(cmd, *args, **kwargs):
        if cmd == DOCKER_PS_ALL_NAMES_IDS_CMD:
            return _ssh_result(exit_status=0, stdout="".join(f"{name}\n" for name in container_names))
        return answer_like_a_clean_host(cmd, *args, **kwargs)

    ssh_client.run.side_effect = run
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.delattr(svc, "clean_existing_containers")
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
    assert any(f"filler_{filler.pod_id}" in targets for targets in _removals(ssh_client))
    assert not any(_CUSTOMER_POD in targets for targets in _removals(ssh_client))


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
