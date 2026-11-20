import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from docker.errors import NotFound
from payload_models.payloads import CustomOptions, WorkloadKind
from test_deploy_optimizations import _payload
from test_filler_removed_at_customer_create import _CUSTOMER_CONTAINER_ID, _FILLER_ID, _listing

from services.docker_service import DockerService, _FillerRefusedBesideUnlistedPod
from services.rental_docker_sdk import GpuDockerConfig, RentalDockerSdkClient


@pytest.mark.asyncio
@pytest.mark.parametrize("customer_at", ["before_filler_create", "after_filler_snapshot"])
async def test_separate_connectors_cannot_start_filler_beside_customer(customer_at):
    # Arrange: independent services share only the external Docker host, without a Redis lock.
    customer = DockerService(ssh_service=Mock(), redis_service=Mock(get=AsyncMock(return_value=None)), attestation_service=Mock())
    filler_connector = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())
    filler = _payload(workload_kind=WorkloadKind.FILLER)
    host = {}
    started = []
    loop = asyncio.get_running_loop()

    def listing():
        return "".join(f"NAME\t{name} {container_id}\n" for name, container_id in host.items()) + "PS\t0\n"

    async def ssh_run(command, **kwargs):
        if command.startswith("/usr/bin/docker rm -fv"):
            assert _FILLER_ID in command
            host.pop("filler_late")
            return _listing("RM\t0\n" + listing())
        return _listing(_CUSTOMER_CONTAINER_ID[:12] + "\n" + listing())

    async def finish_customer():
        host["pod_customer"] = _CUSTOMER_CONTAINER_ID
        assert await customer.check_container_running(
            Mock(run=AsyncMock(side_effect=ssh_run)), "pod_customer", remove_every_filler=True
        )

    def customer_on_other_connector():
        asyncio.run_coroutine_threadsafe(finish_customer(), loop).result(timeout=5)

    def create_container(**kwargs):
        if customer_at == "before_filler_create":
            customer_on_other_connector()
        host["filler_late"] = _FILLER_ID
        return {"Id": _FILLER_ID}

    def containers(all):
        snapshot = [{"Names": ["/" + name]} for name in host]
        if customer_at == "after_filler_snapshot":
            customer_on_other_connector()
        return snapshot

    def start(container_id):
        if container_id not in host.values():
            raise NotFound("customer removed the created filler")
        started.append(container_id)

    api = Mock(create_container=create_container, containers=containers, start=start)
    spec = filler_connector._build_rental_container_run_spec(
        payload=filler, container_name="filler_late", custom_options=CustomOptions(),
        port_maps=[], local_volume="volume_filler", local_volume_path="/root",
        encrypted_local_volume=False, external_volume_name=None, gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None, cpu_count=None,
    )

    # Act: run the real SDK create/list/start and the real customer running check.
    with pytest.raises((_FillerRefusedBesideUnlistedPod, NotFound)):
        await asyncio.to_thread(RentalDockerSdkClient(api)._run_container_sync, replace(spec, network=None))

    # Assert: either the filler sees the customer, or the customer removes its checked ID.
    assert host["pod_customer"] == _CUSTOMER_CONTAINER_ID
    assert not started
