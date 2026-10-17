"""A customer's create that removes a Lium filler restores that filler's GPU power the way its delete
would, raising a capped GPU whose restore record is gone to its default, before the renter starts."""

from unittest.mock import AsyncMock

import pytest
from payload_models.payloads import ContainerCreated, WorkloadKind
from services.docker_service import _removed_filler_pod_ids
from test_deploy_optimizations import _patch_happy, _payload, _run, _ssh_client, svc  # noqa: F401


def test_removed_filler_pod_ids_keeps_only_filler_containers():
    assert _removed_filler_pod_ids(["filler_pod-9", "pod_abc", "filler_", "filler_pod-7"]) == ["pod-9", "pod-7"]
    assert _removed_filler_pod_ids(None) == []


@pytest.mark.asyncio
async def test_customer_create_restores_the_power_of_a_filler_it_removed(svc, monkeypatch):  # noqa: F811
    ssh_client = _ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "clean_existing_containers", AsyncMock(return_value=["filler_pod-9", "pod_old"]))
    restore = AsyncMock(return_value=0)
    monkeypatch.setattr("services.docker_service.restore_filler_pod_gpu_power_limits", restore)
    payload = _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    restore.assert_awaited_once()
    assert restore.await_args.args[2] == "pod-9"
    assert restore.await_args.kwargs["executor_id"] == payload.executor_id


@pytest.mark.asyncio
async def test_a_filler_create_does_not_restore_its_sibling_bundles(svc, monkeypatch):  # noqa: F811
    ssh_client = _ssh_client()
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.setattr(svc, "clean_existing_containers", AsyncMock(return_value=["filler_pod-9"]))
    restore = AsyncMock(return_value=0)
    monkeypatch.setattr("services.docker_service.restore_filler_pod_gpu_power_limits", restore)

    await _run(svc, _payload(workload_kind=WorkloadKind.FILLER))

    restore.assert_not_awaited()
