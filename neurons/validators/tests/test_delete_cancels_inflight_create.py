"""DAH-2728 — a delete must cancel the create it raced, not assume the pod is absent.

A delete arriving mid-create used to see no container yet ("No such container"), report the pod
deleted, and let the create finish minutes later — the orphan then held the host ports and blocked
the next paying rental.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import services.docker_service as ds_module
from payload_models.payloads import ContainerCreateRequest, PayloadPortMapping, WorkloadKind
from services.docker_service import DockerService, _CreateCancelledByDelete, inflight_creates


@pytest.fixture
def svc() -> DockerService:
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
    )


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


@pytest.mark.asyncio
async def test_delete_during_create_marks_the_create_cancelled(svc: DockerService) -> None:
    payload = _payload()
    seen_by_create: dict[str, bool] = {}

    async def fake_create(inner_payload: ContainerCreateRequest, *args) -> str:
        # A delete for the same pod lands right here, while the create is still running.
        seen_by_create["delete_found_the_create"] = inflight_creates.cancel(inner_payload.pod_id)
        seen_by_create["checkpoint"] = inflight_creates.is_cancelled(inner_payload.pod_id)
        return "created"

    svc._create_container = fake_create

    assert await svc.create_container(payload, Mock(), Mock(), "key") == "created"
    assert seen_by_create == {"delete_found_the_create": True, "checkpoint": True}
    # The registration must not outlive the create, or a later pod id reuse would abort at once.
    assert inflight_creates.is_cancelled(payload.pod_id) is False


def test_delete_with_no_create_running_cancels_nothing() -> None:
    assert inflight_creates.cancel(str(uuid4())) is False


@pytest.mark.asyncio
async def test_abort_restores_filler_power_before_raising(svc: DockerService, monkeypatch) -> None:
    payload = _payload(workload_kind=WorkloadKind.FILLER)
    restore = AsyncMock()
    monkeypatch.setattr(ds_module, "restore_filler_pod_gpu_power_limits", restore)

    with pytest.raises(_CreateCancelledByDelete):
        await svc._abort_create_cancelled_by_delete(Mock(), payload, "filler_x", {})

    restore.assert_awaited_once()


@pytest.mark.asyncio
async def test_abort_leaves_rental_power_alone(svc: DockerService, monkeypatch) -> None:
    restore = AsyncMock()
    monkeypatch.setattr(ds_module, "restore_filler_pod_gpu_power_limits", restore)

    with pytest.raises(_CreateCancelledByDelete):
        await svc._abort_create_cancelled_by_delete(Mock(), _payload(), "pod_x", {})

    restore.assert_not_awaited()
