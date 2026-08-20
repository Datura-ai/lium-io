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
from payload_models.payloads import (
    ContainerCreateRequest,
    ContainerDeleteRequest,
    PayloadPortMapping,
    WorkloadKind,
)
from services.docker_service import DockerService, _CreateCancelledByDelete, inflight_creates
from services.miner_service import MinerService


@pytest.fixture
def svc() -> DockerService:
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
    )


@pytest.fixture
def miner_service() -> MinerService:
    return MinerService(
        ssh_service=Mock(),
        task_service=Mock(),
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
async def test_delete_during_create_marks_the_create_cancelled(miner_service: MinerService) -> None:
    payload = _payload()
    seen_by_create: dict[str, bool] = {}

    async def fake_route(routed: ContainerCreateRequest) -> str:
        # A delete for the same pod lands right here, while the create is still running.
        seen_by_create["delete_found_the_create"] = inflight_creates.cancel(routed.pod_id)
        seen_by_create["checkpoint"] = inflight_creates.is_cancelled(routed.pod_id)
        return "created"

    miner_service._route_container = fake_route

    assert await miner_service.handle_container(payload) == "created"
    assert seen_by_create == {"delete_found_the_create": True, "checkpoint": True}
    # The registration must not outlive the create, or a later pod id reuse would abort at once.
    assert inflight_creates.is_cancelled(payload.pod_id) is False


def test_delete_with_no_create_running_cancels_nothing() -> None:
    assert inflight_creates.cancel(str(uuid4())) is False


@pytest.mark.parametrize(
    ("workload_kind", "power_restored"),
    [(WorkloadKind.FILLER, True), (WorkloadKind.CUSTOMER_RENTAL, False)],
)
@pytest.mark.asyncio
async def test_abort_restores_only_filler_power(
    svc: DockerService,
    monkeypatch: pytest.MonkeyPatch,
    workload_kind: WorkloadKind,
    power_restored: bool,
) -> None:
    payload = _payload(workload_kind=workload_kind)
    restore_power_limits = AsyncMock()
    monkeypatch.setattr(ds_module, "restore_filler_pod_gpu_power_limits", restore_power_limits)

    with inflight_creates.track(payload.pod_id):
        inflight_creates.cancel(payload.pod_id)
        with pytest.raises(_CreateCancelledByDelete):
            await svc._abort_if_cancelled_by_delete(Mock(), payload, {})

    assert restore_power_limits.await_count == (1 if power_restored else 0)


@pytest.mark.asyncio
async def test_a_create_nobody_cancelled_runs_on(svc: DockerService) -> None:
    payload = _payload()

    with inflight_creates.track(payload.pod_id):
        assert await svc._abort_if_cancelled_by_delete(Mock(), payload, {}) is None


@pytest.mark.parametrize(
    ("workload_kind", "prefix"),
    [(WorkloadKind.CUSTOMER_RENTAL, "pod_"), (WorkloadKind.FILLER, "filler_")],
)
def test_container_name_is_derived_when_the_delete_carries_none(
    svc: DockerService, workload_kind: WorkloadKind, prefix: str
) -> None:
    # A delete racing an in-flight create arrives before the backend knows the name.
    pod_id = str(uuid4())
    payload = ContainerDeleteRequest(
        miner_hotkey="miner", executor_id=str(uuid4()), pod_id=pod_id,
        workload_kind=workload_kind, container_name="",
    )

    assert svc.get_container_name(payload) == f"{prefix}{pod_id}"
