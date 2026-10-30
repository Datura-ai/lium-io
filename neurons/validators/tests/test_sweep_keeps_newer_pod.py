"""DAH-5392: a delayed create's stale sweep never removes a newer create's `pod_*` on a split node.

Create A's active_container_names are built by the backend before it sends A; if A waits (the per-node
create exclusion) and create B runs first, B's pod is not on A's list. The sweep removes an unlisted
`pod_*` (and its volume) only when the backend confirms its rental closed; anything else keeps it.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

import services.docker_service as ds_module
from protocol.vc_protocol.compute_requests import PodRentalActiveResponse
from services.docker_service import DockerService, own_sweep_removals

NEWER = "11111111-1111-1111-1111-111111111111"
LEFTOVER = "22222222-2222-2222-2222-222222222222"
CLOSED = PodRentalActiveResponse(active=False)
ACTIVE = PodRentalActiveResponse(active=True, status="PENDING")


@pytest.fixture(autouse=True)
def _no_sweeps_from_other_tests():
    own_sweep_removals.clear()


@pytest.fixture
def retry_ssh_mock(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(ds_module, "retry_ssh_command", mock)
    return mock


async def _sweep(answers: dict, retry_ssh_mock, backend: bool = True) -> tuple[list[str], str]:
    async def rental_active(pod_id):
        answer = answers[pod_id]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    backend_client = Mock(get_pod_rental_active=AsyncMock(side_effect=rental_active)) if backend else None
    svc = DockerService(
        ssh_service=Mock(),
        redis_service=Mock(get=AsyncMock(return_value=None)),
        attestation_service=Mock(),
        backend_client=backend_client,
    )
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout=f"pod_target\npod_{NEWER}\npod_{LEFTOVER}\n", stderr=""))
    removed = await svc.clean_existing_containers(
        ssh_client=ssh_client, default_extra={}, pod_name="pod_target", active_container_names=[]
    )
    if backend_client is not None:
        # the create's own container is never asked about: its handling is unchanged
        assert sorted(call.args[0] for call in backend_client.get_pod_rental_active.await_args_list) == [NEWER, LEFTOVER]
    return removed, " ".join(call.args[1] for call in retry_ssh_mock.await_args_list)


@pytest.mark.asyncio
async def test_an_active_pod_is_kept_and_a_closed_leftover_removed_with_its_volume(retry_ssh_mock):
    removed, sent = await _sweep({NEWER: ACTIVE, LEFTOVER: CLOSED}, retry_ssh_mock)

    assert removed == ["pod_target", f"pod_{LEFTOVER}"]
    assert f"pod_{NEWER}" not in sent and f"volume_{NEWER}" not in sent
    assert f"volume_{LEFTOVER}" in sent


@pytest.mark.parametrize(
    "answer",
    [None, ACTIVE, RuntimeError("backend down"), TimeoutError()],
    ids=["no-answer", "active", "error", "timeout"],
)
@pytest.mark.asyncio
async def test_a_pod_whose_rental_is_not_confirmed_closed_is_kept_with_its_volume(retry_ssh_mock, answer):
    removed, sent = await _sweep({NEWER: answer, LEFTOVER: answer}, retry_ssh_mock)

    assert removed == ["pod_target"]
    assert f"volume_{NEWER}" not in sent and f"volume_{LEFTOVER}" not in sent


@pytest.mark.asyncio
async def test_no_backend_client_keeps_every_unlisted_pod(retry_ssh_mock):
    removed, sent = await _sweep({}, retry_ssh_mock, backend=False)

    assert removed == ["pod_target"]
    assert "volume_1" not in sent and "volume_2" not in sent


@pytest.mark.asyncio
async def test_a_name_that_is_no_backend_pod_is_removed_without_asking(retry_ssh_mock):
    backend_client = Mock(get_pod_rental_active=AsyncMock())
    svc = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock(), backend_client=backend_client)
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout="pod_stale\npod_stale__prev\n", stderr=""))

    removed = await svc.clean_existing_containers(ssh_client=ssh_client, default_extra={}, pod_name="pod_target")

    assert removed == ["pod_stale", "pod_stale__prev"]
    backend_client.get_pod_rental_active.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_hung_backend_answer_is_bounded_and_keeps_the_pod(retry_ssh_mock, monkeypatch):
    monkeypatch.setattr(ds_module, "_POD_RENTAL_ACTIVE_TIMEOUT_SECONDS", 0.01)
    never = asyncio.get_running_loop().create_future()

    async def hang(_pod_id):
        return await never

    svc = DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        backend_client=Mock(get_pod_rental_active=hang),
    )
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout=f"pod_{NEWER}\n", stderr=""))

    assert await svc.clean_existing_containers(ssh_client=ssh_client, default_extra={}, pod_name="pod_target") == []


@pytest.mark.asyncio
async def test_lookups_are_capped_per_sweep_and_unchecked_pods_kept(retry_ssh_mock):
    """A provider can leave thousands of stopped pod_<uuid> containers: one sweep asks about at most the cap."""
    cap = ds_module.MAX_UNLISTED_POD_LOOKUPS
    ids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(cap + 5)]
    backend_client = Mock(get_pod_rental_active=AsyncMock(return_value=CLOSED))
    svc = DockerService(
        ssh_service=Mock(),
        redis_service=Mock(get=AsyncMock(return_value=None)),
        attestation_service=Mock(),
        backend_client=backend_client,
    )
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout="".join(f"pod_{i}\n" for i in ids), stderr=""))

    removed = await svc.clean_existing_containers(
        ssh_client=ssh_client, default_extra={}, pod_name="pod_target", active_container_names=[]
    )

    asked = {call.args[0] for call in backend_client.get_pod_rental_active.await_args_list}
    assert len(asked) == backend_client.get_pod_rental_active.await_count == cap
    assert sorted(removed) == sorted(f"pod_{i}" for i in asked)
    sent = " ".join(call.args[1] for call in retry_ssh_mock.await_args_list)
    for unchecked in set(ids) - asked:
        assert f"pod_{unchecked}" not in removed and f"pod_{unchecked}" not in sent
