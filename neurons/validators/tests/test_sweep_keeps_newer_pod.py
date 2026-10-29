"""DAH-5392: a delayed create's stale sweep never removes a newer create's `pod_*` on a split node.

Create A's active_container_names are built by the backend before it sends A; if A waits (the per-node
create exclusion) and create B runs first, B's pod is not on A's list. The sweep keeps an unlisted
`pod_*` the host created after A's list could have been built, and still removes an older leftover.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

import services.docker_service as ds_module
from payload_models.payloads import ContainerCreateRequest
from services.docker_service import DockerService, own_sweep_removals

HOST_NOW = 4_000_000_000  # the host's clock, far from this validator's: only differences on it are compared


def _result(stdout: str):
    return Mock(exit_status=0, stdout=stdout, stderr="")


def _created(seconds_ago: int) -> str:
    return datetime.fromtimestamp(HOST_NOW - seconds_ago, UTC).strftime("%Y-%m-%dT%H:%M:%S.123456789Z")


@pytest.fixture(autouse=True)
def _no_sweeps_from_other_tests():
    own_sweep_removals.clear()


@pytest.fixture
def retry_ssh_mock(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(ds_module, "retry_ssh_command", mock)
    return mock


async def _sweep(inspect_stdout: str) -> tuple[list[str], AsyncMock]:
    svc = DockerService(ssh_service=Mock(), redis_service=Mock(get=AsyncMock(return_value=None)), attestation_service=Mock())
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_result("pod_target\npod_newer\npod_leftover\n"), _result(inspect_stdout)])
    removed = await svc.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={},
        pod_name="pod_target",
        active_container_names=[],
        request_received_monotonic=time.monotonic(),
    )
    return removed, ssh_client


@pytest.mark.asyncio
async def test_unlisted_pod_created_after_the_request_arrived_is_kept_with_its_volume(retry_ssh_mock):
    removed, ssh_client = await _sweep(
        f"{HOST_NOW}\n/pod_newer {_created(10)}\n/pod_leftover {_created(3600)}\n"
    )

    assert removed == ["pod_target", "pod_leftover"]
    inspect_command = ssh_client.run.await_args_list[1].args[0]
    # the create's own container is never asked about: its handling is unchanged
    assert "pod_target" not in inspect_command
    sent = " ".join(call.args[1] for call in retry_ssh_mock.await_args_list)
    assert "pod_newer" not in sent and "volume_newer" not in sent
    assert "volume_leftover" in sent


@pytest.mark.asyncio
async def test_unreadable_creation_times_remove_unlisted_pods_as_before(retry_ssh_mock):
    removed, _ = await _sweep("no date here\n")

    assert removed == ["pod_target", "pod_newer", "pod_leftover"]


def test_received_time_is_recorded_when_the_request_is_parsed():
    before = time.monotonic()
    payload = ContainerCreateRequest(miner_hotkey="hk", executor_id="ex", pod_id="p", docker_image="i", gpu_uuids=[])
    assert before <= payload._received_monotonic <= time.monotonic()
    assert "_received_monotonic" not in payload.model_dump_json()
