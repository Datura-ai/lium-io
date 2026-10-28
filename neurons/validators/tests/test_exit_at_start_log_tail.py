"""DAH-5037: a default-job (filler) container that exits at start carries its last log lines."""

from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import WorkloadKind
from services.docker_service import (
    ImageExitedDuringKeyInjection,
    _attach_filler_log_tail,
    _explain_add_public_keys_failure,
)
from services.rental_docker_sdk import ContainerStateSnapshot, RentalDockerSdkClient

EXITED = ContainerStateSnapshot(
    status="exited", running=False, restarting=False, exit_code=1, restart_count=0, error=None, oom_killed=False
)


def _client(log_tail: str | None = None, log_error: Exception | None = None) -> Mock:
    client = Mock()
    client.inspect_container_state = AsyncMock(return_value=EXITED)
    client.container_log_tail = AsyncMock(return_value=log_tail, side_effect=log_error)
    return client


async def _explain(
    client: Mock, workload_kind: WorkloadKind, environment: dict[str, str] | None = None
) -> Exception:
    explained = await _explain_add_public_keys_failure(
        client, container_name="c1", image="jetskipool/jetski-miner:latest", cause=RuntimeError("exec failed"), log_extra={}
    )
    await _attach_filler_log_tail(
        explained, client, container_name="c1", workload_kind=workload_kind, environment=environment, log_extra={}
    )
    return explained


@pytest.mark.asyncio
async def test_filler_exit_at_start_carries_the_log_tail_outside_the_error() -> None:
    explained = await _explain(_client(log_tail="error: missing --pool\n"), WorkloadKind.FILLER)

    assert isinstance(explained, ImageExitedDuringKeyInjection)
    assert "exited right after start (exit_code=1)" in str(explained)
    assert explained.log_tail == "error: missing --pool"
    # the backend's GPU-fault check reads `error`; a printed NVML line there could deactivate the node
    assert "missing --pool" not in str(explained)


@pytest.mark.asyncio
async def test_filler_log_tail_redacts_the_job_environment() -> None:
    client = _client(log_tail="key=sk-secret-123 pool=stratum://p:1 empty= gpus=1 of 1\n")

    explained = await _explain(
        client,
        WorkloadKind.FILLER,
        environment={"DOLPHIN_API_KEY": "sk-secret-123", "POOL": "stratum://p:1", "E": "", "N": "1"},
    )

    assert explained.log_tail == "key=*** pool=*** empty= gpus=1 of 1"


@pytest.mark.asyncio
async def test_customer_rental_exit_at_start_never_reads_the_log() -> None:
    client = _client(log_tail="renter secret output")

    explained = await _explain(client, WorkloadKind.CUSTOMER_RENTAL)

    assert isinstance(explained, ImageExitedDuringKeyInjection)
    assert explained.log_tail == ""
    client.container_log_tail.assert_not_called()


@pytest.mark.asyncio
async def test_unreadable_log_keeps_the_exit_message() -> None:
    explained = await _explain(_client(log_error=RuntimeError("logs 500")), WorkloadKind.FILLER)

    assert isinstance(explained, ImageExitedDuringKeyInjection)
    assert explained.log_tail == ""


@pytest.mark.asyncio
async def test_sdk_log_tail_asks_docker_for_both_streams_and_decodes_bad_bytes() -> None:
    api_client = Mock()
    api_client.logs = Mock(return_value=b"bad \xff byte\n")

    tail = await RentalDockerSdkClient(api_client).container_log_tail(container_name="c1", lines=20)

    assert tail == "bad � byte\n"
    api_client.logs.assert_called_once_with("c1", stdout=True, stderr=True, tail=20)
