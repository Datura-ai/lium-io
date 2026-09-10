"""DAH-3338: the container state of every rented pod travels to the backend as
ExecutorSpecRequest.pod_states — ctx.state → JobResult → MACHINE_SPEC_CHANNEL → the
websocket request — and the field is optional at every hop, so a validator and a
backend of different ages keep talking.
"""
import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from clients.compute_client import ComputeClient
from helpers import build_state, make_context
from protocol.vc_protocol.validator_requests import (
    ContainerState,
    ExecutorSpecRequest,
    PodContainerState,
)
from services.miner_service import MinerService
from services.redis_service import MACHINE_SPEC_CHANNEL
from services.task.result_handler import ResultHandler

pytest_plugins = ["fixtures.incentive_fixtures"]

OBSERVED_AT = datetime(2026, 9, 10, 12, 30, tzinfo=UTC)


def _spec_kwargs() -> dict[str, Any]:
    """The ExecutorSpecRequest exactly as the redis→WS bridge builds it, without pod_states."""
    return dict(
        miner_hotkey="hk",
        miner_coldkey="ck",
        validator_hotkey="vk",
        executor_uuid="exec-1",
        executor_ip="10.0.0.1",
        executor_port=8080,
        specs={},
        score=0.0,
        synthetic_job_score=0.0,
        log_text="ok",
        log_status="success",
        job_batch_id="2026-09-10 12:00:00",
        collateral_deposited=False,
    )


async def _bridge_machine_spec(payload: dict[str, Any]) -> ExecutorSpecRequest:
    async def listen():
        yield {
            "channel": MACHINE_SPEC_CHANNEL.encode(),
            "data": json.dumps(payload).encode(),
        }
        raise asyncio.CancelledError

    pubsub = MagicMock(listen=listen, aclose=AsyncMock())
    client = ComputeClient.__new__(ComputeClient)
    client.keypair = MagicMock(ss58_address="validator-hotkey")
    client.lock = asyncio.Lock()
    client.message_queue = []
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.miner_service = MagicMock()
    client.miner_service.redis_service.subscribe = AsyncMock(return_value=pubsub)

    with pytest.raises(asyncio.CancelledError):
        await client.subscribe_mesages_from_redis()

    return client.message_queue[0]


def _publisher() -> tuple[MinerService, MagicMock]:
    redis_service = MagicMock()
    redis_service.publish = AsyncMock()
    service = MinerService(
        ssh_service=MagicMock(),
        task_service=MagicMock(),
        redis_service=redis_service,
        attestation_service=MagicMock(),
    )
    return service, redis_service


def test_request_serialises_with_and_without_pod_states() -> None:
    without = ExecutorSpecRequest(**_spec_kwargs())
    with_states = ExecutorSpecRequest(
        **_spec_kwargs(),
        pod_states=[
            PodContainerState(
                pod_id="pod-1", container_state=ContainerState.EXITED, observed_at=OBSERVED_AT
            )
        ],
    )

    assert json.loads(without.model_dump_json())["pod_states"] is None
    assert json.loads(with_states.model_dump_json())["pod_states"] == [
        {"pod_id": "pod-1", "container_state": "exited", "observed_at": "2026-09-10T12:30:00Z"}
    ]


def test_request_ignores_a_key_it_does_not_know() -> None:
    # The backend's ExecutorSpecRequest is the same kind of plain pydantic model (lium-platform
    # protocol/base.py BaseRequest, default extra="ignore"), so this is what an older backend does
    # with pod_states: reads past it.
    parsed = ExecutorSpecRequest.model_validate(
        {**_spec_kwargs(), "message_type": "ExecutorSpecRequest", "field_from_the_future": [1]}
    )

    assert parsed.pod_states is None
    assert not hasattr(parsed, "field_from_the_future")


@pytest.mark.asyncio
async def test_publisher_carries_pod_states_to_the_websocket_request(
    create_job_result, mock_settings
) -> None:
    job = create_job_result()
    job.pod_states = [
        PodContainerState(
            pod_id="pod-1", container_state=ContainerState.RUNNING, observed_at=OBSERVED_AT
        ),
        PodContainerState(
            pod_id="orphan-9", container_state=ContainerState.REAPED, observed_at=OBSERVED_AT
        ),
    ]
    service, redis_service = _publisher()

    await service.publish_machine_specs([job], miner_hotkey="hk", miner_coldkey="ck")

    _, payload = redis_service.publish.await_args.args
    spec = await _bridge_machine_spec(payload)

    assert [(s.pod_id, s.container_state) for s in spec.pod_states] == [
        ("pod-1", ContainerState.RUNNING),
        ("orphan-9", ContainerState.REAPED),
    ]
    assert spec.pod_states[0].observed_at == OBSERVED_AT


@pytest.mark.asyncio
async def test_publisher_sends_null_when_the_cycle_observed_no_pod(
    create_job_result, mock_settings
) -> None:
    service, redis_service = _publisher()

    await service.publish_machine_specs([create_job_result()], miner_hotkey="hk", miner_coldkey="ck")

    _, payload = redis_service.publish.await_args.args
    assert payload["pod_states"] is None
    assert (await _bridge_machine_spec(payload)).pod_states is None


@pytest.mark.asyncio
async def test_payload_from_an_older_validator_parses_with_pod_states_none() -> None:
    payload = {
        **_spec_kwargs(),
        "executor_ssh_port": None,
        "price_per_gpu": None,
        "ssh_pub_keys": None,
    }
    payload.pop("validator_hotkey")

    assert (await _bridge_machine_spec(payload)).pod_states is None


async def _job_result_for(ctx):
    # The JobResult construction is the only thing under test; dry_run skips the redis persist.
    handler = ResultHandler(redis_service=MagicMock(), dry_run=True)
    return await handler.handle_result(
        context=ctx,
        miner_info=MagicMock(job_batch_id="2026-09-10 12:00:00", miner_hotkey="hk"),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="ok",
        success=True,
    )


@pytest.mark.asyncio
async def test_result_handler_copies_the_states_from_the_context() -> None:
    states = [
        PodContainerState(
            pod_id="pod-1", container_state=ContainerState.UNKNOWN, observed_at=OBSERVED_AT
        )
    ]

    with_states = await _job_result_for(make_context(state=build_state(pod_states=states)))
    empty = await _job_result_for(make_context(state=build_state()))

    assert with_states.pod_states == states
    assert empty.pod_states is None
