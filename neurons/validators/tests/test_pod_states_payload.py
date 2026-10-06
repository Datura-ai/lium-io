"""DAH-3338: the container state of every rented pod travels to the backend as
ExecutorSpecRequest.pod_states — ctx.state → JobResult → MACHINE_SPEC_CHANNEL → the
websocket request — and the field is optional at every hop, so a validator and a
backend of different ages keep talking. With POD_STATES_REPORT_ENABLED the whole list
also goes out as PodStatesReport chunks (POD_STATES_CHANNEL → the websocket) after the
spec, so a node with more states than the spec's bound still reports them all in one cycle.
"""
import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pydantic
import pytest
from clients.compute_client import ComputeClient
from helpers import build_state, make_context
from protocol.vc_protocol.validator_requests import (
    POD_STATES_MAX_ITEMS,
    ContainerState,
    ExecutorSpecRequest,
    PodContainerState,
    PodStatesReport,
    chunk_pod_states,
)
from services.miner_service import MinerService
from services.redis_service import MACHINE_SPEC_CHANNEL, POD_STATES_CHANNEL
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
    # DAH-2748's availability_errors rides the same publish → bridge → request path; the two
    # keys sit side by side in the publisher dict, the bridge and ExecutorSpecRequest.
    job.availability_errors = [{"reason_code": "EXECUTOR_SSH_UNREACHABLE"}]
    service, redis_service = _publisher()

    await service.publish_machine_specs([job], miner_hotkey="hk", miner_coldkey="ck")

    _, payload = redis_service.publish.await_args.args
    spec = await _bridge_machine_spec(payload)

    assert [(s.pod_id, s.container_state) for s in spec.pod_states] == [
        ("pod-1", ContainerState.RUNNING),
        ("orphan-9", ContainerState.REAPED),
    ]
    assert spec.pod_states[0].observed_at == OBSERVED_AT
    assert spec.availability_errors == [{"reason_code": "EXECUTOR_SSH_UNREACHABLE"}]


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


def _states(prefix: str, state: ContainerState, n: int) -> list[PodContainerState]:
    return [
        PodContainerState(pod_id=f"{prefix}-{i}", container_state=state, observed_at=OBSERVED_AT) for i in range(n)
    ]


@pytest.mark.asyncio
async def test_result_handler_keeps_every_state_for_the_publisher() -> None:
    # the cleanup check runs before the rented-state check, so the list has the reaped ids first;
    # the publisher, not the handler, decides what fits the spec and what goes in report chunks
    states = [*_states("reaped", ContainerState.REAPED, 10), *_states("seen", ContainerState.RUNNING, 300)]

    result = await _job_result_for(make_context(state=build_state(pod_states=states)))

    assert result.pod_states == states


async def _publish_one(job) -> tuple[list[dict], list[dict]]:
    """Publish one result; back the payloads on the spec channel and on the report channel."""
    service, redis_service = _publisher()
    await service.publish_machine_specs([job], miner_hotkey="hk", miner_coldkey="ck")
    specs = [call.args[1] for call in redis_service.publish.await_args_list if call.args[0] == MACHINE_SPEC_CHANNEL]
    reports = [call.args[1] for call in redis_service.publish.await_args_list if call.args[0] == POD_STATES_CHANNEL]
    return specs, reports


@pytest.mark.asyncio
async def test_without_the_report_the_spec_carries_the_bound_and_nothing_else_is_sent(
    create_job_result, mock_settings, monkeypatch
) -> None:
    """The old-backend path (POD_STATES_REPORT_ENABLED off, the default): the spec is the only
    carrier. It never holds more than the backend accepts (a longer list fails the backend's
    validation and drops the whole spec), the reaped ids at the head of the list all fit, and the
    last observed states are the ones cut (the backend keeps those rows' last state)."""
    monkeypatch.setattr(mock_settings, "POD_STATES_REPORT_ENABLED", False)
    reaped = _states("reaped", ContainerState.REAPED, 32)
    observed = _states("seen", ContainerState.RUNNING, 256)
    job = create_job_result()
    job.pod_states = [*reaped, *observed]

    specs, reports = await _publish_one(job)

    assert reports == []
    (payload,) = specs
    spec = await _bridge_machine_spec(payload)
    assert len(spec.pod_states) == POD_STATES_MAX_ITEMS == 256
    assert [s.pod_id for s in spec.pod_states] == [s.pod_id for s in [*reaped, *observed][:256]]


@pytest.mark.asyncio
async def test_with_the_report_on_every_state_reaches_the_backend_in_the_same_cycle(
    create_job_result, mock_settings, monkeypatch
) -> None:
    """Regression (review): with 256 rented pods the spec had no room for a reaped id, and a queued
    id could expire unsent. With the report the spec keeps its bounded copy and the whole list goes
    out in chunks of 256 right after it: 256 observed + 40 reaped → two chunks, every id sent."""
    monkeypatch.setattr(mock_settings, "POD_STATES_REPORT_ENABLED", True)
    reaped = _states("reaped", ContainerState.REAPED, 40)
    observed = _states("seen", ContainerState.RUNNING, 256)
    job = create_job_result()
    job.pod_states = [*reaped, *observed]

    specs, reports = await _publish_one(job)

    assert len((await _bridge_machine_spec(specs[0])).pod_states) == 256
    assert [(r["chunk_index"], r["chunk_total"], len(r["pod_states"])) for r in reports] == [
        (0, 2, 256),
        (1, 2, 40),
    ]
    sent = [s["pod_id"] for r in reports for s in r["pod_states"]]
    assert sent == [s.pod_id for s in [*reaped, *observed]]
    assert {s.pod_id for s in reaped} <= set(sent)
    assert all(r["executor_uuid"] == job.executor_info.uuid and r["job_batch_id"] == job.job_batch_id for r in reports)


@pytest.mark.asyncio
async def test_a_failed_spec_or_chunk_publish_does_not_stop_the_next_result(
    create_job_result, mock_settings, monkeypatch
) -> None:
    """A redis error on one result's spec skips that result's report too (a chunk for a spec that
    never went out is noise); a redis error on one chunk loses that chunk only. The next result's
    spec and report still go out."""
    monkeypatch.setattr(mock_settings, "POD_STATES_REPORT_ENABLED", True)
    first, second = create_job_result(), create_job_result()
    first.pod_states = _states("a", ContainerState.REAPED, 1)
    second.pod_states = _states("b", ContainerState.REAPED, POD_STATES_MAX_ITEMS + 1)
    service, redis_service = _publisher()
    calls: list[str] = []

    async def publish(channel, payload):
        calls.append(channel)
        # the first result's spec, then the second result's second chunk
        if len(calls) == 1 or (channel == POD_STATES_CHANNEL and payload["chunk_index"] == 1):
            raise ConnectionError("redis away")

    redis_service.publish = AsyncMock(side_effect=publish)

    await service.publish_machine_specs([first, second], miner_hotkey="hk", miner_coldkey="ck")

    assert calls == [MACHINE_SPEC_CHANNEL, MACHINE_SPEC_CHANNEL, POD_STATES_CHANNEL, POD_STATES_CHANNEL]


@pytest.mark.asyncio
async def test_a_cycle_that_observed_nothing_sends_no_report(create_job_result, mock_settings, monkeypatch) -> None:
    monkeypatch.setattr(mock_settings, "POD_STATES_REPORT_ENABLED", True)

    _, reports = await _publish_one(create_job_result())

    assert reports == []


def test_chunk_pod_states_cuts_in_order_at_the_bound() -> None:
    states = _states("s", ContainerState.RUNNING, 2 * POD_STATES_MAX_ITEMS + 1)

    chunks = chunk_pod_states(states)

    assert [len(c) for c in chunks] == [256, 256, 1]
    assert [s for c in chunks for s in c] == states
    assert chunk_pod_states([]) == []
    for index, chunk in enumerate(chunks):
        PodStatesReport(
            validator_hotkey="vk",
            miner_hotkey="hk",
            executor_uuid="exec-1",
            job_batch_id="2026-09-10 12:00:00",
            chunk_index=index,
            chunk_total=len(chunks),
            pod_states=chunk,
        )
    with pytest.raises(pydantic.ValidationError):
        PodStatesReport(
            validator_hotkey="vk",
            miner_hotkey="hk",
            executor_uuid="exec-1",
            job_batch_id="b",
            chunk_index=0,
            chunk_total=1,
            pod_states=_states("s", ContainerState.RUNNING, POD_STATES_MAX_ITEMS + 1),
        )


@pytest.mark.asyncio
async def test_the_bridge_turns_a_report_payload_into_the_websocket_message() -> None:
    payload = {
        "miner_hotkey": "hk",
        "executor_uuid": "exec-1",
        "job_batch_id": "2026-09-10 12:00:00",
        "chunk_index": 1,
        "chunk_total": 2,
        "pod_states": [{"pod_id": "pod-1", "container_state": "reaped", "observed_at": OBSERVED_AT.isoformat()}],
        "sent_at": 1.0,
    }

    async def listen():
        yield {"channel": POD_STATES_CHANNEL.encode(), "data": json.dumps(payload).encode()}
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

    (report,) = client.message_queue
    assert isinstance(report, PodStatesReport)
    assert report.validator_hotkey == "validator-hotkey"
    assert (report.chunk_index, report.chunk_total) == (1, 2)
    assert report.pod_states[0].container_state is ContainerState.REAPED
    assert json.loads(report.model_dump_json())["message_type"] == "PodStatesReport"
