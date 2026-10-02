"""DAH-3980 liumd (staging prototype): a rent diverted to the node's agent through compute-app's relay, and every
way back to today's path (untracked/epics/fast-rent/design-liumd-rent-path.md §3)."""

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest
from clients.compute_client import ComputeClient, OutgoingMessages
from datura.requests.miner_requests import AcceptSSHKeyRequest
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    CustomOptions,
    FailedContainerRequest,
    LiumdAgentFrame,
    LiumdRentRequest,
    PayloadPortMapping,
)
from services import liumd_rent as liumd_rent_module
from services import miner_service as miner_module
from services.docker_service import create_steps_after_reply, inflight_creates
from services.gpu_power_limit import GpuPowerRestoreRecord
from services.liumd_rent import LiumdAgentFrames, LiumdRentService
from services.miner_service import MinerService

BENCH_EXECUTOR_ID = "df044c30-b8b4-4f4f-8860-9d451c16090c"
MINER_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
GPU_UUID = "GPU-6a1f0c52-3e9b-4d7a-8f21-0b9c4e5d7a13"
RENTER_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIrenterkey renter@laptop"
TODAYS_PATH_REPLY = Mock(name="todays_path_reply")
RESULT_STEPS = {
    "validated": 4,
    "preempt_listed": 3,
    "fillers_removed": 560,
    "fillers_gone": 49,
    "volume_created": 35,
    "container_created": 59,
    "gpu_power_restored": 180,
    "started": 644,
    "running": 2,
    "exec_volume_setup": 260,
    "exec_environment": 21,
}


def make_rent_payload(**overrides) -> ContainerCreateRequest:
    fields = dict(
        miner_hotkey=MINER_HOTKEY,
        executor_id=BENCH_EXECUTOR_ID,
        miner_address="10.0.0.1",
        miner_port=8000,
        pod_id=str(uuid4()),
        docker_image="daturaai/pytorch:2.11.0-py3.12-cuda12.8-devel-ubuntu24.04-dind-lium1",
        user_public_keys=[RENTER_KEY],
        gpu_uuids=[GPU_UUID],
        cpu_count=4,
        memory_gb=16,
        custom_options=CustomOptions(internal_ports=[], environment={"EXAMPLE_VAR": "example-value"}, shm_size="6g"),
        local_volume="",
        volume_limit_gb=5,
        storage_limit_gb=2,
        disk_share=1.0,
        is_sysbox=True,
        enable_volume_encryption=True,
        ships_sshd=True,
        available_ports=[PayloadPortMapping(internal_port=port, external_port=port) for port in range(20290, 20300)],
        pod_mapping=[],
        active_container_names=[],
        active_volume_names=[],
        timestamp=1,
    )
    return ContainerCreateRequest(**{**fields, **overrides})


def step_frame(rent: dict, step: str) -> dict:
    return {
        "type": "step",
        "rent_id": rent["rent_id"],
        "attempt": rent["attempt"],
        "step": step,
        "ms": RESULT_STEPS[step],
        "t_ms": 1,
        "host_touched": False,
        "detail": {},
    }


def result_frame(rent: dict) -> dict:
    return {
        "type": "result",
        "rent_id": rent["rent_id"],
        "attempt": rent["attempt"],
        "container_name": rent["container"]["name"],
        "container_id": "c" * 64,
        "volume_name": rent["volume"]["name"],
        "port_maps": rent["container"]["ports"],
        "image": {"id": "sha256:" + "a" * 64, "repo_digests": [], "user": ""},
        "preempted": ["filler_2cb46600-19a2-42d6-bf91-6d2c14d5900e"],
        "volume_encryption_status": "ENABLED",
        "total_ms": 1610,
        "steps": RESULT_STEPS,
    }


def error_frame(rent: dict, code: str, *, host_touched: bool, cleaned: bool, failed_step: str | None = None) -> dict:
    return {
        "type": "error",
        "rent_id": rent["rent_id"],
        "attempt": rent["attempt"],
        "code": code,
        "failed_step": failed_step,
        "message": "renter-safe text",
        "host_touched": host_touched,
        "cleaned": cleaned,
        "detail": {},
        "steps": {},
    }


class FakeRelay:
    """compute-app's relay and the agent behind it: records every frame the connector sends, answers by script."""

    def __init__(self, agent_frames: LiumdAgentFrames, answer: Callable[[dict], list[dict]]) -> None:
        self.agent_frames = agent_frames
        self.answer = answer
        self.sent: list[dict] = []
        agent_frames.send_to_backend = self.send

    def send(self, request: LiumdRentRequest) -> bool:
        self.sent.append(request.frame)
        loop = asyncio.get_running_loop()
        for frame in self.answer(request.frame):
            message = LiumdAgentFrame(
                message_type="LiumdAgentFrame", executor_id=request.executor_id, pod_id=request.pod_id, frame=frame
            )
            loop.call_soon(self.agent_frames.deliver, message)
        return True

    def frames_of_type(self, frame_type: str) -> list[dict]:
        return [frame for frame in self.sent if frame["type"] == frame_type]


def answer_rent_with(*replies: Callable[[dict], dict]) -> Callable[[dict], list[dict]]:
    return lambda frame: [reply(frame) for reply in replies] if frame["type"] == "rent" else []


@pytest.fixture
def miner_service(mocker) -> MinerService:
    mocker.patch("core.config.settings.USE_REST_API", True)
    mocker.patch("core.config.settings.ENABLE_VOLUME_ENCRYPTION", True)
    mocker.patch(
        "core.config.Settings.get_bittensor_wallet",
        return_value=Mock(get_hotkey=Mock(return_value=Mock(ss58_address="5TestValidator", sign=Mock(return_value=b"s")))),
    )
    redis_service = AsyncMock()
    redis_service.renting_in_progress = AsyncMock(return_value=False)
    redis_service.get = AsyncMock(return_value=None)
    redis_service.acquire_executor_lock = MagicMock(side_effect=lambda *_: _async_null_context())
    redis_service.executor_create_exclusion = MagicMock(side_effect=lambda *_: _async_null_context())
    ssh_service = Mock()
    ssh_service.generate_ssh_key.return_value = (b"---PRIV---", b"ssh-ed25519 validator-pub")
    service = MinerService(
        ssh_service=ssh_service,
        task_service=Mock(),
        redis_service=redis_service,
        attestation_service=Mock(),
    )
    service.liumd_rent = LiumdRentService(
        redis_service=redis_service, docker_service=service.liumd_rent.docker_service, agent_frames=LiumdAgentFrames()
    )
    # today's path on staging (USE_REST_API): the miner key exchange and the SSH create
    service._handle_container = AsyncMock(return_value=TODAYS_PATH_REPLY)
    service._make_rest_request = AsyncMock(return_value=(500, None))
    return service


@contextlib.asynccontextmanager
async def _async_null_context():
    yield True


def relay_for(miner_service: MinerService, answer: Callable[[dict], list[dict]]) -> FakeRelay:
    return FakeRelay(miner_service.liumd_rent.agent_frames, answer)


@pytest.mark.asyncio
async def test_connected_agent_answers_the_rent_before_any_miner_key_exchange(miner_service):
    relay = relay_for(
        miner_service,
        answer_rent_with(lambda rent: step_frame(rent, "fillers_gone"), result_frame),
    )
    payload = make_rent_payload()

    reply = await miner_service.handle_container(payload)
    key_exchanges_before_reply = miner_service._make_rest_request.await_count
    await create_steps_after_reply.wait_until_done(payload.pod_id, 5)

    assert isinstance(reply, ContainerCreated)
    assert reply.container_name == f"pod_{payload.pod_id}"
    assert reply.port_maps[0] == (22, 20299)
    profiler_names = {step.name.value for step in reply.profilers}
    assert {"liumd round trip", "liumd fillers_gone", "liumd started", "liumd exec_volume_setup"} <= profiler_names
    assert key_exchanges_before_reply == 0
    assert miner_service._make_rest_request.await_args_list[0].kwargs["url"].endswith("/api/validator/ssh-pubkey-submit")
    miner_service._handle_container.assert_not_awaited()
    assert [frame["type"] for frame in relay.sent] == ["rent", "ack"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_error",
    [
        pytest.param(lambda rent: error_frame(rent, "not_connected", host_touched=False, cleaned=True), id="not_connected"),
        pytest.param(lambda rent: error_frame(rent, "ineligible", host_touched=False, cleaned=True), id="ineligible"),
        pytest.param(
            lambda rent: error_frame(rent, "step_failed", host_touched=False, cleaned=True, failed_step="validated"),
            id="host_not_touched",
        ),
        pytest.param(
            lambda rent: error_frame(rent, "step_failed", host_touched=True, cleaned=True, failed_step="started"),
            id="host_touched_and_cleaned",
        ),
    ],
)
async def test_agent_error_that_left_nothing_on_the_host_runs_todays_path_once(miner_service, agent_error):
    relay = relay_for(miner_service, answer_rent_with(agent_error))

    reply = await miner_service.handle_container(make_rent_payload())

    assert reply is TODAYS_PATH_REPLY
    miner_service._handle_container.assert_awaited_once()
    assert len(relay.frames_of_type("rent")) == 1


@pytest.mark.asyncio
async def test_agent_error_that_could_not_clean_up_fails_the_rent_without_todays_path(miner_service):
    relay_for(
        miner_service,
        answer_rent_with(
            lambda rent: error_frame(rent, "step_failed", host_touched=True, cleaned=False, failed_step="started")
        ),
    )

    reply = await miner_service.handle_container(make_rent_payload())

    assert isinstance(reply, FailedContainerRequest)
    assert reply.failure_step == "liumd_started"
    miner_service._handle_container.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_reply_fails_the_rent_without_todays_path(miner_service):
    relay_for(miner_service, answer_rent_with(lambda rent: error_frame(rent, "reply_lost", host_touched=True, cleaned=False)))

    reply = await miner_service.handle_container(make_rent_payload())

    assert isinstance(reply, FailedContainerRequest)
    assert reply.failure_step == "liumd_reply_lost"
    miner_service._handle_container.assert_not_awaited()


@pytest.mark.asyncio
async def test_silent_agent_times_out_cancels_and_fails_without_todays_path(miner_service, monkeypatch):
    monkeypatch.setattr(liumd_rent_module, "LIUMD_RENT_DEADLINE_MS", 50)
    monkeypatch.setattr(liumd_rent_module, "LIUMD_REPLY_GRACE_SECONDS", 0.05)
    relay = relay_for(miner_service, lambda frame: [])

    reply = await miner_service.handle_container(make_rent_payload())

    assert isinstance(reply, FailedContainerRequest)
    assert reply.failure_step == "liumd_timeout"
    assert [frame["reason"] for frame in relay.frames_of_type("cancel")] == ["timeout"]
    miner_service._handle_container.assert_not_awaited()


@pytest.mark.asyncio
async def test_pod_deleted_during_the_create_sends_cancel_and_never_falls_back(miner_service):
    def agent(frame: dict) -> list[dict]:
        if frame["type"] == "cancel":
            return [error_frame(frame, "cancelled", host_touched=True, cleaned=True)]
        return []

    relay = relay_for(miner_service, agent)
    payload = make_rent_payload()

    async def delete_pod_soon() -> None:
        await asyncio.sleep(0.05)
        inflight_creates.cancel(payload.pod_id)

    reply, _ = await asyncio.gather(miner_service.handle_container(payload), delete_pod_soon())

    assert isinstance(reply, FailedContainerRequest)
    assert reply.failure_step == "cancelled_by_delete"
    assert [frame["reason"] for frame in relay.frames_of_type("cancel")] == ["pod_deleted"]
    miner_service._handle_container.assert_not_awaited()


@pytest.mark.asyncio
async def test_customer_rent_on_a_node_with_a_filler_carries_preempt_and_gpu_power(miner_service):
    restore_record = GpuPowerRestoreRecord(
        gpu_uuid=GPU_UUID, watts=140, pod_id=str(uuid4()), executor_id=BENCH_EXECUTOR_ID, capped_at=1.0
    )
    miner_service.redis_service.get = AsyncMock(return_value=restore_record.model_dump_json())
    relay = relay_for(miner_service, answer_rent_with(result_frame))

    reply = await miner_service.handle_container(make_rent_payload())

    rent = relay.frames_of_type("rent")[0]
    assert isinstance(reply, ContainerCreated)
    assert rent["validator_hotkey"] == MINER_HOTKEY
    assert rent["preempt"] == {"fillers": True}
    assert rent["gpu_power"] == [{"gpu_uuid": GPU_UUID, "restore_watts": 140}]
    assert rent["gpu_power_floor_ratio"] == 0.9
    assert rent["volume"]["driver"] == "vloopback"
    assert {"container_port": 22, "host_port": 20299, "protocol": "tcp"} in rent["container"]["ports"]
    assert [exec_["name"] for exec_ in rent["execs"]] == ["volume_setup", "environment"]
    assert RENTER_KEY in base64.b64decode(rent["execs"][0]["stdin_b64"]).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"enable_volume_encryption": False}, id="no_encryption"),
        pytest.param({"executor_id": str(uuid4())}, id="another_executor"),
        pytest.param({"ships_sshd": False}, id="image_without_sshd"),
        pytest.param({"local_volume": "volume_existing"}, id="edit_of_existing_pod"),
        pytest.param({"miner_hotkey": ""}, id="empty_miner_hotkey"),
        pytest.param({"miner_hotkey": "5TestMiner"}, id="miner_hotkey_not_ss58"),
    ],
)
async def test_ineligible_rent_never_sends_a_frame(miner_service, overrides):
    relay = relay_for(miner_service, answer_rent_with(result_frame))

    reply = await miner_service.handle_container(make_rent_payload(**overrides))

    assert reply is TODAYS_PATH_REPLY
    assert relay.sent == []
    miner_service._handle_container.assert_awaited_once()


@pytest.mark.asyncio
async def test_rent_logs_one_result_line_and_never_the_frame(miner_service, caplog):
    caplog.set_level(logging.DEBUG)
    relay = relay_for(miner_service, answer_rent_with(result_frame))
    payload = make_rent_payload()

    await miner_service.handle_container(payload)
    await create_steps_after_reply.wait_until_done(payload.pod_id, 5)

    rent = relay.frames_of_type("rent")[0]
    # the message and its structured extra, as the JSON formatter ships them to Loki
    logged = "\n".join(
        record.msg.to_full_string() if hasattr(record.msg, "to_full_string") else record.getMessage()
        for record in caplog.records
    )
    assert [record.getMessage() for record in caplog.records].count("LIUMD_RENT_RESULT") == 1
    assert "stdin_b64" not in logged
    assert rent["execs"][0]["stdin_b64"] not in logged
    assert "gocryptfs" not in logged
    assert "stdin_b64" not in repr(LiumdRentRequest(executor_id=BENCH_EXECUTOR_ID, pod_id=payload.pod_id, frame=rent))


@pytest.mark.asyncio
async def test_compute_client_routes_an_agent_frame_to_the_waiting_rent(monkeypatch):
    agent_frames = LiumdAgentFrames()
    monkeypatch.setattr("clients.compute_client.liumd_agent_frames", agent_frames)
    compute_client = ComputeClient(keypair=Mock(ss58_address="5TestValidator"), compute_app_uri="ws://x", miner_service=Mock())
    frame = {"type": "result", "rent_id": "pod", "attempt": "9f2c4e7a1b3d5f60"}

    with agent_frames.waiting_for(BENCH_EXECUTOR_ID, "pod", "9f2c4e7a1b3d5f60") as replies:
        await compute_client.handle_message(
            json.dumps({"message_type": "LiumdAgentFrame", "executor_id": BENCH_EXECUTOR_ID, "pod_id": "pod", "frame": frame})
        )
        delivered = replies.get_nowait()

    assert delivered == frame
    assert compute_client.queue_liumd_request(LiumdRentRequest(executor_id=BENCH_EXECUTOR_ID, pod_id="pod", frame={})) is False
    await compute_client.miner_drivers.put(None)


# Review findings LIUM-115 / LIUM-128 (untracked/epics/fast-rent/experiments/LIUM-128-liumd-platform-review.md); the
# reviewers' red tests from .factory/LIUM-128/ are kept under their own names where the safe behaviour is the same.


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_step", ["_build_rent_frame", "add_pending_pod"])
async def test_pre_send_error_with_no_agent_preserves_todays_path(miner_service, failing_step):
    # F1, from test_lium128_review.py; add_pending_pod: the mark half-taken is still undone
    if failing_step == "_build_rent_frame":
        miner_service.liumd_rent._build_rent_frame = AsyncMock(side_effect=ValueError("injected builder failure"))
    else:
        miner_service.redis_service.add_pending_pod = AsyncMock(side_effect=ConnectionError("injected redis failure"))
    relay = relay_for(miner_service, answer_rent_with(result_frame))

    reply = await miner_service.handle_container(make_rent_payload())

    assert reply is TODAYS_PATH_REPLY
    assert relay.sent == []
    expected_pending_removals = 1 if failing_step == "add_pending_pod" else 0
    assert miner_service.redis_service.remove_pending_pod.await_count == expected_pending_removals


@pytest.mark.asyncio
async def test_cancellation_releases_pending_mark_and_sends_cancel(miner_service):
    # F1, from test_lium128_review.py
    payload = make_rent_payload()
    relay = relay_for(miner_service, lambda frame: [])
    entered = asyncio.Event()
    released = []

    @contextlib.asynccontextmanager
    async def exclusion(*args):
        try:
            yield True
        finally:
            released.append(True)

    miner_service.redis_service.executor_create_exclusion = exclusion

    async def waiting(*args):
        entered.set()
        await asyncio.Event().wait()

    miner_service.liumd_rent._wait_for_reply = waiting

    task = asyncio.create_task(miner_service.handle_container(payload))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert released == [True]
    assert not inflight_creates.is_running(payload.pod_id)
    assert miner_service.redis_service.remove_pending_pod.await_count == 1
    assert [(frame["type"], frame.get("action")) for frame in relay.sent] == [("rent", None), ("cancel", None), ("settle", "rollback")]
    attempt = relay.frames_of_type("rent")[0]["attempt"]
    miner_service.redis_service.set.assert_awaited_once_with(f"liumd:attempt:{attempt}", "failed", ex=3600)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spoil_result",
    [
        pytest.param(lambda frame: frame.pop("volume_name"), id="missing_volume_name"),
        pytest.param(lambda frame: frame.update(container_name=["pod"]), id="container_name_not_a_string"),
        pytest.param(lambda frame: frame.update(volume_name="volume_of_another_pod"), id="another_volume"),
    ],
)
async def test_missing_result_field_does_not_ack_before_validation(miner_service, spoil_result):
    # F1, replaces test_lium128_review.py's version that expected the KeyError: the result is rolled back
    def bad_result(rent: dict) -> dict:
        frame = result_frame(rent)
        spoil_result(frame)
        return frame

    relay = relay_for(miner_service, answer_rent_with(bad_result))

    reply = await miner_service.handle_container(make_rent_payload())

    assert isinstance(reply, FailedContainerRequest)
    assert reply.failure_step == "liumd_bad_result"
    assert [frame["type"] for frame in relay.sent] == ["rent", "settle"]
    assert relay.frames_of_type("settle")[0]["action"] == "rollback"
    miner_service._handle_container.assert_not_awaited()
    miner_service.redis_service.remove_pending_pod.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_error",
    [
        pytest.param(
            lambda rent: {"type": "error", "attempt": rent["attempt"], "rent_id": rent["rent_id"], "code": "protocol"},
            id="no_host_state",
        ),
        pytest.param(lambda rent: {**error_frame(rent, "step_failed", host_touched=False, cleaned=True), "host_touched": "false"}, id="host_touched_not_bool"),
        pytest.param(lambda rent: {**error_frame(rent, "step_failed", host_touched=True, cleaned=True), "cleaned": 1}, id="cleaned_not_bool"),
        pytest.param(lambda rent: error_frame(rent, "send_failed", host_touched=False, cleaned=True), id="send_failed"),
        pytest.param(lambda rent: error_frame(rent, "reply_lost", host_touched=False, cleaned=True), id="reply_lost"),
    ],
)
async def test_error_without_host_state_never_authorizes_fallback(miner_service, agent_error):
    # F2, from test_lium128_review.py; send_failed / reply_lost are unknown whatever their flags say
    relay = relay_for(miner_service, answer_rent_with(agent_error))

    reply = await miner_service.handle_container(make_rent_payload())

    assert isinstance(reply, FailedContainerRequest)
    miner_service._handle_container.assert_not_awaited()
    assert len(relay.frames_of_type("cancel")) == 1


@pytest.mark.asyncio
async def test_unsent_rent_is_not_replayed_after_its_waiter_times_out(miner_service, monkeypatch):
    # F3 (a), replaces test_lium128_transport.py's version: the rent stays queued and is dropped at send time
    client = object.__new__(ComputeClient)
    client.ws = object()
    client.message_queue = OutgoingMessages()
    client.lock = asyncio.Lock()
    client.logging_extra = {}
    frames = miner_service.liumd_rent.agent_frames

    def enqueue_then_disconnect(request: LiumdRentRequest) -> bool:
        queued = client.queue_liumd_request(request)
        client.ws = None
        return queued

    frames.send_to_backend = enqueue_then_disconnect
    monkeypatch.setattr(liumd_rent_module, "LIUMD_RENT_DEADLINE_MS", 10)
    monkeypatch.setattr(liumd_rent_module, "LIUMD_REPLY_GRACE_SECONDS", 0)
    reply = await miner_service.handle_container(make_rent_payload())
    sent_to_backend: list[str] = []
    client.ws = Mock(send=AsyncMock(side_effect=lambda raw: sent_to_backend.append(json.loads(raw)["frame"]["type"])))

    sender = asyncio.create_task(client.handle_send_messages())
    while client.message_queue:
        await asyncio.sleep(0)
    sender.cancel()

    assert reply.failure_step == "liumd_timeout"
    assert sent_to_backend == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("mark", "action"), [(b"acked", "ack"), (b"failed", "rollback"), (None, "rollback")])
async def test_unacked_attempt_gets_a_settlement(miner_service, mark, action):
    # F3 (b), from test_lium128_review.py
    frames = miner_service.liumd_rent.agent_frames
    miner_service.redis_service.get = AsyncMock(return_value=mark)
    relay = relay_for(miner_service, lambda frame: [])
    message = LiumdAgentFrame(
        message_type="LiumdAgentFrame",
        executor_id=BENCH_EXECUTOR_ID,
        pod_id="pod",
        frame={"type": "attempt_state", "rent_id": "pod", "attempt": "old", "state": "unacked"},
    )

    frames.deliver(message)
    await asyncio.sleep(0)

    miner_service.redis_service.get.assert_awaited_once_with("liumd:attempt:old")
    assert relay.frames_of_type("settle") == [{"type": "settle", "rent_id": "pod", "attempt": "old", "action": action}]


@pytest.mark.asyncio
async def test_ack_and_acked_mark_follow_the_handed_on_container_created(miner_service):
    # F3 (b, c): nothing is acked while handle_container has not yet returned ContainerCreated to its caller
    relay = relay_for(miner_service, answer_rent_with(result_frame))
    payload = make_rent_payload()

    reply = await miner_service.handle_container(payload)
    frames_at_hand_off = [frame["type"] for frame in relay.sent]
    marks_at_hand_off = miner_service.redis_service.set.await_count
    await create_steps_after_reply.wait_until_done(payload.pod_id, 5)

    assert isinstance(reply, ContainerCreated)
    assert (frames_at_hand_off, marks_at_hand_off) == (["rent"], 0)
    assert [frame["type"] for frame in relay.sent] == ["rent", "ack"]
    attempt = relay.sent[0]["attempt"]
    miner_service.redis_service.set.assert_awaited_once_with(f"liumd:attempt:{attempt}", "acked", ex=3600)
    miner_service.redis_service.remove_pending_pod.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_arriving_while_waiting_is_observed_before_result(miner_service):
    # F7, from test_lium128_review.py: call_soon lands after the poll, before the result is taken
    payload = make_rent_payload()

    def agent(frame: dict) -> list[dict]:
        if frame["type"] != "rent":
            return []
        asyncio.get_running_loop().call_soon(inflight_creates.cancel, payload.pod_id)
        return [result_frame(frame)]

    relay = relay_for(miner_service, agent)

    reply = await miner_service.handle_container(payload)
    await create_steps_after_reply.wait_until_done(payload.pod_id, 1)

    assert isinstance(reply, FailedContainerRequest)
    assert reply.failure_step == "cancelled_by_delete"
    assert [(frame["type"], frame.get("action")) for frame in relay.sent] == [("rent", None), ("settle", "rollback")]


@pytest.mark.asyncio
async def test_observed_delete_crossing_result_is_rolled_back_not_acked(miner_service):
    # F4, replaces test_lium128_lifecycle.py's characterization (`rent, cancel, ack`) and its by-name delete test:
    # the agent removes the container and the volume of the attempt by label on `settle rollback`
    payload = make_rent_payload()
    rent_frame = {}

    def agent(frame: dict) -> list[dict]:
        if frame["type"] == "rent":
            rent_frame.update(frame)
            asyncio.get_running_loop().call_soon(inflight_creates.cancel, payload.pod_id)
        if frame["type"] == "cancel":
            return [result_frame(rent_frame)]
        return []

    relay = relay_for(miner_service, agent)

    reply = await miner_service.handle_container(payload)

    assert isinstance(reply, FailedContainerRequest)
    assert reply.failure_step == "cancelled_by_delete"
    assert [frame["type"] for frame in relay.sent] == ["rent", "cancel", "settle"]
    assert relay.frames_of_type("settle")[0]["action"] == "rollback"
    miner_service.redis_service.remove_pending_pod.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("slow", [False, True], ids=["ssh_failure", "slow_inspector"])
async def test_post_reply_failure_or_slowness_keeps_reply_and_removes_key(
    miner_service, sample_executor_info, monkeypatch, slow
):
    # F5, from test_lium128_lifecycle.py
    payload = make_rent_payload()
    sample_executor_info.uuid = payload.executor_id
    miner_service._make_rest_request = AsyncMock(return_value=(200, {"synthetic": True}))
    miner_service._remove_ssh_key_via_rest = AsyncMock(return_value=True)
    monkeypatch.setattr(miner_module, "_parse_miner_response", lambda _: AcceptSSHKeyRequest(executors=[sample_executor_info]))
    monkeypatch.setattr(miner_module.asyncssh, "import_private_key", lambda _: object())
    monkeypatch.setattr(miner_module.settings, "ENABLE_INSPECTOR", True)
    docker = miner_service.liumd_rent.docker_service
    docker._cache_rented_pod_best_effort = AsyncMock()
    docker._prepare_known_hosts_policy = AsyncMock(return_value=None)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def inspector(*args):
        entered.set()
        await release.wait()

    docker._run_inspector_collector_lifecycle = inspector

    @contextlib.asynccontextmanager
    async def connect(**kwargs):
        if not slow:
            raise RuntimeError("injected SSH failure")
        yield Mock()

    monkeypatch.setattr(miner_module.asyncssh, "connect", connect)
    relay_for(miner_service, answer_rent_with(result_frame))

    reply = await miner_service.handle_container(payload)
    if slow:
        await asyncio.wait_for(entered.wait(), 1)
        assert not inflight_creates.is_running(payload.pod_id)
        release.set()
    await create_steps_after_reply.wait_until_done(payload.pod_id, 1)

    assert isinstance(reply, ContainerCreated)
    miner_service._remove_ssh_key_via_rest.assert_awaited_once()


@pytest.mark.asyncio
async def test_inspector_submit_cancel_still_attempts_key_removal(miner_service):
    # F5, from test_lium128_review.py
    submitted = asyncio.Event()

    async def submit(**kwargs):
        submitted.set()
        await asyncio.Event().wait()

    miner_service._make_rest_request = submit
    miner_service._remove_ssh_key_via_rest = AsyncMock(return_value=True)

    task = asyncio.create_task(miner_service._start_inspector_after_agent_rent(make_rent_payload(), "pod_test"))
    await submitted.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    miner_service._remove_ssh_key_via_rest.assert_awaited_once()


@pytest.mark.asyncio
async def test_inspector_submit_timeout_still_removes_possibly_installed_key(miner_service):
    # F5, from test_lium128_transport.py
    miner_service._make_rest_request = AsyncMock(side_effect=asyncio.TimeoutError)
    miner_service._remove_ssh_key_via_rest = AsyncMock(return_value=True)

    await miner_service._start_inspector_after_agent_rent(make_rent_payload(), "pod_test")

    miner_service._remove_ssh_key_via_rest.assert_awaited_once()


@pytest.mark.asyncio
async def test_hung_submit_is_cut_at_the_time_limit_and_a_failed_removal_retried_once(miner_service, monkeypatch, caplog):
    # F5: the one time limit on the step, then two removals that both answer False
    async def hung_submit(**kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(miner_module, "AGENT_RENT_INSPECTOR_START_TIMEOUT", 0.05)
    miner_service._make_rest_request = hung_submit
    miner_service._remove_ssh_key_via_rest = AsyncMock(return_value=False)
    payload = make_rent_payload()
    caplog.set_level(logging.ERROR)

    await asyncio.wait_for(miner_service._start_inspector_after_agent_rent(payload, "pod_test"), 1)

    assert miner_service._remove_ssh_key_via_rest.await_count == 2
    key_left = [record for record in caplog.records if record.getMessage() == "Validator key may be left on the node after an agent rent"]
    assert len(key_left) == 1
    assert payload.executor_id in key_left[0].msg.to_full_string()


@pytest.mark.asyncio
async def test_echoed_setup_data_is_not_put_into_logs_or_failure_detail(miner_service, caplog):
    # F6, from test_lium128_boundaries.py; the marker also rides in `message` and under an unknown step name
    marker = "U1lOVEhFVElDX1NFVFVQX0RBVEE="

    def error(rent: dict) -> dict:
        reply = error_frame(rent, "step_failed", host_touched=True, cleaned=False, failed_step="exec_volume_setup")
        reply["steps"] = {"exec_volume_setup": {"stdin_b64": marker}, marker: 5, "started": 644}
        reply["detail"] = {"stderr_tail": marker, "exit_code": 91}
        reply["message"] = marker
        return reply

    relay_for(miner_service, answer_rent_with(error))
    caplog.set_level(logging.INFO)

    reply = await miner_service.handle_container(make_rent_payload())
    logged = "\n".join(
        record.msg.to_full_string() if hasattr(record.msg, "to_full_string") else record.getMessage()
        for record in caplog.records
    )

    assert marker not in logged
    assert marker not in reply.msg
    assert marker not in (reply.detail or "")
    assert reply.failure_step == "liumd_exec_volume_setup"
    assert '"exit_code": 91' in reply.detail
    assert "'started': 644" in logged or '"started": 644' in logged


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("executor_id", "pod_id", "rent_id"),
    [
        pytest.param("different-executor", "pod", "pod", id="other_executor"),
        pytest.param(BENCH_EXECUTOR_ID, "different-pod", "pod", id="other_envelope_pod"),
        pytest.param(BENCH_EXECUTOR_ID, "pod", "different-pod", id="other_rent_id"),
    ],
)
async def test_reply_correlation_binds_executor_and_pod_as_well_as_attempt(executor_id, pod_id, rent_id):
    # F10, from test_lium128_boundaries.py
    frames = LiumdAgentFrames()
    with frames.waiting_for(BENCH_EXECUTOR_ID, "pod", "known-attempt") as replies:
        message = LiumdAgentFrame(
            message_type="LiumdAgentFrame",
            executor_id=executor_id,
            pod_id=pod_id,
            frame={"type": "result", "attempt": "known-attempt", "rent_id": rent_id},
        )

        frames.deliver(message)

        assert replies.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame",
    [
        pytest.param({"type": "result", "attempt": {"invalid": "type"}, "rent_id": "pod"}, id="dict_attempt"),
        pytest.param({"type": "result", "attempt": "known-attempt", "rent_id": ["pod"]}, id="list_rent_id"),
        pytest.param({"type": "attempt_state", "attempt": {"invalid": "type"}, "state": "unacked"}, id="attempt_state"),
    ],
)
async def test_bad_attempt_type_is_contained_at_frame_boundary(frame):
    # F10, from test_lium128_transport.py
    frames = LiumdAgentFrames()
    frames.send_to_backend = Mock()
    with frames.waiting_for(BENCH_EXECUTOR_ID, "pod", "known-attempt") as replies:
        frames.deliver(LiumdAgentFrame(message_type="LiumdAgentFrame", executor_id=BENCH_EXECUTOR_ID, pod_id="pod", frame=frame))
        await asyncio.sleep(0)

        assert replies.empty()
    frames.send_to_backend.assert_not_called()


@pytest.mark.asyncio
async def test_connector_error_after_the_send_fails_the_rent_without_todays_path(miner_service):
    # F1: past the send the host's state is unknown, so the error is not a fallback
    relay = relay_for(miner_service, lambda frame: [])
    miner_service.liumd_rent._wait_for_reply = AsyncMock(side_effect=RuntimeError("injected connector failure"))

    reply = await miner_service.handle_container(make_rent_payload())

    assert isinstance(reply, FailedContainerRequest)
    assert reply.failure_step == "liumd_connector_error"
    miner_service._handle_container.assert_not_awaited()
    assert [frame["type"] for frame in relay.sent] == ["rent", "cancel", "settle"]
    miner_service.redis_service.remove_pending_pod.assert_awaited_once()
