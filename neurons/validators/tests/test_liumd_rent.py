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
from clients.compute_client import ComputeClient
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
from services.docker_service import create_steps_after_reply, inflight_creates
from services.gpu_power_limit import GpuPowerRestoreRecord
from services.liumd_rent import LiumdAgentFrames
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
    service.liumd_rent.agent_frames = LiumdAgentFrames()
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
        pytest.param(lambda rent: error_frame(rent, "send_failed", host_touched=False, cleaned=True), id="send_failed"),
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

    with agent_frames.waiting_for("9f2c4e7a1b3d5f60") as replies:
        await compute_client.handle_message(
            json.dumps({"message_type": "LiumdAgentFrame", "executor_id": BENCH_EXECUTOR_ID, "pod_id": "pod", "frame": frame})
        )
        delivered = replies.get_nowait()

    assert delivered == frame
    assert compute_client.queue_liumd_request(LiumdRentRequest(executor_id=BENCH_EXECUTOR_ID, pod_id="pod", frame={})) is False
    await compute_client.miner_drivers.put(None)
