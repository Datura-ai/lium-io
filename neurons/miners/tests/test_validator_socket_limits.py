"""
The validator WebSocket bounds a connection until its sign-in completes: only a validator this miner serves is
accepted, the sign-in has a deadline, the pre-sign-in queue and frame size are bounded, pre-sign-in connections are
capped in total and per source address, and a sign-in timestamp may not lie in the future.

The socket is a small in-memory stand-in; the consumer and its sign-in check are the real ones.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import bittensor
import pytest
from datura.requests.validator_requests import (
    AuthenticateRequest,
    AuthenticationPayload,
    GetPodLogsRequest,
)
from starlette import status

import consumers.validator_consumer as consumer_module
from consumers.validator_consumer import ValidatorConsumer
from services.executor_service import ExecutorService
from services.ssh_service import MinerSSHService
from services.validator_service import ValidatorService

_EXECUTOR_ID = "0b2a6d1e-9e2e-4e2e-8e2e-000000000e2e"
_POLICY = status.WS_1008_POLICY_VIOLATION


class FakeSocket:
    def __init__(self, host: str = "198.51.100.7"):
        self.client = SimpleNamespace(host=host, port=40000)
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.accepted = False
        self.close_codes: list[int] = []
        self.sent: list[str] = []

    async def accept(self):
        self.accepted = True

    async def close(self, code: int = 1000, reason=None):
        self.close_codes.append(code)

    async def receive_text(self) -> str:
        if self.close_codes:
            raise RuntimeError("socket closed")
        return await self.inbox.get()

    async def send_text(self, data: str):
        self.sent.append(data)


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidatorSocket")


@pytest.fixture(scope="module")
def miner_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestMinerSocket")


@pytest.fixture(autouse=True)
def _module_state(miner_keypair, monkeypatch):
    wallet = MagicMock()
    wallet.get_hotkey.return_value = miner_keypair
    settings = MagicMock()
    settings.get_bittensor_wallet.return_value = wallet
    settings.CENTRAL_MODE = True
    settings.RENTAL_REQUEST_HOOK = None
    monkeypatch.setattr(consumer_module, "settings", settings)
    monkeypatch.setattr(consumer_module, "_pre_sign_in_by_source", consumer_module.Counter())


@pytest.fixture()
def make(validator_keypair):
    def make(key: str | None = None, host: str = "198.51.100.7") -> ValidatorConsumer:
        validator_service = MagicMock(spec=ValidatorService)
        validator_service.is_valid_validator.side_effect = lambda k: (
            k == validator_keypair.ss58_address
        )
        executor_service = MagicMock(spec=ExecutorService)
        executor_service.get_executors_for_validator = AsyncMock(
            return_value=[
                SimpleNamespace(uuid=UUID(_EXECUTOR_ID), address="203.0.113.10", port=8001)
            ]
        )
        return ValidatorConsumer(
            websocket=FakeSocket(host),
            validator_key=key or validator_keypair.ss58_address,
            ssh_service=MagicMock(spec=MinerSSHService),
            validator_service=validator_service,
            executor_service=executor_service,
        )

    return make


def _auth(
    validator_keypair, miner_hotkey: str, timestamp: int | None = None
) -> AuthenticateRequest:
    payload = AuthenticationPayload(
        validator_hotkey=validator_keypair.ss58_address,
        miner_hotkey=miner_hotkey,
        timestamp=int(time.time()) if timestamp is None else timestamp,
    )
    return AuthenticateRequest(
        payload=payload, signature=f"0x{validator_keypair.sign(payload.blob_for_signing()).hex()}"
    )


def _logs_json(miner_hotkey: str) -> str:
    return GetPodLogsRequest(
        miner_hotkey=miner_hotkey, executor_id=_EXECUTOR_ID, container_name="pod_x"
    ).json()


async def test_an_unknown_validator_key_is_refused_before_accept(make):
    c = make(key=bittensor.Keypair.create_from_uri("//LiumTestUnknownSocket").ss58_address)
    assert await c.connect() is False
    assert c.websocket.accepted is False
    assert c.websocket.close_codes == [_POLICY]


async def test_a_normal_sign_in_is_served_and_frees_its_pre_sign_in_slot(
    make, validator_keypair, miner_keypair
):
    c = make()
    assert await c.connect() is True and c.websocket.accepted
    assert consumer_module._pre_sign_in_by_source["198.51.100.7"] == 1
    c.websocket.inbox.put_nowait(_auth(validator_keypair, miner_keypair.ss58_address).json())
    task = asyncio.create_task(c.handle())
    for _ in range(100):
        if c.websocket.sent:
            break
        await asyncio.sleep(0.01)
    assert c.validator_authenticated is True
    reply = json.loads(c.websocket.sent[0])
    assert (
        reply["message_type"] == "AcceptJobRequest"
        and reply["executors"][0]["uuid"] == _EXECUTOR_ID
    )
    assert consumer_module._pre_sign_in_by_source["198.51.100.7"] == 0
    task.cancel()


async def test_a_connection_without_sign_in_is_closed_at_the_deadline(make, monkeypatch):
    monkeypatch.setattr(consumer_module, "SIGN_IN_DEADLINE_SECONDS", 0.05)
    c = make()
    assert await c.connect() is True
    await asyncio.wait_for(c.handle(), 2)
    assert _POLICY in c.websocket.close_codes
    assert sum(consumer_module._pre_sign_in_by_source.values()) == 0


async def test_a_full_pre_sign_in_queue_closes_the_connection(make, miner_keypair):
    c = make()
    assert await c.connect() is True
    for _ in range(consumer_module.MAX_MESSAGE_COUNT + 1):
        c.websocket.inbox.put_nowait(_logs_json(miner_keypair.ss58_address))
    await asyncio.wait_for(c.handle(), 2)
    assert _POLICY in c.websocket.close_codes
    assert len(c.msg_queue) <= consumer_module.MAX_MESSAGE_COUNT


async def test_an_oversized_pre_sign_in_frame_is_refused(make):
    c = make()
    assert await c.connect() is True
    c.websocket.inbox.put_nowait("x" * (consumer_module.MAX_PRE_SIGN_IN_MESSAGE_BYTES + 1))
    await asyncio.wait_for(c.handle(), 2)
    assert _POLICY in c.websocket.close_codes


def test_a_sign_in_timestamp_in_the_future_is_refused(make, validator_keypair, miner_keypair):
    c = make()
    future = int(time.time()) + consumer_module.AUTH_MESSAGE_MAX_AGE + 60
    msg = _auth(validator_keypair, miner_keypair.ss58_address, future)
    ok, reason = c.verify_auth_msg(msg)
    assert ok is False and "future" in reason


async def test_pre_sign_in_connections_are_capped_per_source(make, monkeypatch):
    monkeypatch.setattr(consumer_module, "MAX_PRE_SIGN_IN_CONNECTIONS_PER_SOURCE", 2)
    held = [make(host="198.51.100.7") for _ in range(2)]
    for c in held:
        assert await c.connect() is True
    extra = make(host="198.51.100.7")
    assert await extra.connect() is False
    assert extra.websocket.accepted is False and extra.websocket.close_codes == [_POLICY]
    # another source address is still served
    assert await make(host="203.0.113.9").connect() is True


async def test_pre_sign_in_connections_are_capped_in_total(make, monkeypatch):
    monkeypatch.setattr(consumer_module, "MAX_PRE_SIGN_IN_CONNECTIONS", 3)
    for i in range(3):
        assert await make(host=f"203.0.113.{i + 1}").connect() is True
    extra = make(host="203.0.113.50")
    assert await extra.connect() is False
    assert extra.websocket.accepted is False and extra.websocket.close_codes == [_POLICY]
