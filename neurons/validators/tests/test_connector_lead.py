import asyncio
import os
import signal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import redis.asyncio as aioredis
import redis.exceptions
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

import clients.compute_client as compute_client
from clients.compute_client import ComputeClient


def _client(server: FakeServer) -> ComputeClient:
    client = ComputeClient.__new__(ComputeClient)
    client.logging_extra = {}
    client.miner_service = MagicMock()
    client.miner_service.redis_service.redis = FakeRedis(server=server)
    client.in_flight_drivers = set()
    client.message_queue = compute_client.OutgoingMessages()
    return client


@pytest.mark.asyncio
async def test_a_new_connector_waits_until_the_old_one_stops_leading(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(compute_client, "CONNECTOR_READY_FILE", str(tmp_path / "ready"))
    monkeypatch.setattr(compute_client, "CONNECTOR_LEAD_LOCK_TTL_SECONDS", 1)
    # shortcut: fakeredis runs no Lua without lupa (not a dependency); renew with a plain PEXPIRE here
    monkeypatch.setattr(
        aioredis.lock.Lock,
        "reacquire",
        lambda self: self.redis.pexpire(self.name, int(self.timeout * 1000)),
    )
    server = FakeServer()
    old, new = _client(server), _client(server)

    await old.take_connector_lead()
    taking = asyncio.create_task(new.take_connector_lead())
    await asyncio.sleep(1.5)  # past the TTL: the old one's renewal keeps the lead
    assert not taking.done()
    assert (tmp_path / "ready").exists()  # the new pod is ready while it waits

    old._lead_task.cancel()  # the old pod is gone; its lead expires
    await asyncio.wait_for(taking, 3)
    new._lead_task.cancel()


@pytest.mark.asyncio
async def test_a_redis_error_while_taking_the_lead_unreadies_the_pod_and_retries(
    tmp_path, monkeypatch
) -> None:
    ready = tmp_path / "ready"
    monkeypatch.setattr(compute_client, "CONNECTOR_READY_FILE", str(ready))
    ready_while_failed = []
    real_unready = compute_client.mark_connector_unready

    def unready() -> None:
        real_unready()
        ready_while_failed.append(ready.exists())

    monkeypatch.setattr(compute_client, "mark_connector_unready", unready)
    attempts = []

    async def acquire(self, *args, **kwargs):
        attempts.append(ready.exists())
        if len(attempts) == 1:
            raise redis.exceptions.ConnectionError("failover")
        return True

    monkeypatch.setattr(aioredis.lock.Lock, "acquire", acquire)
    client = _client(FakeServer())

    await asyncio.wait_for(client.take_connector_lead(), 5)
    client._lead_task.cancel()

    assert attempts == [True, True]  # ready only while actually waiting on the lock
    assert ready_while_failed == [False]  # and not while the error is retried
    assert ready.exists()


def test_an_unready_connector_has_no_ready_file(tmp_path, monkeypatch) -> None:
    ready = tmp_path / "ready"
    monkeypatch.setattr(compute_client, "CONNECTOR_READY_FILE", str(ready))
    ready.touch()
    compute_client.mark_connector_unready()
    assert not ready.exists()
    compute_client.mark_connector_unready()  # already gone: no error


def _no_lua_release(monkeypatch) -> None:
    # shortcut: fakeredis runs no Lua without lupa; release with a plain compare-and-delete here
    async def do_release(self, expected_token):
        if await self.redis.get(self.name) == expected_token:
            await self.redis.delete(self.name)

    monkeypatch.setattr(aioredis.lock.Lock, "do_release", do_release)


def _fake_exit(monkeypatch) -> tuple[asyncio.Event, list[int]]:
    exited = asyncio.Event()
    codes: list[int] = []

    def fake_exit(code):
        codes.append(code)
        exited.set()

    monkeypatch.setattr(compute_client.os, "_exit", fake_exit)
    return exited, codes


async def _sigterm(client: ComputeClient, exited: asyncio.Event, timeout: float) -> None:
    loop = asyncio.get_running_loop()
    client.exit_on_sigterm(loop)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(exited.wait(), timeout)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


@pytest.mark.asyncio
async def test_sigterm_releases_the_lead_and_exits_0(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(compute_client, "CONNECTOR_READY_FILE", str(tmp_path / "ready"))
    _no_lua_release(monkeypatch)
    exited, codes = _fake_exit(monkeypatch)
    server = FakeServer()
    client = _client(server)
    await client.take_connector_lead()

    await _sigterm(client, exited, 3)

    assert codes == [0]
    assert not await FakeRedis(server=server).exists(compute_client.CONNECTOR_LEAD_LOCK_KEY)  # the new pod leads now
    assert client._lead_task.cancelled() or client._lead_task.done()
    assert not (tmp_path / "ready").exists()


@pytest.mark.asyncio
async def test_sigterm_lets_in_flight_work_finish_after_releasing_the_lead(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(compute_client, "CONNECTOR_READY_FILE", str(tmp_path / "ready"))
    _no_lua_release(monkeypatch)
    exited, codes = _fake_exit(monkeypatch)
    server = FakeServer()
    client = _client(server)
    await client.take_connector_lead()
    lead_held_while_working = []
    release_work = asyncio.Event()

    async def create() -> None:
        await release_work.wait()
        lead_held_while_working.append(await FakeRedis(server=server).exists(compute_client.CONNECTOR_LEAD_LOCK_KEY))
        client.message_queue.append(MagicMock())  # its reply
        await asyncio.sleep(0.2)
        client.message_queue.pop()  # the send loop sent it

    client.track_in_flight_driver(asyncio.create_task(create()))
    asyncio.get_running_loop().call_later(0.5, release_work.set)

    await _sigterm(client, exited, 5)

    assert codes == [0]
    assert lead_held_while_working == [0]  # the lead went first, the create still finished
    assert not client.in_flight_drivers and not client.message_queue  # the create and its reply both done


@pytest.mark.asyncio
async def test_sigterm_exits_at_the_drain_bound_when_work_hangs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(compute_client, "CONNECTOR_READY_FILE", str(tmp_path / "ready"))
    monkeypatch.setattr(compute_client, "SIGTERM_DRAIN_SECONDS", 0.5)
    _no_lua_release(monkeypatch)
    exited, codes = _fake_exit(monkeypatch)
    client = _client(FakeServer())
    await client.take_connector_lead()
    hung = asyncio.create_task(asyncio.Event().wait())
    client.track_in_flight_driver(hung)

    await _sigterm(client, exited, 3)

    assert codes == [0]
    assert not hung.done()  # left for the exit, as the SIGKILL would
    hung.cancel()


def test_run_sh_execs_the_connector() -> None:
    """Without exec the shell is PID 1 and swallows the rollout's SIGTERM, so the kubelet SIGKILLs the pod."""
    run_sh = (Path(__file__).parents[1] / "run.sh").read_text()
    assert "exec pdm run src/connector.py" in run_sh
    assert "exec pdm run src/validator.py" in run_sh
