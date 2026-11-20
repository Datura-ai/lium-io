import asyncio
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
