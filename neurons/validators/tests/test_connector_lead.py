import asyncio
from unittest.mock import MagicMock

import pytest
import redis.asyncio as aioredis
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
        aioredis.lock.Lock, "reacquire", lambda self: self.redis.pexpire(self.name, int(self.timeout * 1000))
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
