import asyncio
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.apis as apis
from routes.apis import apis_router


@pytest.fixture()
def container_logs_client(monkeypatch):
    async def verify_signature(*_args, **_kwargs):
        return None

    monkeypatch.setattr(apis, "verify_container_logs_signature", verify_signature)

    app = FastAPI()
    app.include_router(apis_router)
    return TestClient(app)


def test_live_container_logs_returns_429_when_stream_limit_is_reached(
    container_logs_client,
    monkeypatch,
):
    monkeypatch.setattr(apis, "MAX_FOLLOW_LOG_STREAMS", 1)
    monkeypatch.setattr(apis, "_active_follow_log_streams", 1)

    response = container_logs_client.get(
        "/containers/test-container/logs",
        params={"follow": "true"},
        headers={"X-Signature": "test", "X-Timestamp": "1"},
    )

    assert response.status_code == 429
    assert response.json() == {"detail": "Too many active live log streams"}


def test_non_follow_container_logs_preserves_all_tail_default(container_logs_client, monkeypatch):
    class FakeContainer:
        def __init__(self):
            self.logs = MagicMock(return_value=iter([b"log line\n"]))

    fake_container = FakeContainer()
    fake_client = MagicMock()
    fake_client.containers.get.return_value = fake_container

    monkeypatch.setattr(apis.docker, "from_env", lambda: fake_client)
    monkeypatch.setattr(apis, "_active_follow_log_streams", 0)

    response = container_logs_client.get(
        "/containers/test-container/logs",
        headers={"X-Signature": "test", "X-Timestamp": "1"},
    )

    assert response.status_code == 200
    assert response.content == b"log line\n"
    assert apis._active_follow_log_streams == 0
    fake_container.logs.assert_called_once_with(
        stream=True,
        follow=False,
        tail="all",
        since=None,
        stdout=True,
        stderr=True,
    )


def test_live_container_logs_duration_limit_releases_stream_slot(monkeypatch):
    async def run_test():
        async def verify_signature(*_args, **_kwargs):
            return None

        class FakeContainer:
            def logs(self, **_kwargs):
                while True:
                    time.sleep(0.05)
                    yield b"log line\n"

        fake_client = MagicMock()
        fake_client.containers.get.return_value = FakeContainer()

        monkeypatch.setattr(apis, "verify_container_logs_signature", verify_signature)
        monkeypatch.setattr(apis.docker, "from_env", lambda: fake_client)
        monkeypatch.setattr(apis, "MAX_FOLLOW_LOG_STREAMS", 1)
        monkeypatch.setattr(apis, "FOLLOW_LOG_STREAM_MAX_SECONDS", 0.001)
        monkeypatch.setattr(apis, "_active_follow_log_streams", 0)

        response = await apis.stream_container_logs(
            "test-container",
            x_signature="test",
            x_timestamp=1,
            follow=True,
            tail=100,
        )

        async for _chunk in response.body_iterator:
            pass

        for _ in range(20):
            if apis._active_follow_log_streams == 0:
                break
            await asyncio.sleep(0.01)

        assert apis._active_follow_log_streams == 0
        fake_client.containers.get.assert_called_once_with("test-container")

    asyncio.run(run_test())
