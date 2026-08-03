from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from services import storage_operations

OPERATION_ID = UUID("11111111-1111-4111-8111-111111111111")


def _spec() -> dict[str, object]:
    return {
        "reporter": {
            "api_url": "https://api.example",
            "auth_token": "token",
            "resource": "backup",
        }
    }


@pytest.mark.asyncio
async def test_launch_failure_is_reported_before_the_error_is_reraised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ssh_client = AsyncMock()
    ssh_client.run.side_effect = RuntimeError("ssh failed")
    report_failure = AsyncMock()
    monkeypatch.setattr(storage_operations, "_report_launch_failure", report_failure)

    with pytest.raises(RuntimeError, match="ssh failed"):
        await storage_operations.start_storage_operation(
            ssh_client,
            "/usr/bin/python3",
            OPERATION_ID,
            _spec(),
        )

    report_failure.assert_awaited_once()


@pytest.mark.asyncio
async def test_launch_failure_uses_the_operation_progress_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def read(self) -> bytes:
            return b"{}"

    class FakeSession:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        def put(self, url: str, **kwargs: object) -> FakeResponse:
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse()

    monkeypatch.setattr(storage_operations.aiohttp, "ClientSession", FakeSession)

    await storage_operations._report_launch_failure(
        OPERATION_ID,
        _spec(),
        RuntimeError("ssh failed"),
    )

    assert captured["url"] == f"https://api.example/backup-logs/{OPERATION_ID}/progress"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["status"] == "FAILED"
    assert payload["stage"] == "LAUNCH"
