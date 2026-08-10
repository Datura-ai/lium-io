from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from services import storage_operations

OPERATION_ID = UUID("11111111-1111-4111-8111-111111111111")


def _spec() -> dict[str, object]:
    return {
        "engine": "restic",
        "reporter": {
            "api_url": "https://api.example",
            "auth_token": "token",
            "resource": "backup",
        },
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
            retain_terminal_artifacts=False,
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


@pytest.mark.asyncio
async def test_missing_runner_capability_is_reported_as_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ssh_client = AsyncMock()
    ssh_client.run.return_value = SimpleNamespace(exit_status=1, stdout="", stderr="")
    report_failure = AsyncMock()
    monkeypatch.setattr(storage_operations, "_report_launch_failure", report_failure)

    with pytest.raises(storage_operations.StorageOperationLaunchError, match="does not support"):
        await storage_operations.start_storage_operation(
            ssh_client,
            "/usr/bin/python3",
            OPERATION_ID,
            _spec(),
            retain_terminal_artifacts=False,
        )

    report_failure.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_allows_long_running_operation_and_accepts_terminal_status() -> None:
    ssh_client = AsyncMock()
    ssh_client.run.side_effect = [
        SimpleNamespace(exit_status=0, stdout="RUNNING\n", stderr=""),
        SimpleNamespace(exit_status=0, stdout="STATUS:0\n", stderr=""),
        SimpleNamespace(exit_status=0, stdout="", stderr=""),
    ]

    await storage_operations.wait_for_storage_operation(
        ssh_client,
        storage_operations.StorageOperationFiles.for_operation(OPERATION_ID),
        poll_interval_seconds=0,
    )


@pytest.mark.asyncio
async def test_wait_fails_when_seen_runner_disappears_without_status() -> None:
    ssh_client = AsyncMock()
    ssh_client.run.side_effect = [
        SimpleNamespace(exit_status=0, stdout="RUNNING\n", stderr=""),
        SimpleNamespace(exit_status=0, stdout="EXITED\n", stderr=""),
        SimpleNamespace(exit_status=0, stdout="runner log", stderr=""),
        SimpleNamespace(exit_status=0, stdout="", stderr=""),
    ]

    with pytest.raises(storage_operations.StorageOperationLaunchError, match="disappeared"):
        await storage_operations.wait_for_storage_operation(
            ssh_client,
            storage_operations.StorageOperationFiles.for_operation(OPERATION_ID),
            poll_interval_seconds=0,
        )


@pytest.mark.asyncio
async def test_cancel_waits_for_runner_then_reports_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ssh_client = AsyncMock()
    ssh_client.run.side_effect = [
        SimpleNamespace(exit_status=0, stdout='{"reporter":{}}', stderr=""),
        SimpleNamespace(exit_status=0, stdout="", stderr=""),
        SimpleNamespace(exit_status=0, stdout="", stderr=""),
    ]
    report_cancelled = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_operations, "_report_cancelled", report_cancelled)

    await storage_operations.cancel_storage_operation(ssh_client, OPERATION_ID)

    report_cancelled.assert_awaited_once_with(OPERATION_ID, {"reporter": {}})
    cancel_command = ssh_client.run.await_args_list[1].args[0]
    assert f"{OPERATION_ID}.cancel" in cancel_command
    assert "kill -0" in cancel_command
