"""Tests for BackendClient."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import bittensor
import pytest
from neurons.validators.src.clients.backend_client import BackendClient
from pydantic import BaseModel
from tenacity import wait_none

from protocol.vc_protocol.compute_requests import (
    FillerRunActiveResponse,
    PodRentalActiveResponse,
    RentedExecutorsResponse,
)


class SampleResponse(BaseModel):
    data: str
    count: int


@pytest.fixture
def reset_session():
    BackendClient._session = None
    yield
    BackendClient._session = None


@pytest.fixture
def mock_keypair():
    keypair = MagicMock()
    keypair.ss58_address = "5FakeValidatorHotkey"
    keypair.sign = MagicMock(return_value=b"\x00" * 64)
    return keypair


@pytest.fixture
def client(mock_keypair):
    return BackendClient(base_url="https://api.example.com", keypair=mock_keypair)


def create_mock_response(status: int, json_data: dict | None = None):
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.text = AsyncMock(return_value="error")
    mock_resp.content_type = "application/json"
    return mock_resp


def create_mock_session(mock_response):
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()

    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(return_value=mock_response)
    async_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session.request = MagicMock(return_value=async_cm)

    return mock_session


@pytest.mark.asyncio
async def test_get_success(reset_session, client):
    response_data = {"data": "test", "count": 42}
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.get("/data", SampleResponse)

        assert result is not None
        assert result.data == "test"
        assert result.count == 42


@pytest.mark.asyncio
async def test_get_pod_rental_active_uses_internal_endpoint(client):
    expected = MagicMock()

    with patch.object(client, "_signed_request", AsyncMock(return_value=expected)) as mock_get:
        result = await client.get_pod_rental_active("pod-1")

    assert result is expected
    mock_get.assert_awaited_once_with(
        "GET",
        "/internal/pods/pod-1/rental-active",
        PodRentalActiveResponse,
        timeout=10,
    )


@pytest.mark.asyncio
async def test_get_filler_run_active_flags_missing_container(client):
    # container_missing=True must ride on the query string — the backend's zombie-run
    # reconciliation (DAH-2471) is fed by exactly this flag; a plain re-check sends none.
    expected = MagicMock()

    with patch.object(client, "_signed_request", AsyncMock(return_value=expected)) as mock_get:
        result = await client.get_filler_run_active("run-1", container_missing=True)

    assert result is expected
    mock_get.assert_awaited_once_with(
        "GET",
        "/internal/filler-runs/run-1/active?container_missing=true",
        FillerRunActiveResponse,
        timeout=10,
    )


@pytest.mark.asyncio
async def test_get_filler_run_active_without_flag_keeps_plain_path(client):
    with patch.object(client, "_signed_request", AsyncMock(return_value=None)) as mock_get:
        await client.get_filler_run_active("run-1")

    mock_get.assert_awaited_once_with(
        "GET",
        "/internal/filler-runs/run-1/active",
        FillerRunActiveResponse,
        timeout=10,
    )


@pytest.mark.asyncio
async def test_get_non_200_status(reset_session, client):
    mock_response = create_mock_response(500)
    mock_session = create_mock_session(mock_response)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.get("/data", SampleResponse)
        assert result is None


@pytest.mark.asyncio
async def test_get_validation_error(reset_session, client):
    invalid_data = {"wrong_field": "value"}
    mock_response = create_mock_response(200, invalid_data)
    mock_session = create_mock_session(mock_response)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.get("/data", SampleResponse)
        assert result is None


@pytest.mark.asyncio
async def test_get_timeout(reset_session, client):
    mock_session = AsyncMock()
    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(side_effect=TimeoutError())
    async_cm.__aexit__ = AsyncMock()
    mock_session.request = MagicMock(return_value=async_cm)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.get("/data", SampleResponse, timeout=1)
        assert result is None


@pytest.mark.asyncio
async def test_get_with_signature(reset_session, client):
    response_data = {"data": "test", "count": 42}
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        await client.get("/data", SampleResponse, add_signature=True)

        call_args = mock_session.request.call_args
        headers = call_args.kwargs.get("headers", {})
        assert "hotkey" in headers
        assert "timestamp" in headers
        assert "signature" in headers


@pytest.mark.asyncio
async def test_post_success(reset_session, client):
    request_data = {"action": "create"}
    response_data = {"data": "created", "count": 1}
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.post("/submit", SampleResponse, json_data=request_data)

        assert result is not None
        assert result.data == "created"


@pytest.mark.asyncio
async def test_check_executor_health_sends_loopback_plugin_only_when_given(client):
    with patch.object(client, "_signed_request", AsyncMock(return_value=None)) as post:
        await client.check_executor_health("1.2.3.4", 8001, "hotkey", 9001)
        await client.check_executor_health("1.2.3.4", 8001, "hotkey", 9001, loopback_plugin="vloopback:v2")

    assert "loopback_plugin" not in post.await_args_list[0].kwargs["json_data"]
    assert post.await_args_list[1].kwargs["json_data"]["loopback_plugin"] == "vloopback:v2"


@pytest.mark.asyncio
async def test_check_executor_health_bounds_the_whole_call_including_retries(reset_session, client):
    # the first attempt drops late, the retry would start a fresh full-length wait
    async def drop_late():
        await asyncio.sleep(0.15)
        raise aiohttp.ClientOSError(104, "Connection reset by peer")

    async def hang():
        await asyncio.sleep(10)

    attempts = []
    for enter in (drop_late, hang):
        async_cm = AsyncMock()
        async_cm.__aenter__ = AsyncMock(side_effect=enter)
        async_cm.__aexit__ = AsyncMock(return_value=None)
        attempts.append(async_cm)
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.request = MagicMock(side_effect=attempts)
    client.CONNECTION_RETRY_BACKOFF_SECONDS = (0.0, 0.0)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        started = asyncio.get_running_loop().time()
        result = await client.check_executor_health("1.2.3.4", 8001, "hotkey", 9001, timeout=0.2)

    assert result is None
    assert asyncio.get_running_loop().time() - started < 0.35


@pytest.mark.asyncio
async def test_url_construction(reset_session, client):
    response_data = {"data": "test", "count": 42}
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        await client.get("/some/path", SampleResponse)

        call_args = mock_session.request.call_args
        assert call_args[0] == ("GET", "https://api.example.com/some/path")


@pytest.mark.asyncio
async def test_session_has_no_total_timeout_cap(reset_session):
    """Regression for DAH-1939: session-level total timeout must not cap per-request timeouts."""
    session = await BackendClient.get_session()
    try:
        assert session.timeout.total is None, (
            "Session-level total timeout must be None so per-request timeouts apply as authored"
        )
    finally:
        await BackendClient.close_session()


@pytest.mark.asyncio
async def test_per_request_timeout_outlives_old_session_cap(reset_session, mock_keypair):
    """Regression for DAH-1939: a request slower than the old 30s cap must complete.

    Spins up a local aiohttp server that delays response by ~1.5s, then issues a POST
    with explicit timeout=5. Before the fix, the 30s session cap would have allowed
    this; the analogous production case (35s response, 300s per-request timeout) was
    being killed at 30s. We use compressed durations here to keep the test fast, and
    additionally assert the session timeout is None (see the test above).
    """
    from aiohttp import web

    async def slow_handler(_request: web.Request) -> web.Response:
        await asyncio.sleep(1.5)
        return web.json_response({"data": "ok", "count": 1})

    app = web.Application()
    app.router.add_post("/slow", slow_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        client = BackendClient(base_url=f"http://127.0.0.1:{port}", keypair=mock_keypair)
        result = await client.post("/slow", SampleResponse, timeout=5)
        assert result is not None
        assert result.data == "ok"
    finally:
        await BackendClient.close_session()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_per_request_timeout_still_enforced(reset_session, mock_keypair):
    """Per-request timeout is honoured (returns None on timeout)."""
    from aiohttp import web

    async def slow_handler(_request: web.Request) -> web.Response:
        await asyncio.sleep(2.0)
        return web.json_response({"data": "ok", "count": 1})

    app = web.Application()
    app.router.add_post("/slow", slow_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    try:
        client = BackendClient(base_url=f"http://127.0.0.1:{port}", keypair=mock_keypair)
        result = await client.post("/slow", SampleResponse, timeout=1)
        assert result is None
    finally:
        await BackendClient.close_session()
        await runner.cleanup()


def test_rented_executors_response_defaults_filler_mapping():
    result = RentedExecutorsResponse.model_validate({"executors": {}})

    assert result.executors == {}
    assert result.filler_containers_by_executor == {}


def test_rented_executors_response_keeps_customer_rentals_separate_from_filler():
    result = RentedExecutorsResponse.model_validate(
        {
            "executors": {
                "executor-123": {
                    "miner_hotkey": "miner-hotkey",
                    "executor_ip_address": "127.0.0.1",
                    "executor_ip_port": "22",
                    "pods": [
                        {
                            "pod_id": "pod-1",
                            "container_name": "pod_pod-1",
                            "rented_ports": [8080],
                        }
                    ],
                }
            },
            "filler_containers_by_executor": {"executor-123": "filler_active"},
        }
    )

    assert result.executors["executor-123"].pods[0].container_name == "pod_pod-1"
    assert result.filler_containers_by_executor == {"executor-123": "filler_active"}
    assert result.get_filler_container("executor-123") == "filler_active"
    assert result.get_filler_container("executor-456") is None


def test_rented_executors_response_filters_non_filler_container_names():
    result = RentedExecutorsResponse.model_validate(
        {
            "executors": {},
            "filler_containers_by_executor": {
                "executor-123": "filler_active",
                "executor-456": "pod_customer-rental",
                "executor-789": "container_arbitrary",
            },
        }
    )

    assert result.filler_containers_by_executor == {"executor-123": "filler_active"}


@pytest.mark.asyncio
async def test_get_all_rented_executors_parses_filler_mapping(reset_session, client):
    response_data = {
        "executors": {},
        "filler_containers_by_executor": {
            "executor-123": "filler_5703f4c9-c2f4-4fae-a652-3dee4753030a",
            "executor-456": "pod_customer-rental",
        },
    }
    mock_response = create_mock_response(200, response_data)
    mock_session = create_mock_session(mock_response)

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.get_all_rented_executors()

    assert result is not None
    assert result.executors == {}
    assert result.filler_containers_by_executor == {
        "executor-123": "filler_5703f4c9-c2f4-4fae-a652-3dee4753030a"
    }


def create_mock_session_with_attempts(side_effects):
    """Session whose request yields one side effect per attempt (exception or response)."""
    mock_session = AsyncMock()
    mock_session.closed = False

    cms = []
    for effect in side_effects:
        async_cm = AsyncMock()
        if isinstance(effect, BaseException):
            async_cm.__aenter__ = AsyncMock(side_effect=effect)
        else:
            async_cm.__aenter__ = AsyncMock(return_value=effect)
        async_cm.__aexit__ = AsyncMock(return_value=None)
        cms.append(async_cm)

    mock_session.request = MagicMock(side_effect=cms)

    return mock_session


@pytest.mark.asyncio
async def test_get_session_bounds_keepalive_timeout(reset_session):
    """DAH-2360: pooled connections must not idle past the keepalive window."""
    session = await BackendClient.get_session()
    try:
        assert session.connector._keepalive_timeout == BackendClient.KEEPALIVE_TIMEOUT_SECONDS
    finally:
        await BackendClient.close_session()


@pytest.mark.asyncio
async def test_post_retries_connection_error_then_succeeds(reset_session, client):
    """DAH-2360: broken pipe on a stale pooled connection is retried, not surfaced."""
    ok_response = create_mock_response(200, {"data": "ok", "count": 1})
    mock_session = create_mock_session_with_attempts(
        [aiohttp.ClientOSError(32, "Broken pipe"), ok_response]
    )

    with (
        patch.object(BackendClient, "get_session", return_value=mock_session),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        result = await client.post("/submit", SampleResponse)

    assert result is not None
    assert result.data == "ok"
    assert mock_session.request.call_count == 2


@pytest.mark.asyncio
async def test_get_retries_server_disconnected_then_succeeds(reset_session, client):
    ok_response = create_mock_response(200, {"data": "ok", "count": 1})
    mock_session = create_mock_session_with_attempts(
        [aiohttp.ServerDisconnectedError(), ok_response]
    )

    with (
        patch.object(BackendClient, "get_session", return_value=mock_session),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        result = await client.get("/data", SampleResponse)

    assert result is not None
    assert result.data == "ok"
    assert mock_session.request.call_count == 2


@pytest.mark.asyncio
async def test_post_returns_none_when_connection_retries_exhausted(reset_session, client):
    attempts = 1 + BackendClient.CONNECTION_RETRY_ATTEMPTS
    mock_session = create_mock_session_with_attempts(
        [aiohttp.ClientOSError(32, "Broken pipe") for _ in range(attempts)]
    )
    sleep_mock = AsyncMock()

    with (
        patch.object(BackendClient, "get_session", return_value=mock_session),
        patch("asyncio.sleep", new=sleep_mock),
    ):
        result = await client.post("/submit", SampleResponse)

    assert result is None
    assert mock_session.request.call_count == attempts
    assert [c.args[0] for c in sleep_mock.await_args_list] == list(
        BackendClient.CONNECTION_RETRY_BACKOFF_SECONDS
    )


@pytest.mark.asyncio
async def test_post_timeout_is_not_retried(reset_session, client):
    """Timeouts may have reached the backend — retrying could double-run long operations."""
    mock_session = create_mock_session_with_attempts([TimeoutError()])

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.post("/submit", SampleResponse, timeout=1)

    assert result is None
    assert mock_session.request.call_count == 1


@pytest.mark.asyncio
async def test_post_generic_client_error_is_not_retried(reset_session, client):
    mock_session = create_mock_session_with_attempts([aiohttp.ClientError("boom")])

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        result = await client.post("/submit", SampleResponse)

    assert result is None
    assert mock_session.request.call_count == 1


def test_get_filler_containers_prefers_list_field_and_falls_back_to_legacy():
    # DAH-2465 additive protocol: the list field wins when present; an older backend that only sends
    # the single legacy map still yields its one filler (so deploy order never breaks protection).
    new_backend = RentedExecutorsResponse(
        executors={},
        filler_containers_by_executor={"exec-legacy-too": "filler_ignored"},
        all_filler_containers_by_executor={"exec-split": ["filler_a", "filler_b"]},
    )
    assert new_backend.get_filler_containers("exec-split") == ["filler_a", "filler_b"]

    old_backend = RentedExecutorsResponse(
        executors={},
        filler_containers_by_executor={"exec-single": "filler_only"},
    )
    assert old_backend.get_filler_containers("exec-single") == ["filler_only"]
    assert old_backend.get_filler_container("exec-single") == "filler_only"
    assert old_backend.get_filler_containers("exec-unknown") == []


@pytest.mark.asyncio
async def test_get_all_rented_executors_returns_none_after_three_empty_answers(client):
    """The retried read gives None after three empty answers."""
    rented_read_mock = AsyncMock(return_value=None)
    with patch.object(BackendClient, "get_rented_executors_now", rented_read_mock):
        result = await BackendClient.get_all_rented_executors.retry_with(wait=wait_none())(client)

    assert result is None
    assert rented_read_mock.await_count == 3


SIGNED_CALLS = [
    ("GET", "/internal/executors/rented", lambda c: c.get_rented_executors_now()),
    ("GET", "/internal/executors/rented?light=true", lambda c: c.get_rented_pod_targets()),
    ("GET", "/internal/pods/pod-1/rental-active", lambda c: c.get_pod_rental_active("pod-1")),
    (
        "POST",
        "/internal/pods/pod-1/host-reboot-recovered",
        lambda c: c.report_pod_host_reboot_recovered("pod-1", "2026-11-11T00:00:00Z"),
    ),
    (
        "POST",
        "/internal/pods/pod-1/gpu-drop",
        lambda c: c.report_rented_gpu_drop(
            "pod-1",
            state="fault",
            executor_id="e",
            first_seen_at="t0",
            consecutive_cycles=1,
            expected_gpu_count=8,
            visible_gpu_count=7,
            missing_uuids=["GPU-1"],
            nvml_error_code=None,
            faults=[],
            pod_gpu_count=8,
            rented_gpu_count=8,
            nvml_gpu_count=7,
        ),
    ),
    (
        "GET",
        "/internal/filler-runs/run-1/active?container_missing=true",
        lambda c: c.get_filler_run_active("run-1", container_missing=True),
    ),
    (
        "POST",
        "/validator/{hk}/executor-health-check",
        lambda c: c.check_executor_health("1.2.3.4", 8001, "hotkey", 9001),
    ),
    (
        "POST",
        "/validator/{hk}/verification-started",
        lambda c: c.report_verification_started(
            job_batch_id="b",
            miner_hotkey="5Miner",
            executor_uuids=["3f2504e0-4f89-11d3-9a0c-0305e82c3301"],
            started_at=datetime(2026, 11, 11, tzinfo=UTC),
        ),
    ),
    ("POST", "/v1/nvml-driver/report-unknown", lambda c: c.report_unknown_driver("580.1")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path", "call"), SIGNED_CALLS, ids=[p for _, p, _ in SIGNED_CALLS])
async def test_backend_calls_sign_the_whole_request(reset_session, method, path, call):
    # DAH-6169: the v1 signature covers method, path with query, body hash and timestamp, and the bytes sent are
    # the bytes signed, so the backend's whole-request check accepts every one of these calls
    keypair = bittensor.Keypair.create_from_uri("//DAH6169")
    client = BackendClient(base_url="https://api.example.com/api", keypair=keypair)
    mock_session = create_mock_session(create_mock_response(500))

    with patch.object(BackendClient, "get_session", return_value=mock_session):
        await call(client)

    kwargs = mock_session.request.call_args.kwargs
    path = path.format(hk=keypair.ss58_address)
    body = kwargs["data"] or b""
    assert mock_session.request.call_args.args == (method, f"https://api.example.com/api{path}")
    assert kwargs["json"] is None
    assert (method == "POST") == bool(body)
    headers = kwargs["headers"]
    message = BackendClient.signed_request_message(method, path, body, headers["timestamp"])
    assert headers["hotkey"] == keypair.ss58_address
    assert keypair.verify(message, bytes.fromhex(headers["signature"].removeprefix("0x")))
