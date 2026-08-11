"""DAH-2580: this validator relays renter provisioning and never originates it.

Three claims, and each is a thing another component relies on.

**The bytes go out unchanged.** The platform's signature covers the method, the request target
and the body bytes, so anything altered in transit is a request the host refuses. That is only
observable from the outgoing side, which is why the session is injected here rather than the
response mocked.

**A host that was never reached is a different answer from a host that refused.** A launch that
timed out on this side may still be running there, so it must not come back as "the rental
failed" — the two are told apart by whether a status is present.

**A CVM node does not serve a container rental.** Creating one means opening an SSH session and
a docker API call against the node the customer has been told we never enter, while their CVM
is running on it. The two paths are made exclusive at the single point every container request
passes through.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from clients.compute_client import ComputeClient
from payload_models.payloads import (
    ContainerCreateRequest,
    ContainerDeleteRequest,
    CvmProvisioned,
    CvmProvisionRequest,
    CvmTeardownRequest,
    CvmTornDown,
    FailedContainerErrorCodes,
    FailedCvmRequest,
    SignedCvmdCall,
)
from services.cvmd_relay import CvmdRelay, CvmdRelayError, RelayResult
from services.miner_service import MinerService

# Every test here is async; the suite runs pytest-asyncio in strict mode.
pytestmark = pytest.mark.asyncio

EXECUTOR_ID = "00000000-0000-0000-0000-000000000000"

SIGNED_BODY = '{"kind":"renter","compose_hash":"' + "a" * 64 + '"}'
SIGNED_HEADERS = {
    "X-Cvmd-Hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    "X-Cvmd-Timestamp": "1754800000000000000",
    "X-Cvmd-Nonce": "ab" * 16,
    "X-Cvmd-Signature": "cd" * 32,
}


class FakeResponse:
    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self._text = text

    async def text(self) -> str:
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Records exactly what went out. The claim under test is that nothing was altered."""

    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict] = []

    def __call__(self, *, timeout=None):
        self.timeout = timeout
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def request(self, method, url, *, data, headers, ssl):
        self.calls.append(
            {"method": method, "url": url, "data": data, "headers": headers, "ssl": ssl}
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TestTheRelayForwardsVerbatim:
    async def test_the_body_bytes_and_headers_are_the_ones_that_were_signed(self):
        session = FakeSession(FakeResponse(201, '{"instance_id":"abc"}'))

        result = await CvmdRelay(session_factory=session).forward(
            base_url="https://10.0.0.1:8443",
            method="POST",
            path="/v1/cvm",
            body=SIGNED_BODY,
            headers=SIGNED_HEADERS,
            timeout_seconds=60,
        )

        sent = session.calls[0]
        assert sent["data"] == SIGNED_BODY.encode()
        assert sent["url"] == "https://10.0.0.1:8443/v1/cvm"
        assert sent["method"] == "POST"
        for header, value in SIGNED_HEADERS.items():
            assert sent["headers"][header] == value
        assert result.ok is True
        assert result.body == {"instance_id": "abc"}

    async def test_a_trailing_slash_on_the_base_url_does_not_change_the_signed_target(self):
        """The request target is inside the signature, so `//v1/cvm` would be refused."""
        session = FakeSession(FakeResponse(200, "{}"))

        await CvmdRelay(session_factory=session).forward(
            base_url="https://10.0.0.1:8443/",
            method="DELETE",
            path="/v1/cvm",
            body="",
            headers=SIGNED_HEADERS,
            timeout_seconds=60,
        )

        assert session.calls[0]["url"] == "https://10.0.0.1:8443/v1/cvm"

    async def test_certificate_verification_is_off_because_there_is_no_ca_to_check_against(self):
        """cvmd's certificate is self-generated. What authenticates this exchange is the
        signature going out and the measurements coming back, not the transport."""
        session = FakeSession(FakeResponse(200, "{}"))

        await CvmdRelay(session_factory=session).forward(
            base_url="https://10.0.0.1:8443",
            method="GET",
            path="/v1/state",
            body="",
            headers=SIGNED_HEADERS,
            timeout_seconds=5,
        )

        assert session.calls[0]["ssl"] is False


class TestWhatTheRelayReportsBack:
    async def test_a_refusal_carries_the_hosts_own_reason(self):
        session = FakeSession(FakeResponse(422, '{"detail":"os_image_hash is not approved"}'))

        result = await CvmdRelay(session_factory=session).forward(
            base_url="https://h:8443",
            method="POST",
            path="/v1/cvm",
            body=SIGNED_BODY,
            headers=SIGNED_HEADERS,
            timeout_seconds=60,
        )

        assert result.ok is False
        assert result.reason() == "os_image_hash is not approved"

    async def test_a_refusal_with_no_detail_still_says_what_happened(self):
        session = FakeSession(FakeResponse(500, "{}"))
        result = await CvmdRelay(session_factory=session).forward(
            base_url="https://h:8443",
            method="POST",
            path="/v1/cvm",
            body="",
            headers=SIGNED_HEADERS,
            timeout_seconds=60,
        )
        assert "500" in result.reason()

    async def test_an_unreachable_host_raises_rather_than_reporting_a_failed_launch(self):
        session = FakeSession(TimeoutError("timed out"))

        with pytest.raises(CvmdRelayError, match="could not be reached"):
            await CvmdRelay(session_factory=session).forward(
                base_url="https://h:8443",
                method="POST",
                path="/v1/cvm",
                body="",
                headers=SIGNED_HEADERS,
                timeout_seconds=1,
            )

    async def test_an_answer_that_is_not_json_is_reported_rather_than_guessed_at(self):
        session = FakeSession(FakeResponse(200, "<html>proxy error</html>"))

        with pytest.raises(CvmdRelayError, match="not JSON"):
            await CvmdRelay(session_factory=session).forward(
                base_url="https://h:8443",
                method="POST",
                path="/v1/cvm",
                body="",
                headers=SIGNED_HEADERS,
                timeout_seconds=60,
            )

    async def test_an_empty_body_on_a_success_is_an_empty_report_not_an_error(self):
        session = FakeSession(FakeResponse(204, ""))
        result = await CvmdRelay(session_factory=session).forward(
            base_url="https://h:8443",
            method="DELETE",
            path="/v1/cvm",
            body="",
            headers=SIGNED_HEADERS,
            timeout_seconds=60,
        )
        assert result.ok is True
        assert result.body == {}


def _client() -> ComputeClient:
    client = ComputeClient.__new__(ComputeClient)
    client.lock = asyncio.Lock()
    client.message_queue = []
    client.logging_extra = {"validator_hotkey": "validator-hotkey"}
    client.miner_service = MagicMock()
    return client


def _provision_request() -> CvmProvisionRequest:
    return CvmProvisionRequest(
        miner_hotkey="miner",
        executor_id=EXECUTOR_ID,
        pod_id="pod-1",
        cvmd_url="https://10.0.0.1:8443",
        call=SignedCvmdCall(
            method="POST", path="/v1/cvm", body=SIGNED_BODY, headers=SIGNED_HEADERS
        ),
        expectations={"compose_hash": "a" * 64},
    )


class TestTheDriver:
    async def test_a_successful_launch_passes_the_hosts_report_back_unchanged(self, monkeypatch):
        """Unchanged because the backend compares the measurements in it against what it
        derived for the order — a validator that summarized them would be standing between two
        parties who are checking each other."""
        report = {
            "instance_id": "abc",
            "kind": "renter",
            "measurements": {"compose_hash": "a" * 64},
        }
        monkeypatch.setattr(
            CvmdRelay, "forward", AsyncMock(return_value=RelayResult(status=201, body=report))
        )
        client = _client()

        await client.cvm_driver(_provision_request())

        (response,) = client.message_queue
        assert isinstance(response, CvmProvisioned)
        assert response.report == report
        assert response.pod_id == "pod-1"

    async def test_a_teardown_answers_with_its_own_message_type(self, monkeypatch):
        monkeypatch.setattr(
            CvmdRelay,
            "forward",
            AsyncMock(return_value=RelayResult(status=200, body={"torn_down": True})),
        )
        client = _client()

        await client.cvm_driver(
            CvmTeardownRequest(
                miner_hotkey="miner",
                executor_id=EXECUTOR_ID,
                pod_id="pod-1",
                cvmd_url="https://10.0.0.1:8443",
                call=SignedCvmdCall(method="DELETE", path="/v1/cvm", headers=SIGNED_HEADERS),
            )
        )

        (response,) = client.message_queue
        assert isinstance(response, CvmTornDown)
        assert response.report == {"torn_down": True}

    async def test_a_host_that_refused_reports_its_status_and_its_reason(self, monkeypatch):
        monkeypatch.setattr(
            CvmdRelay,
            "forward",
            AsyncMock(
                return_value=RelayResult(status=409, body={"detail": "a guest is already running"})
            ),
        )
        client = _client()

        await client.cvm_driver(_provision_request())

        (response,) = client.message_queue
        assert isinstance(response, FailedCvmRequest)
        assert response.status == 409
        assert "already running" in response.detail

    async def test_a_host_that_was_never_reached_carries_no_status(self, monkeypatch):
        """The distinction matters: a launch that timed out here may still be running there, so
        it must not read as a launch that failed."""
        monkeypatch.setattr(
            CvmdRelay, "forward", AsyncMock(side_effect=CvmdRelayError("connection refused"))
        )
        client = _client()

        await client.cvm_driver(_provision_request())

        (response,) = client.message_queue
        assert isinstance(response, FailedCvmRequest)
        assert response.status is None
        assert "could not be reached" in response.msg

    async def test_the_two_operations_get_different_timeouts(self, monkeypatch):
        """A verified teardown holds until the node's memory is back, which on a large guest is
        tens of minutes — far longer than a launch waits for a guest to boot."""
        forward = AsyncMock(return_value=RelayResult(status=200, body={}))
        monkeypatch.setattr(CvmdRelay, "forward", forward)
        client = _client()

        await client.cvm_driver(_provision_request())
        provision_timeout = forward.await_args.kwargs["timeout_seconds"]

        await client.cvm_driver(
            CvmTeardownRequest(
                miner_hotkey="miner",
                executor_id=EXECUTOR_ID,
                pod_id="pod-1",
                cvmd_url="https://10.0.0.1:8443",
                call=SignedCvmdCall(method="DELETE", path="/v1/cvm", headers=SIGNED_HEADERS),
            )
        )
        teardown_timeout = forward.await_args.kwargs["timeout_seconds"]

        assert teardown_timeout > provision_timeout


def _miner_service(is_cvm=False, redis_raises=False) -> MinerService:
    redis = Mock()
    if redis_raises:
        redis.is_elem_exists_in_set = AsyncMock(side_effect=RuntimeError("redis is down"))
    else:
        redis.is_elem_exists_in_set = AsyncMock(return_value=is_cvm)
    return MinerService(
        ssh_service=Mock(), task_service=Mock(), redis_service=redis, attestation_service=Mock()
    )


def _create_payload() -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="miner",
        executor_id=EXECUTOR_ID,
        pod_id="pod-1",
        user_public_keys=[],
        docker_image="alpine:3.20",
        docker_image_tag="3.20",
        gpu_uuids=[],
    )


class TestACvmNodeServesNoContainerRental:
    async def test_a_container_rental_on_an_attested_node_is_refused(self):
        response = await _miner_service(is_cvm=True).handle_container(_create_payload())

        assert response.error_code == FailedContainerErrorCodes.CvmNodeNotContainerRentable
        assert response.pod_id == "pod-1"

    async def test_the_refusal_names_no_host_details_in_the_customer_facing_headline(self):
        """`msg` reaches renter-visible events, so it must never carry executor host details."""
        response = await _miner_service(is_cvm=True).handle_container(_create_payload())

        assert "10." not in response.msg
        assert "miner" not in response.msg

    async def test_a_node_that_has_never_attested_is_left_alone(self, monkeypatch):
        service = _miner_service(is_cvm=False)
        monkeypatch.setattr(
            MinerService, "_handle_container", AsyncMock(return_value="ordinary path")
        )
        monkeypatch.setattr("services.miner_service.settings.USE_REST_API", True)

        assert await service.handle_container(_create_payload()) == "ordinary path"

    async def test_only_a_create_is_refused_a_delete_still_has_to_run(self, monkeypatch):
        """A CVM node can still have a stale container record from before it was converted, and
        refusing the delete would leave it there forever."""
        service = _miner_service(is_cvm=True)
        monkeypatch.setattr(
            MinerService, "_handle_container", AsyncMock(return_value="ordinary path")
        )
        monkeypatch.setattr("services.miner_service.settings.USE_REST_API", True)

        delete = ContainerDeleteRequest(
            miner_hotkey="miner",
            executor_id=EXECUTOR_ID,
            pod_id="pod-1",
            container_name="c",
            volume_name="v",
        )
        assert await service.handle_container(delete) == "ordinary path"

    async def test_redis_being_down_does_not_stop_every_ordinary_rental(self, monkeypatch):
        """Unknown is not "yes". A Redis outage must not refuse the whole fleet's rentals, and
        nothing about a CVM's confidentiality rests on this check alone — the customer verifies
        the CVM itself."""
        service = _miner_service(redis_raises=True)
        monkeypatch.setattr(
            MinerService, "_handle_container", AsyncMock(return_value="ordinary path")
        )
        monkeypatch.setattr("services.miner_service.settings.USE_REST_API", True)

        assert await service.handle_container(_create_payload()) == "ordinary path"
