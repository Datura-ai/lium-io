"""DAH-2675, the hardware-free half: the attest-agent probe is wired, and it is log-only.

The agent (FR-E6) is the one thing the platform may talk to inside a rental. This suite pins
the validator's side of that conversation as far as it can go without a TDX host:

  * the probe resolves the agent's guest port through the launch report's forward list and
    reads /health through it — the same transport stance as every cvmd call (self-signed
    TLS, unauthenticated read);
  * the flag defaults off, and off makes the probe a no-op;
  * a missing forward, an agent error, an unreachable agent, or a malformed answer all end
    in a LOG LINE — the probe has no other output, which is what "log-only" means here and
    why it cannot move a score no matter what it finds;
  * the nonce-bound quote (POST /v1/attest) is deliberately not requested — verifying a
    quote against a real TDX host is the hardware half of this task.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from bittensor_wallet import Keypair

from core.config import settings
from services.cvm_lifecycle import (
    CvmdHost,
    CvmLifecycleService,
    SwitchAssessment,
)
from services.cvmd_relay import CvmdRelayError, RelayResult

HOST = CvmdHost(
    executor_uuid="e-rented",
    address="203.0.113.7",
    miner_hotkey="5Miner",
    gpu_model="NVIDIA H200",
    gpu_count=8,
    updated_at=0.0,
)

HEALTH = {
    "version": "0.3.0",
    "tls_public_key": "ab" * 32,
    "gpu_uuids": ["GPU-1", "GPU-2"],
    "gpu_uuid_digest": "cd" * 32,
    "gpu_detail": "ok",
}


def assessment_with_forwards(ports):
    return SwitchAssessment(
        reachable=True,
        state="RENTER_RUNNING",
        has_cvm=True,
        cvm={"instance_id": "cvm-1", "ports": ports},
    )


FORWARDS = [
    {"protocol": "tcp", "address": "0.0.0.0", "host_port": 12200, "guest_port": 2200},
    {"protocol": "tcp", "address": "0.0.0.0", "host_port": 18451, "guest_port": 8451},
]


def make_service(relay_result=None, relay_error=None):
    service = CvmLifecycleService(MagicMock(), MagicMock(), Keypair.create_from_uri("//Alice"))
    service.agent_relay = MagicMock()
    if relay_error is not None:
        service.agent_relay.forward = AsyncMock(side_effect=relay_error)
    else:
        service.agent_relay.forward = AsyncMock(
            return_value=relay_result or RelayResult(status=200, body=dict(HEALTH))
        )
    return service


@pytest.fixture
def probe_on(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CVM_ATTEST_PROBE", True, raising=False)
    monkeypatch.setattr(settings, "CVM_ATTEST_AGENT_GUEST_PORT", 8451, raising=False)
    monkeypatch.setattr(settings, "CVM_ATTEST_PROBE_TIMEOUT_SECONDS", 15, raising=False)


class TestTheHappyPath:
    @pytest.mark.asyncio
    async def test_health_is_read_through_the_agents_forward(self, probe_on):
        service = make_service()

        await service.probe_attest_agent(HOST, assessment_with_forwards(FORWARDS))

        call = service.agent_relay.forward.await_args.kwargs
        assert call["base_url"] == "https://203.0.113.7:18451"
        assert call["method"] == "GET"
        assert call["path"] == "/health"
        assert call["timeout_seconds"] == 15

    @pytest.mark.asyncio
    async def test_what_the_agent_said_lands_in_the_log(self, probe_on, caplog):
        service = make_service()

        with caplog.at_level("INFO"):
            await service.probe_attest_agent(HOST, assessment_with_forwards(FORWARDS))

        assert "CVM_RENTER_ATTEST_PROBE_OK" in caplog.text


class TestLogOnly:
    @pytest.mark.asyncio
    async def test_flag_off_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_ATTEST_PROBE", False, raising=False)
        service = make_service()

        await service.probe_attest_agent(HOST, assessment_with_forwards(FORWARDS))

        service.agent_relay.forward.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_loopback_bound_forward_is_never_dialed(self, probe_on, caplog):
        """cvmd's port parser defaults the bind address to 127.0.0.1; dialing the public
        address for a loopback forward would time out every cycle and read, fleet-wide, as
        "the agent is never up" — the exact wrong rollout signal."""
        service = make_service()
        loopback = [
            {"protocol": "tcp", "address": "127.0.0.1", "host_port": 18451, "guest_port": 8451}
        ]

        with caplog.at_level("INFO"):
            await service.probe_attest_agent(HOST, assessment_with_forwards(loopback))

        service.agent_relay.forward.assert_not_awaited()
        assert "CVM_RENTER_ATTEST_PROBE_FAILED" in caplog.text
        assert "loopback" in caplog.text

    @pytest.mark.asyncio
    async def test_no_forward_for_the_agent_port_is_a_log_line(self, probe_on, caplog):
        service = make_service()
        only_ssh = [FORWARDS[0]]

        with caplog.at_level("INFO"):
            await service.probe_attest_agent(HOST, assessment_with_forwards(only_ssh))

        service.agent_relay.forward.assert_not_awaited()
        assert "CVM_RENTER_ATTEST_PROBE_FAILED" in caplog.text

    @pytest.mark.asyncio
    async def test_an_unreachable_agent_never_raises(self, probe_on, caplog):
        """The renter has root and can firewall the agent; a probe that raised would hand
        them a way to break the sweep that scores their provider."""
        service = make_service(relay_error=CvmdRelayError("connection refused"))

        with caplog.at_level("INFO"):
            await service.probe_attest_agent(HOST, assessment_with_forwards(FORWARDS))

        assert "CVM_RENTER_ATTEST_PROBE_FAILED" in caplog.text

    @pytest.mark.asyncio
    async def test_an_agent_error_answer_never_raises(self, probe_on, caplog):
        service = make_service(
            relay_result=RelayResult(status=503, body={"detail": "no quote provider"})
        )

        with caplog.at_level("INFO"):
            await service.probe_attest_agent(HOST, assessment_with_forwards(FORWARDS))

        assert "CVM_RENTER_ATTEST_PROBE_FAILED" in caplog.text

    @pytest.mark.asyncio
    async def test_a_malformed_state_body_never_raises(self, probe_on):
        service = make_service()
        mangled = SwitchAssessment(
            reachable=True,
            state="RENTER_RUNNING",
            has_cvm=True,
            cvm={"ports": [{"guest_port": "not-a-number"}, None, 42]},
        )

        await service.probe_attest_agent(HOST, mangled)

        service.agent_relay.forward.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_quote_endpoint_is_never_requested(self, probe_on):
        """POST /v1/attest is the hardware half — a quote nobody verifies is noise, and a
        mocked verifier here would be false confidence."""
        service = make_service()

        await service.probe_attest_agent(HOST, assessment_with_forwards(FORWARDS))

        for call in service.agent_relay.forward.await_args_list:
            assert call.kwargs["path"] == "/health"
            assert call.kwargs["method"] == "GET"
