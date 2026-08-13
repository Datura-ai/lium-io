"""DAH-2675: a rented CVM's live quote decides its score.

The claims pinned here:

  * a quote relayed through cvmd, bound to THIS cycle's fresh nonce and accepted by the
    verifier against the agent recipe (sha256(TLS key ‖ gpu_uuid_digest) ‖ nonce) AND the
    order's expectations, scores the node with NO feature flag — a verified quote is full
    evidence, not a rollout experiment;
  * a quote the verifier examined and refused — bad identity binding, wrong nonce, or a
    measurement that contradicts what was sold — earns nothing, even with the DAH-2674
    fallback flag on: evidence of a false claim outranks the host-side signals;
  * only when NO quote could be examined (agent unreachable, verifier down, attestation
    disabled) does the node fall back to the flag-gated host-side pass — what a missing
    quote MEANS is DAH-2676's question, so absence must never be judged here;
  * the rented sweep is driven by the BACKEND's response, not by validator-side state: a
    rental with no registry entry is still found, and the backend's GPU facts outrank the
    registry's (the validator-stays-stateless rule);
  * the verification math itself: identity binding, nonce echo, expectations comparison,
    and the three-way verdict encoding.
"""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from cvm_helpers import PAYLOAD, cvmd_host, make_miner_service, rented_data_for

from core.config import settings
from services.attestation_service import (
    RENTAL_QUOTE_REJECTED,
    RENTAL_QUOTE_UNAVAILABLE,
    RENTAL_QUOTE_VERIFIED,
    AttestationError,
    AttestationNonce,
    AttestationService,
)
from services.cvm_lifecycle import CvmLifecycleService, SwitchAssessment
from services.cvmd_client import ATTEST_PATH, CvmdClient
from services.cvmd_relay import CvmdRelayError, RelayResult

TLS_KEY = bytes.fromhex("aa" * 32)
GPU_UUIDS = ["GPU-bbb", "GPU-aaa"]
COMPOSE_HASH = "c" * 64
OS_IMAGE_HASH = "d" * 64


def expectations_for(**overrides):
    base = {
        "qemu": "10.1.0",
        "os_image_hash": OS_IMAGE_HASH,
        "compose_hash": COMPOSE_HASH,
        "gpu_model": "NVIDIA H200",
        "gpu_count": 2,
        "agent_port": 8451,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def identity_half(tls_key: bytes = TLS_KEY, uuids: list[str] | None = None) -> bytes:
    gpu_digest = hashlib.sha256(",".join(sorted(uuids or GPU_UUIDS)).encode()).digest()
    return hashlib.sha256(tls_key + gpu_digest).digest()


def agent_answer(**overrides) -> dict:
    answer = {
        "version": "0.1.0",
        "tls_public_key": TLS_KEY.hex(),
        "gpu_uuids": list(GPU_UUIDS),
        "quote": '{"quote": "raw-bytes"}',
    }
    answer.update(overrides)
    return answer


def verifier_payload(
    nonce: AttestationNonce, *, identity: bytes | None = None, **details_overrides
) -> dict:
    allowed_tcb = next(iter(settings.get_allowed_tcb_statuses()))
    details = {
        "quote_verified": True,
        "report_data": "0x" + ((identity or identity_half()) + nonce.value_bytes).hex(),
        "tcb_status": allowed_tcb,
        "advisory_ids": [],
        "event_log_verified": True,
        "os_image_hash_verified": True,
        "app_info": {"compose_hash": COMPOSE_HASH, "os_image_hash": OS_IMAGE_HASH},
    }
    details.update(details_overrides)
    return {"is_valid": True, "details": details}


@pytest.fixture
def attestation(monkeypatch) -> AttestationService:
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True, raising=False)
    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", "http://verifier.test", raising=False)
    return AttestationService(redis_service=None)


async def judge(attestation, answer, payload, *, nonce=None, expectations=None):
    nonce = nonce or AttestationNonce.issue()
    if isinstance(payload, Exception):
        attestation._call_verifier = AsyncMock(side_effect=payload)
    else:
        attestation._call_verifier = AsyncMock(return_value=payload)
    return await attestation.verify_rented_cvm_quote(
        answer,
        nonce=nonce,
        expectations=expectations if expectations is not None else expectations_for(),
        address="203.0.113.7",
        port=8443,
    )


class TestTheVerificationMath:
    @pytest.mark.asyncio
    async def test_a_bound_fresh_matching_quote_verifies(self, attestation):
        nonce = AttestationNonce.issue()
        verdict, reason = await judge(
            attestation, agent_answer(), verifier_payload(nonce), nonce=nonce
        )
        assert verdict == RENTAL_QUOTE_VERIFIED, reason

    @pytest.mark.asyncio
    async def test_an_identity_that_binds_a_different_key_is_rejected(self, attestation):
        """The recipe makes the proof and the channel the same thing: a quote relayed from
        some OTHER genuine CVM carries that CVM's TLS key in its identity half, and the
        answer's self-stated key cannot repair the mismatch."""
        nonce = AttestationNonce.issue()
        other_identity = identity_half(tls_key=bytes.fromhex("bb" * 32))
        verdict, reason = await judge(
            attestation,
            agent_answer(),
            verifier_payload(nonce, identity=other_identity),
            nonce=nonce,
        )
        assert verdict == RENTAL_QUOTE_REJECTED
        assert "identity" in reason

    @pytest.mark.asyncio
    async def test_a_gpu_set_the_hardware_did_not_sign_is_rejected(self, attestation):
        """An agent stating GPUs it does not hold produces an identity half this validator
        will not reproduce — FR-G6 held by arithmetic, not by trust."""
        nonce = AttestationNonce.issue()
        verdict, reason = await judge(
            attestation,
            agent_answer(gpu_uuids=["GPU-claimed-1", "GPU-claimed-2"]),
            verifier_payload(nonce),  # hardware signed the REAL set
            nonce=nonce,
        )
        assert verdict == RENTAL_QUOTE_REJECTED
        assert "identity" in reason

    @pytest.mark.asyncio
    async def test_a_quote_for_someone_elses_nonce_is_rejected(self, attestation):
        nonce = AttestationNonce.issue()
        stale = AttestationNonce.issue()
        verdict, reason = await judge(
            attestation, agent_answer(), verifier_payload(stale), nonce=nonce
        )
        assert verdict == RENTAL_QUOTE_REJECTED
        assert "nonce" in reason

    @pytest.mark.asyncio
    async def test_a_quote_the_verifier_refused_is_rejected(self, attestation):
        nonce = AttestationNonce.issue()
        payload = verifier_payload(nonce)
        payload["is_valid"] = False
        verdict, _ = await judge(attestation, agent_answer(), payload, nonce=nonce)
        assert verdict == RENTAL_QUOTE_REJECTED

    @pytest.mark.asyncio
    async def test_an_answer_without_a_quote_is_rejected_not_unavailable(self, attestation):
        """The agent ANSWERED — with nothing checkable. Treating that as unavailable would
        hand the fallback path to any guest that strips the quote field."""
        verdict, _ = await judge(
            attestation, agent_answer(quote=""), verifier_payload(AttestationNonce.issue())
        )
        assert verdict == RENTAL_QUOTE_REJECTED


class TestTheOrderChecks:
    @pytest.mark.asyncio
    async def test_a_measured_compose_that_disagrees_with_the_order_is_rejected(self, attestation):
        nonce = AttestationNonce.issue()
        payload = verifier_payload(
            nonce, app_info={"compose_hash": "e" * 64, "os_image_hash": OS_IMAGE_HASH}
        )
        verdict, reason = await judge(attestation, agent_answer(), payload, nonce=nonce)
        assert verdict == RENTAL_QUOTE_REJECTED
        assert "compose_hash" in reason

    @pytest.mark.asyncio
    async def test_fewer_gpus_than_sold_is_rejected(self, attestation):
        """One GPU attested, two paid for. The identity half matches the one-GPU set the
        agent really holds — this is the check that catches a short delivery."""
        nonce = AttestationNonce.issue()
        one_gpu = ["GPU-aaa"]
        verdict, reason = await judge(
            attestation,
            agent_answer(gpu_uuids=one_gpu),
            verifier_payload(nonce, identity=identity_half(uuids=one_gpu)),
            nonce=nonce,
        )
        assert verdict == RENTAL_QUOTE_REJECTED
        assert "gpu_count" in reason

    @pytest.mark.asyncio
    async def test_no_expectations_still_verifies_the_binding(self, attestation):
        """An older backend sends no side-map entry; the hardware checks still apply, only
        the order comparison has nothing to compare."""
        nonce = AttestationNonce.issue()
        verdict, _ = await judge(
            attestation, agent_answer(), verifier_payload(nonce), nonce=nonce, expectations=0
        )
        assert verdict == RENTAL_QUOTE_VERIFIED


class TestUnavailability:
    @pytest.mark.asyncio
    async def test_attestation_disabled_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", False, raising=False)
        service = AttestationService(redis_service=None)
        verdict, _ = await service.verify_rented_cvm_quote(
            agent_answer(),
            nonce=AttestationNonce.issue(),
            expectations=expectations_for(),
            address="203.0.113.7",
            port=8443,
        )
        assert verdict == RENTAL_QUOTE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_a_verifier_that_cannot_be_reached_is_unavailable(self, attestation):
        verdict, _ = await judge(attestation, agent_answer(), AttestationError("verifier 502"))
        assert verdict == RENTAL_QUOTE_UNAVAILABLE


class TestTheScoringDecision:
    def renter_running(self):
        return SwitchAssessment(
            reachable=True,
            state="RENTER_RUNNING",
            cvm={"instance_id": "cvm-1", "supervisor_alive": True, "ports": []},
            supervisor_alive=True,
        )

    @pytest.fixture
    def lifecycle_on(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", False, raising=False)

    @pytest.mark.asyncio
    async def test_a_verified_quote_scores_with_no_flag(self, lifecycle_on):
        """The flag stays OFF here. A verified quote is full evidence by itself — this is
        the sentence in the task that says 'if the quote is passed, we can give it full
        score; treat it as valid'."""
        host = cvmd_host("e-rented")
        service, lifecycle = make_miner_service([host], self.renter_running())
        lifecycle.request_renter_quote = AsyncMock(return_value=agent_answer())
        service.attestation_service.verify_rented_cvm_quote = AsyncMock(
            return_value=(RENTAL_QUOTE_VERIFIED, "bound to this cycle's nonce")
        )

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for(expectations=expectations_for())
        )

        assert len(results) == 1
        passed = results[0]
        assert passed.score == 1.0
        assert passed.spec is None
        assert passed.is_rented is True
        assert passed.tdx_attestation_passed is True
        assert "CVM_RENTAL_QUOTE_VERIFIED" in passed.log_text

    @pytest.mark.asyncio
    async def test_a_rejected_quote_earns_nothing_even_with_the_fallback_flag_on(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", True, raising=False)
        host = cvmd_host("e-rented")
        service, lifecycle = make_miner_service([host], self.renter_running())
        lifecycle.request_renter_quote = AsyncMock(return_value=agent_answer())
        service.attestation_service.verify_rented_cvm_quote = AsyncMock(
            return_value=(RENTAL_QUOTE_REJECTED, "identity mismatch")
        )

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for(expectations=expectations_for())
        )

        assert results == [], "evidence of a false claim outranks the host-side fallback"

    @pytest.mark.asyncio
    async def test_an_unexaminable_quote_falls_back_to_the_flagged_pass(self, monkeypatch):
        """request_renter_quote -> None is the make_miner_service default: with the flag on,
        the node keeps the DAH-2674 host-side pass; what absence MEANS stays with DAH-2676."""
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", True, raising=False)
        host = cvmd_host("e-rented")
        service, _ = make_miner_service([host], self.renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD, existing=[], rented_data=rented_data_for()
        )

        assert len(results) == 1
        assert "CVM_RENTAL_FORCED_PASS" in results[0].log_text

    @pytest.mark.asyncio
    async def test_the_agent_port_rides_the_backend_expectations(self, lifecycle_on):
        host = cvmd_host("e-rented")
        service, lifecycle = make_miner_service([host], self.renter_running())
        lifecycle.request_renter_quote = AsyncMock(return_value=agent_answer())
        service.attestation_service.verify_rented_cvm_quote = AsyncMock(
            return_value=(RENTAL_QUOTE_VERIFIED, "ok")
        )

        await service._record_and_grace_cvm_hosts(
            PAYLOAD,
            existing=[],
            rented_data=rented_data_for(expectations=expectations_for(agent_port=9451)),
        )

        assert lifecycle.request_renter_quote.await_args.kwargs["agent_port"] == 9451


class TestTheBackendDrivesTheRentedSweep:
    def renter_running(self):
        return SwitchAssessment(
            reachable=True,
            state="RENTER_RUNNING",
            cvm={"instance_id": "cvm-1", "supervisor_alive": True, "ports": []},
            supervisor_alive=True,
        )

    @pytest.mark.asyncio
    async def test_a_rental_with_no_registry_entry_is_still_swept(self, monkeypatch):
        """The stateless claim: the registry answers nothing here — an empty registry plus
        the backend's response still finds, assesses and scores the rented node."""
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", False, raising=False)
        service, lifecycle = make_miner_service([], self.renter_running())
        lifecycle.request_renter_quote = AsyncMock(return_value=agent_answer())
        service.attestation_service.verify_rented_cvm_quote = AsyncMock(
            return_value=(RENTAL_QUOTE_VERIFIED, "ok")
        )

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD,
            existing=[],
            rented_data=rented_data_for(expectations=expectations_for(gpu_count=2)),
        )

        assert len(results) == 1
        assert results[0].gpu_model == "NVIDIA H200"
        assert results[0].gpu_count == 2
        # And the registry was refilled from the backend's facts for the post-rental
        # launch path — written, never read.
        lifecycle.record_host.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_backend_gpu_facts_outrank_the_registry_entry(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        monkeypatch.setattr(settings, "ENABLE_CVM_RENTAL_SCORING", False, raising=False)
        stale_registry_host = cvmd_host("e-rented", gpu_model="NVIDIA H200", gpu_count=8)
        service, lifecycle = make_miner_service([stale_registry_host], self.renter_running())
        lifecycle.request_renter_quote = AsyncMock(return_value=agent_answer())
        service.attestation_service.verify_rented_cvm_quote = AsyncMock(
            return_value=(RENTAL_QUOTE_VERIFIED, "ok")
        )

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD,
            existing=[],
            rented_data=rented_data_for(expectations=expectations_for(gpu_count=4)),
        )

        assert results[0].gpu_count == 4, "the order the customer paid for is the fact"

    @pytest.mark.asyncio
    async def test_a_backend_rental_for_another_miner_is_not_synthesized(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        service, lifecycle = make_miner_service([], self.renter_running())

        results = await service._record_and_grace_cvm_hosts(
            PAYLOAD,
            existing=[],
            rented_data=rented_data_for(
                miner_hotkey="5SomeoneElse", expectations=expectations_for()
            ),
        )

        assert results == []
        lifecycle.assess.assert_not_awaited()


class TestTheTransport:
    @pytest.mark.asyncio
    async def test_attest_renter_signs_a_post_to_the_attest_path(self):
        from bittensor_wallet import Keypair

        relay = MagicMock()
        relay.forward = AsyncMock(return_value=RelayResult(status=200, body={}))
        client = CvmdClient(Keypair.create_from_uri("//Alice"), relay=relay)

        nonce_hex = "ab" * 32
        await client.attest_renter("https://203.0.113.7:8443", nonce_hex=nonce_hex, agent_port=9451)

        sent = relay.forward.await_args.kwargs
        assert sent["method"] == "POST"
        assert sent["path"] == ATTEST_PATH
        assert sent["body"] == f'{{"nonce":"{nonce_hex}","agent_port":9451}}'
        for header in ("X-Cvmd-Hotkey", "X-Cvmd-Timestamp", "X-Cvmd-Nonce", "X-Cvmd-Signature"):
            assert sent["headers"][header]

    @pytest.mark.asyncio
    async def test_request_renter_quote_returns_the_answer_body(self):
        from bittensor_wallet import Keypair

        service = CvmLifecycleService(MagicMock(), MagicMock(), Keypair.create_from_uri("//Alice"))
        service.client.attest_renter = AsyncMock(
            return_value=RelayResult(status=200, body=agent_answer())
        )
        answer = await service.request_renter_quote(
            cvmd_host("e-rented"), nonce_hex="ab" * 32, agent_port=8451
        )
        assert answer == agent_answer()

    @pytest.mark.asyncio
    async def test_no_answer_and_error_answers_are_both_none(self):
        from bittensor_wallet import Keypair

        service = CvmLifecycleService(MagicMock(), MagicMock(), Keypair.create_from_uri("//Alice"))
        service.client.attest_renter = AsyncMock(side_effect=CvmdRelayError("down"))
        assert (
            await service.request_renter_quote(cvmd_host("e-rented"), nonce_hex="ab" * 32) is None
        )

        service.client.attest_renter = AsyncMock(
            return_value=RelayResult(status=502, body={"detail": "agent unreachable"})
        )
        assert (
            await service.request_renter_quote(cvmd_host("e-rented"), nonce_hex="ab" * 32) is None
        )
