"""
Tests for MinerService._sign_validator_pubkey.

Focus: verify that the validator signs SSH public keys correctly and that
the resulting validator_signature is in the right format and is verifiable
by bittensor — matching the format expected by the executor's
_validate_validator_signature check.
"""

import logging
import time
from types import SimpleNamespace

import bittensor
import pytest
from datura.requests.validator_requests import upload_nonce_issued_at

from clients.subtensor_client import SubtensorClient
from core.config import settings
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.attestation_service import AttestationService
from services.miner_service import MIN_DATED_NONCE_MINER_AXON_VERSION, MinerService

pytest_plugins = ["fixtures.rest_miner_fixtures"]

_SSH_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey123456789abcdef user@host"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def validator_keypair():
    return bittensor.Keypair.create_from_uri("//LiumTestValidator")


@pytest.fixture(scope="module")
def miner_service():
    """MinerService instance with no dependencies — _sign_validator_pubkey
    only calls _normalize_public_key (a @staticmethod), so __init__ is skipped."""
    return MinerService.__new__(MinerService)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sign_validator_pubkey_str_produces_verifiable_signature(miner_service, validator_keypair):
    """Signing a str pubkey produces a 0x-prefixed signature verifiable by bittensor."""
    # Arrange
    public_key = _SSH_KEY

    # Act
    signature = miner_service._sign_validator_pubkey(validator_keypair, public_key)

    # Assert — hex-encoded with 0x prefix (format expected by executor's verification)
    assert isinstance(signature, str)
    assert signature.startswith("0x")

    # Assert — the signature is cryptographically valid over the original key
    assert validator_keypair.verify(public_key, signature)


def _subtensor(monkeypatch, *, axon_versions=None, central=()):
    neurons = [
        SimpleNamespace(hotkey=hotkey, axon_info=SimpleNamespace(version=version))
        for hotkey, version in (axon_versions or {}).items()
    ]
    fake = SimpleNamespace(miners=neurons, hotkeys_routed_via_central_miner=frozenset(central))
    monkeypatch.setattr(SubtensorClient, "get_instance", classmethod(lambda cls: fake))


def test_upload_nonce_dated_for_a_miner_on_bittensor_10_5(miner_service, monkeypatch, caplog):
    # I-78: a miner announcing bittensor 10.5.0+ (miner-v1.003+) relays the dated nonce
    _subtensor(
        monkeypatch, axon_versions={"new": MIN_DATED_NONCE_MINER_AXON_VERSION, "newer": 10_006_001}
    )
    with caplog.at_level(logging.INFO):
        for hotkey in ("new", "newer"):
            assert (
                abs(
                    upload_nonce_issued_at(miner_service._upload_nonce_for_miner(hotkey))
                    - time.time()
                )
                < 5
            )
    assert "dated_upload_nonce_skipped" not in caplog.text


def test_upload_nonce_dated_for_the_central_miner_whatever_its_axon(miner_service, monkeypatch):
    _subtensor(monkeypatch, axon_versions={"opted": 0}, central={"opted"})
    assert upload_nonce_issued_at(miner_service._upload_nonce_for_miner("opted")) is not None


@pytest.mark.parametrize("hotkey", ["old", "zero", "unknown"])
def test_upload_nonce_skipped_and_logged_for_an_unproven_miner(
    miner_service, monkeypatch, caplog, hotkey
):
    _subtensor(monkeypatch, axon_versions={"old": 9_010_001, "zero": 0})
    with caplog.at_level(logging.INFO):
        assert miner_service._upload_nonce_for_miner(hotkey) is None
    [skipped] = [r.msg.extra for r in caplog.records if str(r.msg) == "dated_upload_nonce_skipped"]
    assert skipped["miner_hotkey"] == hotkey
    assert skipped["axon_version"] == {"old": 9_010_001, "zero": 0, "unknown": None}[hotkey]


@pytest.fixture
def miner_task_result():
    return lambda executor_id: None


@pytest.mark.asyncio
@pytest.mark.parametrize("axon_version", [9_010_001, MIN_DATED_NONCE_MINER_AXON_VERSION])
async def test_the_nonce_create_task_gets_equals_the_nonce_sent(rest_miner_service, monkeypatch, axon_version):
    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", "https://verifier.example/verify")
    _subtensor(monkeypatch, axon_versions={"miner-a": axon_version})
    rest_miner_service.attestation_service.maybe_issue_nonce = AttestationService().maybe_issue_nonce
    rest_miner_service.miner_returns("exec-1")
    await rest_miner_service.request_job_to_miner(
        payload=MinerJobRequestPayload(
            job_batch_id="b",
            miner_hotkey="miner-a",
            miner_coldkey="c",
            miner_address="192.0.2.10",
            miner_port=8091,
        ),
        encrypted_files=MinerJobEnryptedFiles(
            encrypt_key="k",
            all_keys={},
            tmp_directory="/tmp",
            machine_scrape_file_name="s",
            machine_scrape_source="",
        ),
        rented_data=RentedExecutorsResponse(executors={}),
        default_docker_image_digests={},
    )
    [submitted] = [
        body for name, body in rest_miner_service.rest_calls if name == "ssh-pubkey-submit"
    ]
    [task_call] = rest_miner_service.task_service.create_task.await_args_list
    expected = task_call.kwargs["attestation_nonce"].value_hex
    assert submitted["nonce"] == (expected if axon_version >= MIN_DATED_NONCE_MINER_AXON_VERSION else None)


# --- DAH-6740: the key is also signed for each other hotkey of the listing owner, as the portal lists them -------


def _portal(node_id, hotkeys):
    from clients.validator_portal_api import PortalExecutor

    return PortalExecutor(id=node_id, validator_hotkey="v", node_hotkeys=hotkeys)


@pytest.fixture
def portal_owners(monkeypatch):
    from clients import validator_portal_api

    monkeypatch.setattr(validator_portal_api, "_owner_hotkeys", {})
    return validator_portal_api


def test_the_portals_owner_hotkeys_are_each_signed(miner_service, validator_keypair, portal_owners):
    from datura.requests.validator_requests import ssh_pubkey_signing_blob

    nonce = "ab" * 32
    assert miner_service._owner_hotkey_signatures(validator_keypair, _SSH_KEY, nonce, "listing") is None

    portal_owners.remember_owner_hotkeys({"listing": [_portal("n1", ["listing", "5Old"]), _portal("n2", ["5Old", "listing"])]})
    signatures = miner_service._owner_hotkey_signatures(validator_keypair, _SSH_KEY, nonce, "listing")

    assert list(signatures) == ["5Old"]
    assert validator_keypair.verify(ssh_pubkey_signing_blob(_SSH_KEY, nonce, "5Old"), signatures["5Old"])


@pytest.mark.parametrize(
    "nodes",
    [
        # nodes of two owners under one listing (the pool): no owner's hotkeys are lent to the other's nodes
        [_portal("n1", ["listing", "5OwnerA"]), _portal("n2", ["listing", "5OwnerB"])],
        [_portal("n1", ["listing", "5Old"]), _portal("n2", None)],
        [_portal("n1", ["listing"] + [f"5Key{i}" for i in range(17)])],
    ],
)
def test_a_listing_without_one_bounded_owner_set_signs_nothing(miner_service, validator_keypair, portal_owners, nodes):
    portal_owners.remember_owner_hotkeys({"listing": nodes})

    assert miner_service._owner_hotkey_signatures(validator_keypair, _SSH_KEY, None, "listing") is None


@pytest.mark.asyncio
async def test_a_hotkey_only_the_miner_reports_gets_no_signature(rest_miner_service, portal_owners, monkeypatch):
    """The owner set comes from the validator's own portal read: a miner naming another hotkey for its nodes gets
    nothing signed for it, in this cycle or the next."""
    from datura.requests.miner_requests import AcceptSSHKeyRequest

    from fixtures.rest_miner_fixtures import executor_info

    _subtensor(monkeypatch, axon_versions={"miner-a": MIN_DATED_NONCE_MINER_AXON_VERSION})
    portal_owners.remember_owner_hotkeys({"miner-a": [_portal("exec-1", ["miner-a", "5PortalOld"])]})
    reported = executor_info("exec-1").model_copy(update={"expected_node_hotkeys": ["miner-a", "5Victim"]})
    submitted = []

    async def _make_rest_request(method, url, json_data, headers, timeout, log_extra, operation_name):
        if url.endswith("ssh-pubkey-submit"):
            submitted.append(json_data)
            return 200, AcceptSSHKeyRequest(executors=[reported]).model_dump(mode="json")
        return 200, {"message_type": "SSHKeyRemoved"}

    rest_miner_service._make_rest_request = _make_rest_request
    for _ in range(2):
        await rest_miner_service.request_job_to_miner(
            payload=MinerJobRequestPayload(
                job_batch_id="b", miner_hotkey="miner-a", miner_coldkey="c", miner_address="192.0.2.10", miner_port=8091
            ),
            encrypted_files=MinerJobEnryptedFiles(
                encrypt_key="k", all_keys={}, tmp_directory="/tmp", machine_scrape_file_name="s",
                machine_scrape_source="",
            ),
            rented_data=RentedExecutorsResponse(executors={}),
            default_docker_image_digests={},
        )

    assert [list(body["owner_hotkey_signatures"]) for body in submitted] == [["5PortalOld"], ["5PortalOld"]]
