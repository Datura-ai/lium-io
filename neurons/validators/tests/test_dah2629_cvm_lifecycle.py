"""DAH-2629: this validator brings validation CVMs up itself, via cvmd.

Four claims carry the task, and each gets its own class below:

  * the validator's signer is byte-identical to the platform's — one wire protocol, two
    keys with two scopes. The blob formula is restated by hand exactly as the backend's
    own suite restates it, so a drift in either mirror breaks a test rather than a host;
  * the launch triple comes from the signed catalog and nowhere else. No catalog, no
    validation entry, no qemu fields — each one refuses the launch instead of guessing;
  * a launch happens only on an idle, empty host, at most once per cooldown, and its 201
    is believed only when the reported measurements equal the pinned triple;
  * the whole thing sits behind ENABLE_CVM_LIFECYCLE, default off.
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from bittensor_wallet import Keypair

from core.config import settings
from services.cvm_lifecycle import CvmdHost, CvmLifecycleService, SwitchAssessment
from services.cvm_whitelist import DOMAIN_SEPARATOR as CATALOG_DOMAIN_SEPARATOR
from services.cvm_whitelist import parse_manifest
from services.cvmd_client import (
    HOTKEY_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    CvmdClient,
    sign_request,
    signing_blob,
)
from services.cvmd_relay import RelayResult

OS_IMAGE = "a" * 64
VALIDATION_COMPOSE = "b" * 64
RENTER_COMPOSE = "c" * 64


@pytest.fixture
def signer() -> Keypair:
    return Keypair.create_from_uri("//Alice")


class TestTheSigningMirror:
    def test_the_blob_is_the_documented_formula(self):
        """Restated as bytes, not by calling the function twice: this is the wire contract
        with cvmd, shared with the backend's signer."""

        def lp(value: bytes) -> bytes:
            return len(value).to_bytes(4, "big") + value

        expected = hashlib.sha256(
            b"cvmd-v1\x00" + lp(b"POST") + lp(b"/v1/cvm") + lp(b"{}") + lp(b"7") + lp(b"beef")
        ).digest()
        assert (
            signing_blob(method="POST", request_target="/v1/cvm", body=b"{}", timestamp="7", nonce="beef")
            == expected
        )

    def test_the_signature_verifies_over_exactly_what_is_sent(self, signer):
        signed = sign_request(signer, method="POST", path="/v1/cvm", body='{"kind":"validation"}')
        blob = signing_blob(
            method="POST",
            request_target="/v1/cvm",
            body=signed.body.encode(),
            timestamp=signed.headers[TIMESTAMP_HEADER],
            nonce=signed.headers[NONCE_HEADER],
        )
        assert Keypair(ss58_address=signer.ss58_address).verify(
            blob, bytes.fromhex(signed.headers[SIGNATURE_HEADER])
        )
        assert signed.headers[HOTKEY_HEADER] == signer.ss58_address

    @pytest.mark.asyncio
    async def test_launch_validation_sends_the_canonical_validation_body(self, signer):
        """The body names the kind and the pinned triple, serialized with pinned separators
        — the same bytes the signature covers."""
        relay = MagicMock()
        relay.forward = AsyncMock(return_value=RelayResult(status=201, body={}))
        client = CvmdClient(signer, relay=relay)

        await client.launch_validation(
            "https://127.0.0.1:8443",
            qemu="10.1.0",
            os_image_hash=OS_IMAGE,
            compose_hash=VALIDATION_COMPOSE,
        )

        kwargs = relay.forward.call_args.kwargs
        assert kwargs["method"] == "POST"
        assert kwargs["path"] == "/v1/cvm"
        assert kwargs["body"] == (
            '{"kind":"validation","qemu":"10.1.0",'
            f'"os_image_hash":"{OS_IMAGE}","compose_hash":"{VALIDATION_COMPOSE}"}}'
        )
        assert set(kwargs["headers"]) == {
            HOTKEY_HEADER,
            TIMESTAMP_HEADER,
            NONCE_HEADER,
            SIGNATURE_HEADER,
        }

    @pytest.mark.asyncio
    async def test_a_state_read_is_a_signed_get_with_no_body(self, signer):
        relay = MagicMock()
        relay.forward = AsyncMock(return_value=RelayResult(status=200, body={"state": "RECONCILING"}))
        client = CvmdClient(signer, relay=relay)

        await client.state("https://127.0.0.1:8443")

        kwargs = relay.forward.call_args.kwargs
        assert kwargs["method"] == "GET"
        assert kwargs["path"] == "/v1/state"
        assert kwargs["body"] == ""
        assert SIGNATURE_HEADER in kwargs["headers"]


def manifest_payload(artifacts) -> str:
    now = datetime.now(UTC)
    return json.dumps(
        {
            "version": 1,
            "serial": 1,
            "issued_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=86400)).isoformat(),
            "floors": {"os_image": 1, "qemu": 1, "compose": 1},
            "artifacts": artifacts,
        }
    )


def signed_envelope(text: str, keypair: Keypair) -> bytes:
    return json.dumps(
        {
            "schema": "lium-cvm-catalog/1",
            "payload": text,
            "signer": keypair.ss58_address,
            "signature": "0x" + keypair.sign(CATALOG_DOMAIN_SEPARATOR + text.encode()).hex(),
        }
    ).encode()


def entry(kind: str, *, qemu="10.1.0", compose_hash=VALIDATION_COMPOSE, versions=None, id_="e"):
    return {
        "id": id_,
        "kind": kind,
        "qemu": qemu,
        "os_image_hash": OS_IMAGE,
        "compose_hash": compose_hash,
        "versions": versions if versions is not None else {"os_image": 1, "qemu": 1, "compose": 1},
    }


class TestTheLaunchTripleComesFromTheCatalog:
    def test_the_newest_validation_entry_wins(self, signer):
        artifacts = [
            entry("validation", id_="old", versions={"os_image": 1, "qemu": 1, "compose": 1}),
            entry("validation", id_="new", compose_hash="d" * 64, versions={"os_image": 2, "qemu": 1, "compose": 2}),
            entry("renter", id_="renter", compose_hash=RENTER_COMPOSE, versions={"os_image": 3, "qemu": 3, "compose": 3}),
        ]
        catalog = parse_manifest(
            signed_envelope(manifest_payload(artifacts), signer), signer=signer.ss58_address
        )

        chosen = catalog.newest_validation_entry()
        assert chosen is not None
        assert chosen.id == "new"
        assert chosen.compose_hash == "d" * 64

    def test_a_renter_only_catalog_offers_nothing_to_launch(self, signer):
        catalog = parse_manifest(
            signed_envelope(manifest_payload([entry("renter", compose_hash=RENTER_COMPOSE)]), signer),
            signer=signer.ss58_address,
        )
        assert catalog.newest_validation_entry() is None

    def test_a_manifest_without_launch_fields_still_checks_but_cannot_launch(self, signer):
        """Older manifests carry only the hashes. Checking measurements keeps working;
        launching answers None instead of guessing a qemu."""
        bare = [{"id": "e", "kind": "validation", "os_image_hash": OS_IMAGE, "compose_hash": VALIDATION_COMPOSE}]
        catalog = parse_manifest(
            signed_envelope(manifest_payload(bare), signer), signer=signer.ss58_address
        )
        assert catalog.approves(os_image_hash=OS_IMAGE, compose_hash=VALIDATION_COMPOSE, kind="validation")
        assert catalog.newest_validation_entry() is None


HOST = CvmdHost(
    executor_uuid="e-1",
    address="203.0.113.7",
    miner_hotkey="5Miner",
    gpu_model="NVIDIA H200",
    gpu_count=8,
    updated_at=0.0,
)

IDLE = SwitchAssessment(reachable=True, state="RECONCILING", has_cvm=False)


def make_lifecycle(catalog_entry, *, cooldown_active=False):
    redis = MagicMock()
    redis.get = AsyncMock(return_value="1" if cooldown_active else None)
    redis.set_with_expiration = AsyncMock()
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    whitelist = MagicMock()
    catalog = MagicMock()
    catalog.newest_validation_entry = MagicMock(return_value=catalog_entry)
    whitelist.current = MagicMock(return_value=catalog if catalog_entry is not None else None)
    service = CvmLifecycleService(redis, whitelist, Keypair.create_from_uri("//Alice"))
    service.client = MagicMock()
    return service


def catalog_entry(**overrides):
    from services.cvm_whitelist import CvmCatalogEntry

    defaults = {
        "id": "validation-v1+img+qemu",
        "kind": "validation",
        "qemu": "10.1.0",
        "os_image_hash": OS_IMAGE,
        "compose_hash": VALIDATION_COMPOSE,
        "versions": {"os_image": 1, "qemu": 1, "compose": 1},
    }
    defaults.update(overrides)
    return CvmCatalogEntry(**defaults)


def launch_report(**overrides):
    report = {
        "instance_id": "abc123",
        "measurements": {
            "qemu": "10.1.0",
            "os_image_hash": OS_IMAGE,
            "compose_hash": VALIDATION_COMPOSE,
        },
    }
    report.update(overrides)
    return report


class TestEnsureValidationCvm:
    @pytest.mark.asyncio
    async def test_flag_off_never_launches(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", False, raising=False)
        service = make_lifecycle(catalog_entry())
        service.client.launch_validation = AsyncMock()

        assert await service.ensure_validation_cvm(HOST, assessment=IDLE) is False
        service.client.launch_validation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_idle_empty_host_gets_the_pinned_triple(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        service = make_lifecycle(catalog_entry())
        service.client.launch_validation = AsyncMock(
            return_value=RelayResult(status=201, body=launch_report())
        )

        assert await service.ensure_validation_cvm(HOST, assessment=IDLE) is True

        kwargs = service.client.launch_validation.call_args.kwargs
        assert kwargs == {
            "qemu": "10.1.0",
            "os_image_hash": OS_IMAGE,
            "compose_hash": VALIDATION_COMPOSE,
        }

    @pytest.mark.asyncio
    async def test_no_catalog_refuses_rather_than_guessing(self, monkeypatch):
        """A validator launching from anything but the signed catalog would create exactly
        the unattributable CVM the design forbids."""
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        service = make_lifecycle(None)
        service.client.launch_validation = AsyncMock()

        assert await service.ensure_validation_cvm(HOST, assessment=IDLE) is False
        service.client.launch_validation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_host_that_is_not_idle_and_empty_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        service = make_lifecycle(catalog_entry())
        service.client.launch_validation = AsyncMock()

        for assessment in (
            SwitchAssessment(reachable=True, state="RENTER_RUNNING", has_cvm=True),
            SwitchAssessment(reachable=True, state="SWITCHING", has_cvm=True, switching=True),
            SwitchAssessment(reachable=True, state="FAILED", has_cvm=False),
            SwitchAssessment(reachable=False),
        ):
            assert await service.ensure_validation_cvm(HOST, assessment=assessment) is False
        service.client.launch_validation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_201_measuring_something_else_is_not_a_success(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        service = make_lifecycle(catalog_entry())
        bad = launch_report()
        bad["measurements"]["compose_hash"] = "9" * 64
        service.client.launch_validation = AsyncMock(return_value=RelayResult(status=201, body=bad))

        assert await service.ensure_validation_cvm(HOST, assessment=IDLE) is False

    @pytest.mark.asyncio
    async def test_the_cooldown_stops_a_second_attempt(self, monkeypatch):
        """A launch takes minutes; a failing one should be read, not hammered."""
        monkeypatch.setattr(settings, "ENABLE_CVM_LIFECYCLE", True, raising=False)
        service = make_lifecycle(catalog_entry(), cooldown_active=True)
        service.client.launch_validation = AsyncMock()

        assert await service.ensure_validation_cvm(HOST, assessment=IDLE) is False
        service.client.launch_validation.assert_not_awaited()


class TestCvmdUrl:
    def test_built_from_the_node_address_and_the_fleet_port(self, monkeypatch):
        monkeypatch.setattr(settings, "CVMD_URL_OVERRIDE", "", raising=False)
        monkeypatch.setattr(settings, "CVMD_PORT", 8443, raising=False)
        assert HOST.cvmd_url() == "https://203.0.113.7:8443"

    def test_the_harness_override_wins(self, monkeypatch):
        monkeypatch.setattr(settings, "CVMD_URL_OVERRIDE", "https://127.0.0.1:18443", raising=False)
        assert HOST.cvmd_url() == "https://127.0.0.1:18443"
