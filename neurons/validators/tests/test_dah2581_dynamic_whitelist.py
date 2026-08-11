"""DAH-2581: what a CVM may measure as comes from the platform, and both recipes are accepted.

Four things are checked here, and each of them is a way the fleet could be broken by getting
this wrong.

**The catalog is verified, not merely fetched.** It decides what every CVM in the fleet is
allowed to be, so an unsigned or wrongly-signed document must never become the whitelist. The
signature is taken with a real keypair rather than a mock: a signature checked through a mock
proves nothing about whether a real backend's manifest would verify.

**A backend that is unreachable is not a fleet-wide rejection.** The manifest in force stays in
force. A validator that failed every CVM because the backend blinked would be a far worse
outage than one accepting a slightly stale catalog — and the catalog's own expiry bounds how
stale it can get.

**Both report_data recipes are accepted during the transition.** A recipe change is a lockstep
change: the node produces one value and the validator compares against it, so a validator that
accepted only the new one would fail every node in the fleet the moment it shipped.

**Every new check is default-off.** Each of them can fail a node the fleet is currently paying,
so the tests state both behaviours — what is logged when the flag is off, and what is refused
when it is on.
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from bittensor_wallet import Keypair
from datura.requests.miner_requests import ExecutorSSHInfo
from protocol.vc_protocol.compute_requests import CvmExpectations
from services.attestation_service import AttestationError, AttestationService
from services.cvm_whitelist import (
    DOMAIN_SEPARATOR,
    CvmCatalogError,
    CvmWhitelistSource,
    parse_manifest,
)

from core.config import settings

pytestmark = pytest.mark.asyncio

OS_IMAGE = "a" * 64
VALIDATION_COMPOSE = "b" * 64
RENTER_COMPOSE = "c" * 64


@pytest.fixture
def signer() -> Keypair:
    return Keypair.create_from_uri("//Alice")


@pytest.fixture
def stranger() -> Keypair:
    return Keypair.create_from_uri("//Charlie")


def payload(*, serial: int = 1, ttl_seconds: int = 86400, artifacts=None) -> str:
    now = datetime.now(UTC)
    return json.dumps(
        {
            "version": 1,
            "serial": serial,
            "issued_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
            "floors": {"os_image": 1, "qemu": 1, "compose": 1},
            "artifacts": artifacts
            if artifacts is not None
            else [
                {
                    "id": "validation-v1+img+qemu",
                    "kind": "validation",
                    "qemu": "10.1.0",
                    "os_image_hash": OS_IMAGE,
                    "compose_hash": VALIDATION_COMPOSE,
                },
                {
                    "id": "renter-v1+img+qemu",
                    "kind": "renter",
                    "qemu": "10.1.0",
                    "os_image_hash": OS_IMAGE,
                    "compose_hash": RENTER_COMPOSE,
                },
            ],
        }
    )


def envelope(text: str, keypair: Keypair, *, schema: str = "lium-cvm-catalog/1") -> bytes:
    """Written out by hand rather than built by the module under test, so a change to the wire
    shape breaks these tests instead of moving with them."""
    signature = keypair.sign(DOMAIN_SEPARATOR + text.encode()).hex()
    return json.dumps(
        {
            "schema": schema,
            "payload": text,
            "signer": keypair.ss58_address,
            "signature": f"0x{signature}",
        }
    ).encode()


class TestVerifyingAManifest:
    def test_a_manifest_the_platform_signed_is_accepted(self, signer):
        catalog = parse_manifest(envelope(payload(), signer), signer=signer.ss58_address)

        assert catalog.serial == 1
        assert catalog.approves(os_image_hash=OS_IMAGE, compose_hash=VALIDATION_COMPOSE)

    def test_kind_separates_a_renter_compose_from_a_validation_one(self, signer):
        """Otherwise a renter compose would satisfy a validation check, and the two stacks are
        approved for different things."""
        catalog = parse_manifest(envelope(payload(), signer), signer=signer.ss58_address)

        assert catalog.approves(
            os_image_hash=OS_IMAGE, compose_hash=RENTER_COMPOSE, kind="renter"
        )
        assert not catalog.approves(
            os_image_hash=OS_IMAGE, compose_hash=RENTER_COMPOSE, kind="validation"
        )

    def test_a_manifest_signed_by_anyone_else_is_refused(self, signer, stranger):
        with pytest.raises(CvmCatalogError, match="not signed by the configured signer"):
            parse_manifest(envelope(payload(), stranger), signer=signer.ss58_address)

    def test_the_signer_named_inside_the_document_is_not_what_is_checked(self, signer, stranger):
        """A document that names its own signer proves only that whoever wrote it also chose
        who to blame."""
        forged = json.loads(envelope(payload(), stranger))
        forged["signer"] = signer.ss58_address

        with pytest.raises(CvmCatalogError, match="not signed"):
            parse_manifest(json.dumps(forged).encode(), signer=signer.ss58_address)

    def test_an_edited_payload_no_longer_verifies(self, signer):
        tampered = json.loads(envelope(payload(), signer))
        tampered["payload"] = tampered["payload"].replace(OS_IMAGE, "9" * 64)

        with pytest.raises(CvmCatalogError, match="not signed"):
            parse_manifest(json.dumps(tampered).encode(), signer=signer.ss58_address)

    def test_an_expired_manifest_is_refused(self, signer):
        """Without an expiry a revocation could be defeated by never delivering the next one."""
        with pytest.raises(CvmCatalogError, match="expired"):
            parse_manifest(
                envelope(payload(ttl_seconds=-1), signer), signer=signer.ss58_address
            )

    def test_an_unknown_payload_version_is_refused_rather_than_guessed_at(self, signer):
        document = json.loads(payload())
        document["version"] = 2

        with pytest.raises(CvmCatalogError, match="version 2 is not the supported 1"):
            parse_manifest(
                envelope(json.dumps(document), signer), signer=signer.ss58_address
            )

    def test_a_wrong_schema_is_refused(self, signer):
        with pytest.raises(CvmCatalogError, match="schema"):
            parse_manifest(
                envelope(payload(), signer, schema="something/else"), signer=signer.ss58_address
            )

    @pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"schema":"lium-cvm-catalog/1"}'])
    def test_a_malformed_manifest_is_refused(self, signer, raw):
        with pytest.raises(CvmCatalogError):
            parse_manifest(raw, signer=signer.ss58_address)


class FakeSession:
    def __init__(self, body: bytes | Exception, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.gets = 0

    def __call__(self, *, timeout=None):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url):
        self.gets += 1
        if isinstance(self.body, Exception):
            raise self.body
        return _FakeGet(self.status, self.body)


class _FakeGet:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def enabled(monkeypatch, signer):
    monkeypatch.setattr(settings, "ENABLE_DYNAMIC_CVM_WHITELIST", True)
    monkeypatch.setattr(settings, "CVM_CATALOG_MANIFEST_URL", "https://backend/cvm/manifest")
    monkeypatch.setattr(settings, "CVM_CATALOG_SIGNER_SS58", signer.ss58_address)
    monkeypatch.setattr(settings, "CVM_CATALOG_REFRESH_SECONDS", 300)


class TestHoldingTheCatalog:
    async def test_it_holds_what_verified(self, enabled, signer):
        session = FakeSession(envelope(payload(serial=7), signer))
        source = CvmWhitelistSource(session_factory=session)

        await source.refresh()

        assert source.current().serial == 7

    async def test_the_feature_being_off_never_fetches(self, monkeypatch, signer):
        monkeypatch.setattr(settings, "ENABLE_DYNAMIC_CVM_WHITELIST", False)
        session = FakeSession(envelope(payload(), signer))
        source = CvmWhitelistSource(session_factory=session)

        await source.refresh()

        assert session.gets == 0
        assert source.current() is None

    async def test_an_unreachable_backend_leaves_the_manifest_in_force(self, enabled, signer):
        """The whole reason the static list stays as a fallback: a briefly unreachable backend
        must not become a fleet-wide attestation failure."""
        source = CvmWhitelistSource(session_factory=FakeSession(envelope(payload(serial=3), signer)))
        await source.refresh()

        source._session_factory = FakeSession(ConnectionError("backend is down"))
        await source.refresh(force=True)

        assert source.current().serial == 3

    async def test_a_manifest_that_fails_verification_never_becomes_the_whitelist(
        self, enabled, signer, stranger
    ):
        source = CvmWhitelistSource(session_factory=FakeSession(envelope(payload(serial=3), signer)))
        await source.refresh()

        source._session_factory = FakeSession(envelope(payload(serial=99), stranger))
        await source.refresh(force=True)

        assert source.current().serial == 3

    async def test_an_older_serial_is_refused(self, enabled, signer):
        """The within-process rollback ratchet: a backend that starts replaying an old manifest
        cannot re-approve something this validator has already seen retired."""
        source = CvmWhitelistSource(session_factory=FakeSession(envelope(payload(serial=9), signer)))
        await source.refresh()

        source._session_factory = FakeSession(envelope(payload(serial=4), signer))
        await source.refresh(force=True)

        assert source.current().serial == 9

    async def test_it_does_not_refetch_before_the_interval(self, enabled, signer):
        session = FakeSession(envelope(payload(), signer))
        source = CvmWhitelistSource(session_factory=session)

        await source.refresh()
        await source.refresh()
        await source.refresh()

        assert session.gets == 1

    async def test_an_expired_manifest_stops_being_in_force(self, enabled, signer, monkeypatch):
        source = CvmWhitelistSource(session_factory=FakeSession(envelope(payload(), signer)))
        await source.refresh()
        assert source.current() is not None

        # Reach past the manifest's own expiry rather than re-parsing an expired one: the point
        # is that `current()` re-checks, not only that `parse_manifest` refuses at fetch time.
        expired = source._catalog.__class__(
            serial=source._catalog.serial,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
            os_image_hashes=source._catalog.os_image_hashes,
            compose_hashes=source._catalog.compose_hashes,
        )
        source._catalog = expired

        assert source.current() is None


def _executor(**overrides) -> ExecutorSSHInfo:
    defaults = dict(
        uuid="executor-1",
        address="10.0.0.1",
        port=8000,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python",
        root_dir="/opt/executor",
        ssh_host_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey user@host",
        tdx_quote='{"quote": "00"}',
    )
    defaults.update(overrides)
    return ExecutorSSHInfo(**defaults)


def _service() -> AttestationService:
    service = AttestationService(redis_service=None)
    service.enabled = True
    service.verifier_url = "https://verifier/verify"
    return service


def v1_digest(executor: ExecutorSSHInfo) -> bytes:
    return hashlib.sha256(
        AttestationService.REPORT_PREFIX + executor.ssh_host_key.encode("utf-8")
    ).digest()


class TestReportDataRecipes:
    def test_v1_is_the_default_so_todays_fleet_keeps_verifying(self, monkeypatch):
        monkeypatch.setattr(settings, "TDX_REPORT_DATA_RECIPES", "v1")
        executor = _executor()

        digests = _service()._expected_identity_digests(executor)

        assert [recipe for recipe, _ in digests] == ["v1"]
        assert digests[0][1] == v1_digest(executor)

    def test_v2_folds_in_the_gpu_set_the_cvm_claims_to_hold(self, monkeypatch):
        """Which GPUs a CVM holds becomes something the hardware signs, rather than something
        software on the host asserts."""
        monkeypatch.setattr(settings, "TDX_REPORT_DATA_RECIPES", "v2")
        uuids = ["GPU-bbbb", "GPU-aaaa"]
        digest = AttestationService.gpu_uuid_digest(uuids)
        executor = _executor(gpu_uuid_digest=digest.hex())

        digests = _service()._expected_identity_digests(executor)

        material = AttestationService.REPORT_PREFIX + executor.ssh_host_key.encode("utf-8")
        assert digests == [("v2", hashlib.sha256(material + digest).digest())]

    def test_the_gpu_digest_does_not_depend_on_enumeration_order(self):
        """PCI topology changes the order NVML reports, and the same node must attest the same
        way after a reboot."""
        assert AttestationService.gpu_uuid_digest(
            ["GPU-a", "GPU-b"]
        ) == AttestationService.gpu_uuid_digest(["GPU-b", "GPU-a"])

    def test_a_cvm_with_no_gpus_digests_to_the_empty_string(self):
        """A validation CVM on a host with none is legal, and every quote carries a GPU binding
        of the same shape rather than a special case."""
        assert (
            AttestationService.gpu_uuid_digest([]) == hashlib.sha256(b"").digest()
        )

    def test_both_recipes_are_accepted_during_the_transition(self, monkeypatch):
        """A validator that accepted only the new one would fail every node in the fleet the
        moment it shipped."""
        monkeypatch.setattr(settings, "TDX_REPORT_DATA_RECIPES", "v2,v1")
        executor = _executor(gpu_uuid_digest=AttestationService.gpu_uuid_digest([]).hex())

        recipes = [recipe for recipe, _ in _service()._expected_identity_digests(executor)]

        assert recipes == ["v2", "v1"]

    def test_v2_is_skipped_for_a_node_that_supplies_no_digest(self, monkeypatch):
        """Not a failure — a node that has not started producing v2 has nothing to bind. It
        still has to satisfy some accepted recipe, so this narrows the set rather than widening
        it."""
        monkeypatch.setattr(settings, "TDX_REPORT_DATA_RECIPES", "v2,v1")

        recipes = [recipe for recipe, _ in _service()._expected_identity_digests(_executor())]

        assert recipes == ["v1"]

    def test_a_digest_that_is_not_hex_is_ignored_rather_than_crashing_the_cycle(
        self, monkeypatch
    ):
        monkeypatch.setattr(settings, "TDX_REPORT_DATA_RECIPES", "v2,v1")
        executor = _executor(gpu_uuid_digest="not-hex")

        recipes = [recipe for recipe, _ in _service()._expected_identity_digests(executor)]

        assert recipes == ["v1"]

    def test_an_unknown_recipe_is_ignored_and_never_accepts_anything(self, monkeypatch):
        monkeypatch.setattr(settings, "TDX_REPORT_DATA_RECIPES", "v9")

        assert _service()._expected_identity_digests(_executor()) == []

    def test_a_quote_bound_to_a_different_gpu_set_does_not_match(self, monkeypatch):
        """The point of v2: a node that claims one set and holds another produces a quote whose
        identity half the validator will not match."""
        monkeypatch.setattr(settings, "TDX_REPORT_DATA_RECIPES", "v2")
        executor = _executor(gpu_uuid_digest=AttestationService.gpu_uuid_digest(["GPU-a"]).hex())
        service = _service()

        (_, expected), = service._expected_identity_digests(executor)
        other = _executor(
            gpu_uuid_digest=AttestationService.gpu_uuid_digest(["GPU-a", "GPU-b"]).hex()
        )
        (_, different), = service._expected_identity_digests(other)

        assert expected != different


class TestRenterExpectations:
    def details(self, **app_info) -> dict:
        base = {
            "os_image_hash": OS_IMAGE,
            "compose_hash": RENTER_COMPOSE,
            "instance_id": "abc123",
        }
        base.update(app_info)
        return {"app_info": base}

    def test_a_matching_renter_cvm_passes(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_RENTER_CVM_VERIFICATION", True)
        expectations = CvmExpectations(
            os_image_hash=OS_IMAGE, compose_hash=RENTER_COMPOSE, gpu_count=2
        )
        service = _service()
        service._last_gpu_ueids = ["ueid-1", "ueid-2"]

        service._assert_renter_expectations(self.details(), expectations, _executor())

    def test_a_compose_that_is_not_the_one_derived_for_the_order_is_refused(self, monkeypatch):
        """The compose hash is what the backend computed for this customer's order, so a node
        running any other stack measures differently and is caught here."""
        monkeypatch.setattr(settings, "ENABLE_RENTER_CVM_VERIFICATION", True)
        expectations = CvmExpectations(compose_hash=RENTER_COMPOSE)

        with pytest.raises(AttestationError, match="compose_hash"):
            _service()._assert_renter_expectations(
                self.details(compose_hash="9" * 64), expectations, _executor()
            )

    def test_an_off_catalog_image_is_refused(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_RENTER_CVM_VERIFICATION", True)
        expectations = CvmExpectations(os_image_hash=OS_IMAGE)

        with pytest.raises(AttestationError, match="os_image_hash"):
            _service()._assert_renter_expectations(
                self.details(os_image_hash="9" * 64), expectations, _executor()
            )

    def test_fewer_gpus_than_were_sold_is_refused(self, monkeypatch):
        """Read from the attested evidence, not from the node's scraped specs: specs are
        asserted by software the host controls, the ueids came out of signed GPU evidence."""
        monkeypatch.setattr(settings, "ENABLE_RENTER_CVM_VERIFICATION", True)
        expectations = CvmExpectations(gpu_count=8)
        service = _service()
        service._last_gpu_ueids = ["ueid-1"]

        with pytest.raises(AttestationError, match="gpu_count: sold 8, attested 1"):
            service._assert_renter_expectations(self.details(), expectations, _executor())

    def test_every_mismatch_is_named_at_once(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_RENTER_CVM_VERIFICATION", True)
        expectations = CvmExpectations(
            os_image_hash="1" * 64, compose_hash="2" * 64, gpu_count=4
        )
        service = _service()
        service._last_gpu_ueids = ["ueid-1"]

        with pytest.raises(AttestationError) as raised:
            service._assert_renter_expectations(self.details(), expectations, _executor())

        message = str(raised.value)
        assert "compose_hash" in message and "os_image_hash" in message and "gpu_count" in message

    def test_the_check_fails_nothing_while_the_flag_is_off(self):
        """Which is how the blast radius gets measured before it is taken."""
        expectations = CvmExpectations(compose_hash=RENTER_COMPOSE)

        _service()._assert_renter_expectations(
            self.details(compose_hash="9" * 64), expectations, _executor()
        )

    def test_a_node_with_no_expectations_is_not_a_renter_cvm_and_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(settings, "ENABLE_RENTER_CVM_VERIFICATION", True)

        _service()._assert_renter_expectations(self.details(), None, _executor())


class TestTheRentalReattestCadence:
    async def test_a_rented_miner_is_re_attested_on_the_tighter_interval(self, monkeypatch):
        """A rental is exactly when a node has something to gain from swapping the stack
        underneath a proof it gave once at launch."""
        monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
        monkeypatch.setattr(settings, "ENABLE_ATTESTATION_NONCE", True)
        monkeypatch.setattr(settings, "TDX_REATTEST_INTERVAL_SECONDS", 3600)
        monkeypatch.setattr(settings, "TDX_RENTAL_REATTEST_INTERVAL_SECONDS", 900)

        redis = MagicMock()
        # Last event was 1000 s ago: past the 900 s rental interval, inside the 3600 s idle one.
        redis.get = AsyncMock(return_value=str(__import__("time").time() - 1000))
        redis.set = AsyncMock()
        service = AttestationService(redis_service=redis)
        service.enabled = True

        assert await service.maybe_issue_nonce("miner", rented=True) is not None
        assert await service.maybe_issue_nonce("miner", rented=False) is None
