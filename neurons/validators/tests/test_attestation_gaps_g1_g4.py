"""
Unit tests for the CVM attestation gap remediation (DAH-2338, G1–G4).

Covers, per the remediation-plan test matrix (§6):
- G4: tcb_status / advisory_ids / event_log_verified / os_image_hash_verified
  rejection when enforced, warn-only telemetry when not.
- G3: nonce round-trip in report_data[32:64], mismatch/stale rejection, legacy
  (no-nonce) behavior preserved.
- G1: NRAS overall-result / eat_nonce / missing / malformed payload handling,
  observe-vs-enforce modes, and the attestation_passed invariant
  (TDX-pass + GPU-fail is never "passed").
- Minimal-G5: the CVM ratchet fail-closed on an omitted quote and the score gate.
- G2: monotonic compose version floor in the whitelist check.
- DAH-2861: the PROD compose whitelist contents — the latest runner is accepted,
  the July staging-hotkey build is gone.
"""

import base64
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR / ".." / ".." / ".."
VALIDATOR_SRC = REPO_ROOT / "neurons" / "validators" / "src"
if str(VALIDATOR_SRC) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

if "celium_collateral_contracts" not in sys.modules:
    module = types.ModuleType("celium_collateral_contracts")

    class CollateralContract:  # type: ignore
        ...

    module.CollateralContract = CollateralContract
    sys.modules["celium_collateral_contracts"] = module

from datura.requests.miner_requests import ExecutorSSHInfo  # noqa: E402
from datura.requests.validator_requests import ssh_pubkey_signing_blob  # noqa: E402

from core.config import settings  # noqa: E402
from services.attestation_service import (  # noqa: E402
    TDX_ATTESTED_EXECUTOR_SET,
    AttestationError,
    AttestationNonce,
    AttestationService,
    HostPolicyResult,
)
from services.task.score_calculator import calculate_scores  # noqa: E402

VERIFIER_RESPONSE_PATH = THIS_DIR / "fixtures" / "verifier_response.json"
TDX_FIXTURE_PATH = THIS_DIR / "fixtures" / "tdx_quote.json"

# The validators suite runs pytest-asyncio in strict mode; mark the whole module
# (pytest-asyncio only applies the marker to coroutine tests).
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _verifier_response() -> dict:
    return json.loads(VERIFIER_RESPONSE_PATH.read_text())


def _make_executor(**overrides) -> ExecutorSSHInfo:
    defaults = dict(
        uuid="test-executor",
        address="executor.example",
        port=2222,
        ssh_username="exec-user",
        ssh_port=2222,
        python_path="/usr/bin/python",
        root_dir="/opt/executor",
        ssh_host_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestKey user@host",
        tdx_quote='{"quote": "00"}',
    )
    defaults.update(overrides)
    return ExecutorSSHInfo(**defaults)


class FakeRedis:
    def __init__(self):
        self.sets: dict[str, set] = {}
        self.kv: dict[str, str] = {}

    async def sadd(self, key, elem):
        self.sets.setdefault(key, set()).add(elem)

    async def is_elem_exists_in_set(self, key, elem):
        return elem in self.sets.get(key, set())

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value):
        self.kv[key] = value


def _jwt(claims: dict) -> str:
    def seg(obj):
        raw = base64.urlsafe_b64encode(json.dumps(obj).encode()).decode()
        return raw.rstrip("=")

    return f"{seg({'alg': 'none'})}.{seg(claims)}.sig"


def _nras_response(
    overall: bool = True,
    nonce: str | None = None,
    per_gpu: dict | None = None,
) -> list:
    overall_claims = {"x-nvidia-overall-att-result": overall}
    if nonce is not None:
        overall_claims["eat_nonce"] = nonce
    response = [["JWT", _jwt(overall_claims)]]
    if per_gpu is not None:
        response.append({name: _jwt(claims) for name, claims in per_gpu.items()})
    return response


@pytest.fixture()
def service(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", "https://verifier.example/verify")
    monkeypatch.setattr(settings, "ENABLE_ATTESTATION_WHITELIST", False)
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", False)
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION_ENFORCEMENT", False)
    monkeypatch.setattr(settings, "ENABLE_ATTESTATION_NONCE", False)
    return AttestationService()


# ---------------------------------------------------------------------------
# G4 — TCB / advisory enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_status", ["OutOfDate", "SWHardeningNeeded", "ConfigurationNeeded"])
def test_g4_bad_tcb_status_rejected_when_enforced(service, monkeypatch, bad_status):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", True)
    response = _verifier_response()
    response["details"]["tcb_status"] = bad_status
    expected = "0x" + response["details"]["report_data"][:64]

    # Act / Assert
    with pytest.raises(AttestationError, match="tcb_status_not_allowed"):
        service._validate_verifier_response(response, expected, _make_executor())


def test_g4_bad_tcb_status_warn_only_when_not_enforced(service):
    # Arrange — enforcement off (fixture default): violations must not reject
    response = _verifier_response()
    response["details"]["tcb_status"] = "OutOfDate"
    expected = "0x" + response["details"]["report_data"][:64]

    # Act — must not raise
    service._validate_verifier_response(response, expected, _make_executor())


def test_g4_allowlisted_tcb_status_accepted(service, monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", True)
    monkeypatch.setattr(settings, "TDX_ALLOWED_TCB_STATUSES", "UpToDate,SWHardeningNeeded")
    response = _verifier_response()
    response["details"]["tcb_status"] = "SWHardeningNeeded"
    expected = "0x" + response["details"]["report_data"][:64]

    # Act — must not raise
    service._validate_verifier_response(response, expected, _make_executor())


def test_g4_unexpected_advisory_rejected_when_enforced(service, monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", True)
    response = _verifier_response()
    response["details"]["advisory_ids"] = ["INTEL-SA-00837"]
    expected = "0x" + response["details"]["report_data"][:64]

    # Act / Assert
    with pytest.raises(AttestationError, match="advisory_present"):
        service._validate_verifier_response(response, expected, _make_executor())


def test_g4_allowlisted_advisory_accepted(service, monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", True)
    monkeypatch.setattr(settings, "TDX_ALLOWED_ADVISORY_IDS", "INTEL-SA-00837")
    response = _verifier_response()
    response["details"]["advisory_ids"] = ["INTEL-SA-00837"]
    expected = "0x" + response["details"]["report_data"][:64]

    # Act — must not raise
    service._validate_verifier_response(response, expected, _make_executor())


@pytest.mark.parametrize("flag", ["event_log_verified", "os_image_hash_verified"])
def test_g4_unverified_flags_rejected_when_enforced(service, monkeypatch, flag):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", True)
    response = _verifier_response()
    response["details"][flag] = False
    expected = "0x" + response["details"]["report_data"][:64]

    # Act / Assert
    with pytest.raises(AttestationError):
        service._validate_verifier_response(response, expected, _make_executor())


# ---------------------------------------------------------------------------
# G3 — freshness nonce
# ---------------------------------------------------------------------------


def test_g3_nonce_round_trip_accepted(service):
    # Arrange — report_data[32:64] echoes the issued nonce
    nonce = AttestationNonce.issue()
    response = _verifier_response()
    prefix = response["details"]["report_data"][:64]
    response["details"]["report_data"] = prefix + nonce.value_hex

    # Act — must not raise
    service._validate_verifier_response(response, "0x" + prefix, _make_executor(), nonce=nonce)


def test_g3_nonce_mismatch_rejected(service):
    # Arrange — fixture report_data carries zeros in [32:64], not the nonce
    nonce = AttestationNonce.issue()
    response = _verifier_response()
    expected = "0x" + response["details"]["report_data"][:64]

    # Act / Assert
    with pytest.raises(AttestationError, match="does not echo"):
        service._validate_verifier_response(response, expected, _make_executor(), nonce=nonce)


def test_g3_stale_nonce_rejected(service, monkeypatch):
    # Arrange — nonce issued beyond the TTL window
    monkeypatch.setattr(settings, "ATTESTATION_NONCE_TTL_SECONDS", 600)
    nonce = AttestationNonce.issue()
    nonce.issued_at = time.time() - 601
    response = _verifier_response()
    prefix = response["details"]["report_data"][:64]
    response["details"]["report_data"] = prefix + nonce.value_hex

    # Act / Assert
    with pytest.raises(AttestationError, match="nonce expired"):
        service._validate_verifier_response(response, "0x" + prefix, _make_executor(), nonce=nonce)


def test_g3_legacy_no_nonce_still_accepted(service):
    # Arrange — no nonce issued: the legacy prefix-only check applies
    response = _verifier_response()
    expected = "0x" + response["details"]["report_data"][:64]

    # Act — must not raise
    service._validate_verifier_response(response, expected, _make_executor(), nonce=None)


async def test_g3_maybe_issue_nonce_respects_interval(monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", "https://verifier.example/verify")
    monkeypatch.setattr(settings, "ENABLE_ATTESTATION_NONCE", True)
    monkeypatch.setattr(settings, "TDX_REATTEST_INTERVAL_SECONDS", 3600)
    service = AttestationService(redis_service=FakeRedis())

    # Act
    first = await service.maybe_issue_nonce("miner-hotkey")
    second = await service.maybe_issue_nonce("miner-hotkey")

    # Assert — one event per interval per miner
    assert first is not None
    assert len(first.value_bytes) == 32
    assert second is None


async def test_g3_maybe_issue_nonce_disabled_returns_none(service):
    # Arrange — ENABLE_ATTESTATION_NONCE is off in the service fixture

    # Act / Assert
    assert await service.maybe_issue_nonce("miner-hotkey") is None


def test_g3_signing_blob_covers_nonce():
    # Arrange
    nonce = AttestationNonce.issue()

    # Act
    legacy_blob = ssh_pubkey_signing_blob("ssh-ed25519 AAAA key")
    nonce_blob = ssh_pubkey_signing_blob("ssh-ed25519 AAAA key", nonce.value_hex)

    # Assert — stripping or swapping the nonce changes the signed message
    assert legacy_blob == "ssh-ed25519 AAAA key"
    assert nonce_blob != legacy_blob
    assert nonce.value_hex in nonce_blob
    assert ssh_pubkey_signing_blob("ssh-ed25519 AAAA key", "ff" * 32) != nonce_blob


# ---------------------------------------------------------------------------
# G1 — NVIDIA GPU attestation (validator verification side)
# ---------------------------------------------------------------------------


def _gpu_payload(nonce_hex: str) -> str:
    return json.dumps(
        {"nonce": nonce_hex, "evidence_list": [{"evidence": "ZXY=", "certificate": "chain"}],
         "arch": "HOPPER"}
    )


async def test_g1_missing_payload_rejected_when_enforced(service, monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION_ENFORCEMENT", True)
    nonce = AttestationNonce.issue()
    executor = _make_executor(nvidia_payload=None)

    # Act / Assert
    with pytest.raises(AttestationError, match="did not provide NVIDIA GPU"):
        await service._verify_gpu(executor, nonce)


async def test_g1_missing_payload_none_when_not_enforced(service):
    # Act / Assert — observe-only: not performed
    assert await service._verify_gpu(_make_executor(nvidia_payload=None), None) is None


async def test_g1_malformed_payload_rejected_when_enforced(service, monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION_ENFORCEMENT", True)
    nonce = AttestationNonce.issue()
    executor = _make_executor(nvidia_payload="not-json{")

    # Act / Assert
    with pytest.raises(AttestationError, match="Malformed"):
        await service._verify_gpu(executor, nonce)


async def test_g1_unbound_nonce_rejected_when_enforced(service, monkeypatch):
    # Arrange — payload carries a self-generated nonce, not the issued challenge
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION_ENFORCEMENT", True)
    nonce = AttestationNonce.issue()
    executor = _make_executor(nvidia_payload=_gpu_payload("ab" * 32))

    # Act / Assert
    with pytest.raises(AttestationError, match="not\nbound|not bound"):
        await service._verify_gpu(executor, nonce)


async def test_g1_nras_overall_failure_rejected_when_enforced(service, monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION_ENFORCEMENT", True)
    nonce = AttestationNonce.issue()
    executor = _make_executor(nvidia_payload=_gpu_payload(nonce.value_hex))

    async def fake_post(payload, executor_):
        return _nras_response(overall=False, nonce=nonce.value_hex)

    monkeypatch.setattr(service, "_post_nras", fake_post)

    # Act / Assert
    with pytest.raises(AttestationError, match="overall_result"):
        await service._verify_gpu(executor, nonce)


async def test_g1_nras_overall_failure_observed_as_false(service, monkeypatch):
    # Arrange — enforcement off: verified-bad is recorded, never raised
    nonce = AttestationNonce.issue()
    executor = _make_executor(nvidia_payload=_gpu_payload(nonce.value_hex))

    async def fake_post(payload, executor_):
        return _nras_response(overall=False, nonce=nonce.value_hex)

    monkeypatch.setattr(service, "_post_nras", fake_post)

    # Act / Assert
    assert await service._verify_gpu(executor, nonce) is False


async def test_g1_eat_nonce_mismatch_rejected(service, monkeypatch):
    # Arrange — NRAS echoes a different nonce than the payload claimed
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION_ENFORCEMENT", True)
    nonce = AttestationNonce.issue()
    executor = _make_executor(nvidia_payload=_gpu_payload(nonce.value_hex))

    async def fake_post(payload, executor_):
        return _nras_response(overall=True, nonce="cd" * 32)

    monkeypatch.setattr(service, "_post_nras", fake_post)

    # Act / Assert
    with pytest.raises(AttestationError, match="eat_nonce_mismatch"):
        await service._verify_gpu(executor, nonce)


async def test_g1_per_gpu_claims_checked(service, monkeypatch):
    # Arrange — one GPU reports measres failure
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION_ENFORCEMENT", True)
    nonce = AttestationNonce.issue()
    executor = _make_executor(nvidia_payload=_gpu_payload(nonce.value_hex))

    async def fake_post(payload, executor_):
        return _nras_response(
            overall=True,
            nonce=nonce.value_hex,
            per_gpu={"GPU-0": {"measres": "success", "eat_nonce": nonce.value_hex},
                     "GPU-1": {"measres": "fail", "eat_nonce": nonce.value_hex}},
        )

    monkeypatch.setattr(service, "_post_nras", fake_post)

    # Act / Assert
    with pytest.raises(AttestationError, match="GPU-1:measres"):
        await service._verify_gpu(executor, nonce)


async def test_g1_nras_pass_returns_true(service, monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION_ENFORCEMENT", True)
    nonce = AttestationNonce.issue()
    executor = _make_executor(nvidia_payload=_gpu_payload(nonce.value_hex))

    async def fake_post(payload, executor_):
        return _nras_response(
            overall=True,
            nonce=nonce.value_hex,
            per_gpu={"GPU-0": {"measres": "success", "eat_nonce": nonce.value_hex,
                               "dbgstat": "disabled", "secboot": True}},
        )

    monkeypatch.setattr(service, "_post_nras", fake_post)

    # Act / Assert
    assert await service._verify_gpu(executor, nonce) is True


async def test_g1_nras_unreachable_fail_closed_when_enforced(service, monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION_ENFORCEMENT", True)
    nonce = AttestationNonce.issue()
    executor = _make_executor(nvidia_payload=_gpu_payload(nonce.value_hex))

    async def fake_post(payload, executor_):
        raise AttestationError("NRAS unreachable for executor x: boom")

    monkeypatch.setattr(service, "_post_nras", fake_post)

    # Act / Assert — enforcement: fail closed
    with pytest.raises(AttestationError, match="NRAS unreachable"):
        await service._verify_gpu(executor, nonce)

    # Arrange — observe-only: undeterminable, not verified-bad
    monkeypatch.setattr(settings, "ENABLE_GPU_ATTESTATION_ENFORCEMENT", False)

    # Act / Assert
    assert await service._verify_gpu(executor, nonce) is None


def test_g1_attestation_passed_invariant():
    # Assert — TDX pass + GPU fail can never surface as passed
    assert HostPolicyResult(attestation_digest="d", gpu_attestation_passed=False).attestation_passed is False
    assert HostPolicyResult(attestation_digest="d", gpu_attestation_passed=True).attestation_passed is True
    assert HostPolicyResult(attestation_digest="d", gpu_attestation_passed=None).attestation_passed is True
    assert HostPolicyResult(attestation_digest=None).attestation_passed is False


# ---------------------------------------------------------------------------
# Minimal-G5 — CVM ratchet + score gate
# ---------------------------------------------------------------------------


async def test_g5_ratcheted_executor_omitting_quote_fails_closed(monkeypatch):
    # Arrange — executor previously attested (in the ratchet set), now no quote
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", "https://verifier.example/verify")
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", True)
    redis = FakeRedis()
    await redis.sadd(TDX_ATTESTED_EXECUTOR_SET, "test-executor")
    service = AttestationService(redis_service=redis)
    executor = _make_executor(tdx_quote=None)

    # Act / Assert
    with pytest.raises(AttestationError, match="omitted its TDX quote"):
        await service.prepare_host_policy(executor)


async def test_g5_unknown_executor_without_quote_unaffected(monkeypatch):
    # Arrange — bare-metal executor (never attested) stays on the legacy path
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", "https://verifier.example/verify")
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", True)
    import asyncssh

    monkeypatch.setattr(asyncssh, "import_known_hosts", lambda value: object())
    service = AttestationService(redis_service=FakeRedis())
    executor = _make_executor(tdx_quote=None)

    # Act
    host_policy = await service.prepare_host_policy(executor)

    # Assert — no attestation claimed, no rejection
    assert host_policy.attestation_digest is None
    assert host_policy.gpu_attestation_passed is None


def _score_ctx(tdx_quote, attestation_passed):
    executor = SimpleNamespace(price_per_gpu=None, tdx_quote=tdx_quote)
    state = SimpleNamespace(
        gpu_model="",
        specs={"network": {"ema_verifyx_download_speed": 500.0}},
    )
    return SimpleNamespace(
        state=state,
        collateral_deposited=True,
        collateral_error_message=None,
        contract_version=None,
        executor=executor,
        tdx_attestation_passed=attestation_passed,
        cpu_truth_passed=True,
        provider_side_load_passed=True,
    )


def test_g5_score_gate_zeroes_failed_cvm(monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", True)

    # Act
    actual, job, warning = calculate_scores(_score_ctx('{"quote":"00"}', False), rented=False)

    # Assert
    assert actual == 0.0
    assert job == 0.0
    assert "CVM attestation not passed" in warning


def test_g5_score_gate_passes_attested_cvm_and_bare_metal(monkeypatch):
    # Arrange
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", True)

    # Act / Assert — attested CVM keeps its score
    actual, _, _ = calculate_scores(_score_ctx('{"quote":"00"}', True), rented=False)
    assert actual == 1.0

    # Act / Assert — bare-metal node (no quote) is untouched by the gate
    actual, _, _ = calculate_scores(_score_ctx(None, False), rented=False)
    assert actual == 1.0


def test_g5_score_gate_inert_when_enforcement_off(monkeypatch):
    # Arrange — safe-off default: gate must not fire
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    monkeypatch.setattr(settings, "ENABLE_TCB_ENFORCEMENT", False)

    # Act / Assert
    actual, _, _ = calculate_scores(_score_ctx('{"quote":"00"}', False), rented=False)
    assert actual == 1.0


# ---------------------------------------------------------------------------
# G2 — monotonic compose version floor
# ---------------------------------------------------------------------------


async def test_g2_compose_version_floor(service, monkeypatch):
    # Arrange
    from services import const

    monkeypatch.setattr(settings, "DEPLOY_ENV", "LOCAL")
    monkeypatch.setitem(
        const.TDX_WHITELIST["COMPOSE_HASH"], "LOCAL", {"old-hash": 1, "new-hash": 2}
    )
    os_image = next(iter(const.TDX_WHITELIST["OS_IMAGE_HASH"]))
    executor = _make_executor()

    # Act / Assert — floor 0 (default): every whitelisted hash accepted
    monkeypatch.setattr(settings, "TDX_MINIMUM_COMPOSE_VERSION", 0)
    assert await service._check_whitelist("old-hash", os_image, executor) is True

    # Act / Assert — floor 2: version-1 release rejected, version-2 accepted
    monkeypatch.setattr(settings, "TDX_MINIMUM_COMPOSE_VERSION", 2)
    assert await service._check_whitelist("old-hash", os_image, executor) is False
    assert await service._check_whitelist("new-hash", os_image, executor) is True

    # Act / Assert — unknown hash always rejected
    assert await service._check_whitelist("unknown-hash", os_image, executor) is False

    # Act / Assert — unknown OS image always rejected
    assert await service._check_whitelist("new-hash", "bad-os-image", executor) is False


# ---------------------------------------------------------------------------
# DAH-2861 — the PROD compose whitelist itself
# ---------------------------------------------------------------------------

# executor-v1.123 runner sha256:8c07d3a9… (2026-08-19), the prod build.
LATEST_RUNNER_COMPOSE_HASH = "8224d58801af6333561f116e2d566b179b399f1d1d700f0e8a5ab9326ae901d9"
# executor-v1.108 runner sha256:f85b948b… (2026-07-07), which bakes the staging validator hotkey.
JULY_STAGING_RUNNER_COMPOSE_HASH = "ab4d14336f0762c0d8ec7631a69148246661de84ceead7a215f8a33b74fd43e6"


async def test_prod_whitelist_accepts_latest_runner_compose(service, monkeypatch):
    # Arrange
    from services import const

    monkeypatch.setattr(settings, "DEPLOY_ENV", "PROD")
    os_image = next(iter(const.TDX_WHITELIST["OS_IMAGE_HASH"]))
    executor = _make_executor()

    # Act / Assert — default floor
    monkeypatch.setattr(settings, "TDX_MINIMUM_COMPOSE_VERSION", 0)
    assert await service._check_whitelist(LATEST_RUNNER_COMPOSE_HASH, os_image, executor) is True

    # Act / Assert — floor raised to this release: still at/above it
    monkeypatch.setattr(settings, "TDX_MINIMUM_COMPOSE_VERSION", 4)
    assert await service._check_whitelist(LATEST_RUNNER_COMPOSE_HASH, os_image, executor) is True


async def test_prod_whitelist_rejects_july_staging_runner_compose(service, monkeypatch):
    # Arrange — the July runner is gone from the dict, so rejection cannot depend on the floor
    from services import const

    monkeypatch.setattr(settings, "DEPLOY_ENV", "PROD")
    monkeypatch.setattr(settings, "TDX_MINIMUM_COMPOSE_VERSION", 0)
    os_image = next(iter(const.TDX_WHITELIST["OS_IMAGE_HASH"]))
    executor = _make_executor()

    # Act / Assert
    assert JULY_STAGING_RUNNER_COMPOSE_HASH not in const.TDX_WHITELIST["COMPOSE_HASH"]["PROD"]
    assert (
        await service._check_whitelist(JULY_STAGING_RUNNER_COMPOSE_HASH, os_image, executor)
        is False
    )
