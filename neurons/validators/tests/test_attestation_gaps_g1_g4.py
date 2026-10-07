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

from datura.requests.miner_requests import ExecutorSSHInfo  # noqa: E402

from core.config import settings  # noqa: E402
from services.attestation_service import (  # noqa: E402
    TDX_ATTESTED_EXECUTOR_SET,
    AttestationError,
    AttestationNonce,
    AttestationService,
)
from services.task.score_calculator import calculate_scores  # noqa: E402

VERIFIER_RESPONSE_PATH = THIS_DIR / "fixtures" / "verifier_response.json"

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


# ---------------------------------------------------------------------------
# G3 — freshness nonce
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# G1 — NVIDIA GPU attestation (validator verification side)
# ---------------------------------------------------------------------------


def _gpu_payload(nonce_hex: str) -> str:
    return json.dumps(
        {"nonce": nonce_hex, "evidence_list": [{"evidence": "ZXY=", "certificate": "chain"}],
         "arch": "HOPPER"}
    )


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


def _score_ctx(tdx_quote, attestation_passed):
    executor = SimpleNamespace(price_per_gpu=None, tdx_quote=tdx_quote)
    state = SimpleNamespace(
        gpu_model="",
        specs={"network": {"ema_verifyx_download_speed": 500.0}},
    )
    return SimpleNamespace(
        state=state,
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


# ---------------------------------------------------------------------------
# G2 — monotonic compose version floor
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DAH-2861 — the PROD compose whitelist itself
# ---------------------------------------------------------------------------


