import json
import os
import sys
import types
from pathlib import Path
from typing import Any
import asyncssh  # noqa: E402
import pytest
from datura.requests.miner_requests import ExecutorSSHInfo  # noqa: E402
from neurons.validators.src.services.attestation_service import AttestationService  # noqa: E402


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

FIXTURE_PATH = THIS_DIR / "fixtures" / "tdx_quote.json"
VERIFIER_RESPONSE_PATH = THIS_DIR / "fixtures" / "verifier_response.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.mark.asyncio
async def test_attestation_service_accepts_fixture_quote(monkeypatch):
    fixture = _fixture()
    quote_json = json.dumps(fixture["quote_response"])

    original_call_verifier = AttestationService._call_verifier

    async def fake_call_verifier(self, quote_payload: str, executor: ExecutorSSHInfo) -> dict:
        if VERIFIER_RESPONSE_PATH.exists():
            return json.loads(VERIFIER_RESPONSE_PATH.read_text())
        response = await original_call_verifier(self, quote_payload, executor)
        VERIFIER_RESPONSE_PATH.write_text(json.dumps(response))
        return response

    monkeypatch.setattr(AttestationService, "_call_verifier", fake_call_verifier)
    monkeypatch.setattr(asyncssh, "import_public_key", lambda value: object())

    service = AttestationService()
    assert service.enabled is True

    executor_info = ExecutorSSHInfo(
        uuid="test-executor",
        address="executor.example",
        port=2222,
        ssh_username="exec-user",
        ssh_port=2222,
        python_path="/usr/bin/python",
        root_dir="/opt/executor",
        port_range=None,
        port_mappings=None,
        price=None,
        ssh_host_key=fixture["host_key"],
        tdx_quote=quote_json,
    )

    policy, digest, _ = await service.prepare_host_policy(executor_info)
    assert policy is not None
