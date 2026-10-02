import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from neurons.validators.src.services.attestation_service import AttestationService  # noqa: E402

from core.config import settings  # noqa: E402

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


DEFAULT_URL = "https://verifier.default/verify"
QEMU10_URL = "https://verifier.qemu10/verify"

FIXTURE_PATH = THIS_DIR / "fixtures" / "tdx_quote.json"


def _make_service(monkeypatch, *, qemu10_url: str | None) -> AttestationService:
    """Construct the service with settings patched BEFORE __init__ reads them.

    __init__ snapshots TDX_VERIFIER_URL / TDX_VERIFIER_QEMU10_URL, so the patch
    has to be in place at construction time.
    """
    monkeypatch.setattr(settings, "ENABLE_TDX_ATTESTATION", True)
    monkeypatch.setattr(settings, "TDX_VERIFIER_URL", DEFAULT_URL)
    monkeypatch.setattr(settings, "TDX_VERIFIER_QEMU10_URL", qemu10_url)
    return AttestationService()


def _quote(qemu_version: Any, *, vm_config_as_string: bool = True) -> str:
    """Build a verifier-style quote JSON string carrying a qemu_version."""
    vm_config: dict[str, Any] = {"spec_version": 1, "image": "dstack-x"}
    if qemu_version is not _OMIT:
        vm_config["qemu_version"] = qemu_version
    payload = {
        "quote": "deadbeef",
        "vm_config": json.dumps(vm_config) if vm_config_as_string else vm_config,
    }
    return json.dumps(payload)


_OMIT = object()  # sentinel: leave qemu_version out of vm_config entirely


# ---------------------------------------------------------------------------
# _select_verifier_url — routing decisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["10.2.1", "10", "11.0.0", "12.1"])
def test_qemu10_or_newer_routes_to_qemu10_verifier(monkeypatch, version):
    service = _make_service(monkeypatch, qemu10_url=QEMU10_URL)
    assert service._select_verifier_url(_quote(version)) == QEMU10_URL


@pytest.mark.parametrize("version", ["9.2.1", "8.2.2", "9", "2.0"])
def test_qemu9_or_older_routes_to_default_verifier(monkeypatch, version):
    service = _make_service(monkeypatch, qemu10_url=QEMU10_URL)
    assert service._select_verifier_url(_quote(version)) == DEFAULT_URL


def test_qemu10_unset_always_routes_to_default(monkeypatch):
    service = _make_service(monkeypatch, qemu10_url=None)
    # Even a QEMU 10 quote must fall back to the default when no qemu10 URL exists.
    assert service._select_verifier_url(_quote("10.2.1")) == DEFAULT_URL
    assert service._select_verifier_url(_quote("9.2.1")) == DEFAULT_URL


def test_vm_config_as_dict_and_as_string_both_route(monkeypatch):
    service = _make_service(monkeypatch, qemu10_url=QEMU10_URL)
    assert service._select_verifier_url(_quote("10.2.1", vm_config_as_string=True)) == QEMU10_URL
    assert service._select_verifier_url(_quote("10.2.1", vm_config_as_string=False)) == QEMU10_URL
    assert service._select_verifier_url(_quote("9.2.1", vm_config_as_string=True)) == DEFAULT_URL
    assert service._select_verifier_url(_quote("9.2.1", vm_config_as_string=False)) == DEFAULT_URL


# ---------------------------------------------------------------------------
# _select_verifier_url — defensive fallbacks (must never raise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quote_json",
    [
        "{ this is not json",              # malformed JSON
        "",                                # empty string
        "null",                            # decodes to None, not a dict
        "[]",                              # decodes to a list, not a dict
        "\"just-a-string\"",               # decodes to a str, not a dict
        json.dumps({"quote": "x"}),        # vm_config missing entirely
        json.dumps({"vm_config": "{ not json"}),      # vm_config is unparseable JSON
        json.dumps({"vm_config": "[]"}),              # vm_config decodes to non-dict
        json.dumps({"vm_config": {}}),                # vm_config present, no qemu_version
    ],
)
def test_malformed_or_missing_falls_back_to_default(monkeypatch, quote_json):
    service = _make_service(monkeypatch, qemu10_url=QEMU10_URL)
    assert service._select_verifier_url(quote_json) == DEFAULT_URL


@pytest.mark.parametrize("version", ["q35", "", "  ", "abc", "x.y.z", "v10", "10a"])
def test_weird_version_strings_fall_back_to_default(monkeypatch, version):
    service = _make_service(monkeypatch, qemu10_url=QEMU10_URL)
    assert service._select_verifier_url(_quote(version)) == DEFAULT_URL


@pytest.mark.parametrize("version", [None, 10, 10.2, True, ["10"], _OMIT])
def test_non_string_or_absent_version_falls_back_to_default(monkeypatch, version):
    service = _make_service(monkeypatch, qemu10_url=QEMU10_URL)
    # None / non-string / omitted qemu_version must never raise and must default.
    assert service._select_verifier_url(_quote(version)) == DEFAULT_URL


def test_select_never_raises_on_arbitrary_garbage(monkeypatch):
    service = _make_service(monkeypatch, qemu10_url=QEMU10_URL)
    for garbage in ["{}", "0", "true", "{\"vm_config\": 123}", "{\"vm_config\": null}"]:
        assert service._select_verifier_url(garbage) == DEFAULT_URL


def test_real_fixture_quote_routes_to_default(monkeypatch):
    # The committed fixture stamps qemu_version 8.2.2 → default verifier.
    service = _make_service(monkeypatch, qemu10_url=QEMU10_URL)
    quote_response = json.loads(FIXTURE_PATH.read_text())["quote_response"]
    assert service._select_verifier_url(json.dumps(quote_response)) == DEFAULT_URL


# ---------------------------------------------------------------------------
# _parse_qemu_version — best-effort extraction
# ---------------------------------------------------------------------------


def test_parse_qemu_version_string_and_dict_forms():
    assert AttestationService._parse_qemu_version(_quote("10.2.1")) == "10.2.1"
    assert AttestationService._parse_qemu_version(_quote("8.2.2", vm_config_as_string=False)) == "8.2.2"


@pytest.mark.parametrize(
    "quote_json",
    ["{ bad json", "", "null", json.dumps({"quote": "x"}), json.dumps({"vm_config": {}})],
)
def test_parse_qemu_version_returns_none_on_failure(quote_json):
    assert AttestationService._parse_qemu_version(quote_json) is None
