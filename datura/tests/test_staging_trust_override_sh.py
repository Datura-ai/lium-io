"""scripts/staging_trust_override.sh: what a non-prod image build appends to its config_override, read back the way the
executor, miner and watchtower read it (override_trust)."""

import subprocess
from pathlib import Path

import pytest
from bittensor_wallet import Keypair

from datura.validator_trust import SignedTrustMessage, override_trust, trust_line

HELPER = Path(__file__).resolve().parents[2] / "scripts" / "staging_trust_override.sh"
VALIDATOR = Keypair.create_from_uri("//HelperValidator6670").ss58_address


def _run(tmp_path, env):
    override = tmp_path / "helper_override_6670.py"
    override.write_text('_VALIDATOR_HOTKEY_SS58 = "x"\n')
    r = subprocess.run(["bash", "-c", '. "$0" && staging_trust_override "$1"', str(HELPER), str(override)],
                       env={"PATH": "/usr/bin:/bin", **env}, capture_output=True, text=True, timeout=30)
    return r, override


def test_the_helper_writes_an_override_the_services_follow(tmp_path, monkeypatch):
    owner = Keypair.create_from_uri("//HelperOwner6670")
    line = trust_line(VALIDATOR, 1)
    sig = "0x" + owner.sign(line).hex()
    r, _ = _run(tmp_path, {"STAGING_TRUST_OWNER_SS58": owner.ss58_address, "STAGING_TRUST_MESSAGE": line,
                           "STAGING_TRUST_SIGNATURE": sig})
    assert r.returncode == 0, r.stderr
    monkeypatch.syspath_prepend(str(tmp_path))
    assert override_trust("helper_override_6670") == {
        "owner": owner.ss58_address, "read_chain": False, "message": SignedTrustMessage(line, sig)}


def test_unset_leaves_the_override_as_it_was(tmp_path):
    r, override = _run(tmp_path, {})
    assert r.returncode == 0 and override.read_text() == '_VALIDATOR_HOTKEY_SS58 = "x"\n'


@pytest.mark.parametrize("bad", [
    {"STAGING_TRUST_OWNER_SS58": VALIDATOR},   # partial
    {"STAGING_TRUST_OWNER_SS58": VALIDATOR, "STAGING_TRUST_MESSAGE": 'x"\nimport os\n', "STAGING_TRUST_SIGNATURE": "0x" + "a" * 128},
    {"STAGING_TRUST_OWNER_SS58": 'a"; import os #', "STAGING_TRUST_MESSAGE": trust_line(VALIDATOR, 1),
     "STAGING_TRUST_SIGNATURE": "0x" + "a" * 128},
])
def test_a_partial_or_malformed_value_fails_and_writes_nothing(tmp_path, bad):
    r, override = _run(tmp_path, bad)
    assert r.returncode != 0 and override.read_text() == '_VALIDATOR_HOTKEY_SS58 = "x"\n'
