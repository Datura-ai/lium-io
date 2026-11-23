"""scripts/staging_trust_override.sh: what a non-prod image build appends to its config_override, read back the way the
executor, miner and watchtower read it (override_trust)."""

import subprocess
from pathlib import Path

import pytest
from bittensor_wallet import Keypair

from datura.validator_trust import SignedTrustMessage, override_trust, trust_line

HELPER = Path(__file__).resolve().parents[2] / "scripts" / "staging_trust_override.sh"
VALIDATOR = Keypair.create_from_uri("//HelperValidator6670").ss58_address
STAGING_API = "https://staging.lium.io/api"
GOOD = {"STAGING_TRUST_OWNER_SS58": VALIDATOR, "STAGING_TRUST_MESSAGE": trust_line(VALIDATOR, 1, "test"),
        "STAGING_TRUST_SIGNATURE": "0x" + "a" * 128, "STAGING_TRUST_API_URL": STAGING_API}


def _run(tmp_path, env):
    override = tmp_path / "helper_override_6670.py"
    override.write_text('_VALIDATOR_HOTKEY_SS58 = "x"\n')
    r = subprocess.run(["bash", "-c", '. "$0" && staging_trust_override "$1"', str(HELPER), str(override)],
                       env={"PATH": "/usr/bin:/bin", "LC_ALL": "en_US.UTF-8", **env}, capture_output=True, text=True, timeout=30)
    return r, override


def test_the_helper_writes_an_override_the_services_follow(tmp_path, monkeypatch):
    owner = Keypair.create_from_uri("//HelperOwner6670")
    line = trust_line(VALIDATOR, 1, "test")
    sig = "0x" + owner.sign(line).hex()
    r, _ = _run(tmp_path, {"STAGING_TRUST_OWNER_SS58": owner.ss58_address, "STAGING_TRUST_MESSAGE": line,
                           "STAGING_TRUST_SIGNATURE": sig, "STAGING_TRUST_API_URL": STAGING_API})
    assert r.returncode == 0, r.stderr
    monkeypatch.syspath_prepend(str(tmp_path))
    assert override_trust("helper_override_6670") == {
        "owner": owner.ss58_address, "read_chain": False, "message": SignedTrustMessage(line, sig), "api_url": STAGING_API,
        "network": "test"}


def test_unset_leaves_the_override_as_it_was(tmp_path):
    r, override = _run(tmp_path, {})
    assert r.returncode == 0 and override.read_text() == '_VALIDATOR_HOTKEY_SS58 = "x"\n'


@pytest.mark.parametrize("bad", [
    {"STAGING_TRUST_OWNER_SS58": VALIDATOR},   # partial
    {"STAGING_TRUST_OWNER_SS58": VALIDATOR, "STAGING_TRUST_MESSAGE": 'x"\nimport os\n', "STAGING_TRUST_SIGNATURE": "0x" + "a" * 128},
    {"STAGING_TRUST_OWNER_SS58": 'a"; import os #', "STAGING_TRUST_MESSAGE": trust_line(VALIDATOR, 1, "test"),
     "STAGING_TRUST_SIGNATURE": "0x" + "a" * 128, "STAGING_TRUST_API_URL": STAGING_API},
    {k: v for k, v in GOOD.items() if k != "STAGING_TRUST_API_URL"},   # no backend for the messages
    {**GOOD, "STAGING_TRUST_API_URL": "http://staging.lium.io/api"},
    {**GOOD, "STAGING_TRUST_API_URL": 'https://x"\nimport os\n'},
    {**GOOD, "STAGING_TRUST_MESSAGE": trust_line(VALIDATOR, 1)},   # a release line: a staging signature must not make one
    {**GOOD, "STAGING_TRUST_OWNER_SS58": VALIDATOR[:-1] + "\u00e9"},   # non-ASCII letter: the C locale class refuses it
])
def test_a_partial_or_malformed_value_fails_and_writes_nothing(tmp_path, bad):
    r, override = _run(tmp_path, bad)
    assert r.returncode != 0 and override.read_text() == '_VALIDATOR_HOTKEY_SS58 = "x"\n'


@pytest.mark.parametrize("key", sorted(GOOD))
@pytest.mark.parametrize("char", ['"', "\n", "\\"])
def test_a_quote_newline_or_backslash_in_any_value_is_refused(tmp_path, key, char):
    # the override is Python source: no value may carry a character that ends or escapes its string
    r, override = _run(tmp_path, {**GOOD, key: GOOD[key] + char})
    assert r.returncode != 0 and override.read_text() == '_VALIDATOR_HOTKEY_SS58 = "x"\n'


WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


@pytest.mark.parametrize("workflow", ["executor_cd_dev.yml", "miner_cd_staging.yml", "watchtower_image.yml"])
def test_every_staging_image_build_names_the_same_signed_owner_message(workflow):
    # the staging executor, miner and watchtower must follow one owner; a build without the values drops the trust
    import re

    text = (WORKFLOWS / workflow).read_text()
    values = {k: re.search(rf'^\s+{k}: "([^"]+)"$', text, re.M).group(1)
              for k in ("STAGING_TRUST_OWNER_SS58", "STAGING_TRUST_MESSAGE", "STAGING_TRUST_SIGNATURE",
                        "STAGING_TRUST_API_URL")}
    assert values["STAGING_TRUST_OWNER_SS58"] == "5EeYEmUGfAJeTmVvYSmkpsUmVhiySd2Zch5nyt4tE1HygFX8"
    assert values["STAGING_TRUST_API_URL"] == STAGING_API   # the backend that serves this owner's messages
    assert values["STAGING_TRUST_MESSAGE"].startswith("lium-validator:test:")   # never a release line
    assert Keypair(ss58_address=values["STAGING_TRUST_OWNER_SS58"]).verify(
        values["STAGING_TRUST_MESSAGE"], bytes.fromhex(values["STAGING_TRUST_SIGNATURE"][2:]))
