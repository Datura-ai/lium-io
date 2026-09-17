"""The CVM a provider creates from this checkout must pass the validator's compose-hash whitelist.

Regression (DAH-3602): lium-io#1339 edited app/init_script.sh, one of the three files dstack measures
into compose_hash, without adding the new hash to TDX_WHITELIST. CI stayed green and every CVM created
from executor-v1.128 to v1.130 scored zero. These tests rebuild the hash the way `lium-cvm.sh new`
does (scripts/compose_hash.py, same builder as dstack.py) and fail the PR that moves it without
whitelisting it.
"""

import importlib.util
import sys
from pathlib import Path

from services.const import TDX_WHITELIST

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "neurons" / "executor" / "dstacktee" / "scripts"


def _load_compose_hash_module():
    # scripts/ is not a package: dstack.py is run as a script on the CVM host and imports host_api
    # from its own directory, so load it the way the host does.
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location("compose_hash", SCRIPTS_DIR / "compose_hash.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compose_hash = _load_compose_hash_module()
PROD_WHITELIST: dict[str, int] = TDX_WHITELIST["COMPOSE_HASH"]["PROD"]


def test_prod_compose_hash_of_this_checkout_is_whitelisted():
    measured = compose_hash.compose_hash("prod")
    assert measured in PROD_WHITELIST, (
        f"compose hash {measured} of app/docker-compose.yml + init_script.sh + pre_launch_script.sh "
        f"with runner {compose_hash.APPROVED_RUNNER_IMAGE_DIGEST} is not in "
        'TDX_WHITELIST["COMPOSE_HASH"]["PROD"]: a CVM created from this tree scores zero. '
        f"Add it as version {max(PROD_WHITELIST.values()) + 1} in services/const.py."
    )


def test_prod_compose_hash_of_this_checkout_is_the_newest_version():
    # newest-wins: the checkout is the release providers deploy next, so its hash carries the highest
    # version, or raising TDX_MINIMUM_COMPOSE_VERSION to retire an old release would retire this one
    measured = compose_hash.compose_hash("prod")
    assert PROD_WHITELIST.get(measured) == max(PROD_WHITELIST.values())


def test_release_notes_section_carries_digest_and_hash():
    section = compose_hash.release_notes_section("prod")
    assert section.startswith(compose_hash.RELEASE_NOTES_HEADING)
    assert compose_hash.APPROVED_RUNNER_IMAGE_DIGEST in section
    assert compose_hash.compose_hash("prod") in section
    assert "sha256sum run/vms/<name>/shared/app-compose.json" in section


def test_check_flags_an_app_compose_that_differs(tmp_path, capsys):
    good = tmp_path / "app-compose.json"
    good.write_text(compose_hash.measured_app_compose("prod"))
    assert compose_hash.main(["--check", str(good)]) == 0
    assert capsys.readouterr().out.rstrip().endswith("OK")

    # the provider edited a measured file (host-setup.md says not to): one byte moves the hash
    edited = tmp_path / "edited-app-compose.json"
    edited.write_text(
        compose_hash.measured_app_compose("prod").replace("secure_time", "secure_tine")
    )
    assert compose_hash.main(["--check", str(edited)]) == 1
    assert "MISMATCH" in capsys.readouterr().err
