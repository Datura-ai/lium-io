"""The CVM a provider creates from this checkout must pass the validator's compose-hash whitelist.

Regression (DAH-3602): lium-io#1339 edited app/init_script.sh, one of the three files dstack measures
into compose_hash, without adding the new hash to TDX_WHITELIST. CI stayed green and every CVM created
from executor-v1.128 to v1.130 would score zero with the whitelist on. These tests rebuild the hash the way `lium-cvm.sh new`
does (scripts/compose_hash.py, same builder as dstack.py) and fail the PR that moves it without
whitelisting it.
"""

import importlib.util
import sys
from pathlib import Path

from services.const import TDX_WHITELIST

REPO_ROOT = Path(__file__).resolve().parents[3]
DSTACKTEE_DIR = REPO_ROOT / "neurons" / "executor" / "dstacktee"
SCRIPTS_DIR = DSTACKTEE_DIR / "scripts"


def _load_compose_hash():
    # scripts/ is not a package: compose_hash.py puts its own directory on sys.path and imports
    # dstack (which imports host_api) from there, the way the CVM host runs it
    spec = importlib.util.spec_from_file_location("compose_hash", SCRIPTS_DIR / "compose_hash.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compose_hash = _load_compose_hash()
dstack = sys.modules["dstack"]
PROD_WHITELIST: dict[str, int] = TDX_WHITELIST["COMPOSE_HASH"]["PROD"]


def test_prod_compose_hash_of_this_checkout_is_whitelisted():
    measured = compose_hash.compose_hash("prod")
    assert measured in PROD_WHITELIST, (
        f"compose hash {measured} of app/docker-compose.yml + init_script.sh + pre_launch_script.sh "
        f"with runner {compose_hash.APPROVED_RUNNER_IMAGE_DIGEST} is not in "
        'TDX_WHITELIST["COMPOSE_HASH"]["PROD"]: a CVM created from this tree scores zero. '
        f"Add it as version {max(PROD_WHITELIST.values()) + 1} in services/const.py."
    )


