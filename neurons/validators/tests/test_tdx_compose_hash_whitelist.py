"""The CVM a provider creates from this checkout must pass the validator's compose-hash whitelist.

Regression (DAH-3602): lium-io#1339 edited app/init_script.sh, one of the three files dstack measures
into compose_hash, without adding the new hash to TDX_WHITELIST. CI stayed green and every CVM created
from executor-v1.128 to v1.130 scored zero. These tests rebuild the hash the way `lium-cvm.sh new`
does (scripts/compose_hash.py, same builder as dstack.py) and fail the PR that moves it without
whitelisting it.
"""

import argparse
import hashlib
import importlib.util
import re
import sys
import types
from pathlib import Path

from services.const import TDX_WHITELIST

REPO_ROOT = Path(__file__).resolve().parents[3]
DSTACKTEE_DIR = REPO_ROOT / "neurons" / "executor" / "dstacktee"
SCRIPTS_DIR = DSTACKTEE_DIR / "scripts"


def _load_script(name: str):
    # scripts/ is not a package: dstack.py runs as a script on the CVM host and imports host_api
    # from its own directory, so that directory goes on sys.path the way the host has it
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compose_hash = _load_script("compose_hash")
dstack = _load_script("dstack")
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


def test_dstack_new_writes_the_bytes_compose_hash_rebuilds(tmp_path):
    # the rebuild is only worth anything while it equals what `dstack.py new` writes: a key added in
    # setup_instance alone (or a different serializer) moves every real CVM's hash and must fail here
    digest = compose_hash.APPROVED_RUNNER_IMAGE_DIGEST
    resolved = tmp_path / "resolved-docker-compose.yml"
    resolved.write_text(
        (DSTACKTEE_DIR / "app" / "docker-compose.yml")
        .read_text()
        .replace(compose_hash.DIGEST_PLACEHOLDER, digest)
    )
    manager = dstack.DStackManager.__new__(dstack.DStackManager)
    manager.run_path = str(tmp_path)
    manager.config = types.SimpleNamespace(docker_registry=None)
    manager.setup_instance(
        argparse.Namespace(
            dir=str(tmp_path / "vm"),
            compose_file=str(resolved),
            image=str(tmp_path / "image"),
            vcpus=1,
            memory="2G",
            disk="20G",
            gpu=None,
            port=None,
            env_file=None,
            pin_numa=False,
            hugepages=False,
            local_key_provider=True,
            enable_logs=False,
            enable_sysinfo=False,
            init_script=str(DSTACKTEE_DIR / "app" / "init_script.sh"),
            pre_launch_script=str(DSTACKTEE_DIR / "app" / "pre_launch_script.sh"),
        )
    )
    written = (tmp_path / "vm" / "shared" / "app-compose.json").read_bytes()
    assert written == compose_hash.measured_app_compose("prod", digest).encode()
    assert hashlib.sha256(written).hexdigest() == compose_hash.compose_hash("prod", digest)


def test_lium_cvm_sh_new_passes_the_flags_the_rebuild_assumes():
    # compose_hash.py hardcodes the flags lium-cvm.sh gives `dstack.py new`; a new default flag there
    # (say --enable-logs) moves the real hash while the rebuild stays put, so pin the contract
    script = (DSTACKTEE_DIR / "lium-cvm.sh").read_text()
    invocation = re.search(r"python3 \$SCRIPTS_DIR/dstack\.py new .*?(?=\n\n)", script, re.S).group(
        0
    )
    flags = set(re.findall(r"(--[a-z-]+|\$\w+_args?)\b", invocation))
    assert flags == {
        "--init-script",
        "--pre-launch-script",
        "--dir",
        "--image",
        "--vcpus",
        "--memory",
        "--disk",
        "--env-file",
        "$gpu_args",
        "$port_args",
        "$lkp_args",
        "$logs_arg",
        "$sysinfo_arg",
    }, invocation
    assert 'local lkp_args="--local-key-provider"' in script
    assert 'local logs_arg=""' in script and 'local sysinfo_arg=""' in script
    assert "envsubst '${EXECUTOR_RUNNER_IMAGE_DIGEST}'" in script


def test_release_notes_section_carries_digest_and_hash():
    section = compose_hash.release_notes_section("prod")
    assert section.startswith(compose_hash.RELEASE_NOTES_HEADING)
    assert compose_hash.APPROVED_RUNNER_IMAGE_DIGEST in section
    assert compose_hash.compose_hash("prod") in section


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
