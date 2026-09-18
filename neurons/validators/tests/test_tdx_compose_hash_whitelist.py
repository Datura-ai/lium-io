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
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
from services.const import TDX_WHITELIST

REPO_ROOT = Path(__file__).resolve().parents[3]
DSTACKTEE_DIR = REPO_ROOT / "neurons" / "executor" / "dstacktee"
SCRIPTS_DIR = DSTACKTEE_DIR / "scripts"
RELEASE_SCRIPT = SCRIPTS_DIR / "release_notes_update.sh"


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


def test_prod_compose_hash_of_this_checkout_is_the_newest_version():
    # newest-wins: the checkout is the release providers deploy next, so its hash carries the highest
    # version, or raising TDX_MINIMUM_COMPOSE_VERSION to retire an old release would retire this one
    # strictly above every other hash: two hashes sharing the top version could not be retired apart
    measured = compose_hash.compose_hash("prod")
    others = max((v for h, v in PROD_WHITELIST.items() if h != measured), default=0)
    assert PROD_WHITELIST.get(measured, 0) > others, (
        f"compose hash {measured} of this checkout is not the newest entry in "
        'TDX_WHITELIST["COMPOSE_HASH"]["PROD"]: give it a version above every other hash '
        "(a revert to an older tree lands here)."
    )


def test_dstack_new_writes_the_bytes_compose_hash_rebuilds(tmp_path):
    # the rebuild is only worth anything while it equals what `dstack.py new` writes: a key added in
    # setup_instance alone (or a different serializer) moves every real CVM's hash and must fail here
    digest = compose_hash.APPROVED_RUNNER_IMAGE_DIGEST
    resolved = tmp_path / "resolved-docker-compose.yml"
    resolved.write_text(
        (DSTACKTEE_DIR / "app" / "docker-compose.yml")
        .read_text(encoding="utf-8")
        .replace(compose_hash.DIGEST_PLACEHOLDER, digest),
        encoding="utf-8",
    )
    manager = dstack.DStackManager.__new__(dstack.DStackManager)
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
    script = (DSTACKTEE_DIR / "lium-cvm.sh").read_text(encoding="utf-8")
    match = re.search(r"python3 \$SCRIPTS_DIR/dstack\.py new .*?(?=\n\n)", script, re.S)
    assert match, (
        "the `dstack.py new` block in lium-cvm.sh moved: update this test and compose_hash.py"
    )
    invocation = match.group(0)
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
    # the paths behind the flags too: a changed INIT_SCRIPT or prod compose path moves the real hash
    # while the rebuild (APP_DIR / "init_script.sh", COMPOSE_FILES["prod"]) stays put
    assert 'INIT_SCRIPT="$THIS_DIR/app/init_script.sh"' in script
    assert 'PRE_LAUNCH_SCRIPT="$THIS_DIR/app/pre_launch_script.sh"' in script
    assert 'local compose_file="$THIS_DIR/app/docker-compose.yml"' in script


def test_release_notes_section_carries_digest_and_hash():
    section = compose_hash.release_notes_section("prod")
    # the release script cuts the body at this heading to find the section a release already carries
    script = RELEASE_SCRIPT.read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github" / "workflows" / "executor_cd_prod.yml").read_text(
        encoding="utf-8"
    )
    assert section.startswith(compose_hash.RELEASE_NOTES_HEADING)
    assert f'heading="{compose_hash.RELEASE_NOTES_HEADING}"' in script
    assert RELEASE_SCRIPT.relative_to(REPO_ROOT).as_posix() in workflow
    assert compose_hash.APPROVED_RUNNER_IMAGE_DIGEST in section
    assert compose_hash.compose_hash("prod") in section
    assert "\n## " not in section[len(compose_hash.RELEASE_NOTES_HEADING) :], (
        "a second '## ' heading inside the section would end it early for release_notes_update.sh"
    )
    # the digest is the 2026-08-19 runner, older than any tag that ships it: the section says so, and
    # says what an unknown hash costs under each setting of the validator's whitelist flag
    assert f"pushed {compose_hash.APPROVED_RUNNER_IMAGE_PUSHED}" in section
    assert "ENABLE_ATTESTATION_WHITELIST" in section
    preview = compose_hash.release_notes_section("prod", "sha256:" + "0" * 64)
    assert "not the approved runner" in preview and "pushed" not in preview


def test_digest_must_be_64_hex():
    # sha256: plus 64 of anything used to pass; Docker cannot pull such a digest, so refuse it (argparse exits 2)
    with pytest.raises(SystemExit) as exc:
        compose_hash.main(["--digest", "sha256:" + "z" * 64])
    assert exc.value.code == 2
    assert compose_hash.main(["--digest", compose_hash.APPROVED_RUNNER_IMAGE_DIGEST]) == 0


TAG = "executor-v9.999"
GENERATED_NOTES = "## What's Changed\n* something by @someone in #1\n\n**Full Changelog**: a...b"
FAKE_GH = r"""#!/usr/bin/env bash
# a `gh` stand-in for release_notes_update.sh: releases are files under $FAKE_RELEASES/<tag>.md;
# every call is appended to $FAKE_GH_LOG as "<sub> <sub2> <tag>"
set -euo pipefail
echo "$1 $2 ${3:-}" >> "$FAKE_GH_LOG"
case "$1 $2" in
  "release view") [ -z "${FAKE_VIEW_ERROR:-}" ] || { echo "$FAKE_VIEW_ERROR" >&2; exit 1; }
                  [ -f "$FAKE_RELEASES/$3.md" ] || { echo "release not found" >&2; exit 1; }; cat "$FAKE_RELEASES/$3.md" ;;
  "release edit") [ -f "$FAKE_RELEASES/$3.md" ] || exit 1; cp "$5" "$FAKE_RELEASES/$3.md" ;;
  "release create") [ ! -f "$FAKE_RELEASES/$3.md" ] || exit 1; [ "$4" = --verify-tag ] || exit 1; cp "$6" "$FAKE_RELEASES/$3.md" ;;
  "api repos/{owner}/{repo}/releases/generate-notes") printf '%s\n' "$FAKE_GENERATED" ;;
  *) echo "fake gh: unexpected call: $*" >&2; exit 64 ;;
esac
"""


def _run_release_script(tmp_path, body, view_error=""):
    """Run release_notes_update.sh for TAG against the fake gh; returns (completed process, body after, gh log)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(FAKE_GH, encoding="utf-8")
    (bin_dir / "gh").chmod(0o755)
    # the script calls `python3`: make that this interpreter (dstack.py needs 3.10+; a box's system
    # python3 may be older), the way the ubuntu-latest runner's python3 is 3.12
    (bin_dir / "python3").symlink_to(sys.executable)
    releases = tmp_path / "releases"
    releases.mkdir()
    if body is not None:
        (releases / f"{TAG}.md").write_bytes(body.encode("utf-8"))  # CRLF kept as given
    log = tmp_path / "gh.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_RELEASES": str(releases),
        "FAKE_GH_LOG": str(log),
        "FAKE_GENERATED": GENERATED_NOTES,
        "FAKE_VIEW_ERROR": view_error,
    }
    proc = subprocess.run(
        ["bash", str(RELEASE_SCRIPT), TAG],
        env=env,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    release = releases / f"{TAG}.md"
    after = (
        release.read_bytes().decode("utf-8") if release.exists() else None
    )  # no newline translation
    calls = log.read_text(encoding="utf-8").split("\n") if log.exists() else []
    return proc, after, calls


def _section_of(body: str) -> str:
    start = body.index(compose_hash.RELEASE_NOTES_HEADING)
    rest = body[start + len(compose_hash.RELEASE_NOTES_HEADING) :]
    end = rest.find("\n## ")
    if end < 0:
        return body[start:]
    return body[start : start + len(compose_hash.RELEASE_NOTES_HEADING) + end]


def test_release_script_appends_the_section_to_a_release_without_one(tmp_path):
    proc, after, calls = _run_release_script(tmp_path, GENERATED_NOTES)
    assert proc.returncode == 0, proc.stderr
    assert after.startswith(GENERATED_NOTES)
    assert _section_of(after).rstrip() == compose_hash.release_notes_section("prod")
    assert f"release edit {TAG}" in calls and not any(c.startswith("release create") for c in calls)


def test_release_script_creates_a_missing_release_with_generated_notes_then_the_section(tmp_path):
    proc, after, calls = _run_release_script(tmp_path, None)
    assert proc.returncode == 0, proc.stderr
    assert f"release create {TAG}" in calls and not any(c.startswith("release edit") for c in calls)
    assert after.startswith(GENERATED_NOTES)
    assert _section_of(after).rstrip() == compose_hash.release_notes_section("prod")


def test_release_script_leaves_a_release_that_already_carries_this_hash(tmp_path):
    body = (
        GENERATED_NOTES
        + "\r\n\r\n"
        + compose_hash.release_notes_section("prod").replace("\n", "\r\n")
    )
    proc, after, calls = _run_release_script(tmp_path, body)
    assert proc.returncode == 0, proc.stderr
    assert after == body, (
        "a release edited in the web UI (CRLF) with the right hash is not rewritten"
    )
    assert calls == [f"release view {TAG}", ""]


def test_release_script_replaces_a_stale_section_even_when_the_hash_appears_elsewhere(tmp_path):
    # the tag was moved to a tree with another hash: the old section must go, the job must not fail,
    # and the new hash quoted in the human-written notes above the section must not make the stale
    # section pass (the hash is searched only inside the section)
    current = compose_hash.compose_hash("prod")
    stale = compose_hash.release_notes_section("prod", "sha256:" + "0" * 64)
    assert current not in stale
    human = GENERATED_NOTES + f"\n\n## Notes\nthe new hash is {current}, see below\n"
    trailer = "## Thanks\n* everyone\n"
    proc, after, calls = _run_release_script(tmp_path, human + "\n" + stale + "\n\n" + trailer)
    assert proc.returncode == 0, proc.stderr
    assert "replacing it" in proc.stdout
    assert f"release edit {TAG}" in calls
    assert stale not in after
    assert after.startswith(human)
    assert trailer.rstrip() in after, (
        "text a human wrote after the section survives the replacement"
    )
    assert _section_of(after).rstrip() == compose_hash.release_notes_section("prod")
    assert after.count(compose_hash.RELEASE_NOTES_HEADING) == 1


def test_release_script_replaces_a_section_with_the_current_hash_but_a_stale_digest(tmp_path):
    # a section that quotes the current hash next to an old digest used to pass the hash grep and stay:
    # the provider then pins a digest the release does not ship. The whole section is compared now.
    current = compose_hash.release_notes_section("prod")
    stale_digest = current.replace(
        compose_hash.APPROVED_RUNNER_IMAGE_DIGEST, "sha256:" + "2" * 64, 1
    )
    assert compose_hash.compose_hash("prod") in stale_digest and stale_digest != current
    proc, after, calls = _run_release_script(tmp_path, GENERATED_NOTES + "\n\n" + stale_digest)
    assert proc.returncode == 0, proc.stderr
    assert "replacing it" in proc.stdout
    assert f"release edit {TAG}" in calls
    assert "2" * 64 not in after
    assert _section_of(after).rstrip() == current


def test_release_script_rewrites_a_doubled_section_to_one(tmp_path):
    # two copies of the section (a re-run that appended after a hand edit) both carry the current hash;
    # the provider reads two sets of instructions. One heading must come out of the job.
    current = compose_hash.release_notes_section("prod")
    proc, after, calls = _run_release_script(
        tmp_path, GENERATED_NOTES + "\n\n" + current + "\n\n" + current
    )
    assert proc.returncode == 0, proc.stderr
    assert "replacing it" in proc.stdout
    assert f"release edit {TAG}" in calls
    assert after.count(compose_hash.RELEASE_NOTES_HEADING) == 1
    assert after.startswith(GENERATED_NOTES)
    assert _section_of(after).rstrip() == current


def test_check_flags_an_app_compose_that_differs(tmp_path, capsys):
    good = tmp_path / "app-compose.json"
    good.write_text(compose_hash.measured_app_compose("prod"), encoding="utf-8")
    assert compose_hash.main(["--check", str(good)]) == 0
    assert capsys.readouterr().out.rstrip().endswith("OK")

    # the provider edited a measured file (host-setup.md says not to): one byte moves the hash
    edited = tmp_path / "edited-app-compose.json"
    edited.write_text(
        compose_hash.measured_app_compose("prod").replace("secure_time", "secure_tine"),
        encoding="utf-8",
    )
    assert compose_hash.main(["--check", str(edited)]) == 1
    assert "MISMATCH" in capsys.readouterr().err


def test_release_script_matches_a_heading_with_trailing_whitespace(tmp_path):
    # a hand edit that leaves a space after the heading is still the section: replaced, not doubled
    stale = compose_hash.release_notes_section("prod", "sha256:" + "1" * 64)
    proc, after, calls = _run_release_script(
        tmp_path, GENERATED_NOTES + "\n\n" + stale.replace("\n", " \n", 1)
    )
    assert proc.returncode == 0, proc.stderr
    assert "replacing it" in proc.stdout
    assert after.count(compose_hash.RELEASE_NOTES_HEADING) == 1
    assert _section_of(after).rstrip() == compose_hash.release_notes_section("prod")


def test_release_script_does_not_create_when_the_release_cannot_be_read(tmp_path):
    # auth / rate limit / 5xx on `gh release view` is not "no release": creating would collide with one that exists
    proc, after, calls = _run_release_script(
        tmp_path, GENERATED_NOTES, view_error="HTTP 502: bad gateway"
    )
    assert proc.returncode == 1
    assert "bad gateway" in proc.stderr and "re-run the job" in proc.stderr
    assert after == GENERATED_NOTES
    assert not any(c.startswith(("release create", "release edit", "api ")) for c in calls)
