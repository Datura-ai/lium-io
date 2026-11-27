import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "protocol_release_gate.py"
_spec = importlib.util.spec_from_file_location("protocol_release_gate", _PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_a_newer_minor_than_the_prod_backend_waits():
    assert "release the backend first" in gate.refusal("2.20.0", "2.16.0\n")


def test_same_minor_or_older_releases():
    assert gate.refusal("2.16.0", "2.16.0\n") is None
    assert gate.refusal("2.16.3", "2.16.0") is None
    assert gate.refusal("2.15.0", "2.16.0") is None


def test_the_pin_and_protocol_files_parse():
    assert gate.main() in (0, 1)


def test_the_validator_protocol_file_matches_lium_protocol():
    """The image build sees only this directory, so validator_protocol stands in for lium_protocol's version."""
    import re

    init = _PATH.parents[2] / "lium_protocol" / "lium_protocol" / "__init__.py"
    version = re.search(r'^PROTOCOL_VERSION = "([^"]+)"', init.read_text(), re.M).group(1)
    assert gate.VALIDATOR_PROTOCOL.read_text().strip() == version


def _run_docker_build(tmp_path, tag):
    import os
    import subprocess

    fake = tmp_path / "docker"
    fake.write_text('#!/bin/sh\necho "$@" >> "$DOCKER_CALLS"\n')
    fake.chmod(0o755)
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "TAG": tag,
        "DOCKER_CALLS": str(calls),
    }
    done = subprocess.run(
        ["bash", "docker_build.sh"], cwd=_PATH.parent, env=env, capture_output=True, text=True
    )
    return done.returncode, calls.read_text() if calls.exists() else ""


def test_only_the_prod_build_runs_the_check(tmp_path):
    """The prod (latest) build is held while the validator is ahead of the prod backend; any other build is not."""
    ahead = gate.refusal(gate.VALIDATOR_PROTOCOL.read_text(), gate.PIN.read_text()) is not None
    rc, calls = _run_docker_build(tmp_path, "latest")
    assert (rc != 0) == ahead
    assert ("build" in calls) == (not ahead)
    rc, calls = _run_docker_build(tmp_path, "dev")
    assert rc == 0 and "build" in calls
    # every other build of the Dockerfile (staging workflow, dev environments) does not run it
    assert "protocol_release_gate" not in (_PATH.parent / "Dockerfile").read_text()
    prod = (_PATH.parents[2] / ".github" / "workflows" / "validator_cd_prod.yml").read_text()
    assert 'TAG: "latest"' in prod and "bash ./docker_build.sh" in prod
