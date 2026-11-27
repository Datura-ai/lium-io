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


def test_only_the_prod_build_turns_the_gate_on(tmp_path):
    """docker_build.sh passes PROTOCOL_RELEASE_GATE=1 for the prod (latest) image only."""
    rc, calls = _run_docker_build(tmp_path, "latest")
    assert rc == 0 and "--build-arg PROTOCOL_RELEASE_GATE=1 " in calls
    rc, calls = _run_docker_build(tmp_path, "dev")
    assert rc == 0 and "--build-arg PROTOCOL_RELEASE_GATE=0 " in calls
    prod = (_PATH.parents[2] / ".github" / "workflows" / "validator_cd_prod.yml").read_text()
    assert 'TAG: "latest"' in prod and "bash ./docker_build.sh" in prod


def _run_dockerfile_gate(tmp_path, gate_arg, validator, backend):
    """Runs the Dockerfile's own gate step, with the files it copies, the way the build does."""
    import shutil
    import subprocess

    dockerfile = (_PATH.parent / "Dockerfile").read_text()
    step = next(
        line for line in dockerfile.splitlines() if line.startswith("RUN ") and _PATH.name in line
    )
    shutil.copy(_PATH, tmp_path / _PATH.name)
    (tmp_path / "validator_protocol").write_text(validator)
    (tmp_path / "backend_protocol_in_prod").write_text(backend)
    env = {"PATH": "/usr/bin:/bin", "PROTOCOL_RELEASE_GATE": gate_arg}
    return subprocess.run(["sh", "-c", step.removeprefix("RUN ")], cwd=tmp_path, env=env).returncode


def test_the_dockerfile_build_fails_on_a_prod_build_ahead_of_the_backend(tmp_path):
    """The deployment's prod stack builds the Dockerfile directly, so the gate lives in it, not only in docker_build.sh."""
    assert "ARG PROTOCOL_RELEASE_GATE=0" in (_PATH.parent / "Dockerfile").read_text()
    assert _run_dockerfile_gate(tmp_path, "1", "2.20.0\n", "2.16.0\n") != 0
    assert _run_dockerfile_gate(tmp_path, "1", "2.16.0\n", "2.16.0\n") == 0
    assert _run_dockerfile_gate(tmp_path, "0", "2.20.0\n", "2.16.0\n") == 0
