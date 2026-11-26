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


def test_every_image_build_runs_the_check_unless_it_opts_out():
    dockerfile = (_PATH.parent / "Dockerfile").read_text()
    # a build that passes nothing, as prod's does, is checked
    assert "ARG ALLOW_PROTOCOL_AHEAD=0\n" in dockerfile
    assert 'RUN [ "$ALLOW_PROTOCOL_AHEAD" = 1 ] || python3 protocol_release_gate.py' in dockerfile
    assert "protocol_release_gate.py" not in (_PATH.parent / ".dockerignore").read_text()


def test_only_the_prod_build_is_held_by_the_check():
    """Staging and dev run ahead of the prod backend; prod (latest) passes nothing and is checked."""
    repo = _PATH.parents[2]
    staging = (repo / ".github" / "workflows" / "validator_cd_staging.yml").read_text()
    assert "context: ./neurons/validators" in staging
    assert "ALLOW_PROTOCOL_AHEAD=1" in staging
    build = (_PATH.parent / "docker_build.sh").read_text()
    assert 'if [ "$TAG" != "latest" ]; then\n  ALLOW_PROTOCOL_AHEAD=1' in build
    assert "ALLOW_PROTOCOL_AHEAD" not in (repo / ".github" / "workflows" / "validator_cd_prod.yml").read_text()
