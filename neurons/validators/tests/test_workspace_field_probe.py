"""The workspace-field probe, run for real against the validator's storage models staged as on an executor."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest
from services.storage_operations import supports_workspace_field

from services import executor_payload


class _LocalSSH:
    """Runs the probe command locally, its stdin carrying the staged files."""

    def __init__(self, cwd=None):
        self.cwd = cwd

    async def run(self, command, check=False, input=None):
        done = subprocess.run(command, shell=True, input=input, capture_output=True, text=True, cwd=self.cwd)
        return SimpleNamespace(exit_status=done.returncode, stderr=done.stderr)


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(executor_payload, "SERVICE_DIR_PARENT", str(tmp_path))
    return tmp_path


def _probe(field, python=sys.executable):
    return asyncio.run(supports_workspace_field(_LocalSSH(), python, field))


@pytest.mark.parametrize("field", ["bootstrap", "exclude_patterns"])
def test_the_staged_models_know_the_fields_the_validator_sends(service, field):
    assert _probe(field) is True
    assert not list(service.iterdir())  # the staged copy goes with the probe


def test_an_unknown_field_is_reported_missing(service):
    assert _probe("no_such_field") is False


def test_a_probe_that_cannot_run_fails_rather_than_reading_as_missing(service):
    with pytest.raises(RuntimeError, match="exit 127"):
        _probe("bootstrap", python="/nonexistent/python")


def test_the_probe_imports_nothing_from_the_login_directory(service, tmp_path_factory):
    # modules in the session's working directory, under the names the probe and the staged models import
    cwd = tmp_path_factory.mktemp("login")
    marker = cwd / "imported"
    module = f"open({str(marker)!r}, 'a').write(__name__)\n"
    (cwd / "dataclasses.py").write_text(module)
    (cwd / "storage").mkdir()
    (cwd / "storage" / "__init__.py").write_text(module)
    (cwd / "storage" / "models.py").write_text(module + "class WorkspaceSpec:\n    __dataclass_fields__ = {}\n")

    assert asyncio.run(supports_workspace_field(_LocalSSH(cwd), sys.executable, "bootstrap")) is True
    assert not marker.exists()
