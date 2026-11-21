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

    async def run(self, command, check=False, input=None):
        done = subprocess.run(command, shell=True, input=input, capture_output=True, text=True)
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
