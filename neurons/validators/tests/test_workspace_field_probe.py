"""The workspace-field probe, run for real against fake executor source trees."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest
from services.storage_operations import supports_workspace_field


class _LocalSSH:
    """Runs the probe command locally with `src` on the path, as the executor's /root/app/src would be."""

    def __init__(self, src):
        self.src = src

    async def run(self, command, check=False):
        done = subprocess.run(
            command, shell=True, capture_output=True, text=True, env={"PYTHONPATH": str(self.src)}
        )
        return SimpleNamespace(exit_status=done.returncode, stderr=done.stderr)


def _probe(src, field="exclude_patterns"):
    return asyncio.run(supports_workspace_field(_LocalSSH(src), sys.executable, field))


def _models(tmp_path, body):
    (tmp_path / "storage").mkdir()
    (tmp_path / "storage" / "__init__.py").write_text("")
    (tmp_path / "storage" / "models.py").write_text(body)
    return tmp_path


def test_probe_reads_present_and_missing_field(tmp_path):
    src = _models(
        tmp_path,
        "from dataclasses import dataclass\n@dataclass\nclass WorkspaceSpec:\n    exclude_patterns: list\n",
    )
    assert _probe(src) is True
    assert _probe(src, "bootstrap") is False


def test_image_without_storage_package_is_an_old_image(tmp_path):
    assert _probe(tmp_path) is False


def test_other_import_error_still_fails_the_probe(tmp_path):
    src = _models(tmp_path, "import not_a_real_module_xyz\n")
    with pytest.raises(RuntimeError, match="exit 1"):
        _probe(src)
