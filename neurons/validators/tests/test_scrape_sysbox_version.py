"""get_sysbox_version() in machine_scrape.py: the host's `sysbox-runc --version`, recorded as
specs.sysbox_version. Telemetry only (Conductor 3 Oct 2026 16:07Z): nothing scores or gates on it.
Sysbox absent or the command failing reads None and never fails the scrape.

machine_scrape.py is a script, not a module, so the helper is extracted by ast (helpers.build_scrape_namespace).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

SYSBOX_VERSION_OUTPUT = """sysbox-runc
\tedition: \tCommunity Edition (CE)
\tversion: \t0.6.4
\tcommit: \t0e1a8ab9e5e2d0bbbc1d0d4cb9a2cbbf9e07d1d6
\tbuilt at: \tThu Mar 14 01:29:46 UTC 2024
\tbuilt by: \tRodny Molina
\toci-specs: \t1.1.0+dev
"""


@pytest.fixture
def scrape() -> dict[str, Any]:
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {"SYSBOX_RUNC_CANDIDATES", "SYSBOX_VERSION_PATTERN", "get_sysbox_version"},
        {"re": re, "subprocess": subprocess},
    )
    namespace["HOST_ROOT_PREFIX"] = "/proc/1/root"
    return namespace


def fake_run(answers: dict[str, Any], calls: list[str]):
    def run(cmd, **_kwargs):
        calls.append(cmd[0])
        answer = answers.get(cmd[0], FileNotFoundError(cmd[0]))
        if isinstance(answer, Exception):
            raise answer
        return answer

    return run


def test_present_on_path(scrape):
    calls: list[str] = []
    scrape["subprocess"] = SimpleNamespace(
        PIPE=-1,
        run=fake_run({"sysbox-runc": SimpleNamespace(returncode=0, stdout=SYSBOX_VERSION_OUTPUT)}, calls),
    )
    assert scrape["get_sysbox_version"]() == "0.6.4"
    assert calls == ["sysbox-runc"]


def test_present_only_under_the_host_root(scrape):
    calls: list[str] = []
    scrape["subprocess"] = SimpleNamespace(
        PIPE=-1,
        run=fake_run({"/proc/1/root/usr/bin/sysbox-runc": SimpleNamespace(returncode=0, stdout=SYSBOX_VERSION_OUTPUT)}, calls),
    )
    assert scrape["get_sysbox_version"]() == "0.6.4"
    assert calls == ["sysbox-runc", "/proc/1/root/usr/bin/sysbox-runc"]


def test_absent_is_none(scrape):
    calls: list[str] = []
    scrape["subprocess"] = SimpleNamespace(PIPE=-1, run=fake_run({}, calls))
    assert scrape["get_sysbox_version"]() is None
    assert len(calls) == 3


@pytest.mark.parametrize(
    "answer",
    [
        SimpleNamespace(returncode=1, stdout=""),
        SimpleNamespace(returncode=0, stdout="unexpected output\n"),
        subprocess.TimeoutExpired("sysbox-runc", 10),
        PermissionError("denied"),
    ],
)
def test_command_error_is_none(scrape, answer):
    scrape["subprocess"] = SimpleNamespace(
        PIPE=-1, run=fake_run({candidate: answer for candidate in ("sysbox-runc", "/proc/1/root/usr/bin/sysbox-runc", "/proc/1/root/usr/local/bin/sysbox-runc")}, [])
    )
    assert scrape["get_sysbox_version"]() is None


def test_scrape_key_is_renamed_to_sysbox_version():
    service = (SRC / "services" / "file_encrypt_service.py").read_text()
    assert "'data_sysbox_version': \"sysbox_version\"" in service
