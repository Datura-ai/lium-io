"""get_executor_pid_mode() in machine_scrape.py, extracted by ast like the other scrape helpers."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"


def _read(inspection):
    calls = []

    def docker_api_get(path):
        calls.append(path)
        return inspection

    ns = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", {"get_executor_pid_mode"}, {"socket": socket}
    )
    ns["docker_api_get"] = docker_api_get
    result = ns["get_executor_pid_mode"]()
    assert calls == [f"/containers/{socket.gethostname()}/json"]
    return result


@pytest.mark.parametrize("pid_mode", ["host", ""])
def test_reads_pid_mode(pid_mode):
    assert _read({"HostConfig": {"PidMode": pid_mode}}) == pid_mode


@pytest.mark.parametrize(
    "inspection",
    [{}, {"HostConfig": {}}, {"HostConfig": None}, {"HostConfig": {"PidMode": 1}}, [], None],
)
def test_missing_or_malformed_raises(inspection):
    # raising lands in data_executor_pid_mode_scrape_error: unknown, never a failing mode
    with pytest.raises(ValueError):
        _read(inspection)
