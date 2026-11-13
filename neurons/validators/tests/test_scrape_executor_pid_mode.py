"""get_executor_pid_mode() in machine_scrape.py, extracted by ast like the other scrape helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

ANCESTORS = {4321, 4000, 812}


def _read(inspections: dict[str, object], ancestors=ANCESTORS):
    def docker_api_get(path):
        if path == "/containers/json":
            return [{"Id": container_id} for container_id in inspections]
        return inspections[path.split("/")[2]]

    ns = build_scrape_namespace(SRC / "miner_jobs" / "machine_scrape.py", {"get_executor_pid_mode"}, {})
    ns["docker_api_get"] = docker_api_get
    ns["own_ancestor_pids"] = lambda: set(ancestors)
    return ns["get_executor_pid_mode"]()


def _container(pid: int, pid_mode: object) -> dict[str, object]:
    return {"State": {"Pid": pid}, "HostConfig": {"PidMode": pid_mode}}


@pytest.mark.parametrize("pid_mode", ["host", ""])
def test_reads_the_pid_mode_of_the_container_the_scrape_runs_in(pid_mode):
    assert _read({"burner": _container(900, "host"), "executor": _container(812, pid_mode)}) == pid_mode


def test_a_container_named_by_the_hostname_but_not_an_ancestor_is_not_read():
    # DAH-6251: a provider can set the executor's hostname to a pid-host container's id
    assert _read({"pid-host-burner": _container(900, "host"), "executor": _container(812, "")}) == ""


def test_no_container_on_the_scrape_s_ancestry_is_a_private_pid_namespace():
    # in a private namespace this process sees no host pid, so no container's init is its ancestor
    assert _read({"executor": _container(812, "host")}, ancestors={57, 12}) == ""


@pytest.mark.parametrize("host_config", [{}, None, {"PidMode": 1}])
def test_missing_or_malformed_raises(host_config):
    # raising lands in data_executor_pid_mode_scrape_error: unknown, never a failing mode
    with pytest.raises(ValueError):
        _read({"executor": {"State": {"Pid": 812}, "HostConfig": host_config}})
