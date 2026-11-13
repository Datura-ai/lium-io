"""get_executor_pid_mode() in machine_scrape.py, extracted by ast like the other scrape helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

OWN = os.getpid()


def _read(containers: dict[str, tuple[list[int] | None, object]]):
    def docker_api_get(path):
        if path == "/containers/json":
            return [{"Id": container_id} for container_id in containers]
        container_id = path.split("/")[2]
        pids, inspection = containers[container_id]
        if "/top" in path and pids is None:
            raise RuntimeError(f"docker api {path} returned HTTP 409")
        if "/top" in path:
            return {"Titles": ["PID"], "Processes": [[str(pid)] for pid in pids]}
        return inspection

    ns = build_scrape_namespace(SRC / "miner_jobs" / "machine_scrape.py", {"get_executor_pid_mode"}, {"os": os})
    ns["docker_api_get"] = docker_api_get
    return ns["get_executor_pid_mode"]()


def _mode(pid_mode: object) -> dict[str, object]:
    return {"HostConfig": {"PidMode": pid_mode}}


@pytest.mark.parametrize("pid_mode", ["host", ""])
def test_reads_the_pid_mode_of_the_container_listing_the_scrape_s_pid(pid_mode):
    assert _read({"burner": ([900], _mode("host")), "executor": ([812, OWN], _mode(pid_mode))}) == pid_mode


def test_a_container_named_by_the_hostname_but_not_listing_the_scrape_is_not_read():
    # DAH-6371: a provider can set the executor's hostname to a pid-host container's id
    assert _read({"pid-host-burner": ([900], _mode("host")), "executor": ([OWN], _mode(""))}) == ""


def test_no_container_listing_the_scrape_s_pid_is_a_private_pid_namespace():
    # in a private namespace this process's pid is not the host pid docker lists
    assert _read({"executor": ([812], _mode("host"))}) == ""


def test_a_container_whose_top_fails_is_skipped_not_turned_into_unknown():
    # a restarting (409) or removed (404) container must not make the whole reading unknown, which passes
    assert _read({"restarting": (None, _mode("host")), "executor": ([OWN], _mode(""))}) == ""


@pytest.mark.parametrize("host_config", [{}, None, {"PidMode": 1}])
def test_missing_or_malformed_raises(host_config):
    # raising lands in data_executor_pid_mode_scrape_error: unknown, never a failing mode
    with pytest.raises(ValueError):
        _read({"executor": ([OWN], {"HostConfig": host_config})})
