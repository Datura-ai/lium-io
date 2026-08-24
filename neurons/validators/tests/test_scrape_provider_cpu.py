"""DAH-2734: the scrape's CPU attribution must void itself rather than blame the provider.

The host reading only means something when every container that burned CPU in the SAME window is
subtracted from it. Two things break that pairing on a real host: a container that exits between
`docker stats` and `docker ps`, and a docker daemon too wedged to answer `stats` at all. Both must
end with no host reading in the payload, which makes the gate report NOT_MEASURABLE and withhold
no money.

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the helpers
are compiled out of the source and executed on their own, like test_scrape_disk_breakdown.py does.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"

CPU_HELPERS = {"get_container_cpu_percents", "get_docker_info"}

RENTER = "aaaaaaaaaaaa" + "0" * 52
EXECUTOR = "bbbbbbbbbbbb" + "0" * 52


class _Psutil:
    """`cpu_percent(interval=None)` returns the load since the previous call — the reset returns 0."""

    def __init__(self, host_percent: float) -> None:
        self._readings = [0.0, host_percent]

    def cpu_percent(self, interval=None) -> float:
        return self._readings.pop(0)


def _fake_run_cmd(outputs: dict[str, str]):
    """Match a docker call by the subcommand in it, so the temp binary path does not matter."""

    def run_cmd(cmd: str) -> str:
        for marker, output in outputs.items():
            if marker in cmd:
                if isinstance(output, Exception):
                    raise output
                return output
        return ""

    return run_cmd


def _scrape(outputs: dict[str, str], host_percent: float = 40.0) -> dict:
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        CPU_HELPERS,
        {"json": json, "os": os, "tempfile": tempfile, "psutil": _Psutil(host_percent)},
    )
    namespace["run_cmd"] = _fake_run_cmd(outputs)
    return namespace


def _docker_outputs(stats: str | Exception, ps_ids: list[str]) -> dict:
    return {
        "version --format": "24.0.7\n",
        "stats --no-stream": stats,
        "ps --no-trunc": "\n".join(ps_ids) + "\n",
        "{{.Image}}": "sha256:image\n",
        "json .RepoDigests": '["daturaai/pod@sha256:digest"]\n',
        "{{.Name}}": "/pod_renter\n",
    }


def test_container_cpu_is_attached_and_the_host_reading_is_kept():
    # Arrange — one container, one stats row for it
    scrape = _scrape(_docker_outputs(f"{RENTER[:12]}|158.00%\n", [RENTER]), host_percent=40.0)

    # Act
    data = scrape["get_docker_info"](b"#!/bin/sh\n")

    # Assert
    assert data["docker_host_cpu_percent"] == 40.0
    assert data["docker_containers"][0]["each_cpu_percent"] == 158.0


def test_a_container_that_exits_between_stats_and_ps_voids_the_host_reading():
    # Arrange — stats saw two containers, ps lists only one. The missing container's CPU sits
    # inside the host reading with nothing to subtract it, so the whole reading has to go.
    stats = f"{RENTER[:12]}|158.00%\n{EXECUTOR[:12]}|12.00%\n"
    scrape = _scrape(_docker_outputs(stats, [RENTER]))

    # Act
    data = scrape["get_docker_info"](b"#!/bin/sh\n")

    # Assert
    assert "docker_host_cpu_percent" not in data
    assert data["docker_containers"][0]["each_cpu_percent"] == 158.0


def test_a_wedged_docker_daemon_voids_the_reading_and_keeps_the_scrape_alive():
    # Arrange — `timeout 30 docker stats` returns non-zero, so run_cmd raises
    outputs = _docker_outputs(RuntimeError("run_cmd error: timeout"), [RENTER])
    scrape = _scrape(outputs)

    # Act
    data = scrape["get_docker_info"](b"#!/bin/sh\n")

    # Assert — the scrape is fatal for the executor, so it must survive with the CPU part missing
    assert "docker_host_cpu_percent" not in data
    assert "each_cpu_percent" not in data["docker_containers"][0]
    assert data["docker_version"] == "24.0.7"


def test_an_unparsable_stats_row_raises_so_the_caller_voids_the_reading():
    # Arrange — a row the window cannot account for must not be dropped silently
    scrape = _scrape(_docker_outputs(f"{RENTER[:12]}|--\n", [RENTER]))

    # Act / Assert
    with pytest.raises(ValueError):
        scrape["get_container_cpu_percents"]("/tmp/docker")
