"""A GPU PID whose cgroup cannot be read must still be reported, unattributed (DAH-2735).

An unreadable PID used to be dropped, turning a foreign host workload into an empty
process list, which the usage gate reads as a clean card. A PID that is simply gone is
a different case: it died between the NVML snapshot and the /proc read, so it is dropped
rather than reported as a foreign process.

machine_scrape.py is a script, not a module — the helper is compiled out of the source
(see test_scrape_disk_breakdown.py for the pattern).
"""

import os
from pathlib import Path

import psutil


from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"
SCRAPE = SRC / "miner_jobs" / "machine_scrape.py"

CONTAINERS = [{"each_container_id": "abc123", "each_name": "filler_deadbeef"}]


def _namespace(run_cmd):
    return build_scrape_namespace(
        SCRAPE, {"get_gpu_processes"}, {"run_cmd": run_cmd, "psutil": psutil}
    )


def test_unreadable_cgroup_of_a_live_pid_keeps_it_unattributed():
    live_pid = os.getpid()

    def run_cmd(cmd: str) -> str:
        raise RuntimeError("Permission denied")

    processes = _namespace(run_cmd)["get_gpu_processes"]({live_pid}, CONTAINERS)

    assert processes == [
        {"processes_pid": live_pid, "processes_info": "", "processes_container_name": None}
    ]


def test_pid_that_died_before_the_read_is_dropped():
    # Raced with the NVML snapshot (e.g. a filler restarting) — not a foreign workload.
    def run_cmd(cmd: str) -> str:
        raise RuntimeError("No such process")

    processes = _namespace(run_cmd)["get_gpu_processes"]({2 ** 30}, CONTAINERS)

    assert processes == []


def test_readable_cgroup_still_maps_to_its_container():
    def run_cmd(cmd: str) -> str:
        return "0::/system.slice/docker-abc123.scope"

    processes = _namespace(run_cmd)["get_gpu_processes"]({111}, CONTAINERS)

    assert processes == [
        {
            "processes_pid": 111,
            "processes_info": "0::/system.slice/docker-abc123.scope",
            "processes_container_name": "filler_deadbeef",
        }
    ]
