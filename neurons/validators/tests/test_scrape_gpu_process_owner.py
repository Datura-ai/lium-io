"""A GPU PID whose cgroup cannot be read must still be reported, unattributed (DAH-2735).

The scrape runs inside the executor container; NVML returns host-namespace PIDs that
/proc cannot resolve there. Dropping them silently turned a foreign host workload into
an empty process list, which the usage gate reads as a clean card.

machine_scrape.py is a script, not a module — the helper is compiled out of the source
(see test_scrape_disk_breakdown.py for the pattern).
"""

from pathlib import Path

from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"
SCRAPE = SRC / "miner_jobs" / "machine_scrape.py"

CONTAINERS = [{"each_container_id": "abc123", "each_name": "filler_deadbeef"}]


def _namespace(run_cmd):
    return build_scrape_namespace(SCRAPE, {"get_gpu_processes"}, {"run_cmd": run_cmd})


def test_unreadable_cgroup_keeps_the_pid_unattributed():
    def run_cmd(cmd: str) -> str:
        raise RuntimeError("No such process")

    processes = _namespace(run_cmd)["get_gpu_processes"]({2844137}, CONTAINERS)

    assert processes == [
        {"processes_pid": 2844137, "processes_info": "", "processes_container_name": None}
    ]


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
