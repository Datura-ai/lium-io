"""The scrape reports the command line of GPU processes inside our filler containers only."""

from pathlib import Path

import pytest

from neurons.validators.tests.helpers import build_scrape_namespace

SCRAPE = Path(__file__).resolve().parents[1] / "src" / "miner_jobs" / "machine_scrape.py"


def _scrape(container_name, cmdline_read):
    reads = []

    def run_cmd(cmd):
        if "cmdline" in cmd:
            reads.append(cmd)
            return cmdline_read()
        return "0::/docker/abc\n"

    ns = build_scrape_namespace(SCRAPE, {"get_gpu_processes"}, {"run_cmd": run_cmd})
    [process] = ns["get_gpu_processes"]({7}, [{"each_container_id": "abc", "each_name": container_name}])
    return process, reads


def test_a_filler_gpu_process_carries_its_command_line():
    process, _ = _scrape("filler_x", lambda: "/usr/local/bin/peakminer --gpu 0 \n")

    assert process["processes_cmdline"] == "/usr/local/bin/peakminer --gpu 0"


@pytest.mark.parametrize("container_name", ["pod_abc", "executor", None])
def test_a_renter_pod_command_line_is_never_read(container_name):
    process, reads = _scrape(container_name, lambda: "python train.py --api-key SECRET")

    assert reads == []
    assert process["processes_cmdline"] == ""


def test_an_unreadable_command_line_is_empty_and_keeps_the_process():
    def gone():
        raise RuntimeError("gone")

    process, _ = _scrape("filler_x", gone)

    assert process == {
        "processes_pid": 7,
        "processes_info": "0::/docker/abc",
        "processes_container_name": "filler_x",
        "processes_cmdline": "",
    }
