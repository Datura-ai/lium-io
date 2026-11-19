"""The scrape reports each GPU process's command line, for the filler GPU process allowlist."""

import os
import subprocess
from pathlib import Path

from neurons.validators.tests.helpers import build_scrape_namespace

SCRAPE = Path(__file__).resolve().parents[1] / "src" / "miner_jobs" / "machine_scrape.py"


def test_a_gpu_process_carries_its_command_line():
    ns = build_scrape_namespace(SCRAPE, {"get_gpu_processes", "run_cmd"}, {"os": os, "subprocess": subprocess})

    [process] = ns["get_gpu_processes"]({os.getpid()}, [])

    # NUL-separated argv, joined by spaces
    assert process["processes_cmdline"] == " ".join(
        Path(f"/proc/{os.getpid()}/cmdline").read_bytes().decode().split("\0")
    ).strip()[:512]


def test_an_unreadable_command_line_is_empty_and_keeps_the_process():
    def run_cmd(cmd):
        if "cmdline" in cmd:
            raise RuntimeError("gone")
        return "0::/docker/abc\n"

    ns = build_scrape_namespace(SCRAPE, {"get_gpu_processes"}, {"run_cmd": run_cmd})

    [process] = ns["get_gpu_processes"]({7}, [{"each_container_id": "abc", "each_name": "filler_x"}])

    assert process == {
        "processes_pid": 7,
        "processes_info": "0::/docker/abc",
        "processes_container_name": "filler_x",
        "processes_cmdline": "",
    }
