"""DAH-2787 - see probe_filler_container_entries() in machine_scrape.py.

While an idle node earns the unrented incentive it runs Lium's own job in a filler container.
A process that starts inside that container from the host (`docker exec`, `nsenter`) joins the
container's PID namespace while its parent stays outside it - that pair is the whole signal.

machine_scrape.py is a script, not a module - importing it runs the whole scrape - so the helpers
are extracted by ast and executed in their own namespace (same pattern as
test_scrape_gpu_power_cap_probe.py).
"""

import ast
import os
from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace, dict_literal_keys

SRC = Path(__file__).resolve().parents[1] / "src"

PROBE_HELPERS = {
    "PROC_DIR",
    "FILLER_CONTAINER_NAME_PREFIX",
    "ENTRY_COMMAND_LIMIT",
    "FillerEntryProbe",
    "read_pid_namespace",
    "read_process_parent_and_start_ticks",
    "read_process_command",
    "read_host_uptime_seconds",
    "find_processes_entered_from_outside",
    "probe_filler_container_entries",
}

FILLER_NAMESPACE = "pid:[4026532817]"
HOST_NAMESPACE = "pid:[4026531836]"
UPTIME_SECONDS = 1000.0
CLOCK_TICKS_PER_SECOND = os.sysconf("SC_CLK_TCK")

FILLER_INIT_PID = 100
FILLER_CONTAINER_ID = "d3adb33f"
FILLER_CONTAINER_NAME = "filler_9f2c"


def _write_process(
    proc_dir: Path, pid: int, parent_pid: int, namespace: str, age_seconds: float, command: str
) -> None:
    process_dir = proc_dir / str(pid)
    (process_dir / "ns").mkdir(parents=True)
    (process_dir / "ns" / "pid").symlink_to(namespace)  # readlink gives back this exact string
    start_ticks = int((UPTIME_SECONDS - age_seconds) * CLOCK_TICKS_PER_SECOND)
    # /proc/<pid>/stat: pid, (comm), state, ppid, then 17 fields before start time
    fields_between_parent_and_start = " ".join(["0"] * 17)
    (process_dir / "stat").write_text(
        f"{pid} (a name with spaces) S {parent_pid} {fields_between_parent_and_start} {start_ticks} 0 0\n"
    )
    (process_dir / "cmdline").write_bytes(command.encode() + b"\0")


@pytest.fixture
def proc_dir(tmp_path: Path) -> Path:
    """A fake /proc holding a filler container: its init, one child of its own, nothing else."""
    fake_proc = tmp_path / "proc"
    fake_proc.mkdir()
    (fake_proc / "uptime").write_text(f"{UPTIME_SECONDS} 500.00\n")
    _write_process(fake_proc, FILLER_INIT_PID, 90, FILLER_NAMESPACE, 3600, "/opt/pearl/start.sh")
    _write_process(fake_proc, 101, FILLER_INIT_PID, FILLER_NAMESPACE, 3599, "pearl-miner")
    _write_process(fake_proc, 90, 1, HOST_NAMESPACE, 3601, "containerd-shim")
    # the scrape itself: the executor container runs with pid: host, so it is in the host namespace
    _write_process(fake_proc, os.getpid(), 1, HOST_NAMESPACE, 10, "machine_scrape")
    return fake_proc


@pytest.fixture
def scrape(proc_dir: Path) -> dict[str, Any]:
    """The probe helpers, executed in a namespace of their own, reading the fake /proc."""
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        PROBE_HELPERS,
        {"os": os, "docker_api_get": _docker_api(FILLER_CONTAINER_NAME)},
    )
    namespace["PROC_DIR"] = str(proc_dir)
    return namespace


def _docker_api(container_name: str, init_pid: int = FILLER_INIT_PID):
    def docker_api_get(path: str) -> Any:
        if path == "/containers/json":
            return [
                {"Id": FILLER_CONTAINER_ID, "Names": [f"/{container_name}"]},
                {"Id": "cafe", "Names": ["/pod_1234"]},
            ]
        if path == f"/containers/{FILLER_CONTAINER_ID}/json":
            return {"State": {"Pid": init_pid}}
        return {"State": {"Pid": 555}}

    return docker_api_get


def _entries(scrape: dict[str, Any]) -> list[dict[str, Any]]:
    return scrape["probe_filler_container_entries"]().entries


def test_the_containers_own_processes_are_not_reported(scrape: dict[str, Any]) -> None:
    # Arrange - a filler running only what it started itself

    # Act
    entries = _entries(scrape)

    # Assert
    assert entries == []


def test_a_session_opened_from_the_host_is_reported(scrape: dict[str, Any], proc_dir: Path) -> None:
    # Arrange - `docker exec`: inside the container's PID namespace, parented to the host shim
    _write_process(proc_dir, 200, 90, FILLER_NAMESPACE, 42, "bash")

    # Act
    entries = _entries(scrape)

    # Assert
    assert len(entries) == 1
    assert entries[0]["entry_pid"] == 200
    assert entries[0]["entry_parent_pid"] == 90
    assert entries[0]["entry_container"] == FILLER_CONTAINER_NAME
    assert entries[0]["entry_command"] == "bash"
    assert entries[0]["entry_age_seconds"] == pytest.approx(42, abs=1)


def test_children_of_a_session_are_not_reported_again(
    scrape: dict[str, Any], proc_dir: Path
) -> None:
    # Arrange - one session and the command it runs; the session alone is the entry
    _write_process(proc_dir, 200, 90, FILLER_NAMESPACE, 42, "bash")
    _write_process(proc_dir, 201, 200, FILLER_NAMESPACE, 40, "cat /root/.bittensor/wallet")

    # Act
    entries = _entries(scrape)

    # Assert
    assert [entry["entry_pid"] for entry in entries] == [200]


def test_processes_of_another_container_are_not_reported(
    scrape: dict[str, Any], proc_dir: Path
) -> None:
    # Arrange - a rental pod the provider may enter freely: its own namespace, its own shim
    _write_process(proc_dir, 300, 91, "pid:[4026532999]", 42, "bash")

    # Act
    entries = _entries(scrape)

    # Assert
    assert entries == []


def test_a_filler_sharing_the_host_namespace_reports_nothing(proc_dir: Path) -> None:
    # A container started with --pid=host has no namespace of its own: every host process reads
    # as a member and their parentage tells nothing apart. Report none rather than the host.
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        PROBE_HELPERS,
        {"os": os, "docker_api_get": _docker_api(FILLER_CONTAINER_NAME, init_pid=90)},
    )
    namespace["PROC_DIR"] = str(proc_dir)

    probe = namespace["probe_filler_container_entries"]()

    assert probe.entries == []
    assert probe.scrape_error == ""


def test_a_process_that_exits_during_the_scan_is_skipped(
    scrape: dict[str, Any], proc_dir: Path
) -> None:
    # Arrange - /proc is a moving target: a pid can vanish between listdir and read
    (proc_dir / "999").mkdir()

    # Act
    entries = _entries(scrape)

    # Assert - no entry, no exception
    assert entries == []


def test_only_filler_containers_are_scanned(proc_dir: Path) -> None:
    # Arrange - the same processes, but the container is a customer rental
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        PROBE_HELPERS,
        {"os": os, "docker_api_get": _docker_api("pod_9f2c")},
    )
    namespace["PROC_DIR"] = str(proc_dir)
    _write_process(proc_dir, 200, 90, FILLER_NAMESPACE, 42, "bash")

    # Act
    probe = namespace["probe_filler_container_entries"]()

    # Assert
    assert probe.entries == []
    assert probe.scrape_error == ""


def test_a_stopped_filler_container_is_skipped(proc_dir: Path) -> None:
    # Arrange - a container that is not running reports pid 0
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        PROBE_HELPERS,
        {"os": os, "docker_api_get": _docker_api(FILLER_CONTAINER_NAME, init_pid=0)},
    )
    namespace["PROC_DIR"] = str(proc_dir)

    # Act
    probe = namespace["probe_filler_container_entries"]()

    # Assert
    assert probe.entries == []
    assert probe.scrape_error == ""


def test_a_broken_docker_api_reports_no_entries(proc_dir: Path) -> None:
    # Fail open: the validator must not read a failed probe as "somebody was inside"
    def failing_docker_api_get(path: str) -> Any:
        raise RuntimeError("docker api /containers/json returned HTTP 500")

    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        PROBE_HELPERS,
        {"os": os, "docker_api_get": failing_docker_api_get},
    )
    namespace["PROC_DIR"] = str(proc_dir)

    # Act
    probe = namespace["probe_filler_container_entries"]()

    # Assert
    assert probe.entries == []
    assert "HTTP 500" in probe.scrape_error


def test_probe_keys_are_wired_through_both_obfuscation_tables() -> None:
    """A key missing from either table ships un-renamed/un-mapped - keep all of them in both."""
    # Arrange
    service_module = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    scrape_text = (SRC / "miner_jobs" / "machine_scrape.py").read_text()
    new_keys = [
        "data_filler_entries",
        "data_filler_entry_scrape_error",
        "entry_container",
        "entry_pid",
        "entry_parent_pid",
        "entry_age_seconds",
        "entry_command",
    ]

    # Act
    original_keys = dict_literal_keys(service_module, "ORIGINAL_KEYS")
    all_keys = dict_literal_keys(service_module, "all_keys")

    # Assert
    for key in new_keys:
        assert f'"{key}"' in scrape_text or f"'{key}'" in scrape_text
        assert key in original_keys
        assert key in all_keys
