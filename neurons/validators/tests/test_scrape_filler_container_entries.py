"""DAH-2787 - see probe_filler_container_entries() in machine_scrape.py.

While an idle node earns the unrented incentive it runs Lium's own job in a filler container.
The probe reports two shapes of visit from the host, because one alone is not enough:

- `docker_exec` - the docker daemon's own event log, which keeps finished visits. A provider guard
  script execs for milliseconds, so a live scan almost never meets one.
- `open_session` - a process in the container's PID namespace whose parent is outside it. This is
  the only way to see an `nsenter`, which never reaches the daemon.

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
    "ENTRY_COMMAND_MAX_CHARS",
    "MAX_REPORTED_ENTRIES",
    "EXEC_EVENT_WINDOW_SECONDS",
    "EXEC_CREATE_STATUS_PREFIX",
    "FillerEntryProbe",
    "read_pid_namespace",
    "read_process_parent_and_start_ticks",
    "read_process_command",
    "read_host_uptime_seconds",
    "read_processes_by_pid_namespace",
    "find_open_sessions",
    "find_docker_exec_events",
    "seconds_after_container_start",
    "RunningFillerContainers",
    "read_running_filler_containers",
    "find_filler_container_entries",
    "probe_filler_container_entries",
}

FILLER_NAMESPACE = "pid:[4026532817]"
HOST_NAMESPACE = "pid:[4026531836]"
BOOT_TIME_UNIX = 1_700_000_000.0
UPTIME_SECONDS = 100_000.0
NOW_UNIX = BOOT_TIME_UNIX + UPTIME_SECONDS
CLOCK_TICKS_PER_SECOND = os.sysconf("SC_CLK_TCK")

FILLER_INIT_PID = 100
CONTAINER_AGE = 3600.0  # the filler has been running for an hour
FILLER_CONTAINER_ID = "d3adb33f"
FILLER_CONTAINER_NAME = "filler_9f2c"


class _FakePsutil:
    @staticmethod
    def boot_time() -> float:
        return BOOT_TIME_UNIX


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
    _write_process(
        fake_proc, FILLER_INIT_PID, 90, FILLER_NAMESPACE, CONTAINER_AGE, "/opt/pearl/start.sh"
    )
    _write_process(fake_proc, 101, FILLER_INIT_PID, FILLER_NAMESPACE, CONTAINER_AGE - 1, "peakminer")
    _write_process(fake_proc, 90, 1, HOST_NAMESPACE, CONTAINER_AGE + 1, "containerd-shim")
    # the scrape itself: the executor container runs with pid: host, so it is in the host namespace
    _write_process(fake_proc, os.getpid(), 1, HOST_NAMESPACE, 10, "machine_scrape")
    return fake_proc


def _docker_api(container_name: str = FILLER_CONTAINER_NAME, init_pid: int = FILLER_INIT_PID):
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


def _exec_event(
    container_name: str = FILLER_CONTAINER_NAME,
    command: str = "pgrep -x peakminer",
    seconds_ago: float = 60.0,
    status: str | None = None,
) -> dict[str, Any]:
    return {
        "Type": "container",
        "status": status if status is not None else f"exec_create: {command}",
        "time": int(NOW_UNIX - seconds_ago),
        "Actor": {"ID": FILLER_CONTAINER_ID, "Attributes": {"name": container_name}},
    }


def _events_api(events: list[dict[str, Any]] | None = None):
    def docker_api_get_events(path: str) -> list[dict[str, Any]]:
        assert path.startswith("/events?since=")
        return events or []

    return docker_api_get_events


def _build_probe(
    proc_dir: Path,
    docker_api_get: Any = None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        PROBE_HELPERS,
        {
            "os": os,
            "psutil": _FakePsutil,
            "docker_api_get": docker_api_get or _docker_api(),
            "docker_api_get_events": _events_api(events),
        },
    )
    namespace["PROC_DIR"] = str(proc_dir)
    return namespace


@pytest.fixture
def scrape(proc_dir: Path) -> dict[str, Any]:
    """The probe helpers, executed in a namespace of their own, reading the fake /proc."""
    return _build_probe(proc_dir)


def _entries(scrape: dict[str, Any]) -> list[dict[str, Any]]:
    return scrape["probe_filler_container_entries"]().entries


def test_the_containers_own_work_is_not_reported(scrape: dict[str, Any]) -> None:
    # Arrange - a filler running only what it started itself, nobody at the door

    # Act
    entries = _entries(scrape)

    # Assert
    assert entries == []


def test_a_session_opened_from_the_host_is_reported(scrape: dict[str, Any], proc_dir: Path) -> None:
    # Arrange - `nsenter` or `docker exec -it`: inside the container's PID namespace, parented
    # to a process outside it, and still running when the scrape passes by
    _write_process(proc_dir, 200, 90, FILLER_NAMESPACE, 42, "bash")

    # Act
    entries = _entries(scrape)

    # Assert
    assert len(entries) == 1
    assert entries[0]["entry_kind"] == "open_session"
    assert entries[0]["entry_pid"] == 200
    assert entries[0]["entry_container"] == FILLER_CONTAINER_NAME
    assert entries[0]["entry_command"] == "bash"
    assert entries[0]["entry_seconds_after_start"] == pytest.approx(CONTAINER_AGE - 42, abs=1)


def test_children_of_a_session_are_not_reported_again(
    scrape: dict[str, Any], proc_dir: Path
) -> None:
    # Arrange - one session and the command it runs; the session alone is the visit
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


def test_a_finished_docker_exec_is_reported(proc_dir: Path) -> None:
    # Arrange - the real attack: a timer execs into our filler every few minutes and each visit is
    # over long before the scrape runs (prod 109.174.15.2, pearl-watchdog.sh)
    scrape = _build_probe(proc_dir, events=[_exec_event()])

    # Act
    entries = _entries(scrape)

    # Assert
    assert len(entries) == 1
    assert entries[0]["entry_kind"] == "docker_exec"
    assert entries[0]["entry_pid"] is None
    assert entries[0]["entry_container"] == FILLER_CONTAINER_NAME
    assert entries[0]["entry_command"] == "pgrep -x peakminer"
    assert entries[0]["entry_seconds_after_start"] == pytest.approx(CONTAINER_AGE - 60, abs=1)


def test_exec_events_of_other_containers_are_ignored(proc_dir: Path) -> None:
    # Arrange - a customer execs into their own rental pod, which is none of our business
    scrape = _build_probe(proc_dir, events=[_exec_event(container_name="pod_1234")])

    # Act / Assert
    assert _entries(scrape) == []


@pytest.mark.parametrize(
    "status",
    [
        "exec_start: pgrep -x peakminer",  # the create event already carries this visit
        "exec_die",
        "start",
        "health_status: healthy",
    ],
)
def test_only_the_exec_create_event_counts(proc_dir: Path, status: str) -> None:
    # One visit must produce one entry, and a container's own lifecycle is not a visit at all.
    scrape = _build_probe(proc_dir, events=[_exec_event(status=status)])

    assert _entries(scrape) == []


def test_the_newest_visits_are_kept_when_a_guard_script_floods_the_window(proc_dir: Path) -> None:
    # Arrange - a timer that execs every few minutes fills a whole window with the same visit
    events = [_exec_event(seconds_ago=float(60 * minute)) for minute in range(1, 40)]
    scrape = _build_probe(proc_dir, events=events)

    # Act
    entries = _entries(scrape)

    # Assert - capped, and the newest visit survives the cap
    assert len(entries) == scrape["MAX_REPORTED_ENTRIES"]
    assert entries[0]["entry_seconds_after_start"] == pytest.approx(CONTAINER_AGE - 60, abs=1)


def test_a_process_that_exits_during_the_scan_is_skipped(
    scrape: dict[str, Any], proc_dir: Path
) -> None:
    # Arrange - /proc is a moving target: a pid can vanish between listdir and read
    (proc_dir / "999").mkdir()

    # Act
    entries = _entries(scrape)

    # Assert - no entry, no exception
    assert entries == []


def test_a_filler_sharing_the_host_namespace_reports_no_session(proc_dir: Path) -> None:
    # A container started with --pid=host has no namespace of its own: every host process reads
    # as a member and their parentage tells nothing apart. Report none rather than the host.
    scrape = _build_probe(proc_dir, docker_api_get=_docker_api(init_pid=90))

    probe = scrape["probe_filler_container_entries"]()

    assert probe.entries == []
    assert probe.scrape_error == ""


def test_only_filler_containers_are_scanned(proc_dir: Path) -> None:
    # Arrange - the same processes and the same visit, but the container is a customer rental
    scrape = _build_probe(
        proc_dir,
        docker_api_get=_docker_api(container_name="pod_9f2c"),
        events=[_exec_event(container_name="pod_9f2c")],
    )
    _write_process(proc_dir, 200, 90, FILLER_NAMESPACE, 42, "bash")

    # Act
    probe = scrape["probe_filler_container_entries"]()

    # Assert
    assert probe.entries == []
    assert probe.scrape_error == ""


def test_a_stopped_filler_container_is_skipped(proc_dir: Path) -> None:
    # Arrange - a container that is not running reports pid 0
    scrape = _build_probe(proc_dir, docker_api_get=_docker_api(init_pid=0))

    # Act
    probe = scrape["probe_filler_container_entries"]()

    # Assert
    assert probe.entries == []
    assert probe.scrape_error == ""


def test_a_broken_docker_api_reports_no_entries(proc_dir: Path) -> None:
    # Fail open: the validator must not read a failed probe as "somebody was inside"
    def failing_docker_api_get(path: str) -> Any:
        raise RuntimeError("docker api /containers/json returned HTTP 500")

    scrape = _build_probe(proc_dir, docker_api_get=failing_docker_api_get)

    # Act
    probe = scrape["probe_filler_container_entries"]()

    # Assert
    assert probe.entries == []
    assert "HTTP 500" in probe.scrape_error


def test_a_broken_event_log_still_reports_open_sessions(proc_dir: Path) -> None:
    # Arrange - the daemon refuses the event log, the live scan still works
    def failing_events(path: str) -> list[dict[str, Any]]:
        raise RuntimeError("docker api /events returned HTTP 500")

    scrape = _build_probe(proc_dir)
    scrape["docker_api_get_events"] = failing_events
    _write_process(proc_dir, 200, 90, FILLER_NAMESPACE, 42, "bash")

    # Act
    probe = scrape["probe_filler_container_entries"]()

    # Assert
    assert [entry["entry_kind"] for entry in probe.entries] == ["open_session"]
    assert "docker events" in probe.scrape_error


def test_the_probe_never_raises(proc_dir: Path) -> None:
    # get_machine_specs has no guard around it: a probe that raises kills the whole scrape, and
    # the miner then has no specs and scores 0 on every executor.
    def unreadable_proc() -> dict[str, Any]:
        raise OSError("/proc is mounted with hidepid=2")

    scrape = _build_probe(proc_dir)
    scrape["read_processes_by_pid_namespace"] = unreadable_proc

    probe = scrape["probe_filler_container_entries"]()

    assert probe.entries == []
    assert "hidepid=2" in probe.scrape_error


def test_probe_keys_are_wired_through_both_obfuscation_tables() -> None:
    """A key missing from either table ships un-renamed/un-mapped - keep all of them in both."""
    # Arrange
    service_module = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    scrape_text = (SRC / "miner_jobs" / "machine_scrape.py").read_text()
    new_keys = [
        "data_filler_entries",
        "data_filler_entry_scrape_error",
        "entry_container",
        "entry_kind",
        "entry_pid",
        "entry_seconds_after_start",
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
