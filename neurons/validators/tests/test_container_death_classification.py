"""DAH-2439 follow-up: classify WHY a filler container died.

Only an external kill (container removed, or SIGKILL/SIGTERM-stopped) may cost a provider
incentive. A filler that crashed on its own, never started, was OOM-killed or exited cleanly
is self-heal territory (DAH-2419) and must never be punished. These tests pin the classifier.
"""

from core.docker_utils import (
    ContainerDeathDiagnostics,
    ContainerDeathKind,
    classify_container_death,
    container_uptime_seconds,
)


def test_removed_container_classifies_as_removed():
    diagnostics = ContainerDeathDiagnostics(
        capture_error="inspect: error: no such object: filler_x",
        logs_tail="Error response from daemon: No such container: filler_x",
    )
    assert classify_container_death(diagnostics) is ContainerDeathKind.REMOVED


def test_removed_marker_in_logs_tail_only_does_not_classify_as_removed():
    # The container still exists (a real stop, exit 137) but its own logs printed the docker
    # "no such container" string — must NOT be read as an external removal and punished.
    diagnostics = ContainerDeathDiagnostics(
        status="exited",
        exit_code=137,
        oom_killed=False,
        logs_tail="Error response from daemon: No such container: some-other-name",
    )
    assert classify_container_death(diagnostics) is ContainerDeathKind.STOPPED


def test_sigkill_exit_137_classifies_as_stopped():
    diagnostics = ContainerDeathDiagnostics(status="exited", exit_code=137, oom_killed=False)
    assert classify_container_death(diagnostics) is ContainerDeathKind.STOPPED


def test_sigterm_exit_143_classifies_as_stopped():
    diagnostics = ContainerDeathDiagnostics(status="exited", exit_code=143, oom_killed=False)
    assert classify_container_death(diagnostics) is ContainerDeathKind.STOPPED


def test_oom_kill_beats_exit_code():
    # OOM also reports 137 — the kernel killed it, not the owner.
    diagnostics = ContainerDeathDiagnostics(status="exited", exit_code=137, oom_killed=True)
    assert classify_container_death(diagnostics) is ContainerDeathKind.OOM_KILLED


def test_nonzero_exit_classifies_as_self_crashed():
    diagnostics = ContainerDeathDiagnostics(status="exited", exit_code=1, oom_killed=False)
    assert classify_container_death(diagnostics) is ContainerDeathKind.SELF_CRASHED


def test_zero_exit_classifies_as_clean_exit():
    diagnostics = ContainerDeathDiagnostics(status="exited", exit_code=0, oom_killed=False)
    assert classify_container_death(diagnostics) is ContainerDeathKind.CLEAN_EXIT


def test_created_but_never_run_classifies_as_never_started():
    diagnostics = ContainerDeathDiagnostics(status="created", exit_code=0)
    assert classify_container_death(diagnostics) is ContainerDeathKind.NEVER_STARTED


def test_zero_started_at_classifies_as_never_started():
    diagnostics = ContainerDeathDiagnostics(
        status="exited", exit_code=128, started_at="0001-01-01T00:00:00Z"
    )
    assert classify_container_death(diagnostics) is ContainerDeathKind.NEVER_STARTED


def test_empty_diagnostics_classifies_as_unknown():
    assert classify_container_death(ContainerDeathDiagnostics()) is ContainerDeathKind.UNKNOWN


def test_capture_failure_without_removal_marker_is_unknown():
    diagnostics = ContainerDeathDiagnostics(capture_error="inspect: timed out")
    assert classify_container_death(diagnostics) is ContainerDeathKind.UNKNOWN


def test_uptime_from_docker_timestamps():
    seconds = container_uptime_seconds(
        started_at="2026-07-20T06:00:00.123456789Z",
        finished_at="2026-07-20T06:05:30.500000000Z",
    )
    assert seconds is not None
    assert abs(seconds - 330.4) < 1.0


def test_uptime_is_none_when_timestamps_missing():
    assert container_uptime_seconds(started_at=None, finished_at="2026-07-20T06:05:30Z") is None
    assert container_uptime_seconds(started_at="0001-01-01T00:00:00Z", finished_at="2026-07-20T06:05:30Z") is None


def test_removed_capitalized_docker_casing_still_removed():
    diagnostics = ContainerDeathDiagnostics(
        capture_error="inspect: Error: No such object: filler_x",
    )
    assert classify_container_death(diagnostics) is ContainerDeathKind.REMOVED


def test_sigterm_coinciding_with_executor_restart_is_host_reboot():
    # executor stack (re)started right around the filler's death -> collateral of a reboot/restart.
    diagnostics = ContainerDeathDiagnostics(
        status="exited", exit_code=143, oom_killed=False,
        finished_at="2026-07-20T09:00:00Z",
        host_context={"executor_container_started_at": "2026-07-20T09:00:20Z"},
    )
    assert classify_container_death(diagnostics) is ContainerDeathKind.HOST_REBOOT


def test_sigterm_with_old_executor_start_is_targeted_stop():
    # executor up for days, only the filler was stopped -> targeted external stop.
    diagnostics = ContainerDeathDiagnostics(
        status="exited", exit_code=143, oom_killed=False,
        finished_at="2026-07-20T09:00:00Z",
        host_context={"executor_container_started_at": "2026-07-17T08:00:00Z"},
    )
    assert classify_container_death(diagnostics) is ContainerDeathKind.STOPPED


def test_sigterm_without_host_context_is_stopped():
    diagnostics = ContainerDeathDiagnostics(status="exited", exit_code=143, oom_killed=False)
    assert classify_container_death(diagnostics) is ContainerDeathKind.STOPPED


def test_sigterm_with_executor_started_before_kill_is_targeted_stop():
    # Executor came up 5 min BEFORE the filler died: not a reboot (a reboot restarts the executor
    # after shutdown killed the filler). A miner must not launder a `docker stop` this way.
    diagnostics = ContainerDeathDiagnostics(
        status="exited", exit_code=143, oom_killed=False,
        finished_at="2026-07-20T09:00:00Z",
        host_context={"executor_container_started_at": "2026-07-20T08:55:00Z"},
    )
    assert classify_container_death(diagnostics) is ContainerDeathKind.STOPPED


def test_sigterm_with_executor_restart_long_after_kill_is_targeted_stop():
    # Executor restarted 20 min AFTER the kill: the stop already happened, the late restart does
    # not excuse it.
    diagnostics = ContainerDeathDiagnostics(
        status="exited", exit_code=143, oom_killed=False,
        finished_at="2026-07-20T09:00:00Z",
        host_context={"executor_container_started_at": "2026-07-20T09:20:00Z"},
    )
    assert classify_container_death(diagnostics) is ContainerDeathKind.STOPPED
