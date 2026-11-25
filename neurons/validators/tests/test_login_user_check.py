"""A node whose validator SSH login is not root is refused before any script is installed or run on it."""

from datetime import datetime, timezone

import pytest

from services.task.availability import AvailabilityErrorCode, run_availability_errors
from services.task.checks.login_user import LoginUserCheck
from services.task.checks.node_account import NodeAccountCheck
from services.task.checks.start_gpu_monitor import StartGPUMonitorCheck
from services.task.pipeline import _stops_run
from services.task.pipeline_factory import PipelineFactory
from services.task.runner import SSHCommandResult
from tests.helpers import make_context

pytestmark = pytest.mark.asyncio


class _Runner:
    def __init__(self, stdout: str, success: bool = True):
        self.stdout, self.success, self.commands = stdout, success, []
        self.error_type = None if success else "ssh_error"

    async def run(self, cmd, **_):
        self.commands.append(cmd)
        now = datetime.now(timezone.utc)
        return SSHCommandResult(command=cmd, command_id="c", exit_code=0 if self.success else 255, stdout=self.stdout,
                                stderr="", duration_ms=1, started_at=now, finished_at=now, success=self.success,
                                error_type=self.error_type)


# (uid output, ssh ok, error type) -> (reason, whether the run goes on to the monitor install, what the run sends)
CASES = [
    ("0\n", True, None, "LOGIN_USER_ROOT", True, "sends its errors"),
    ("1000\n", True, None, "LOGIN_USER_NOT_ROOT", False, "hides the node"),
    ("\n", True, None, "LOGIN_USER_UNREAD", False, "keeps what is stored"),
    ("root\n", True, None, "LOGIN_USER_UNREAD", False, "keeps what is stored"),
    ("", False, "timeout", "LOGIN_USER_UNREAD", False, "keeps what is stored"),
]


@pytest.mark.parametrize(("stdout", "ok", "error", "reason", "goes_on", "sends"), CASES)
async def test_only_a_confirmed_root_uid_reaches_the_next_step(stdout, ok, error, reason, goes_on, sends):
    runner = _Runner(stdout, success=ok)
    if error:
        runner.error_type = error
    result = await LoginUserCheck().run(make_context(runner=runner))
    assert result.event.reason_code == reason and result.passed is goes_on
    assert _stops_run(LoginUserCheck(), result) is not goes_on   # the monitor install is the next step
    assert not (result.updates or {}).get("clear_verified_job_info")   # never a reset
    sent = run_availability_errors(result.passed, [result.event])
    if sends == "hides the node":
        assert [e["reason_code"] for e in sent] == [AvailabilityErrorCode.VERIFICATION_FAILED]
    elif sends == "keeps what is stored":
        assert sent is None   # no verdict: the backend keeps the node's stored restrictions, a hold is not cleared
    else:
        assert sent == []


@pytest.mark.parametrize("fast_path", [False, True])
async def test_runs_after_the_account_check_and_before_anything_is_installed(fast_path):
    kinds = [type(c) for c in PipelineFactory.build_checks(fast_path=fast_path)]
    assert kinds.index(NodeAccountCheck) + 1 == kinds.index(LoginUserCheck) == kinds.index(StartGPUMonitorCheck) - 1


class _NodeShell:
    """Records every command the run sends to the node; `id -u` answers `uid`, anything else fails."""

    def __init__(self, uid: str):
        self.uid, self.commands = uid, []

    async def run(self, cmd, *args, **kwargs):
        self.commands.append(cmd)
        from types import SimpleNamespace

        return SimpleNamespace(exit_status=0 if cmd == "id -u" else 1, stdout=self.uid if cmd == "id -u" else "", stderr="")


@pytest.mark.parametrize("uid, reason", [("1000\n", "LOGIN_USER_NOT_ROOT"), ("\n", "LOGIN_USER_UNREAD")])
@pytest.mark.parametrize("build", ["standard", "fast", "dry_run"])
async def test_no_host_command_runs_after_a_login_that_is_not_confirmed_root(monkeypatch, build, uid, reason):
    from unittest.mock import AsyncMock

    from services.task.checks import rental_egress_guard
    from services.task.pipeline import Pipeline

    checks = {
        "standard": PipelineFactory.build_checks,
        "fast": PipelineFactory.build_fast_path_checks,
        "dry_run": PipelineFactory.build_dry_run_checks,
    }[build]()
    # the completion guard is due: only the login decides whether it runs
    monkeypatch.setattr(rental_egress_guard, "task_seconds_left", lambda ctx: 3600)
    monkeypatch.setattr(rental_egress_guard, "rental_of_this_node", lambda ctx: None)
    for check in checks:
        if isinstance(check, rental_egress_guard.RentalEgressGuardCheck):
            monkeypatch.setattr(check, "_last_run", {})
    shell = _NodeShell(uid)
    runner = _Runner(uid)
    runner.commands = shell.commands  # one log, in the order the node saw them
    ok, events, _ = await Pipeline(checks, AsyncMock()).run(make_context(ssh=shell, runner=runner))

    assert ok is False and events[-1].reason_code == reason
    # the login read is the last command: nothing installed, scraped or run on the host after it
    assert shell.commands[-1] == "id -u"


class _DropsWhileReading:
    """A login uid read whose session closes under it, as a recreated executor container does."""

    def __init__(self, ssh):
        self.ssh = ssh

    async def run(self, cmd, **_):
        self.ssh.is_closed.return_value = True
        now = datetime.now(timezone.utc)
        return SSHCommandResult(command=cmd, command_id="c", exit_code=255, stdout="", stderr="", duration_ms=1,
                                started_at=now, finished_at=now, success=False, error_type="ssh_error")


@pytest.mark.parametrize("reconnect_fails", [True, False])
async def test_a_session_that_drops_while_the_uid_is_read_ends_the_run_with_no_verdict(monkeypatch, reconnect_fails):
    """review 4693567618: the unread uid stays the run's end; the transport failure would publish a failed validation"""
    from unittest.mock import MagicMock

    import asyncssh

    from services.task.pipeline import Pipeline

    def session():
        ssh = MagicMock(spec=asyncssh.SSHClientConnection)
        ssh.is_closed.return_value = False
        return ssh

    first, fresh = session(), session()

    async def reconnect():
        if reconnect_fails:
            raise OSError("connection refused")
        return fresh

    class _Sink:
        async def emit(self, event):
            pass

    # reconnected, the rerun's session drops under the read too
    monkeypatch.setattr("services.task.pipeline.SSHCommandRunner", lambda ssh, **_: _DropsWhileReading(ssh))
    ok, events, _ = await Pipeline([LoginUserCheck()], _Sink(), reconnect=reconnect).run(
        make_context(ssh=first, runner=_DropsWhileReading(first))
    )
    assert not ok and events[-1].reason_code == "LOGIN_USER_UNREAD"
    assert "EXECUTOR_TRANSPORT_UNREACHABLE" not in [e.reason_code for e in events]
