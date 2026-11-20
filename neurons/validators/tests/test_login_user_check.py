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


async def test_root_login_passes():
    result = await LoginUserCheck().run(make_context(runner=_Runner("0\n")))
    assert result.passed and result.event.reason_code == "LOGIN_USER_ROOT"


async def test_non_root_login_is_refused_and_ends_the_run_without_a_reset():
    result = await LoginUserCheck().run(make_context(runner=_Runner("1000\n")))
    assert not result.passed and result.fatal
    assert result.event.reason_code == "LOGIN_USER_NOT_ROOT"
    # the run ends without a verified-state reset: there is none to clear
    assert not (result.updates or {}).get("clear_verified_job_info")


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
    assert result.event.reason_code == reason
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
