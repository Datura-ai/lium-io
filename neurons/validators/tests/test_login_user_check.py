"""A node whose validator SSH login is not root is refused before any script is installed or run on it."""

from datetime import datetime, timezone

import pytest

from services.task.checks.login_user import LoginUserCheck
from services.task.checks.node_account import NodeAccountCheck
from services.task.checks.start_gpu_monitor import StartGPUMonitorCheck
from services.task.pipeline_factory import PipelineFactory
from services.task.runner import SSHCommandResult
from tests.helpers import make_context

pytestmark = pytest.mark.asyncio


class _Runner:
    def __init__(self, stdout: str, success: bool = True):
        self.stdout, self.success, self.commands = stdout, success, []

    async def run(self, cmd, **_):
        self.commands.append(cmd)
        now = datetime.now(timezone.utc)
        return SSHCommandResult(command=cmd, command_id="c", exit_code=0 if self.success else 255, stdout=self.stdout,
                                stderr="", duration_ms=1, started_at=now, finished_at=now, success=self.success,
                                error_type=None if self.success else "ssh_error")


async def test_root_login_passes():
    result = await LoginUserCheck().run(make_context(runner=_Runner("0\n")))
    assert result.passed and result.event.reason_code == "LOGIN_USER_ROOT"


async def test_non_root_login_is_refused_and_ends_the_run_without_a_reset():
    result = await LoginUserCheck().run(make_context(runner=_Runner("1000\n")))
    assert not result.passed and result.fatal
    assert result.event.reason_code == "LOGIN_USER_NOT_ROOT"
    # the run ends without a verified-state reset: there is none to clear
    assert not (result.updates or {}).get("clear_verified_job_info")


async def test_unreadable_uid_is_not_a_verdict():
    result = await LoginUserCheck().run(make_context(runner=_Runner("", success=False)))
    assert result.passed and result.event.reason_code == "LOGIN_USER_UNREAD"


@pytest.mark.parametrize("fast_path", [False, True])
async def test_runs_after_the_account_check_and_before_anything_is_installed(fast_path):
    kinds = [type(c) for c in PipelineFactory.build_checks(fast_path=fast_path)]
    assert kinds.index(NodeAccountCheck) + 1 == kinds.index(LoginUserCheck) == kinds.index(StartGPUMonitorCheck) - 1
