"""An open SSH session that drops under a check is reconnected once and the check runs again.

An executor image release makes providers' auto-updaters recreate the executor container, which closes every
open validator session at once. The check that was running then reports the dead session, not the node: the
pipeline reconnects once and runs it again; a failed reconnect or a second drop is no verdict, never a 0.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import asyncssh
import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from services.executor_rollout import withhold_renter_gpu_reset_verdicts
from services.task.messages import TenantEnforcementMessages
from services.task.models import JobResult, ValidationEvent
from services.task.pipeline import CheckResult, Context, Pipeline, SessionDroppedMidCheck

pytestmark = pytest.mark.asyncio

DROPPED = TenantEnforcementMessages.SESSION_DROPPED_MID_CHECK.reason


class _NullSink:
    async def emit(self, event: ValidationEvent) -> None:
        pass


def _session(closed: bool = False) -> MagicMock:
    ssh = MagicMock(spec=asyncssh.SSHClientConnection)
    ssh.is_closed.return_value = closed
    return ssh


def _event(reason: str) -> ValidationEvent:
    return ValidationEvent(
        event=reason,
        reason_code=reason,
        severity="info",
        impact="x",
        check_id="c",
        when=datetime.now(UTC),
    )


class _Check:
    """Fails the first `drops` runs by closing the session it ran on, as a recreated container does."""

    check_id = "executor.validate.port_connectivity"
    fatal = True

    def __init__(self, drops: int, raises: bool = False):
        self.drops = drops
        self.raises = raises
        self.sessions: list[MagicMock] = []

    async def run(self, ctx: Context) -> CheckResult:
        self.sessions.append(ctx.ssh)
        if len(self.sessions) <= self.drops:
            ctx.ssh.is_closed.return_value = True
            if self.raises:
                raise asyncssh.ChannelOpenError(
                    asyncssh.OPEN_CONNECT_FAILED, "SSH connection closed"
                )
            return CheckResult(passed=False, event=_event("EXECUTOR_TRANSPORT_UNREACHABLE"))
        return CheckResult(passed=True, event=_event("PORTS_OK"))


def _ctx(ssh: MagicMock) -> Context:
    return Context.model_construct(ssh=ssh, runner=None, default_extra={})


def _reconnect(sessions: list[MagicMock]):
    calls = []

    async def reconnect():
        calls.append(1)
        session = sessions.pop(0)
        if isinstance(session, Exception):
            raise session
        return session

    return reconnect, calls


@pytest.mark.parametrize("raises", [False, True])
async def test_a_check_whose_session_dropped_runs_again_on_a_new_session(raises):
    check = _Check(drops=1, raises=raises)
    fresh = _session()
    reconnect, calls = _reconnect([fresh])

    ok, events, _ = await Pipeline([check], _NullSink(), reconnect=reconnect).run(_ctx(_session()))

    assert ok and [e.reason_code for e in events] == ["PORTS_OK"]
    assert len(calls) == 1 and check.sessions[1] is fresh


async def test_a_failed_reconnect_is_no_verdict():
    reconnect, _ = _reconnect([OSError("connection refused")])

    with pytest.raises(SessionDroppedMidCheck):
        await Pipeline([_Check(drops=1)], _NullSink(), reconnect=reconnect).run(_ctx(_session()))


async def test_a_second_drop_is_no_verdict_and_there_is_no_third_try():
    check = _Check(drops=2)
    reconnect, calls = _reconnect([_session(), _session()])

    with pytest.raises(SessionDroppedMidCheck):
        await Pipeline([check], _NullSink(), reconnect=reconnect).run(_ctx(_session()))
    assert len(calls) == 1 and len(check.sessions) == 2


async def test_a_failure_on_a_session_that_stayed_open_stands():
    class _Fails(_Check):
        async def run(self, ctx):
            self.sessions.append(ctx.ssh)
            return CheckResult(passed=False, event=_event("PORT_VERIFICATION_FAILED"))

    reconnect, calls = _reconnect([_session()])

    ok, events, _ = await Pipeline([_Fails(drops=0)], _NullSink(), reconnect=reconnect).run(
        _ctx(_session())
    )

    assert not ok and events[-1].reason_code == "PORT_VERIFICATION_FAILED" and not calls


async def test_without_a_reconnect_a_dropped_session_ends_the_run_as_before():
    with pytest.raises(asyncssh.ChannelOpenError):
        await Pipeline([_Check(drops=1, raises=True)], _NullSink()).run(_ctx(_session()))


async def test_the_cycle_withholds_a_session_dropped_result():
    executor = ExecutorSSHInfo(
        uuid="e1",
        address="h",
        port=1,
        ssh_username="root",
        ssh_port=22,
        python_path="p",
        root_dir="/r",
    )
    dropped = JobResult(
        executor_info=executor,
        score=0,
        job_score=0,
        job_batch_id="b",
        log_status="error",
        log_text="",
        failure_reason_code=DROPPED,
    )

    standing, withheld = withhold_renter_gpu_reset_verdicts({"hk": [dropped]})

    assert standing == {"hk": []} and [w.result for w in withheld] == [dropped]
