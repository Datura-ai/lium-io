"""An open SSH session that drops under a check is reconnected once and the check runs again.

An executor image release makes providers' auto-updaters recreate the executor container, which closes every
open validator session at once. The check that was running then reports the dead session, not the node: the
pipeline reconnects once per run and runs it again. A failed reconnect, a second drop or a check that leaves
host state behind keeps today's transport failure, which the bounded image-rollout window withholds.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import asyncssh
import pytest
from services.task.checks.port_connectivity import PortConnectivityCheck
from services.task.checks.rental_probe import RentalProbeCheck
from services.task.checks.rented_machine import TenantEnforcementCheck
from services.task.models import ValidationEvent
from services.task.pipeline import CheckResult, Context, Pipeline

pytestmark = pytest.mark.asyncio


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


class _FailsOnDrop(_Check):
    """Returns a verdict of its own on the dead session, as CapabilityCheck does."""

    async def run(self, ctx: Context) -> CheckResult:
        self.sessions.append(ctx.ssh)
        if len(self.sessions) <= self.drops:
            ctx.ssh.is_closed.return_value = True
            return CheckResult(passed=False, event=_event("GPU_VERIFICATION_FAILED"))
        return CheckResult(passed=True, event=_event("PORTS_OK"))


@pytest.mark.parametrize("check_cls", [_Check, _FailsOnDrop])
async def test_a_failed_reconnect_raises_the_transport_failure(check_cls):
    reconnect, calls = _reconnect([OSError("connection refused")])

    with pytest.raises(asyncssh.ConnectionLost):
        await Pipeline([check_cls(drops=1)], _NullSink(), reconnect=reconnect).run(_ctx(_session()))
    assert len(calls) == 1


async def test_a_second_drop_keeps_the_transport_failure_and_there_is_no_third_try():
    check = _FailsOnDrop(drops=2)
    reconnect, calls = _reconnect([_session(), _session()])

    with pytest.raises(asyncssh.ConnectionLost):
        await Pipeline([check], _NullSink(), reconnect=reconnect).run(_ctx(_session()))
    assert len(calls) == 1 and len(check.sessions) == 2


async def test_a_run_reconnects_at_most_once():
    class _Observe(_Check):
        fatal = False
        check_id = "host.observe.a"

    class _Later(_Check):
        check_id = "host.observe.b"

    first, later = _Observe(drops=2), _Later(drops=1)
    reconnect, calls = _reconnect([_session(), _session()])

    with pytest.raises(asyncssh.ConnectionLost):
        await Pipeline([first, later], _NullSink(), reconnect=reconnect).run(_ctx(_session()))
    assert len(calls) == 1 and len(later.sessions) == 0


async def test_a_check_that_leaves_host_state_is_not_replayed():
    class _StartsContainers(_Check):
        replays_after_session_drop = False

    check = _StartsContainers(drops=1)
    reconnect, calls = _reconnect([_session()])

    with pytest.raises(asyncssh.ConnectionLost):
        await Pipeline([check], _NullSink(), reconnect=reconnect).run(_ctx(_session()))
    assert not calls and len(check.sessions) == 1


def test_checks_that_start_containers_or_recover_pods_are_not_replayable():
    for check in (PortConnectivityCheck, RentalProbeCheck, TenantEnforcementCheck):
        assert check.replays_after_session_drop is False


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
