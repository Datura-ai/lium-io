from __future__ import annotations

import pytest
from neurons.validators.src.services.task.checks.pid_host_required import PidHostRequiredCheck
from neurons.validators.src.services.task.messages import PidHostRequiredMessages as Msg

from tests.helpers import build_state
from tests.test_sysbox_required_check import _rented_data


@pytest.mark.asyncio
@pytest.mark.parametrize("pid_mode", ["", "private", "container:abc"])
async def test_not_host_pid_unrented_fails_with_reason(context_factory, pid_mode):
    ctx = context_factory(state=build_state(specs={"executor_pid_mode": pid_mode}))

    result = await PidHostRequiredCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.MISSING.reason
    assert "pid: host" in Msg.MISSING.remediation and "encrypted pod backups and restores cannot run" in Msg.MISSING.remediation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "specs,rented",
    [
        ({"executor_pid_mode": "host"}, False),
        ({}, False),  # old scrape / inspect failed: unknown, never fails
    ],
)
async def test_passes(context_factory, specs, rented):
    state = build_state(specs=specs, rented_data=_rented_data() if rented else None)

    result = await PidHostRequiredCheck().run(context_factory(state=state))

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
async def test_not_host_pid_rented_warns_and_passes(context_factory):
    state = build_state(specs={"executor_pid_mode": ""}, rented_data=_rented_data())

    result = await PidHostRequiredCheck().run(context_factory(state=state))

    assert result.passed is True
    assert result.event.reason_code == Msg.MISSING.reason
    assert result.event.severity == "warning"
