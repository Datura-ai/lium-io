from __future__ import annotations

from types import SimpleNamespace

import pytest
from neurons.validators.src.services.task.checks import docker_cpu
from neurons.validators.src.services.task.checks.docker_cpu import DockerCpuCheck
from neurons.validators.src.services.task.messages import DockerCpuMessages as Msg
from tests.helpers import build_state
from tests.test_sysbox_required_check import _rented_data


def _ssh(ncpu: str, exit_status: int = 0):
    async def run(cmd, *args, **kwargs):
        assert "NCPU" in cmd
        return SimpleNamespace(exit_status=exit_status, stdout=ncpu, stderr="")

    return SimpleNamespace(run=run)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enforce,host,ncpu,exit_status,rented,passed,reason",
    [
        (True, 12, "10", 0, False, False, Msg.MISMATCH.reason),
        (False, 12, "10", 0, False, True, Msg.WARNING.reason),
        (True, 12, "10", 0, True, True, Msg.WARNING.reason),
        (True, 12, "12", 0, False, True, Msg.OK.reason),
        (True, 12, "", 1, False, True, Msg.OK.reason),
        (True, None, "10", 0, False, True, Msg.OK.reason),
    ],
    ids=["mismatch-refused", "switch-off-logged-only", "rented-warned", "equal-passes", "unreadable-passes", "no-host-count-passes"],
)
async def test_docker_cpu_check(context_factory, monkeypatch, enforce, host, ncpu, exit_status, rented, passed, reason):
    monkeypatch.setattr(docker_cpu.settings, "DOCKER_CPU_MISMATCH_ENFORCEMENT_ENABLED", enforce)
    state = build_state(specs={"cpu": {"count": host}}, rented_data=_rented_data() if rented else None)
    ctx = context_factory(state=state, ssh=_ssh(ncpu, exit_status))

    result = await DockerCpuCheck().run(ctx)

    assert result.passed is passed
    assert result.event.reason_code == reason
    if reason != Msg.OK.reason:
        assert "Docker sees 10 of 12 CPUs; remove the CPU limit on the Docker service." in str(result.event)
