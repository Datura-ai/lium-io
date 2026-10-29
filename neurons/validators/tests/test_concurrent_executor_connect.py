"""DAH-3004: the asyncssh session and the Docker-SDK-over-SSH client are opened together.

container_profiler_events (7 d to 6 Sep 2026, cached-template rentals): "SSH connection"
p50 2.1 s / p90 4.3 s — two serial handshakes to the same host. Entered concurrently the step
costs the slower of the two, and a failure on either side still closes the other.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from unittest.mock import Mock

import pytest
from services.docker_service import DockerService


class _Recorder:
    def __init__(self):
        self.events: list[str] = []
        self.ssh_started = asyncio.Event()
        self.docker_started = asyncio.Event()


# The side that survives connects a moment after the other side has failed: the failure is then already
# known when the survivor registers, which is exactly the case a bare gather leaks and the helper closes.
LATE_SURVIVOR_SECONDS = 0.01


def _ssh_context(rec: _Recorder, *, fail: bool = False, late: bool = False):
    @asynccontextmanager
    async def ctx():
        rec.events.append("ssh:enter")
        rec.ssh_started.set()
        # a serial implementation never gets past this line: docker has not been entered yet
        await asyncio.wait_for(rec.docker_started.wait(), timeout=1)
        if fail:
            raise ConnectionError("sshd refused")
        if late:
            await asyncio.sleep(LATE_SURVIVOR_SECONDS)
        try:
            yield "ssh-client"
        finally:
            rec.events.append("ssh:exit")

    return ctx()


def _docker_context(rec: _Recorder, *, fail: bool = False, late: bool = False):
    @asynccontextmanager
    async def ctx():
        rec.events.append("docker:enter")
        rec.docker_started.set()
        await asyncio.wait_for(rec.ssh_started.wait(), timeout=1)
        if fail:
            raise ConnectionError("docker over ssh refused")
        if late:
            await asyncio.sleep(LATE_SURVIVOR_SECONDS)
        try:
            yield "docker-client"
        finally:
            rec.events.append("docker:exit")

    return ctx()


@pytest.mark.asyncio
async def test_each_leg_logs_its_own_connect_time_even_when_one_fails(monkeypatch):
    # DAH-4677: the step costs the slower leg, so DAH-4210 needs each leg's time, failures included
    import services.docker_service as docker_service

    info = Mock()
    monkeypatch.setattr(docker_service.logger, "info", info)
    rec = _Recorder()

    with pytest.raises(ConnectionError):
        async with AsyncExitStack() as stack:
            await DockerService._connect_ssh_and_docker(
                stack, _ssh_context(rec), _docker_context(rec, fail=True), log_extra={"executor_id": "e1"}
            )

    (line,) = [c.args[0] for c in info.call_args_list if str(c.args[0]) == "Create SSH legs timed"]
    assert line.extra["executor_id"] == "e1"
    assert line.extra["ssh_connect_ms"] >= 0 and line.extra["docker_connect_ms"] >= 0
    assert (line.extra["ssh_connected"], line.extra["docker_connected"]) == (True, False)
