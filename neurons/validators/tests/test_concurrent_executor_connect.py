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
from payload_models.payloads import ContainerCreated, ProfilerStepName
from services.docker_service import DockerService
from test_deploy_optimizations import _patch_happy, _payload, _run, _ssh_client


class _Recorder:
    def __init__(self):
        self.events: list[str] = []
        self.ssh_started = asyncio.Event()
        self.docker_started = asyncio.Event()


def _ssh_context(rec: _Recorder, *, fail: bool = False):
    @asynccontextmanager
    async def ctx():
        rec.events.append("ssh:enter")
        rec.ssh_started.set()
        # a serial implementation never gets past this line: docker has not been entered yet
        await asyncio.wait_for(rec.docker_started.wait(), timeout=1)
        if fail:
            raise ConnectionError("sshd refused")
        try:
            yield "ssh-client"
        finally:
            rec.events.append("ssh:exit")

    return ctx()


def _docker_context(rec: _Recorder, *, fail: bool = False):
    @asynccontextmanager
    async def ctx():
        rec.events.append("docker:enter")
        rec.docker_started.set()
        await asyncio.wait_for(rec.ssh_started.wait(), timeout=1)
        if fail:
            raise ConnectionError("docker over ssh refused")
        try:
            yield "docker-client"
        finally:
            rec.events.append("docker:exit")

    return ctx()


@pytest.mark.asyncio
async def test_both_connections_are_opened_at_the_same_time():
    rec = _Recorder()

    async with AsyncExitStack() as stack:
        ssh_client, docker_client = await DockerService._connect_ssh_and_docker(
            stack, _ssh_context(rec), _docker_context(rec)
        )
        assert (ssh_client, docker_client) == ("ssh-client", "docker-client")
        assert sorted(rec.events) == ["docker:enter", "ssh:enter"]

    assert sorted(rec.events[2:]) == ["docker:exit", "ssh:exit"]


@pytest.mark.asyncio
async def test_a_failed_docker_connect_closes_the_ssh_session_and_raises():
    rec = _Recorder()

    with pytest.raises(ConnectionError, match="docker over ssh refused"):
        async with AsyncExitStack() as stack:
            await DockerService._connect_ssh_and_docker(stack, _ssh_context(rec), _docker_context(rec, fail=True))

    assert "ssh:exit" in rec.events
    assert "docker:exit" not in rec.events


@pytest.mark.asyncio
async def test_a_failed_ssh_connect_closes_the_docker_client_and_raises():
    rec = _Recorder()

    with pytest.raises(ConnectionError, match="sshd refused"):
        async with AsyncExitStack() as stack:
            await DockerService._connect_ssh_and_docker(stack, _ssh_context(rec, fail=True), _docker_context(rec))

    assert "docker:exit" in rec.events


@pytest.mark.asyncio
async def test_create_container_still_completes_with_one_ssh_connection_step(monkeypatch):
    svc = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())
    _patch_happy(svc, monkeypatch, _ssh_client())

    result = await _run(svc, _payload(ships_sshd=True))

    assert isinstance(result, ContainerCreated)
    ssh_steps = [p for p in result.profilers if p.name == ProfilerStepName.SSH_CONNECTION_ESTABLISHED]
    assert len(ssh_steps) == 1
