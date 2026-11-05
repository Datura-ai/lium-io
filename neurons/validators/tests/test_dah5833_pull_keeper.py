"""DAH-5833: a host-side pull keeps an image download alive across a connector restart.

dockerd drops a download once no client watches it, and the create's SDK pull watches through the
connector. Back-to-back validator releases cut one long pull four times; each resend started over.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
from unittest.mock import AsyncMock, Mock

import pytest
from services.docker_service import (
    DockerService,
    inflight_creates,
    pull_keeper_start_command,
    pull_keeper_stop_command,
)
from test_deploy_optimizations import _docker_client, _patch_happy, _payload, _run, _ssh_client


def _sh(cmd: str) -> None:
    subprocess.run(["sh", "-c", cmd], check=True, timeout=10)


def _pid(pod_id: str) -> int:
    with open(f"/tmp/lium-pull-keeper-{pod_id}.pid") as f:
        return int(f.read())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().split()[2] != "Z"


def test_the_keeper_runs_detached_once_per_pod_and_its_stop_kills_only_it(tmp_path):
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/bin/sh\nexec sleep 30\n")
    fake_docker.chmod(0o755)
    pod_id = str(uuid.uuid4())
    start = pull_keeper_start_command(pod_id, "ghcr.io/org/img:1").replace("/usr/bin/docker", str(fake_docker))

    _sh(start)
    first = _pid(pod_id)
    assert _alive(first)
    _sh(start)  # the resent create: the earlier keeper is still pulling, so no second one
    assert _pid(pod_id) == first

    _sh(pull_keeper_stop_command(pod_id))
    for _ in range(50):
        if not _alive(first):
            break
        subprocess.run(["sleep", "0.1"], check=True)
    assert not _alive(first)
    assert not os.path.exists(f"/tmp/lium-pull-keeper-{pod_id}.pid")

    # a pidfile naming a pid that is not this pod's keeper (reused pid) kills nothing
    bystander = subprocess.Popen(["sleep", "30"])
    try:
        with open(f"/tmp/lium-pull-keeper-{pod_id}.pid", "w") as f:
            f.write(str(bystander.pid))
        _sh(pull_keeper_stop_command(pod_id))
        assert bystander.poll() is None
    finally:
        bystander.kill()


@pytest.fixture
def svc():
    service = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())
    service._read_host_pid_max = AsyncMock(return_value=4_194_304)
    return service


def _commands(ssh_client) -> list[str]:
    return [c.args[0] for c in ssh_client.run.call_args_list if c.args]


@pytest.mark.asyncio
async def test_a_pull_starts_the_keeper_and_stops_it_when_done(svc, monkeypatch):
    payload = _payload()
    ssh = _ssh_client(inspect_exit=1)
    _patch_happy(svc, monkeypatch, ssh)

    with inflight_creates.track(payload.pod_id):
        await _run(svc, payload)

    cmds = _commands(ssh)
    assert cmds.index(pull_keeper_start_command(payload.pod_id, payload.docker_image)) < cmds.index(
        pull_keeper_stop_command(payload.pod_id)
    )


@pytest.mark.asyncio
async def test_a_pull_with_registry_credentials_gets_no_keeper(svc, monkeypatch):
    payload = _payload(docker_username="u", docker_password="p")
    ssh = _ssh_client(inspect_exit=1)
    _patch_happy(svc, monkeypatch, ssh)

    with inflight_creates.track(payload.pod_id):
        await _run(svc, payload)

    assert not any("lium-pull-keeper" in c for c in _commands(ssh))


@pytest.mark.asyncio
async def test_a_connector_shutdown_mid_pull_leaves_the_keeper_running(svc, monkeypatch):
    payload = _payload()
    ssh = _ssh_client(inspect_exit=1)
    _patch_happy(svc, monkeypatch, ssh)
    pull_started = asyncio.Event()

    async def endless_pull(*, image):
        pull_started.set()
        await asyncio.Event().wait()

    _docker_client(svc).pull = endless_pull
    create = asyncio.create_task(_run(svc, payload))
    await asyncio.wait_for(pull_started.wait(), 10)
    create.cancel()
    with pytest.raises(asyncio.CancelledError):
        await create

    cmds = _commands(ssh)
    assert pull_keeper_start_command(payload.pod_id, payload.docker_image) in cmds
    assert pull_keeper_stop_command(payload.pod_id) not in cmds


@pytest.mark.asyncio
async def test_a_delete_mid_pull_stops_the_keeper(svc, monkeypatch):
    payload = _payload()
    ssh = _ssh_client(inspect_exit=1)
    _patch_happy(svc, monkeypatch, ssh)
    pull_started = asyncio.Event()

    async def endless_pull(*, image):
        pull_started.set()
        await asyncio.Event().wait()

    docker = _docker_client(svc)
    docker.pull = endless_pull
    docker.aclose = AsyncMock()

    async def tracked_create():
        with inflight_creates.track(payload.pod_id):
            return await _run(svc, payload)

    create = asyncio.create_task(tracked_create())
    await asyncio.wait_for(pull_started.wait(), 10)
    assert inflight_creates.cancel(payload.pod_id)
    await asyncio.wait_for(create, 10)

    assert pull_keeper_stop_command(payload.pod_id) in _commands(ssh)
