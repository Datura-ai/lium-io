"""DAH-2740 — an edit keeps the pod's current container until the replacement runs, and can undo itself.

The edit path force-removed the pod's own container in the stale sweep (it is never among the
backend's `active_container_names`) and only then created the new one; when `docker rm -fv`
wedged ("did not receive an exit event") or anything after it failed, the customer had neither
pod. Now the container is renamed aside and stopped, the replacement is built under the original
name, and a failure renames the old one back and starts it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import ContainerCreated, FailedContainerRequest
from test_deploy_optimizations import _executor_info, _patch_happy, _payload, _ssh_result, svc  # noqa: F401

from core.utils import retry_ssh_command
from services.docker_service import EDIT_PARKED_SUFFIX, DockerService


def _edit_payload(**over):
    return _payload(local_volume="volume_" + "x" * 8, active_container_names=["pod_someone_else"], **over)


def _pod_name(payload) -> str:
    return DockerService.get_container_name(payload)


def _ssh_recording(*, container_present: bool = True, stop_exit: int = 0, rename_back_exit: int = 0):
    """An ssh mock that answers docker like a host with (or without) the pod's container, recording every command."""
    client = AsyncMock()
    client.image_exists_result = True
    client.image_exists_error = None
    client.commands: list[str] = []

    def _side(cmd, *args, **kwargs):
        client.commands.append(cmd)
        if "docker ps -a" in cmd and "--filter name=" in cmd:
            name = cmd.split("--filter name=^")[1].rstrip("$").strip("'")
            return _ssh_result(stdout=f"{name}\n" if container_present else "")
        if "docker stop" in cmd:
            return _ssh_result(exit_status=stop_exit, stderr="tried to kill container, but did not receive an exit event")
        if "docker rename" in cmd and cmd.split()[-2].endswith(EDIT_PARKED_SUFFIX):  # parked -> original name
            return _ssh_result(exit_status=rename_back_exit, stderr="rename back failed")
        return _ssh_result(exit_status=0)

    client.run = AsyncMock(side_effect=_side)
    return client


async def _run(svc, payload):
    return await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


def _docker(commands: list[str], verb: str) -> list[str]:
    return [c for c in commands if f"/usr/bin/docker {verb}" in c]


@pytest.mark.asyncio
async def test_edit_parks_the_current_container_before_the_sweep_and_removes_it_after_success(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    name, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert _docker(ssh.commands, "rename") == [f"/usr/bin/docker rename {name} {parked}"]
    assert _docker(ssh.commands, "stop") == [f"/usr/bin/docker stop -t 10 {parked}"]
    # parked and stopped before the replacement was created
    create_index = ssh.commands.index(f"/usr/bin/docker stop -t 10 {parked}")
    assert svc._run_rental_docker_create_with_port_retry.await_count == 1
    assert create_index < len(ssh.commands)
    # the sweep was told the parked name is not stale
    protected = svc.clean_existing_containers.await_args.kwargs["active_container_names"]
    assert protected == ["pod_someone_else", parked]
    # only after the replacement is up is the old container removed; the new one never is
    assert _docker(ssh.commands, "rm -fv")[-1] == f"/usr/bin/docker rm -fv {parked}"
    assert f"/usr/bin/docker rm -fv {name} 2>/dev/null || true" not in ssh.commands


@pytest.mark.asyncio
async def test_a_failed_edit_restores_the_previous_container(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr(svc, "_run_rental_docker_create_with_port_retry", AsyncMock(side_effect=RuntimeError("gocryptfs: EPERM")))
    name, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_run" and "gocryptfs: EPERM" in result.detail
    tail = ssh.commands[-3:]
    assert tail == [
        f"/usr/bin/docker rm -fv {name} 2>/dev/null || true",
        f"/usr/bin/docker rename {parked} {name}",
        f"/usr/bin/docker start {name}",
    ]


@pytest.mark.asyncio
async def test_an_unstoppable_container_fails_the_edit_before_anything_is_destroyed(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording(stop_exit=1)
    _patch_happy(svc, monkeypatch, ssh)
    name, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "park_current_container"
    assert "could not be stopped" in result.detail and "did not receive an exit event" in result.detail
    assert _docker(ssh.commands, "rename") == [
        f"/usr/bin/docker rename {name} {parked}",
        f"/usr/bin/docker rename {parked} {name}",
    ]
    # the only rm is the best-effort sweep of a leftover parked name from an earlier edit
    destructive = [c for c in _docker(ssh.commands, "rm -fv") if not c.endswith("2>/dev/null || true")]
    assert destructive == []
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    svc.clean_existing_containers.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_edit_without_a_container_on_the_host_behaves_like_a_create(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording(container_present=False)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert _docker(ssh.commands, "rename") == [] and _docker(ssh.commands, "stop") == []
    assert svc.clean_existing_containers.await_args.kwargs["active_container_names"] == ["pod_someone_else"]


@pytest.mark.asyncio
async def test_a_fresh_create_never_parks_anything(svc, monkeypatch):
    payload = _payload(active_container_names=["pod_someone_else"])
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert _docker(ssh.commands, "rename") == [] and _docker(ssh.commands, "stop") == []
    assert svc.clean_existing_containers.await_args.kwargs["active_container_names"] == ["pod_someone_else"]


@pytest.mark.asyncio
async def test_retry_ssh_command_raises_the_last_attempts_error_not_retryerror():
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=_ssh_result(exit_status=1, stderr="cannot remove container: did not receive an exit event"))

    with pytest.raises(Exception) as info:
        await retry_ssh_command(ssh, "/usr/bin/docker rm -fv pod_x", "clean_existing_containers", max_attempts=2, wait_seconds=0)

    assert type(info.value) is Exception
    assert "did not receive an exit event" in str(info.value) and "RetryError" not in str(info.value)
    assert ssh.run.await_count == 2
