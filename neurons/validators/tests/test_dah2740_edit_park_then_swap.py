"""DAH-2740 — an edit keeps the pod's current container until the replacement runs, and can undo itself.

The edit path force-removed the pod's own container in the stale sweep (it is never among the
backend's `active_container_names`) and only then created the new one; when `docker rm -fv`
wedged ("did not receive an exit event") or anything after it failed, the customer had neither
pod. Now the container is renamed aside and stopped, the replacement is built under the original
name, and a failure renames the old one back and starts it.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import ContainerCreated, FailedContainerRequest
from test_deploy_optimizations import _executor_info, _patch_happy, _payload, _ssh_result

from core.utils import retry_ssh_command
from services.docker_service import EDIT_PARKED_SUFFIX, DockerService, create_steps_after_reply, inflight_creates


@pytest.fixture
def svc():
    # the same service test_deploy_optimizations builds; a local fixture rather than an import, which
    # pyflakes reads as a name every `svc` parameter below redefines (F811)
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


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

    client.parked_present = False

    def _side(cmd, *args, **kwargs):
        client.commands.append(cmd)
        if "docker ps -a" in cmd and "--filter name=" in cmd:
            names = [part.split()[0].rstrip("$").strip("'") for part in cmd.split("--filter name=^")[1:]]
            present = [n for n in names if (n.endswith(EDIT_PARKED_SUFFIX) and client.parked_present) or (not n.endswith(EDIT_PARKED_SUFFIX) and container_present)]
            return _ssh_result(stdout="".join(f"{n}\n" for n in present))
        if "docker stop" in cmd:
            return _ssh_result(exit_status=stop_exit, stderr="tried to kill container, but did not receive an exit event")
        if "docker rename" in cmd and cmd.split()[-2].endswith(EDIT_PARKED_SUFFIX):  # parked -> original name
            return _ssh_result(exit_status=rename_back_exit, stderr="rename back failed")
        return _ssh_result(exit_status=0)

    client.run = AsyncMock(side_effect=_side)
    return client


async def _run(svc, payload):
    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )
    # a successful edit removes the parked container after its reply, with the create's session close
    await create_steps_after_reply.wait_until_done(payload.pod_id, timeout=10)
    return result


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
async def test_a_successful_edit_runs_the_same_host_commands(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)

    name, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert ssh.commands == [
        f'/usr/bin/docker ps -a --format "{{{{.Names}}}}" --filter name=^{name}$ --filter name=^{parked}$',
        f"/usr/bin/docker rename {name} {parked}",
        f"/usr/bin/docker stop -t 10 {parked}",
        '/usr/bin/docker volume ls --format "{{.Name}}"',
        # the live power floor read twice: beside the volume create, and again right before docker run
        "nvidia-smi --query-gpu=uuid,power.limit,power.default_limit,power.min_limit,power.max_limit"
        " --format=csv,noheader,nounits",
        "nvidia-smi --query-gpu=uuid,power.limit,power.default_limit,power.min_limit,power.max_limit"
        " --format=csv,noheader,nounits",
        "nohup /usr/bin/python /root/app/src/inspector_executor.py --start-collector >/dev/null 2>&1 &",
        f"/usr/bin/docker rm -fv {parked}",
    ]


@pytest.mark.asyncio
async def test_a_cancelled_step_after_the_reply_keeps_the_edit(svc, monkeypatch):
    """After the reply the edit is a success: a delete's wait timing out, or a shutdown, cancels the
    steps after it, and that removes the parked container instead of undoing the edit."""
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr("services.docker_service.settings.ENABLE_INSPECTOR", True)

    async def inspector_start_that_hangs(**kwargs) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(svc, "_run_inspector_collector_lifecycle", inspector_start_that_hangs)
    name, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX

    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )
    await asyncio.sleep(0)
    steps_finished = await create_steps_after_reply.wait_until_done(payload.pod_id, timeout=0.01)

    assert isinstance(result, ContainerCreated)
    assert steps_finished is False
    assert _docker(ssh.commands, "rm -fv")[-1] == f"/usr/bin/docker rm -fv {parked}"
    assert f"/usr/bin/docker rename {parked} {name}" not in ssh.commands


@pytest.mark.asyncio
async def test_a_second_edit_parks_only_after_the_first_edits_steps_after_reply(svc, monkeypatch):
    """The first edit's steps after its reply remove `<name>_parked`; a second edit of the same pod
    that parked before them would lose its parked container, the pod's only copy until it succeeds."""
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr("services.docker_service.settings.ENABLE_INSPECTOR", True)
    inspector_may_start = asyncio.Event()

    async def inspector_start_held(**kwargs) -> None:
        await inspector_may_start.wait()

    monkeypatch.setattr(svc, "_run_inspector_collector_lifecycle", inspector_start_held)
    name, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX
    create = dict(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    first_edit = await svc.create_container(**create)
    second_edit = asyncio.create_task(svc.create_container(**create))
    await asyncio.sleep(0.1)
    inspector_may_start.set()
    second_edit_result = await asyncio.wait_for(second_edit, 10)
    await create_steps_after_reply.wait_until_done(payload.pod_id, timeout=10)

    assert isinstance(first_edit, ContainerCreated)
    assert isinstance(second_edit_result, ContainerCreated)
    parks = [i for i, command in enumerate(ssh.commands) if command == f"/usr/bin/docker rename {name} {parked}"]
    first_parked_removal = ssh.commands.index(f"/usr/bin/docker rm -fv {parked}")
    assert len(parks) == 2
    assert first_parked_removal < parks[1]


@pytest.mark.asyncio
async def test_a_second_edit_cancelled_by_a_delete_while_it_waits_never_parks(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr("services.docker_service.settings.ENABLE_INSPECTOR", True)
    inspector_may_start = asyncio.Event()

    async def inspector_start_held(**kwargs) -> None:
        await inspector_may_start.wait()

    monkeypatch.setattr(svc, "_run_inspector_collector_lifecycle", inspector_start_held)
    name, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX
    create = dict(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )

    async def tracked_edit():
        with inflight_creates.track(payload.pod_id):
            return await svc.create_container(**create)

    await svc.create_container(**create)
    second_edit = asyncio.create_task(tracked_edit())
    await asyncio.sleep(0.1)
    inflight_creates.cancel(payload.pod_id)
    inspector_may_start.set()
    second_edit_result = await asyncio.wait_for(second_edit, 10)
    await create_steps_after_reply.wait_until_done(payload.pod_id, timeout=10)

    assert isinstance(second_edit_result, FailedContainerRequest)
    assert ssh.commands.count(f"/usr/bin/docker rename {name} {parked}") == 1


@pytest.mark.asyncio
async def test_a_failed_edit_restores_the_previous_container(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr(svc, "_bring_up_existing_container", AsyncMock())
    monkeypatch.setattr(svc, "_run_rental_docker_create_with_port_retry", AsyncMock(side_effect=RuntimeError("gocryptfs: EPERM")))
    name, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_run" and "gocryptfs: EPERM" in result.detail
    tail = ssh.commands[-2:]
    assert tail == [
        f"/usr/bin/docker rm -fv {name} 2>/dev/null || true",
        f"/usr/bin/docker rename {parked} {name}",
    ]
    # park stopped the container, so the restore goes through start + gocryptfs remount + sshd — the
    # start_existing_container steps — at the pod's volume path, never a bare `docker start`
    assert f"/usr/bin/docker start {name}" not in ssh.commands
    bring_up = svc._bring_up_existing_container
    bring_up.assert_awaited_once()
    kwargs = bring_up.await_args.kwargs
    assert kwargs["container_name"] == name and kwargs["pod_id"] == payload.pod_id
    assert kwargs["local_volume_path"] == "/root" and kwargs["ssh_client"] is ssh


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


@pytest.mark.asyncio
async def test_a_parked_container_left_by_a_crashed_edit_is_the_pod_and_is_never_removed(svc, monkeypatch):
    """The original name is gone, `__prev` is there: an earlier edit died between park and restore.
    The parked container is the customer's only copy — it gets its name back and is parked again."""
    payload = _edit_payload()
    ssh = _ssh_recording(container_present=False)
    ssh.parked_present = True
    _patch_happy(svc, monkeypatch, ssh)
    name, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert _docker(ssh.commands, "rename") == [
        f"/usr/bin/docker rename {parked} {name}",   # recovered
        f"/usr/bin/docker rename {name} {parked}",   # parked again for this edit
    ]
    assert not any(c.startswith(f"/usr/bin/docker rm -fv {parked} 2>/dev/null") for c in ssh.commands)
    assert _docker(ssh.commands, "rm -fv")[-1] == f"/usr/bin/docker rm -fv {parked}"  # only after the replacement is up


@pytest.mark.asyncio
async def test_a_leftover_parked_container_next_to_the_pod_is_removed_before_parking(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    ssh.parked_present = True
    _patch_happy(svc, monkeypatch, ssh)
    name, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert ssh.commands.index(f"/usr/bin/docker rm -fv {parked} 2>/dev/null || true") < ssh.commands.index(f"/usr/bin/docker rename {name} {parked}")


def test_the_cycle_cleanup_protects_the_parked_twin_of_every_rented_pod():
    from services.container_cleanup import listed_container_names
    from types import SimpleNamespace

    rented = SimpleNamespace(
        executors={"exec-1": SimpleNamespace(pods=[SimpleNamespace(container_name="pod_aaaa"), SimpleNamespace(container_name="pod_bbbb")])},
        all_filler_containers_by_executor={},
        filler_containers_by_executor={},
    )

    assert listed_container_names(rented) == {"pod_aaaa", "pod_aaaa" + EDIT_PARKED_SUFFIX, "pod_bbbb", "pod_bbbb" + EDIT_PARKED_SUFFIX}


@pytest.mark.asyncio
async def test_a_sibling_create_sweep_keeps_the_parked_twin_of_an_active_pod(svc):
    """clean_existing_containers with the backend's active names: pod_x__prev survives while pod_x is active,
    an unrelated __prev leftover does not."""
    listing = "pod_active\npod_active__prev\npod_gone__prev\npod_gone\n"
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=lambda cmd, *a, **k: _ssh_result(stdout=listing) if "docker ps -a" in cmd else _ssh_result())

    await svc.clean_existing_containers(ssh, {}, pod_name="pod_new", clear_volume=False, active_container_names=["pod_active"])

    rm = next(c for c in [call.args[0] for call in ssh.run.await_args_list] if c.startswith("/usr/bin/docker rm -fv"))
    assert "pod_active__prev" not in rm and "pod_active" not in rm.replace("pod_active__prev", "")
    assert "pod_gone__prev" in rm and "pod_gone" in rm


@pytest.mark.asyncio
async def test_an_undo_whose_ssh_session_died_keeps_the_creates_own_error(svc, monkeypatch):
    """The session that failed the edit is the one the undo runs on: a `run` that raises there must not
    replace the create's failure_step/detail with the undo's exception."""
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr(svc, "_run_rental_docker_create_with_port_retry", AsyncMock(side_effect=RuntimeError("gocryptfs: EPERM")))
    monkeypatch.setattr(svc, "_bring_up_existing_container", AsyncMock())
    inner = ssh.run.side_effect

    def _dies_after_park(cmd, *args, **kwargs):
        if "docker rm -fv" in cmd and cmd.endswith("2>/dev/null || true") and "__prev" not in cmd:
            raise ConnectionResetError("Connection lost")  # the undo's first command on a dead session
        return inner(cmd, *args, **kwargs)

    ssh.run = AsyncMock(side_effect=_dies_after_park)

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_run" and "gocryptfs: EPERM" in result.detail
    assert "Connection lost" not in result.detail
