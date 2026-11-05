"""DAH-2740 — an edit keeps the pod's current container until the replacement runs, and can undo itself.

The edit path force-removed the pod's own container in the stale sweep (it is never among the
backend's `active_container_names`) and only then created the new one; when `docker rm -fv`
wedged ("did not receive an exit event") or anything after it failed, the customer had neither
pod. Now the container is renamed aside and stopped, the replacement is built under the original
name, and a failure renames the old one back and starts it.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from payload_models.payloads import ContainerCreated, ContainerWarningCode, FailedContainerRequest
from test_deploy_optimizations import _executor_info, _patch_happy, _payload, _ssh_result

from core.docker_utils import DockerCommand
from core.utils import retry_ssh_command
from services.docker_service import (
    EDIT_PARKED_SUFFIX,
    DockerService,
    _layer_data_check,
    create_steps_after_reply,
    inflight_creates,
)
from services.gpu_reset import RECENT_RENTER_GPU_RESET_SECONDS
from services.hung_gpu_reset import HUNG_GPU_QUERY_COMMAND, HungGpuResult
from services.redis_service import GPU_RESET_PENDING_KIND
from services.prerun_host_probe import prerun_host_probe_command


@pytest.fixture(autouse=True)
def _hung_gpu_reset_on(monkeypatch):
    # DAH-5301: these cover the reset itself (the flag on); its shadow mode is in test_gpu_clocks/test_hung_gpu_reset
    from core.config import settings

    monkeypatch.setattr(settings, "HUNG_GPU_RESET_ON_REBOOT_ENABLED", True)


@pytest.fixture
def svc():
    # the same service test_deploy_optimizations builds; a local fixture rather than an import, which
    # pyflakes reads as a name every `svc` parameter below redefines (F811)
    service = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())
    # the pre-run kernel.pid_max read is mandatory and the SSH doubles here do not model /proc
    service._read_host_pid_max = AsyncMock(return_value=4_194_304)
    service._read_host_ram_kib = AsyncMock(return_value=None)
    return service


def _edit_payload(**over):
    return _payload(local_volume="volume_" + "x" * 8, active_container_names=["pod_someone_else"], **over)


def _pod_name(payload) -> str:
    return DockerService.get_container_name(payload)


_PARKED_AT = 1700000123


@pytest.fixture(autouse=True)
def _parked_at(monkeypatch):
    # every edit parks under kept_unmounted_<pod id>_<time>: one fixed time names it
    monkeypatch.setattr("services.docker_service.time", SimpleNamespace(time=lambda: _PARKED_AT, monotonic=time.monotonic))


def _kept(payload) -> str:
    return f"kept_unmounted_{payload.pod_id}_{_PARKED_AT}"


def _ssh_recording(*, container_present: bool = True, stop_exit: int = 0, rename_back_exit: int = 0, listed_with_ids: bool = True, kill_stops: bool = False):
    """An ssh mock that answers docker like a host with (or without) the pod's container, recording every command."""
    client = AsyncMock()
    client.image_exists_result = True
    client.image_exists_error = None
    client.commands: list[str] = []

    client.parked_present = False
    # names of kept_unmounted_<pod id>_* containers on the host
    client.kept: list[str] = []

    def _side(cmd, *args, **kwargs):
        client.commands.append(cmd)
        if "docker ps -a" in cmd and "--filter name=" in cmd:
            names = [part.split()[0].rstrip("$").strip("'") for part in cmd.split("--filter name=^")[1:]]
            kept = [k for n in names if n.startswith("kept_unmounted_") for k in client.kept if k.startswith(n)]
            names = [n for n in names if not n.startswith("kept_unmounted_")]
            present = kept + [n for n in names if (n.endswith(EDIT_PARKED_SUFFIX) and client.parked_present) or (not n.endswith(EDIT_PARKED_SUFFIX) and container_present)]
            return _ssh_result(stdout="".join(f"{n} id-{n}\n" if listed_with_ids else f"{n}\n" for n in present))
        if "/usr/bin/docker diff" in cmd:  # the park's layer read: no renter files under the volume path
            return _ssh_result(stdout="drop\n")
        if cmd == HUNG_GPU_QUERY_COMMAND:  # DAH-5301: the pod's GPU is healthy
            return _ssh_result(stdout="GPU-test, 0, 0\n--compute-apps--\n")
        if "docker stop" in cmd:
            return _ssh_result(exit_status=stop_exit, stderr="tried to kill container, but did not receive an exit event")
        if "docker inspect -f '{{.State.Running}}'" in cmd:
            return _ssh_result(stdout="false\n" if kill_stops else "true\n")
        if "docker inspect -f '{{.State.Status}}'" in cmd:
            return _ssh_result(stdout="running\n")
        if "docker rename" in cmd and cmd.split()[-2].endswith(EDIT_PARKED_SUFFIX) or cmd.startswith("/usr/bin/docker rename kept_unmounted_"):  # parked -> original name
            return _ssh_result(exit_status=rename_back_exit, stderr="rename back failed")
        return _ssh_result(exit_status=0)

    client.run = AsyncMock(side_effect=_side)

    # the swap's commands are host-bounded (`timeout ... sh -c '<cmd>'`) and open with create_process;
    # hand the inner command to run so it is recorded and answered as written
    async def _create_process(command: str):
        inner = shlex.split(command)[-1] if command.startswith("timeout ") else command
        return Mock(wait=lambda: client.run(inner), wait_closed=AsyncMock())

    client.create_process = _create_process
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


_POWER_READ = (
    "nvidia-smi --query-gpu=uuid,power.limit,power.default_limit,power.min_limit,power.max_limit"
    " --format=csv,noheader,nounits"
)


def _docker(commands: list[str], verb: str) -> list[str]:
    return [c for c in commands if f"/usr/bin/docker {verb}" in c]


@pytest.mark.asyncio
async def test_edit_parks_the_current_container_before_the_sweep_and_removes_it_after_success(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    name, parked = _pod_name(payload), _kept(payload)

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
    # only after the replacement is up is the old container removed, by its ID: a later edit may park
    # another container under the same name before a late rm lands; the new one is never removed
    assert _docker(ssh.commands, "rm -fv")[-1] == f"/usr/bin/docker rm -fv id-{name}"
    assert f"/usr/bin/docker rm -fv {name} 2>/dev/null || true" not in ssh.commands


@pytest.mark.asyncio
async def test_a_successful_edit_runs_the_same_host_commands(svc, monkeypatch):
    from services import gpu_clocks

    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    # the real clock-lock safety net (conftest stubs it): its -rgc is one of the host commands
    monkeypatch.setattr("services.docker_service.reset_leftover_clock_locks", gpu_clocks.reset_leftover_clock_locks)
    monkeypatch.setattr(svc.redis_service, "get", AsyncMock(return_value=None), raising=False)  # no lock records

    name, parked = _pod_name(payload), _kept(payload)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    host_key = next(cmd for cmd in ssh.commands if "pod_ssh_host_key" in cmd)
    # DAH-5301: the one hung-GPU query while the old container is parked (a healthy GPU adds nothing). It
    # races the early power-floor read, so only its place against the sequential steps is pinned.
    assert ssh.commands.count(HUNG_GPU_QUERY_COMMAND) == 1
    hung_query = ssh.commands.index(HUNG_GPU_QUERY_COMMAND)
    assert ssh.commands.index('/usr/bin/docker volume ls --format "{{.Name}}"') < hung_query
    assert hung_query < len(ssh.commands) - 1 - ssh.commands[::-1].index(_POWER_READ)  # before the last power read
    assert [cmd for cmd in ssh.commands if cmd != HUNG_GPU_QUERY_COMMAND] == [
        f'/usr/bin/docker ps -a --no-trunc --format "{{{{.Names}}}} {{{{.ID}}}}" --filter name=^{name}$ --filter name=^{name}{EDIT_PARKED_SUFFIX}$ '
        f"--filter name=^kept_unmounted_{payload.pod_id}_",
        # the layer read: renter files under the volume path keep it after the edit
        _layer_data_check(name, "/root"),
        f"/usr/bin/docker rename {name} {parked}",
        f"/usr/bin/docker stop -t 10 {parked}",
        # the layer read again on the stopped container, before the replacement exists
        _layer_data_check(f"id-{name}", "/root"),
        # the pre-run host probe (DAH-3257); no image-label section, the volume is not encrypted
        prerun_host_probe_command(
            docker_image=payload.docker_image, image_label=None, with_power=True, miner_hotkey=payload.miner_hotkey
        ),
        '/usr/bin/docker volume ls --format "{{.Name}}"',
        # the live power floor read twice: beside the volume create, and again right before docker run
        "nvidia-smi --query-gpu=uuid,power.limit,power.default_limit,power.min_limit,power.max_limit"
        " --format=csv,noheader,nounits",
        "nvidia-smi --query-gpu=uuid,power.limit,power.default_limit,power.min_limit,power.max_limit"
        " --format=csv,noheader,nounits",
        # DAH-4663: the clock-lock safety net resets the pod's GPUs, inside the failed-create cleanup, before docker run
        "nvidia-smi -i GPU-test -rgc >/dev/null 2>&1; echo GPU-test $?",
        host_key,
        "nohup /usr/bin/python /root/app/src/inspector_executor.py --start-collector >/dev/null 2>&1 &",
        f"/usr/bin/docker rm -fv id-{name}",
    ]


@pytest.mark.asyncio
async def test_a_cancelled_step_after_the_reply_keeps_the_edit(svc, monkeypatch):
    """After the reply the edit is a success: a delete's wait timing out, or a shutdown, cancels the
    steps after it, and that removes the parked container instead of undoing the edit."""
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)

    async def inspector_start_that_hangs(**kwargs) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(svc, "_run_inspector_collector_lifecycle", inspector_start_that_hangs)
    name, parked = _pod_name(payload), _kept(payload)

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
    assert _docker(ssh.commands, "rm -fv")[-1] == f"/usr/bin/docker rm -fv id-{name}"
    assert f"/usr/bin/docker rename {parked} {name}" not in ssh.commands


@pytest.mark.asyncio
async def test_a_second_edit_parks_only_after_the_first_edits_steps_after_reply(svc, monkeypatch):
    """The first edit's steps after its reply (its inspector start, its parked container's removal by ID) finish
    before a second edit of the same pod parks."""
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    inspector_may_start = asyncio.Event()

    async def inspector_start_held(**kwargs) -> None:
        await inspector_may_start.wait()

    monkeypatch.setattr(svc, "_run_inspector_collector_lifecycle", inspector_start_held)
    name, parked = _pod_name(payload), _kept(payload)
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
    first_parked_removal = ssh.commands.index(f"/usr/bin/docker rm -fv id-{name}")
    assert len(parks) == 2
    assert first_parked_removal < parks[1]


@pytest.mark.asyncio
async def test_a_second_edit_cancelled_by_a_delete_while_it_waits_never_parks(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    inspector_may_start = asyncio.Event()

    async def inspector_start_held(**kwargs) -> None:
        await inspector_may_start.wait()

    monkeypatch.setattr(svc, "_run_inspector_collector_lifecycle", inspector_start_held)
    name, parked = _pod_name(payload), _kept(payload)
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
    name, parked = _pod_name(payload), _kept(payload)

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


@pytest.mark.parametrize(
    ("visible", "starts"),
    [(["GPU-new"], False), (["GPU-old-2", "GPU-new"], True), ([], True)],
    ids=["card_set_swapped", "one_card_missing", "procfs_unreadable"],
)
@pytest.mark.asyncio
async def test_a_failed_edit_skips_the_restore_start_only_when_all_pod_gpus_left_the_node(svc, monkeypatch, caplog, visible, starts):
    # after a VM move swapped the whole card set, the old container's device request names cards the
    # node no longer has, so it cannot start; it gets its name back, stopped
    from services.nvidia_devices import MissingRentedGpuError

    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr(svc, "_bring_up_existing_container", AsyncMock())
    requested = ["GPU-old-1", "GPU-old-2"]
    missing = [u for u in requested if u not in visible]
    monkeypatch.setattr(
        "services.docker_service.build_gpu_docker_config_for_executor",
        AsyncMock(side_effect=MissingRentedGpuError(requested_uuids=requested, visible_uuids=visible, missing_uuids=missing)),
    )
    name, parked = _pod_name(payload), _kept(payload)

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert ssh.commands.count(f"/usr/bin/docker rename {parked} {name}") == 1
    assert f"/usr/bin/docker start {name}" not in ssh.commands
    if starts:
        svc._bring_up_existing_container.assert_awaited_once()
    else:
        svc._bring_up_existing_container.assert_not_awaited()
        assert "restore skipped: pod GPUs no longer on node" in caplog.text
        assert "unknown device" not in caplog.text


@pytest.mark.parametrize(
    ("stop_exit", "listed_with_ids", "details", "renamed"),
    [(1, True, ("could not be stopped", "did not receive an exit event"), True), (0, False, ("without its container ID",), False)],
    ids=["unstoppable", "listed_without_its_id"],  # the removal after the reply could only go by name, which a later edit reuses
)
@pytest.mark.asyncio
async def test_an_unstoppable_container_fails_the_edit_before_anything_is_destroyed(svc, monkeypatch, stop_exit, listed_with_ids, details, renamed):
    payload = _edit_payload()
    monkeypatch.setattr("services.docker_service._PARK_KILL_SETTLE_SECONDS", 0)
    ssh = _ssh_recording(stop_exit=stop_exit, listed_with_ids=listed_with_ids)
    _patch_happy(svc, monkeypatch, ssh)
    name, parked = _pod_name(payload), _kept(payload)

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "park_current_container"
    assert all(detail in result.detail for detail in details)
    assert _docker(ssh.commands, "rename") == [
        f"/usr/bin/docker rename {name} {parked}",
        f"/usr/bin/docker rename {parked} {name}",
    ] * renamed
    # the only rm is the best-effort sweep of a leftover parked name from an earlier edit
    destructive = [c for c in _docker(ssh.commands, "rm -fv") if not c.endswith("2>/dev/null || true")]
    assert destructive == []
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    svc.clean_existing_containers.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_stop_that_docker_kill_ends_still_parks_and_edits(svc, monkeypatch):
    # DinD pods: the first `docker stop` misses its exit event, a kill seconds later works
    monkeypatch.setattr("services.docker_service._PARK_KILL_SETTLE_SECONDS", 0)
    payload = _edit_payload()
    ssh = _ssh_recording(stop_exit=1, kill_stops=True)
    _patch_happy(svc, monkeypatch, ssh)
    parked = _kept(payload)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert _docker(ssh.commands, "kill") == [f"/usr/bin/docker kill {parked}"]
    assert _docker(ssh.commands, "rename") == [f"/usr/bin/docker rename {_pod_name(payload)} {parked}"]


@pytest.mark.parametrize("direct_kill_stops", [True, False], ids=["direct_kill_works", "host_fault"])
@pytest.mark.asyncio
async def test_a_kill_without_an_exit_event_falls_back_to_killing_the_processes(svc, monkeypatch, direct_kill_stops):
    # F-1633: `docker kill` got no exit event; kill init + shim like the delete path, else report the host's fault
    monkeypatch.setattr("services.docker_service._PARK_KILL_SETTLE_SECONDS", 0)
    monkeypatch.setattr("services.docker_service._PARK_KILL_WAIT_SECONDS", 1)
    payload = _edit_payload()
    ssh = _ssh_recording(stop_exit=1)
    inner = ssh.run
    killed_directly = []

    async def unkillable(cmd, *args, **kwargs):
        if cmd.startswith("/usr/bin/docker kill"):
            ssh.commands.append(cmd)
            return _ssh_result(exit_status=1, stderr="cannot kill container: tried to kill container, but did not receive an exit event")
        if "{{.State.Pid}}" in cmd:
            killed_directly.append(cmd)
        if "docker inspect -f '{{.State.Running}}'" in cmd and killed_directly:
            return _ssh_result(stdout="false\n" if direct_kill_stops else "true\n")
        return await inner(cmd, *args, **kwargs)

    ssh.run = unkillable
    _patch_happy(svc, monkeypatch, ssh)
    parked = _kept(payload)

    result = await _run(svc, payload)

    assert killed_directly == [DockerCommand.kill_container_processes(parked)]
    if direct_kill_stops:
        assert isinstance(result, ContainerCreated)
    else:
        assert isinstance(result, FailedContainerRequest)
        assert result.failure_step == "container_cleanup"
        svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


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


def _hung_reset(outcome, calls: list | None = None):
    """Stands in for reset_hung_pod_gpus the way it uses `resetting`: a hung find holds the caller's mark
    and hands it the result before letting go; an exception escapes like an SSH error would."""

    async def fake(runner, pod_gpu_uuids, *, resetting, shadow):
        assert not shadow
        if calls is not None:
            calls.append(pod_gpu_uuids)
        if isinstance(outcome, Exception):
            raise outcome
        async with resetting() as held:
            held.append(outcome)
        return outcome

    return fake


@pytest.mark.parametrize(
    ("outcome", "warnings"),
    [
        (HungGpuResult(["GPU-test"], ContainerWarningCode.HungGpuReset), [ContainerWarningCode.HungGpuReset]),
        (RuntimeError("ssh dropped"), []),  # best-effort: the reboot goes on
    ],
)
@pytest.mark.asyncio
async def test_a_reboot_resets_a_hung_gpu_while_the_old_container_is_parked(svc, monkeypatch, outcome, warnings):
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    calls: list = []
    monkeypatch.setattr("services.docker_service.reset_hung_pod_gpus", _hung_reset(outcome, calls))
    svc.redis_service.mark_gpu_reset_recent = AsyncMock()

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated) and result.warnings == warnings
    assert calls == [["GPU-test"]]
    # a reset GPU drops out for a moment: the checks after the reboot must not blame the provider
    if warnings:
        svc.redis_service.mark_gpu_reset_recent.assert_awaited_once_with(
            payload.miner_hotkey, payload.executor_id, payload.pod_id, ["GPU-test"], RECENT_RENTER_GPU_RESET_SECONDS
        )
    else:
        svc.redis_service.mark_gpu_reset_recent.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_hung_gpu_reset_holds_the_gpu_reset_pending_mark(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    svc.redis_service.mark_gpu_reset_recent = AsyncMock()
    held = []

    async def fake_reset(runner, pod_gpu_uuids, *, resetting, shadow):
        assert not shadow
        async with resetting():
            # the rented checks skip the node while the GPUs reset
            held.append(svc.redis_service.add_pending_pod.await_args)
            svc.redis_service.remove_pending_pod.assert_not_awaited()
        return HungGpuResult(pod_gpu_uuids, ContainerWarningCode.HungGpuNotResetMig)

    monkeypatch.setattr("services.docker_service.reset_hung_pod_gpus", fake_reset)
    svc.redis_service.remove_pending_pod.reset_mock()

    assert isinstance(await _run(svc, payload), ContainerCreated)
    assert held[0].kwargs == {"kind": GPU_RESET_PENDING_KIND, "gpu_uuids": ["GPU-test"]}
    # only the gpu_reset mark is dropped: the create's own pending mark for the pod stays until its own removal
    assert call(payload.miner_hotkey, payload.executor_id, payload.pod_id, kind=GPU_RESET_PENDING_KIND) in (
        svc.redis_service.remove_pending_pod.await_args_list
    )


@pytest.mark.asyncio
async def test_the_recent_reset_mark_is_written_before_the_pending_mark_goes(svc, monkeypatch):
    # a check between the two would otherwise see neither mark and blame the provider for the reset
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    order = Mock()
    for name in ("mark_gpu_reset_recent", "remove_pending_pod"):
        setattr(svc.redis_service, name, AsyncMock())
        order.attach_mock(getattr(svc.redis_service, name), name)
    monkeypatch.setattr(
        "services.docker_service.reset_hung_pod_gpus",
        _hung_reset(HungGpuResult(["GPU-test"], ContainerWarningCode.HungGpuReset)),
    )

    assert isinstance(await _run(svc, payload), ContainerCreated)
    names = [name for name, _, kwargs in order.mock_calls if name == "mark_gpu_reset_recent" or kwargs.get("kind")]
    assert names == ["mark_gpu_reset_recent", "remove_pending_pod"]


@pytest.mark.parametrize(
    "warning", [ContainerWarningCode.HungGpuReset, ContainerWarningCode.HungGpuNotResetMig]
)
@pytest.mark.asyncio
async def test_a_failed_recent_reset_mark_does_not_fail_the_reboot(svc, monkeypatch, warning):
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr("services.docker_service.reset_hung_pod_gpus", _hung_reset(HungGpuResult(["GPU-test"], warning)))
    svc.redis_service.mark_gpu_reset_recent = AsyncMock(side_effect=RuntimeError("redis down"))

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated) and result.warnings == [warning]
    # only a GPU that was actually reset gets the mark
    assert svc.redis_service.mark_gpu_reset_recent.await_count == (warning == ContainerWarningCode.HungGpuReset)


@pytest.mark.parametrize("edit", [True, False])
@pytest.mark.asyncio
async def test_nothing_parked_means_no_hung_gpu_check(svc, monkeypatch, edit):
    # an edit with no container on the host, or a fresh create
    payload = _edit_payload() if edit else _payload(active_container_names=["pod_someone_else"])
    ssh = _ssh_recording(container_present=not edit)
    _patch_happy(svc, monkeypatch, ssh)
    reset = AsyncMock()
    monkeypatch.setattr("services.docker_service.reset_hung_pod_gpus", reset)

    assert isinstance(await _run(svc, payload), ContainerCreated)
    reset.assert_not_awaited()


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
    name, prev, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX, _kept(payload)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert _docker(ssh.commands, "rename") == [
        f"/usr/bin/docker rename {prev} {name}",   # recovered
        f"/usr/bin/docker rename {name} {parked}",   # parked again for this edit
    ]
    assert not any(c.startswith(f"/usr/bin/docker rm -fv {prev} 2>/dev/null") for c in ssh.commands)
    assert _docker(ssh.commands, "rm -fv")[-1] == f"/usr/bin/docker rm -fv id-{prev}"  # only after the replacement is up


@pytest.mark.asyncio
async def test_a_leftover_parked_container_next_to_the_pod_is_removed_before_parking(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    ssh.parked_present = True
    _patch_happy(svc, monkeypatch, ssh)
    name, prev, parked = _pod_name(payload), _pod_name(payload) + EDIT_PARKED_SUFFIX, _kept(payload)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert ssh.commands.index(f"/usr/bin/docker rm -fv {prev} 2>/dev/null || true") < ssh.commands.index(f"/usr/bin/docker rename {name} {parked}")


def test_the_cycle_cleanup_protects_the_parked_twin_of_every_rented_pod():
    from services.container_cleanup import listed_container_names
    from types import SimpleNamespace

    from protocol.vc_protocol.compute_requests import RentedPod

    pods = [RentedPod(pod_id=pod_id, container_name=f"pod_{pod_id}") for pod_id in ("aaaa", "bbbb")]
    rented = SimpleNamespace(
        executors={"exec-1": SimpleNamespace(pods=pods)},
        all_filler_containers_by_executor={},
        filler_containers_by_executor={},
        retained_container_names=[],
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
    # the cleanup's retry_ssh_command retries the dead session 5 times 10 s apart: 40 s of CI per run for nothing
    monkeypatch.setattr(
        "services.docker_service.retry_ssh_command", lambda *a, **k: retry_ssh_command(*a, **{**k, "wait_seconds": 0})
    )

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "docker_run" and "gocryptfs: EPERM" in result.detail
    assert "Connection lost" not in result.detail


@pytest.mark.asyncio
async def test_the_undo_removes_the_replacement_by_its_id_so_a_sibling_retry_under_the_pod_name_survives():
    from services.docker_service import _EditSwap

    ssh = _ssh_recording()
    swap = _EditSwap(ssh, "pod_abc", {})
    swap.parked_name = "pod_abc" + EDIT_PARKED_SUFFIX
    swap.replacement_id = "a" * 64

    await swap.restore()

    assert _docker(ssh.commands, "rm -fv") == [f"/usr/bin/docker rm -fv {'a' * 64} 2>/dev/null || true"]
    assert "/usr/bin/docker rm -fv pod_abc 2>/dev/null || true" not in ssh.commands


@pytest.mark.asyncio
async def test_park_waits_for_a_killed_container_that_exits_late(svc, monkeypatch):
    # prod 22 Oct: the killed DinD pod exits tens of seconds later, not within the first re-read
    monkeypatch.setattr("services.docker_service._PARK_KILL_SETTLE_SECONDS", 0)
    payload = _edit_payload()
    ssh = _ssh_recording(stop_exit=1, kill_stops=False)
    reads = []
    inner = ssh.run

    async def exits_on_the_fourth_read(cmd, *args, **kwargs):
        if "docker inspect -f '{{.State.Running}}'" in cmd:
            reads.append(cmd)
            return _ssh_result(stdout="false\n" if len(reads) >= 4 else "true\n")
        return await inner(cmd, *args, **kwargs)

    ssh.run = exits_on_the_fourth_read
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert len(reads) == 4


@pytest.mark.parametrize(
    ("last_read", "park_renames_back"),
    [(_ssh_result(stdout="true\n"), True), (_ssh_result(stdout="false\n"), False), (_ssh_result(exit_status=1, stderr="timeout"), False)],
    ids=["still_running", "exited_after_the_wait", "inconclusive"],
)
@pytest.mark.asyncio
async def test_park_renames_back_only_when_a_last_read_says_it_still_runs(svc, monkeypatch, last_read, park_renames_back):
    # a kill that lands after the wait must not leave the pod stopped under its own name with no
    # replacement: only an explicit "true" renames back; otherwise it stays parked and the undo restores it
    monkeypatch.setattr("services.docker_service._PARK_KILL_SETTLE_SECONDS", 0)
    monkeypatch.setattr("services.docker_service._PARK_KILL_WAIT_SECONDS", 1)
    payload = _edit_payload()
    ssh = _ssh_recording(stop_exit=1)
    reads, after_last_read_at = [], []
    inner = ssh.run

    async def running_until_the_last_read(cmd, *args, **kwargs):
        if "docker inspect -f '{{.State.Running}}'" in cmd:
            reads.append(cmd)
            if len(reads) == 2:
                ssh.commands.append(cmd)
                after_last_read_at.append(len(ssh.commands))
                return last_read
        return await inner(cmd, *args, **kwargs)

    ssh.run = running_until_the_last_read
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr(svc, "_bring_up_existing_container", AsyncMock())
    name, parked = _pod_name(payload), _kept(payload)

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "park_current_container"
    assert len(reads) == 2
    after_last_read = ssh.commands[after_last_read_at[0] :]
    if park_renames_back:
        assert after_last_read == [f"/usr/bin/docker rename {parked} {name}"]
        svc._bring_up_existing_container.assert_not_awaited()
    else:
        assert after_last_read == [f"/usr/bin/docker rm -fv {name} 2>/dev/null || true", f"/usr/bin/docker rename {parked} {name}"]
        svc._bring_up_existing_container.assert_awaited_once()
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


_STALE_MOUNT_START_ERROR = (
    "Docker SDK start failed: 500 Server Error: error while mounting volume '': VolumeDriver.Mount: 01M5: "
    "cannot create mount point dir '/mnt/volume_x': mkdir /mnt/volume_x: file exists"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start_error, repaired, starts, raises",
    [
        (_STALE_MOUNT_START_ERROR, "volume_x", 2, False),  # the 22 Oct undo: repair, start again, pod up
        (_STALE_MOUNT_START_ERROR, None, 1, True),  # nothing safe to repair: the start's own error stands
        ("Docker SDK start failed: 500 no such device", None, 1, True),  # another error is never repaired
    ],
)
async def test_the_undo_repairs_a_stale_vloopback_mountpoint_left_by_park_and_starts_again(svc, start_error, repaired, starts, raises):
    docker_client = Mock()
    docker_client.start = AsyncMock(side_effect=[RuntimeError(start_error), None])
    svc._repair_stale_mountpoint_of_container_volume = AsyncMock(return_value=repaired)
    svc._restore_mount_and_sshd_after_start = AsyncMock()
    ssh = Mock()

    async def bring_up():
        await svc._bring_up_existing_container(
            docker_client=docker_client,
            ssh_client=ssh,
            container_name="pod_abc",
            local_volume_path="/root",
            pod_id="abc",
            default_extra={},
        )

    if raises:
        with pytest.raises(Exception, match="start failed"):
            await bring_up()
        svc._restore_mount_and_sshd_after_start.assert_not_awaited()
    else:
        await bring_up()
        svc._restore_mount_and_sshd_after_start.assert_awaited_once()
    assert docker_client.start.await_count == starts
    if "file exists" in start_error:
        svc._repair_stale_mountpoint_of_container_volume.assert_awaited_once_with(ssh, "pod_abc", {})
    else:
        svc._repair_stale_mountpoint_of_container_volume.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_docker_stop_that_never_returns_fails_the_edit_in_bounded_time_and_keeps_the_pod(svc, monkeypatch):
    # a reboot hung in park: `docker stop` over SSH never returned and nothing bounded it
    monkeypatch.setattr("services.docker_service._PARK_KILL_SETTLE_SECONDS", 0)
    monkeypatch.setattr("services.docker_service._PARK_KILL_WAIT_SECONDS", 1)
    monkeypatch.setattr("services.docker_service._PARK_COMMAND_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("services.docker_service._HOST_TIMEOUT_VALIDATOR_MARGIN_SECONDS", 0)
    payload = _edit_payload()
    ssh = _ssh_recording()
    inner = ssh.run

    async def stop_and_kill_hang(cmd, *args, **kwargs):
        if cmd.startswith(("/usr/bin/docker stop", "/usr/bin/docker kill")):
            ssh.commands.append(cmd)
            await asyncio.Event().wait()
        return await inner(cmd, *args, **kwargs)

    ssh.run = stop_and_kill_hang
    _patch_happy(svc, monkeypatch, ssh)
    monkeypatch.setattr(svc, "_bring_up_existing_container", AsyncMock())
    name, parked = _pod_name(payload), _kept(payload)

    result = await asyncio.wait_for(_run(svc, payload), 10)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "park_current_container"
    # the parked container gets its name back and nothing of the renter's is removed
    assert ssh.commands[-1] == f"/usr/bin/docker rename {parked} {name}"
    assert not any(parked in c for c in _docker(ssh.commands, "rm"))
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_stop_and_kill_the_host_times_out_rename_the_pod_back(svc, monkeypatch):
    # the host's `timeout` ends a wedged stop and kill (exit 124): the pod still runs, so it gets its name back
    monkeypatch.setattr("services.docker_service._PARK_KILL_SETTLE_SECONDS", 0)
    monkeypatch.setattr("services.docker_service._PARK_KILL_WAIT_SECONDS", 1)
    payload = _edit_payload()
    ssh = _ssh_recording()
    inner = ssh.run

    async def host_times_out(cmd, *args, **kwargs):
        if cmd.startswith(("/usr/bin/docker stop", "/usr/bin/docker kill")):
            ssh.commands.append(cmd)
            return _ssh_result(exit_status=124)
        return await inner(cmd, *args, **kwargs)

    ssh.run = host_times_out
    _patch_happy(svc, monkeypatch, ssh)
    name, parked = _pod_name(payload), _kept(payload)

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "park_current_container"
    assert "exit 124" in f"{result.msg} {result.detail}"
    assert _docker(ssh.commands, "kill") == [f"/usr/bin/docker kill {parked}"]
    assert _docker(ssh.commands, "rename")[-1] == f"/usr/bin/docker rename {parked} {name}"
    assert not any(parked in c for c in _docker(ssh.commands, "rm"))


@pytest.mark.asyncio
async def test_a_restore_whose_bring_up_hangs_ends_in_bounded_time_and_keeps_the_pod(svc, monkeypatch):
    # the undo's start/remount/sshd can wedge on the same node: it gives up, the container keeps its name and volume
    monkeypatch.setattr("services.docker_service._RESTORE_BRING_UP_TIMEOUT_SECONDS", 0.05)
    payload = _edit_payload()
    ssh = _ssh_recording()
    _patch_happy(svc, monkeypatch, ssh)

    async def hangs(**kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(svc, "_bring_up_existing_container", AsyncMock(side_effect=hangs))
    monkeypatch.setattr(svc, "_run_rental_docker_create_with_port_retry", AsyncMock(side_effect=RuntimeError("docker run failed")))
    name, parked = _pod_name(payload), _kept(payload)

    result = await asyncio.wait_for(_run(svc, payload), 10)

    assert isinstance(result, FailedContainerRequest)
    svc._bring_up_existing_container.assert_awaited_once()
    assert ssh.commands[-1] == f"/usr/bin/docker rename {parked} {name}"
    assert not any(parked in c or "volume rm" in c for c in ssh.commands if "rm " in c)


@pytest.mark.asyncio
async def test_a_docker_ps_the_host_times_out_fails_the_edit_and_keeps_the_pod(svc, monkeypatch):
    # review 5766107783: exit 124 with empty stdout must not read as "no container" (the sweep would remove the pod)
    payload = _edit_payload()
    ssh = _ssh_recording()
    inner = ssh.run

    async def ps_times_out(cmd, *args, **kwargs):
        if "docker ps -a" in cmd and "--filter name=" in cmd:
            ssh.commands.append(cmd)
            return _ssh_result(exit_status=124)
        return await inner(cmd, *args, **kwargs)

    ssh.run = ps_times_out
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "park_current_container"
    assert "docker ps exited 124" in f"{result.msg} {result.detail}"
    assert _docker(ssh.commands, "rename") == []
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    svc.clean_existing_containers.assert_not_awaited()


def _report_unmounted(ssh, unmounted: bool = True):
    answer = ssh.run.side_effect

    def _side(cmd, *args, **kwargs):
        if "/usr/bin/docker diff" in cmd:
            ssh.commands.append(cmd)
            return _ssh_result(stdout="" if unmounted else "drop\n")
        return answer(cmd, *args, **kwargs)

    ssh.run.side_effect = _side


@pytest.mark.asyncio
@pytest.mark.parametrize("unmounted", [True, False])
async def test_edit_keeps_a_parked_container_that_ran_without_its_encrypted_volume(svc, monkeypatch, unmounted):
    # dockerd restarted the pod by itself, without the gocryptfs mount: the renter's writes to the volume
    # path are in the old container's layer only, so a successful edit keeps it out of every sweep
    payload = _edit_payload()
    ssh = _ssh_recording()
    _report_unmounted(ssh, unmounted)
    _patch_happy(svc, monkeypatch, ssh)
    name = _pod_name(payload)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    renames = _docker(ssh.commands, "rename")
    assert len(renames) == 1
    if unmounted:
        # parked straight under the kept name, before the stop: no later rename can fail and leave it as __prev
        assert renames[0].split()[:3] == ["/usr/bin/docker", "rename", name]
        assert renames[0].split()[3].startswith(f"kept_unmounted_{payload.pod_id}_")
        assert ssh.commands.index(renames[0]) < ssh.commands.index(_docker(ssh.commands, "stop")[0])
        assert _docker(ssh.commands, "rm -fv id-") == []
    else:
        assert _docker(ssh.commands, "rm -fv")[-1] == f"/usr/bin/docker rm -fv id-{name}"


@pytest.mark.asyncio
async def test_a_write_between_the_layer_read_and_the_stop_keeps_the_parked_container(svc, monkeypatch):
    # park read the layer while the container ran; renter files landed before (or during) the stop:
    # the read on the stopped container, before the replacement exists, keeps it; it was parked under the kept name
    payload = _edit_payload()
    ssh = _ssh_recording()
    answer = ssh.run.side_effect
    name = _pod_name(payload)

    def _side(cmd, *args, **kwargs):
        if "/usr/bin/docker diff" in cmd and f"id-{name}" in cmd:
            ssh.commands.append(cmd)
            return _ssh_result(stdout="")
        return answer(cmd, *args, **kwargs)

    ssh.run.side_effect = _side
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    assert _docker(ssh.commands, "rm -fv id-") == []
    # no late rename: it was parked under the kept name before the stop
    assert _docker(ssh.commands, "rename") == [f"/usr/bin/docker rename {name} {_kept(payload)}"]
    stopped_read = ssh.commands.index(next(c for c in ssh.commands if "/usr/bin/docker diff" in c and f"id-{name}" in c))
    assert stopped_read < ssh.commands.index('/usr/bin/docker volume ls --format "{{.Name}}"')  # before the replacement


@pytest.mark.asyncio
async def test_a_write_during_the_stop_with_a_kept_container_on_the_host_fails_the_edit_before_the_replacement(svc, monkeypatch):
    # the one-kept limit holds for a write seen only on the stopped layer: the edit fails and the pod comes back
    payload = _edit_payload()
    ssh = _ssh_recording()
    ssh.kept = [f"kept_unmounted_{payload.pod_id}_1700000000"]
    answer = ssh.run.side_effect
    name = _pod_name(payload)

    def _side(cmd, *args, **kwargs):
        if "/usr/bin/docker diff" in cmd and f"id-{name}" in cmd:
            ssh.commands.append(cmd)
            return _ssh_result(stdout="")
        if "/usr/bin/docker diff" in cmd and "1700000000" in cmd:
            ssh.commands.append(cmd)
            return _ssh_result(stdout="")  # the earlier kept one holds renter files
        return answer(cmd, *args, **kwargs)

    ssh.run.side_effect = _side
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "park_current_container"
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    assert _docker(ssh.commands, "rm -fv id-") == []


@pytest.mark.asyncio
async def test_a_leftover_parked_container_with_renter_files_is_never_removed(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    ssh.parked_present = True
    _report_unmounted(ssh)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert _docker(ssh.commands, "rm -fv") == [] and _docker(ssh.commands, "rename") == []
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


def _kept_not_running(ssh, status):
    answer = ssh.run.side_effect

    def _side(cmd, *args, **kwargs):
        if "docker inspect -f '{{.State.Status}}' kept_unmounted_" in cmd:
            ssh.commands.append(cmd)
            return _ssh_result(stdout=f"{status}\n")
        return answer(cmd, *args, **kwargs)

    ssh.run.side_effect = _side


def _kept_rename_fails(ssh):
    answer = ssh.run.side_effect

    def _side(cmd, *args, **kwargs):
        if "docker rename kept_unmounted_" in cmd:
            ssh.commands.append(cmd)
            return _ssh_result(exit_status=1, stderr="name is already in use")
        return answer(cmd, *args, **kwargs)

    ssh.run.side_effect = _side


def _kept_stop_wedged(ssh):
    """DinD's lost exit event: `docker stop` and `docker kill` fail; only the direct process kill stops it."""
    answer = ssh.run.side_effect
    killed_directly = []

    def _side(cmd, *args, **kwargs):
        wedged = "did not receive an exit event"
        if cmd.startswith("/usr/bin/docker stop -t 10 kept_unmounted_") or cmd.startswith("/usr/bin/docker kill kept_unmounted_"):
            ssh.commands.append(cmd)
            return _ssh_result(exit_status=1, stderr=f"cannot stop container: tried to kill container, but {wedged}")
        if "{{.State.Pid}}" in cmd and "kept_unmounted_" in cmd:
            killed_directly.append(cmd)
            ssh.commands.append(cmd)
            return _ssh_result()
        if "docker inspect -f '{{.State.Running}}' kept_unmounted_" in cmd:
            ssh.commands.append(cmd)
            return _ssh_result(stdout="false\n" if killed_directly else "true\n")
        return answer(cmd, *args, **kwargs)

    ssh.run.side_effect = _side
    return killed_directly


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start_error, status, renamed_back, stop_wedged",
    [
        (None, "running", True, False),  # up with sshd, no remount, then the pod name
        ("Docker SDK start failed: 500 no such device", "running", False, False),  # start failed: stays kept
        (None, "exited", False, False),  # start returned but the entrypoint already exited: stays kept
        (None, "restarting", False, False),  # restart backoff (.State.Running still true): stays kept
        (None, "running", False, False),  # running, but the rename to the pod name failed: stopped, stays kept
        (None, "running", False, True),  # ... and its stop is wedged: killed via the park path's fallback
    ],
)
async def test_a_failed_edit_brings_a_kept_container_back_unmounted(
    svc, monkeypatch, start_error, status, renamed_back, stop_wedged
):
    # the remount would hide its layer files from the renter
    monkeypatch.setattr("services.docker_service._PARK_KILL_SETTLE_SECONDS", 0)
    monkeypatch.setattr("services.docker_service._PARK_KILL_WAIT_SECONDS", 1)
    payload = _edit_payload()
    ssh = _ssh_recording()
    _report_unmounted(ssh)
    if status != "running":
        _kept_not_running(ssh, status)
    elif not renamed_back and not start_error:
        _kept_rename_fails(ssh)
    killed_directly = _kept_stop_wedged(ssh) if stop_wedged else None
    _patch_happy(svc, monkeypatch, ssh)
    docker_client = svc.rental_docker_client_factory.client
    docker_client.start = AsyncMock(side_effect=RuntimeError(start_error) if start_error else None)
    monkeypatch.setattr(svc, "_encrypted_local_volume_name", AsyncMock(return_value="volume_x"))
    monkeypatch.setattr(svc, "setup_encrypted_local_volume", AsyncMock())
    monkeypatch.setattr(svc, "_run_rental_docker_create_with_port_retry", AsyncMock(side_effect=RuntimeError("port clash")))
    name = _pod_name(payload)

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    kept = _docker(ssh.commands, "rename")[0].split()[3]
    # the real bring-up, under the kept name: start, sshd, never the remount
    assert docker_client.start.await_args.kwargs["container_name"] == kept
    svc._encrypted_local_volume_name.assert_not_awaited()
    svc.setup_encrypted_local_volume.assert_not_awaited()
    sshd = svc.install_open_ssh_server_and_start_ssh_service_with_rental_docker
    if start_error:
        sshd.assert_not_awaited()
    else:
        assert sshd.await_args.kwargs["container_name"] == kept
    if renamed_back:
        assert ssh.commands[-2:] == [f"/usr/bin/docker inspect -f '{{{{.State.Status}}}}' {kept}", f"/usr/bin/docker rename {kept} {name}"]
        return
    # not running (or not renamed): it is stopped under its kept name, and the next (successful) edit keeps it
    assert _docker(ssh.commands, "rename")[1:] == ([f"/usr/bin/docker rename {kept} {name}"] if status == "running" and not start_error else [])
    if stop_wedged:
        assert killed_directly == [DockerCommand.kill_container_processes(kept)]
        assert ssh.commands[-1] == f"/usr/bin/docker inspect -f '{{{{.State.Running}}}}' {kept}"
    else:
        assert ssh.commands[-1] == f"/usr/bin/docker stop -t 10 {kept}"
    retry = _ssh_recording(container_present=False)
    retry.kept = [kept]
    _report_unmounted(retry)
    _patch_happy(svc, monkeypatch, retry)

    assert isinstance(await _run(svc, payload), ContainerCreated)
    assert _docker(retry.commands, "rename") == []
    assert _docker(retry.commands, "rm -fv") == []


@pytest.mark.asyncio
async def test_a_second_unmounted_edit_is_refused_while_a_kept_container_is_on_the_host(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording()
    ssh.kept = [f"kept_unmounted_{payload.pod_id}_1700000000"]
    _report_unmounted(ssh)
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, FailedContainerRequest)
    assert _docker(ssh.commands, "rename") == [] and _docker(ssh.commands, "stop") == []
    assert _docker(ssh.commands, "rm -fv id-") == []
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_kept_container_left_by_a_crashed_edit_is_the_pod_and_stays_kept(svc, monkeypatch):
    payload = _edit_payload()
    ssh = _ssh_recording(container_present=False)
    old = f"kept_unmounted_{payload.pod_id}_1700000000"
    ssh.kept = [old]
    _report_unmounted(ssh)  # its layer holds the renter's files
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    # reused under its kept name, never renamed: no rename can fail and leave it unprotected for the next edit
    assert _docker(ssh.commands, "rename") == []
    assert _docker(ssh.commands, "stop") == [f"/usr/bin/docker stop -t 10 {old}"]
    assert _docker(ssh.commands, "rm -fv id-") == []


@pytest.mark.parametrize(
    ("mounts", "diff", "kept"),
    [
        # written while the volume was unmounted, since remounted: the layer still holds it, whatever is mounted now
        ("/lium-cipher\n", "C /\nA /workspace\nA /workspace/data.bin\n", True),
        ("/lium-cipher\n", "C /root\nC /root/.bashrc\n", False),  # nothing under the volume path
        ("/lium-cipher\n", "A /workspace\nC /etc\nA /workspace2/x\n", False),  # the path itself, a sibling
        ("/lium-cipher\n", "D /workspace/old\n", False),  # a deletion holds no renter data
        ("/lium-cipher\n", None, True),  # docker diff failed: kept
        ("/workspace\n", "A /workspace/data.bin\n", False),  # no encrypted volume: nothing hidden in the layer
        (None, "", True),  # docker inspect failed: kept
    ],
)
def test_the_layer_read_runs_in_a_real_shell(tmp_path, mounts, diff, kept):
    import subprocess

    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f'[ "$1" = inspect ] && {{ {"exit 1" if mounts is None else "cat " + str(tmp_path / "mounts")}; exit; }}\n'
        f'[ "$1" = diff ] && {{ {"exit 1" if diff is None else "cat " + str(tmp_path / "diff")}; exit; }}\n'
        "exit 2\n"
    )
    docker.chmod(0o755)
    (tmp_path / "mounts").write_text(mounts or "")
    (tmp_path / "diff").write_text(diff or "")
    command = _layer_data_check("pod_x", "/workspace").replace("/usr/bin/docker", str(docker))
    out = subprocess.run(["sh", "-c", command], capture_output=True, text=True).stdout
    assert (out.strip() != "drop") is kept


@pytest.mark.parametrize(
    ("created_ago", "kept"),
    [
        (0, False),  # written by the image's startup, before the first mount: not the renter's
        (3600, True),  # written an hour after creation: an unmounted restart, the renter's
    ],
)
def test_the_layer_read_skips_files_the_image_wrote_at_startup(tmp_path, created_ago, kept):
    import subprocess
    import time

    upper = tmp_path / "upper"
    (upper / "root" / ".jupyter").mkdir(parents=True)
    (upper / "root" / ".jupyter" / "lab.json").write_text("{}")
    created = time.strftime("%Y-%m-%dT%H:%M:%S.123456789Z", time.gmtime(time.time() - created_ago))
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f'case "$3" in *Created*) echo "{created} {upper}"; exit;; esac\n'
        '[ "$1" = inspect ] && { echo /lium-cipher; exit; }\n'
        # docker diff lists the same file: only the upper-dir read can tell it is startup noise
        '[ "$1" = diff ] && { printf "C /root\\nA /root/.jupyter\\nA /root/.jupyter/lab.json\\n"; exit; }\n'
        "exit 2\n"
    )
    docker.chmod(0o755)
    command = _layer_data_check("pod_x", "/root").replace("/usr/bin/docker", str(docker))
    out = subprocess.run(["sh", "-c", command], capture_output=True, text=True).stdout
    assert (out.strip() != "drop") is kept


@pytest.mark.asyncio
async def test_a_clean_kept_leftover_next_to_the_pod_is_removed_before_parking(svc, monkeypatch):
    # every edit parks under the kept name: a clean one next to the pod is an earlier edit's copy whose removal failed
    payload = _edit_payload()
    ssh = _ssh_recording()
    old = f"kept_unmounted_{payload.pod_id}_1700000000"
    ssh.kept = [old]
    _patch_happy(svc, monkeypatch, ssh)

    result = await _run(svc, payload)

    assert isinstance(result, ContainerCreated)
    removed = ssh.commands.index(f"/usr/bin/docker rm -fv id-{old} 2>/dev/null || true")
    assert removed < ssh.commands.index(f"/usr/bin/docker rename {_pod_name(payload)} {_kept(payload)}")
