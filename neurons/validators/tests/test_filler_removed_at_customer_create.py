"""A customer's create removes every `filler_*` on the host and confirms the removal.

DAH-3706 (20 Sep 2026): 6 of 459 rented nodes carried a live `filler_*` container beside a paying
pod. The backend's filler stop is best-effort, and a stop that did not confirm left the filler on the
create's `active_container_names` — the names `clean_existing_containers` preserves. The backend no
longer lists a filler on a customer create (lium-platform, same ticket); this is the validator's half:
a CUSTOMER_RENTAL create treats every `filler_*` as stale whatever the list says (an older backend
still lists one), re-reads `docker ps -a` after the removal, and writes the typed event
`FILLER_STILL_RUNNING` for a name that survived — the sweep goes on, the event makes it countable.
DAH-3980: the running check after the customer's `docker run` lists the host once more in the same exec;
a `filler_*` there (a sweep survivor, or one created since) is removed by ID and confirmed gone, or the
create fails before it is reported RUNNING.
A FILLER create keeps protecting its listed sibling bundles (DAH-2465).
DAH-3980: the customer's rm, that re-read and the unprotected volumes' rm are one bounded SSH command.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from unittest.mock import AsyncMock, Mock

import asyncssh
import pytest

from payload_models.payloads import ContainerCreated, FailedContainerRequest, WorkloadKind
from test_deploy_optimizations import _executor_info, _patch_happy, _payload, _run, _ssh_client

import services.docker_service as ds_module
from services.docker_service import (
    FILLER_STILL_RUNNING_EVENT,
    ContainerCleanupReport,
    DockerService,
    _remove_and_list_containers_command,
    own_sweep_removals,
)


@pytest.fixture
def svc():
    # local fixtures rather than imports, which pyflakes reads as names every parameter below
    # redefines (F811); same services test_deploy_optimizations / test_docker_service build
    return DockerService(ssh_service=Mock(), redis_service=Mock(get=AsyncMock(return_value=None)), attestation_service=Mock())


@pytest.fixture
def docker_service(svc):
    return svc


@pytest.fixture
def retry_ssh_mock(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(ds_module, "retry_ssh_command", mock)
    return mock


def _listing(stdout: str, exit_status: int = 0, stderr: str = ""):
    result = Mock()
    result.exit_status = exit_status
    result.stdout = stdout
    result.stderr = stderr
    return result


@pytest.fixture(autouse=True)
def _no_sweeps_from_other_tests():
    own_sweep_removals.clear()


def _events(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(getattr(record, "msg", None), "extra", {}).get("event")
        == FILLER_STILL_RUNNING_EVENT
    ]


def _removal(*names_after: str, rm_exit: int = 0, ps_exit: int = 0, stderr: str = ""):
    # what the one removal command prints: the rm's status, the names left, the listing's status
    listed = "".join(f"NAME\t{name}\n" for name in names_after)
    return _listing(f"RM\t{rm_exit}\n{listed}PS\t{ps_exit}\n", stderr=stderr)


async def _clean_for_customer(docker_service, ssh_client, report=None, **over):
    kwargs = dict(
        ssh_client=ssh_client,
        default_extra={"executor_uuid": "exec-1"},
        pod_name="pod_target",
        active_container_names=[],
        remove_every_filler=True,
        report=report,
    )
    return await docker_service.clean_existing_containers(**{**kwargs, **over})


@pytest.mark.asyncio
async def test_customer_create_removes_a_filler_the_backend_still_lists(
    docker_service, retry_ssh_mock
):
    ssh_client = AsyncMock()
    # before: the host listing; then the removal, whose listing shows the filler gone
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_unconfirmed\npod_sibling\n"),
            _removal("pod_target_new", "pod_sibling"),
        ]
    )

    removed = await _clean_for_customer(
        docker_service, ssh_client, active_container_names=["filler_unconfirmed", "pod_sibling"]
    )

    rm_command = ssh_client.run.await_args_list[1].args[0]
    assert "filler_unconfirmed" in rm_command
    assert "pod_sibling" not in rm_command
    assert sorted(removed) == ["filler_unconfirmed", "pod_target"]
    retry_ssh_mock.assert_not_called()


@pytest.mark.asyncio
async def test_a_filler_that_survives_the_sweep_is_logged_and_left_to_the_running_check(
    docker_service, retry_ssh_mock, caplog
):
    ssh_client = AsyncMock()
    # the listing after the rm still names the filler: dockerd said removed, the host says otherwise
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_stuck\nfiller_gone\n"),
            _removal("filler_stuck"),
        ]
    )
    report = ContainerCleanupReport()

    with caplog.at_level(logging.WARNING):
        removed = await _clean_for_customer(
            docker_service,
            ssh_client,
            report,
            active_volume_names=["volume_target", "volume_stuck", "volume_gone"],
        )

    # the sweep does not raise: the running check after `docker run` removes the survivor again or fails
    # the create (test_a_filler_that_cannot_be_removed_fails_the_customer_create); the survivor is
    # reported with typed fields
    assert "filler_stuck" in removed
    [event] = _events(caplog)
    # the create path's default_extra keys the executor as `executor_uuid` (create_container); the backend's
    # rent-path event carries `executor_uuid` too (its `executor_id` is the DB row id) — one Loki query joins on it
    assert event.msg.extra["executor_uuid"] == "exec-1"
    assert event.msg.extra["pod_name"] == "pod_target"
    assert event.msg.extra["container_names"] == ["filler_stuck"]
    assert ssh_client.run.await_count == 2
    assert report.removed_cleanly_without_volume_rm is False


async def _run_like_asyncssh(answer, open_s: float | None, command_s: float, timeout: float | None):
    # asyncssh's run(): create_process (the channel open, no timeout of its own), then wait(check, timeout)
    if open_s is None:
        await asyncio.Event().wait()  # an sshd that never confirms the channel open
    await asyncio.sleep(open_s)
    try:
        await asyncio.wait_for(asyncio.sleep(command_s), timeout)
    except TimeoutError:
        raise asyncssh.TimeoutError(None, None, None, None, None, None, "", "") from None
    return answer


@pytest.mark.asyncio
async def test_a_sweep_beside_fillers_removes_stale_pods_but_no_filler(
    docker_service, retry_ssh_mock
):
    # the rental probe beside a filler: the backend lists no filler to it, and none may go
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_listing("pod_target\nfiller_a\npod_stale\n")])

    removed = await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={"executor_uuid": "exec-1"},
        pod_name="pod_target",
        active_container_names=[],
        keep_every_filler=True,
    )

    assert "filler_a" not in removed
    rm_command = retry_ssh_mock.call_args_list[0][0][1]
    assert "pod_stale" in rm_command
    assert "filler_a" not in rm_command and "volume_a" not in rm_command


def _patch_filler_touching_steps(svc, monkeypatch) -> dict[str, AsyncMock]:
    steps = {
        "remove_fillers_at_ssh_connect": AsyncMock(
            return_value=ds_module.FillerRemovalAtSshConnect(
                listed_id_by_filler_name={}, removed_cleanly_without_volume_rm=False
            )
        ),
        "_restore_gpu_power_for_uncapped_pod": AsyncMock(),
        "reclaim_dphn_cache_for_rental": AsyncMock(return_value=[]),
    }
    for name, mock in steps.items():
        monkeypatch.setattr(svc, name, mock)
    return steps


@pytest.mark.asyncio
async def test_a_create_beside_fillers_removes_no_filler_and_lifts_no_power_cap(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    steps = _patch_filler_touching_steps(svc, monkeypatch)
    payload = _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL)

    result = await svc.create_container(
        payload=payload,
        executor_info=_executor_info(payload),
        keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
        beside_fillers=True,
    )

    assert isinstance(result, ContainerCreated)
    for mock in steps.values():
        mock.assert_not_awaited()
    sweep = svc.clean_existing_containers.await_args.kwargs
    assert sweep["remove_every_filler"] is False and sweep["keep_every_filler"] is True
    # nor does the running check after `docker run` remove a filler, should it learn to
    assert not svc.check_container_running.await_args.kwargs.get("remove_every_filler")


@pytest.mark.asyncio
async def test_a_customer_create_still_removes_fillers_and_restores_power(svc, monkeypatch):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    steps = _patch_filler_touching_steps(svc, monkeypatch)

    result = await _run(svc, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL))

    assert isinstance(result, ContainerCreated)
    for mock in steps.values():
        mock.assert_awaited()
    sweep = svc.clean_existing_containers.await_args.kwargs
    assert sweep["remove_every_filler"] is True and sweep["keep_every_filler"] is False


_FILLER_ID, _CUSTOMER_CONTAINER_ID = "f" * 64, "c" * 64


def _customer_create_host(svc, monkeypatch, *, listing_at_running_check: str, listing_after_rm: str):
    # a customer create whose real running check sees `listing_at_running_check` beside its running container
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)
    monkeypatch.delattr(svc, "check_container_running")
    svc._run_rental_docker_create_with_port_retry.return_value = _CUSTOMER_CONTAINER_ID
    default_run = ssh_client.run.side_effect

    def run(cmd, *args, **kwargs):
        if cmd.startswith("/usr/bin/docker ps -q --filter"):
            return _listing(f"{_CUSTOMER_CONTAINER_ID[:12]}\n{listing_at_running_check}")
        if cmd.startswith("/usr/bin/docker rm -fv") and "printf 'RM" in cmd:
            return _listing(listing_after_rm)
        return default_run(cmd, *args, **kwargs)

    ssh_client.run.side_effect = run
    return ssh_client


def _commands(ssh_client) -> list[str]:
    return [call.args[0] for call in ssh_client.run.await_args_list]


@pytest.mark.asyncio
async def test_a_filler_created_after_the_sweep_is_removed_before_the_customer_create_succeeds(svc, monkeypatch):
    ssh_client = _customer_create_host(
        svc,
        monkeypatch,
        listing_at_running_check=f"NAME\tpod_x {_CUSTOMER_CONTAINER_ID}\nNAME\tfiller_late {_FILLER_ID}\nPS\t0\n",
        listing_after_rm=f"RM\t0\nNAME\tpod_x {_CUSTOMER_CONTAINER_ID}\nPS\t0\n",
    )

    result = await _run(svc, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL))

    assert isinstance(result, ContainerCreated)
    assert _remove_and_list_containers_command([_FILLER_ID], []) in _commands(ssh_client)
    assert own_sweep_removals.sent_rm_for(_FILLER_ID)


@pytest.mark.asyncio
async def test_a_filler_that_cannot_be_removed_fails_the_customer_create(svc, monkeypatch):
    # a survivor of the sweep before `docker run` is still listed here, and its rm leaves it again
    ssh_client = _customer_create_host(
        svc,
        monkeypatch,
        listing_at_running_check=f"NAME\tfiller_stuck {_FILLER_ID}\nPS\t0\n",
        listing_after_rm=f"RM\t1\nNAME\tfiller_stuck {_FILLER_ID}\nPS\t0\n",
    )

    result = await _run(svc, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL))

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "container_cleanup"
    assert "filler_stuck" in result.detail
    # the customer's own container is removed, as on any failed run
    assert f"/usr/bin/docker rm -fv {_CUSTOMER_CONTAINER_ID} 2>/dev/null || true" in _commands(ssh_client)


@pytest.mark.asyncio
async def test_a_clean_node_lists_fillers_in_the_running_checks_own_exec(svc, monkeypatch):
    ssh_client = _customer_create_host(
        svc,
        monkeypatch,
        listing_at_running_check=f"NAME\tpod_x {_CUSTOMER_CONTAINER_ID}\nPS\t0\n",
        listing_after_rm="",
    )

    result = await _run(svc, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL))

    assert isinstance(result, ContainerCreated)
    commands = _commands(ssh_client)
    [running_check] = [cmd for cmd in commands if cmd.startswith("/usr/bin/docker ps -q --filter")]
    assert "/usr/bin/docker ps -a" in running_check
    # nothing after it lists or removes: the clean node's check is the one exec it was before
    after_running_check = commands[commands.index(running_check) + 1:]
    assert not [cmd for cmd in after_running_check if "docker ps" in cmd or "docker rm" in cmd]


@pytest.mark.asyncio
async def test_a_late_fillers_power_cap_is_lifted_once_it_is_confirmed_gone(svc, monkeypatch):
    # a PEARL filler capped its GPUs after the customer's last restore before `docker run`
    filler_pod_id = "0b6c2e4a-7f1d-4c3b-9a5e-2d8f6b1c4e7a"
    ssh_client = _customer_create_host(
        svc,
        monkeypatch,
        listing_at_running_check=f"NAME\tfiller_{filler_pod_id} {_FILLER_ID}\nPS\t0\n",
        listing_after_rm="RM\t0\nPS\t0\n",
    )
    steps: list[str] = []
    default_run = ssh_client.run.side_effect

    def run(cmd, *args, **kwargs):
        if cmd.startswith("/usr/bin/docker rm -fv") and "printf 'RM" in cmd:
            steps.append("filler rm")
        return default_run(cmd, *args, **kwargs)

    ssh_client.run.side_effect = run
    restore = AsyncMock(side_effect=lambda *_args, **_kwargs: steps.append("power restore"))
    monkeypatch.setattr(ds_module, "restore_filler_pod_gpu_power_limits", restore)

    result = await _run(svc, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL))

    assert isinstance(result, ContainerCreated)
    assert restore.await_args.args[2] == filler_pod_id
    assert steps == ["filler rm", "power restore"]


@pytest.mark.asyncio
@pytest.mark.parametrize("late_step", ["running check", "second listing", "filler rm", "power restore"])
async def test_a_step_after_docker_run_waits_for_a_channel_the_host_opens_late(
    svc, monkeypatch, late_step
):
    # a cancelled channel open that the host confirms later stays open, and a MaxSessions=1 sshd then
    # refuses the failed-run cleanup's channels: the steps after `docker run` bound commands, not opens
    filler_listing = f"NAME\tfiller_late {_FILLER_ID}\nPS\t0\n"
    ssh_client = _customer_create_host(
        svc,
        monkeypatch,
        # an unread listing (`docker ps -a` exited 1) makes the check read the host once more
        listing_at_running_check="PS\t1\n" if late_step == "second listing" else filler_listing,
        listing_after_rm="RM\t0\nPS\t0\n",
    )
    default_run = ssh_client.run.side_effect

    async def run(cmd, *args, timeout=None, **kwargs):
        # the second listing, by either listing command
        if cmd == ds_module._RUNNING_CHECK_LISTING_CMD:
            return await _run_like_asyncssh(_listing(filler_listing), 0.15, 0, timeout)
        if cmd == ds_module.DOCKER_PS_ALL_NAMES_IDS_CMD:
            return await _run_like_asyncssh(_listing(f"filler_late {_FILLER_ID}\n"), 0.15, 0, timeout)
        answer = default_run(cmd, *args, **kwargs)
        is_late = (
            late_step == "running check" and cmd.startswith("/usr/bin/docker ps -q --filter")
        ) or (
            late_step == "filler rm"
            and cmd.startswith("/usr/bin/docker rm -fv")
            and "printf 'RM" in cmd
        )
        return await _run_like_asyncssh(answer, 0.15, 0, timeout) if is_late else answer

    async def restore(*_args, **_kwargs):
        if late_step == "power restore":
            await asyncio.sleep(0.15)

    ssh_client.run.side_effect = run
    monkeypatch.setattr(ds_module, "restore_filler_pod_gpu_power_limits", restore)
    monkeypatch.setattr(ds_module, "_PRERUN_HOST_PROBE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(ds_module, "_CUSTOMER_CONTAINER_REMOVAL_TIMEOUT_SECONDS", 0.05)

    result = await asyncio.wait_for(
        _run(svc, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL)), timeout=5
    )

    assert isinstance(result, ContainerCreated)


@pytest.mark.asyncio
async def test_a_running_check_that_never_answers_fails_the_customer_create(svc, monkeypatch):
    ssh_client = _customer_create_host(svc, monkeypatch, listing_at_running_check="PS\t0\n", listing_after_rm="")
    default_run = ssh_client.run.side_effect

    async def run(cmd, *args, timeout=None, **kwargs):
        if cmd.startswith("/usr/bin/docker ps -q --filter"):
            # the channel opens, the command hangs
            return await _run_like_asyncssh(_listing(""), 0, 1, timeout)
        return default_run(cmd, *args, **kwargs)

    ssh_client.run.side_effect = run
    monkeypatch.setattr(ds_module, "_PRERUN_HOST_PROBE_TIMEOUT_SECONDS", 0.05)

    result = await asyncio.wait_for(_run(svc, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL)), timeout=5)

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == "container_cleanup"
    assert f"/usr/bin/docker rm -fv {_CUSTOMER_CONTAINER_ID} 2>/dev/null || true" in _commands(ssh_client)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_state_failure", [False, True])
async def test_failed_late_filler_power_restore_fails_and_cleans_customer(svc, monkeypatch, platform_state_failure):
    # Arrange: removal succeeded, but a GPU remains capped.
    ssh_client = _customer_create_host(svc, monkeypatch,
        listing_at_running_check=f"NAME\tfiller_late {_FILLER_ID}\nPS\t0\n",
        listing_after_rm="RM\t0\nPS\t0\n")
    error_type = ds_module.GpuPowerRestoreStateError if platform_state_failure else RuntimeError
    restore = AsyncMock(side_effect=error_type("GPU power restoration incomplete"))
    monkeypatch.setattr(ds_module, "restore_filler_pod_gpu_power_limits", restore)
    # Act: the real create path must not publish RUNNING.
    result = await _run(svc, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL))
    # Assert: use the cleanup failure and remove the customer's own container.
    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == ("gpu_power_restore_state" if platform_state_failure else "container_cleanup")
    assert restore.await_args.kwargs["require_complete"] is True
    assert f"/usr/bin/docker rm -fv {_CUSTOMER_CONTAINER_ID} 2>/dev/null || true" in _commands(ssh_client)


# DAH-5012: a customer create's rm of our filler times out (ENGY_TEE fillers on 8x B300, 22 Oct: dockerd
# finished one removal 62 s after the rm, the other was still "in progress" at 66 s). The filler processes
# are killed directly and read; exiting ones no longer fail the create, a live one fails it at its own step.
_STUCK_ID = "a" * 64


def _stuck_filler_host(probe_stdout: str, *, listing_after: str = f"filler_stuck {_STUCK_ID}\n"):
    async def run(cmd, *args, timeout=None, **kwargs):
        if cmd.startswith("pgrep -x dockerd"):
            return _listing(probe_stdout)
        if cmd == ds_module.DOCKER_PS_ALL_NAMES_IDS_CMD:
            return _listing(listing_after)
        if "docker rm -fv" in cmd:
            await asyncio.Event().wait()  # the killed filler never exits within the bound
        raise AssertionError(cmd)

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=run)
    return ssh_client


@pytest.fixture
def _short_removal_bound(monkeypatch):
    monkeypatch.setattr(ds_module, "_CUSTOMER_CONTAINER_REMOVAL_TIMEOUT_SECONDS", 0.1)
    ds_module.fillers_exiting_after_kill.clear()
    yield
    ds_module.fillers_exiting_after_kill.clear()


def test_a_process_is_exiting_once_killed_or_a_zombie():
    host_pid_view, processes = ds_module._parse_stuck_filler_processes(
        "HOSTPID\nPROC\tid\t10\tD\t4194564\tuvm_va_space_destroy\nPROC\tid\t11\tZ\t0\t0\n"
        "PROC\tid\t12\tS\t4194304\tep_poll\ngarbage\n"
    )
    assert host_pid_view
    assert [process.exiting for process in processes] == [True, True, False]
    assert processes[0].wchan == "uvm_va_space_destroy"


@pytest.mark.asyncio
async def test_a_killed_filler_still_exiting_does_not_fail_the_customer_create(
    docker_service, retry_ssh_mock, _short_removal_bound, caplog
):
    ssh_client = _stuck_filler_host(
        f"HOSTPID\nPROC\t{_STUCK_ID}\t4242\tD\t4194564\tnv_teardown\n"
        "STACK\t4242\t[<0>] wb_wait_for_completion|[<0>] sync_filesystem|\n"
        "Dirty:          812345678 kB\nWriteback:          1024 kB\n"
    )

    with caplog.at_level(logging.INFO):
        removed = await docker_service.remove_fillers_at_ssh_connect(
            ssh_client, {"executor_uuid": "exec-1"}, "pod_target", []
        )

    assert removed.listed_id_by_filler_name == {"filler_stuck": _STUCK_ID}
    assert _STUCK_ID in ds_module.fillers_exiting_after_kill
    probe = [cmd for cmd in _commands(ssh_client) if cmd.startswith("pgrep -x dockerd")]
    assert len(probe) == 1 and shlex.quote(_STUCK_ID) in probe[0] and "kill -KILL" in probe[0]
    [stuck_log] = [
        record.msg for record in _events(caplog) if record.msg.extra.get("reason") == "rm_timed_out"
    ]
    assert stuck_log.extra["exiting"] is True
    assert stuck_log.extra["processes"][0]["wchan"] == "nv_teardown"
    assert stuck_log.extra["stacks"] == [["4242", "[<0>] wb_wait_for_completion|[<0>] sync_filesystem|"]]
    assert stuck_log.extra["dirty_writeback"] == ["Dirty:          812345678 kB", "Writeback:          1024 kB"]


@pytest.mark.parametrize("exiting", [True, False])
@pytest.mark.asyncio
async def test_a_filler_whose_rm_failed_and_retry_hung_is_settled_the_same_way(
    docker_service, retry_ssh_mock, _short_removal_bound, exiting
):
    """DAH-5012 (22 Oct 22:32Z): the rm of our filler exited 1 after 12 s, its retry hung, and the create
    failed at container_cleanup ("the steps after a failed docker rm -fv did not finish")."""
    state = "D\t4194564\tnv_teardown" if exiting else "S\t4194304\tep_poll"
    ssh_client = _stuck_filler_host(f"HOSTPID\nPROC\t{_STUCK_ID}\t4242\t{state}\n")
    hang = ssh_client.run.side_effect

    async def run(cmd, *args, **kwargs):
        if "docker rm -fv" in cmd:
            return _listing(f"RM\t1\nNAME\tfiller_stuck {_STUCK_ID}\nPS\t0\n")
        return await hang(cmd, *args, **kwargs)

    async def retry_hangs(*args, **kwargs):
        await asyncio.Event().wait()

    ssh_client.run.side_effect = run
    retry_ssh_mock.side_effect = retry_hangs

    if exiting:
        await docker_service.remove_fillers_at_ssh_connect(ssh_client, {"executor_uuid": "exec-1"}, "pod_target", [])
        assert _STUCK_ID in ds_module.fillers_exiting_after_kill
    else:
        with pytest.raises(ds_module._FillerRemovalStuck, match="did not finish"):
            await docker_service.remove_fillers_at_ssh_connect(
                ssh_client, {"executor_uuid": "exec-1"}, "pod_target", []
            )
    assert retry_ssh_mock.await_count == 1


@pytest.mark.parametrize(
    "probe_stdout",
    [
        f"HOSTPID\nPROC\t{_STUCK_ID}\t4242\tS\t4194304\tep_poll\n",  # alive after SIGKILL
        f"PROC\t{_STUCK_ID}\t4242\tD\t4194564\tnv_teardown\n",  # dockerd not visible: proves nothing
    ],
    ids=["not_exiting", "no_host_pid_view"],
)
@pytest.mark.asyncio
async def test_a_filler_that_is_not_exiting_fails_at_its_own_step(
    docker_service, retry_ssh_mock, _short_removal_bound, probe_stdout
):
    ssh_client = _stuck_filler_host(probe_stdout)

    with pytest.raises(ds_module._FillerRemovalStuck, match="did not finish"):
        await docker_service.remove_fillers_at_ssh_connect(ssh_client, {"executor_uuid": "exec-1"}, "pod_target", [])
    assert not ds_module.fillers_exiting_after_kill


@pytest.mark.asyncio
async def test_a_stuck_filler_fails_the_create_at_filler_removal_stuck_not_container_cleanup(svc, monkeypatch):
    _customer_create_host(svc, monkeypatch, listing_at_running_check="PS\t0\n", listing_after_rm="")
    monkeypatch.setattr(
        svc, "clean_existing_containers", AsyncMock(side_effect=ds_module._FillerRemovalStuck("stuck"))
    )

    result = await _run(svc, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL))

    assert isinstance(result, FailedContainerRequest)
    assert result.failure_step == ds_module.FILLER_REMOVAL_STUCK_STEP == "filler_removal_stuck"


@pytest.mark.asyncio
async def test_the_running_check_leaves_a_killed_exiting_filler_to_dockerd(svc, monkeypatch, _short_removal_bound):
    ds_module.fillers_exiting_after_kill.add(_FILLER_ID)
    ssh_client = _customer_create_host(
        svc,
        monkeypatch,
        listing_at_running_check=f"NAME\tpod_x {_CUSTOMER_CONTAINER_ID}\nNAME\tfiller_exiting {_FILLER_ID}\nPS\t0\n",
        listing_after_rm="",
    )

    result = await _run(svc, _payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL))

    assert isinstance(result, ContainerCreated)
    assert not [cmd for cmd in _commands(ssh_client) if _FILLER_ID in cmd and "docker rm" in cmd]
