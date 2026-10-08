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
import os
import shlex
import stat
import subprocess
from unittest.mock import AsyncMock, Mock

import asyncssh
import pytest

from payload_models.payloads import ContainerCreated, FailedContainerRequest, WorkloadKind
from test_deploy_optimizations import _patch_happy, _payload, _run, _ssh_client

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
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


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


@pytest.mark.asyncio
async def test_filler_create_keeps_its_listed_sibling_bundle(docker_service, retry_ssh_mock):
    # DAH-2465: bundle #2's create must not wipe bundle #1 — the default stays the protecting one
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[_listing("filler_bundle_2\nfiller_bundle_1\n"), _listing("filler_bundle_1\n")]
    )

    await docker_service.clean_existing_containers(
        ssh_client=ssh_client,
        default_extra={"executor_uuid": "exec-1"},
        pod_name="filler_bundle_2",
        active_container_names=["filler_bundle_1"],
    )

    rm_command = retry_ssh_mock.call_args_list[0][0][1]
    assert "filler_bundle_2" in rm_command
    assert "filler_bundle_1" not in rm_command


@pytest.mark.asyncio
async def test_rm_failure_on_a_filler_create_still_raises(docker_service, retry_ssh_mock):
    # the tolerant re-read is the customer create's; a FILLER create keeps the old contract
    retry_ssh_mock.side_effect = Exception("[clean_existing_containers] exit_code 1")
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_listing("filler_bundle_2\nfiller_old\n")])

    with pytest.raises(Exception, match="exit_code 1"):
        await docker_service.clean_existing_containers(
            ssh_client=ssh_client,
            default_extra={"executor_uuid": "exec-1"},
            pod_name="filler_bundle_2",
            active_container_names=[],
        )
    assert ssh_client.run.await_count == 1


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
async def test_customer_removal_and_its_confirmation_are_one_bounded_command(
    docker_service, retry_ssh_mock
):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_listing("pod_target\nfiller_x\n"), _removal()])
    report = ContainerCleanupReport()

    await _clean_for_customer(
        docker_service, ssh_client, report, active_volume_names=["volume_target", "volume_x"]
    )

    assert ssh_client.run.await_count == 2
    removal = ssh_client.run.await_args_list[1]
    assert removal.args[0] == _remove_and_list_containers_command(["pod_target", "filler_x"], [])
    assert removal.kwargs["timeout"] == ds_module._CUSTOMER_CONTAINER_REMOVAL_TIMEOUT_SECONDS
    assert removal.kwargs["check"] is False
    retry_ssh_mock.assert_not_called()
    assert report.removed_cleanly_without_volume_rm is True


@pytest.mark.asyncio
async def test_an_unprotected_volume_is_removed_in_the_same_command(docker_service, retry_ssh_mock):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_listing("pod_target\nfiller_x\n"), _removal()])
    report = ContainerCleanupReport()

    await _clean_for_customer(docker_service, ssh_client, report, active_volume_names=["volume_target"])

    assert ssh_client.run.await_count == 2
    assert ssh_client.run.await_args_list[1].args[0] == _remove_and_list_containers_command(
        ["pod_target", "filler_x"], ["volume_x"]
    )
    retry_ssh_mock.assert_not_called()
    assert report.removed_cleanly_without_volume_rm is False


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


@pytest.mark.asyncio
async def test_a_filler_that_survives_the_removal_is_still_recorded_as_ours(docker_service, retry_ssh_mock):
    # its `rm` can still finish after the listing in the same command named it, so that listing does not undo it
    stuck_id, gone_id, target_id = "a" * 64, "b" * 64, "c" * 64
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing(f"pod_target {target_id}\nfiller_stuck {stuck_id}\nfiller_gone {gone_id}\n"),
            _removal(f"filler_stuck {stuck_id}"),
        ]
    )

    await _clean_for_customer(docker_service, ssh_client)

    assert all(own_sweep_removals.sent_rm_for(i) for i in (stuck_id, gone_id, target_id))


@pytest.mark.asyncio
async def test_a_confirmed_removal_writes_no_event(docker_service, retry_ssh_mock, caplog):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_listing("pod_target\nfiller_gone\n"), _removal()])

    with caplog.at_level(logging.WARNING):
        await _clean_for_customer(docker_service, ssh_client)

    assert _events(caplog) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "removal_output",
    [
        _listing("RM\t0\n"),  # the output stops after the rm's status: the listing was never read
        _removal(ps_exit=1, stderr="Cannot connect to the Docker daemon"),  # empty, but not a listing
    ],
)
async def test_an_unread_confirmation_listing_does_not_fail_the_create_and_is_not_clean(
    docker_service, retry_ssh_mock, caplog, removal_output
):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_listing("pod_target\nfiller_gone\n"), removal_output])
    report = ContainerCleanupReport()

    with caplog.at_level(logging.WARNING):
        removed = await _clean_for_customer(
            docker_service, ssh_client, report, active_volume_names=["volume_target", "volume_gone"]
        )

    assert "filler_gone" in removed
    assert _events(caplog) == []
    assert any("Unable to confirm the filler removal" in str(r.msg) for r in caplog.records)
    assert report.removed_cleanly_without_volume_rm is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ambiguous_stdout",
    [
        "RM\t0\nPS\t0\nnoise\n",  # an untagged line
        "RM\t0\nRM\t0\nPS\t0\n",  # the rm's status twice
        "RM\tzero\nPS\t0\n",  # a status that is not a number
    ],
)
async def test_an_ambiguous_removal_output_is_never_clean(
    docker_service, retry_ssh_mock, ambiguous_stdout
):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_x\n"),
            _listing(ambiguous_stdout),
            _listing(""),  # the listing on its own, when the output's could not be read
        ]
    )
    report = ContainerCleanupReport()

    await _clean_for_customer(
        docker_service, ssh_client, report, active_volume_names=["volume_target", "volume_x"]
    )

    assert report.removed_cleanly_without_volume_rm is False


@pytest.mark.asyncio
async def test_rm_that_fails_because_the_filler_is_already_gone_does_not_fail_the_create(
    docker_service, retry_ssh_mock
):
    # the backend's delete landed between the listing and the rm: `docker rm -f` exits non-zero for
    # the vanished name; one attempt, the listing after it says it is gone, the create goes on
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_racing\n"),
            _removal("pod_other", rm_exit=1, stderr="No such container: filler_racing"),
        ]
    )
    report = ContainerCleanupReport()

    removed = await _clean_for_customer(docker_service, ssh_client, report)

    assert sorted(removed) == ["filler_racing", "pod_target"]
    # the removal's own listing decides: no re-read, no second rm (a vanished name must not cost the
    # create the 5x10 s budget); only the volume rm runs again, as it did after a failed rm before
    assert ssh_client.run.await_count == 2
    assert retry_ssh_mock.call_count == 1
    assert "volume rm" in retry_ssh_mock.call_args_list[0][0][1]
    assert report.removed_cleanly_without_volume_rm is False


@pytest.mark.asyncio
async def test_an_ssh_error_on_the_removal_lets_a_fresh_listing_decide(docker_service, retry_ssh_mock):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_x\n"),
            OSError("channel closed"),
            _listing("pod_other\n"),  # the re-read: everything stale is gone
        ]
    )
    report = ContainerCleanupReport()

    removed = await _clean_for_customer(docker_service, ssh_client, report)

    assert sorted(removed) == ["filler_x", "pod_target"]
    assert ssh_client.run.await_count == 3
    assert report.removed_cleanly_without_volume_rm is False


@pytest.mark.asyncio
async def test_rm_retry_budget_goes_only_to_the_names_still_on_the_host(
    docker_service, retry_ssh_mock
):
    # two stale fillers, one vanished, one still there: the full retry budget is spent on the second only
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_gone\nfiller_busy\n"),
            _removal("filler_busy", rm_exit=1),  # filler_gone left, filler_busy still there
            _listing(""),  # the confirmation after the retried rm
        ]
    )
    report = ContainerCleanupReport()

    await _clean_for_customer(docker_service, ssh_client, report)

    retried_rm = retry_ssh_mock.call_args_list[0]
    assert "filler_busy" in retried_rm[0][1]
    assert "filler_gone" not in retried_rm[0][1]
    assert retried_rm.kwargs["max_attempts"] == 5  # the full budget, as before DAH-3706
    assert "volume rm" in retry_ssh_mock.call_args_list[1][0][1]
    assert ssh_client.run.await_count == 3
    assert report.removed_cleanly_without_volume_rm is False


@pytest.mark.asyncio
async def test_rm_that_fails_with_the_container_still_there_raises(docker_service, retry_ssh_mock):
    retry_ssh_mock.side_effect = Exception("[clean_existing_containers] exit_code 1, stderr: busy")
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[_listing("pod_target\nfiller_stuck\n"), _removal("filler_stuck", rm_exit=1)]
    )

    with pytest.raises(Exception, match="busy"):
        await _clean_for_customer(docker_service, ssh_client)
    # the one removal, then the full-budget retry of the name still there
    assert retry_ssh_mock.call_count == 1


@pytest.mark.asyncio
async def test_rm_that_fails_and_cannot_be_re_read_raises_the_rm_error(
    docker_service, retry_ssh_mock
):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_x\n"),
            _removal(rm_exit=1, ps_exit=1, stderr="rm failed"),
            OSError("ssh dropped"),
        ]
    )

    with pytest.raises(Exception, match="rm failed"):
        await _clean_for_customer(docker_service, ssh_client)


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


_SLOW_SSH_ROWS = pytest.mark.parametrize(
    ("open_s", "command_s"),
    [(0, 1), (None, 0), (0.15, 0.15)],
    ids=["command_hangs", "channel_never_opens", "slow_open_then_slow_command"],
)


@pytest.mark.asyncio
@_SLOW_SSH_ROWS
async def test_a_removal_that_times_out_fails_the_cleanup(
    docker_service, retry_ssh_mock, monkeypatch, open_s, command_s
):
    async def run(cmd, *args, timeout=None, **kwargs):
        if cmd == ds_module.DOCKER_PS_ALL_NAMES_IDS_CMD:
            return _listing("pod_target\nfiller_x\n")
        return await _run_like_asyncssh(_removal(), open_s, command_s, timeout)

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=run)
    monkeypatch.setattr(ds_module, "_CUSTOMER_CONTAINER_REMOVAL_TIMEOUT_SECONDS", 0.2)
    started = asyncio.get_running_loop().time()

    with pytest.raises(Exception, match="did not finish"):
        await asyncio.wait_for(_clean_for_customer(docker_service, ssh_client), 2)
    assert asyncio.get_running_loop().time() - started < 0.3
    # a hung dockerd is not asked again
    assert ssh_client.run.await_count == 2
    retry_ssh_mock.assert_not_called()


@pytest.mark.asyncio
@_SLOW_SSH_ROWS
async def test_a_listing_that_times_out_is_unread_within_its_bound(monkeypatch, open_s, command_s):
    async def run(cmd, *args, timeout=None, **kwargs):
        return await _run_like_asyncssh(_listing("filler_x\n"), open_s, command_s, timeout)

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=run)
    monkeypatch.setattr(ds_module, "_PRERUN_HOST_PROBE_TIMEOUT_SECONDS", 0.2)
    started = asyncio.get_running_loop().time()

    assert await asyncio.wait_for(DockerService._list_all_containers(ssh_client), 2) is None
    assert asyncio.get_running_loop().time() - started < 0.3


@pytest.mark.asyncio
async def test_a_removal_that_times_out_still_logs_its_duration(docker_service, retry_ssh_mock, caplog):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(
        side_effect=[
            _listing("pod_target\nfiller_x\n"),
            asyncssh.TimeoutError(None, None, None, None, None, None, "", ""),
        ]
    )

    with caplog.at_level(logging.INFO), pytest.raises(Exception, match="did not finish"):
        await _clean_for_customer(docker_service, ssh_client)

    [removal_log] = [
        record.msg
        for record in caplog.records
        if getattr(record.msg, "message", None) == "customer_container_removal"
    ]
    assert "removal_ms" in removal_log.extra
    assert removal_log.extra["listing_read"] is False


_DOCKER_STUB = """#!/bin/sh
case "$1" in
  rm) shift 2; for arg in "$@"; do printf '%s\\0' "$arg" >> "$RM_ARGS"; done ;;
  ps) printf 'NAME\\tpod_other\\n' ;;
  volume) shift 2; for arg in "$@"; do printf '%s\\0' "$arg" >> "$VOLUME_RM_ARGS"; done ;;
esac
"""


def test_a_hostile_container_name_stays_one_argument_of_the_rm(tmp_path):
    # the names come from the miner's host; each reaches docker as one argument and runs nothing
    hostile_names = ["filler_x'; touch pwned_quote; '", "filler_$(touch pwned_subshell)", "filler_y\nRM\t0"]
    hostile_volumes = ["volume_`touch pwned_backtick`"]
    stub = tmp_path / "docker"
    stub.write_text(_DOCKER_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    command = _remove_and_list_containers_command(hostile_names, hostile_volumes).replace(
        "/usr/bin/docker", str(stub)
    )
    env = {
        **os.environ,
        "RM_ARGS": str(tmp_path / "rm_args"),
        "VOLUME_RM_ARGS": str(tmp_path / "volume_rm_args"),
    }

    stdout = subprocess.run(
        ["sh", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    ).stdout

    assert list(tmp_path.glob("pwned*")) == []
    assert (tmp_path / "rm_args").read_text().split("\0")[:-1] == hostile_names
    assert (tmp_path / "volume_rm_args").read_text().split("\0")[:-1] == hostile_volumes
    assert ds_module._parse_remove_and_list_containers(stdout) == (0, (("pod_other",), {}))


@pytest.mark.asyncio
async def test_a_removal_without_a_filler_is_never_a_clean_filler_removal(docker_service, retry_ssh_mock):
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=[_listing("pod_target\npod_stale\n"), _removal()])
    report = ContainerCleanupReport()

    await _clean_for_customer(
        docker_service, ssh_client, report, active_volume_names=["volume_target", "volume_stale"]
    )

    assert ssh_client.run.await_count == 2
    assert report.removed_cleanly_without_volume_rm is False



_LISTED_ID = "a" * 64
_REPLACEMENT_ID = "b" * 64


def _host_where_a_same_name_container_replaces_the_listed_one(name: str, events: list[str]) -> AsyncMock:
    """A host that lists ``name`` under _LISTED_ID; by the time the rm arrives that container is gone
    and a new one runs under the same name (_REPLACEMENT_ID). `docker rm -fv` removes by ID or name;
    `docker volume rm` keeps the name's volume while either container mounts it."""
    containers = {name: _LISTED_ID}
    volumes = {"volume_" + name.split("_", 1)[1]}
    listed = False

    def remove_volume_unless_mounted() -> None:
        if name not in containers:
            volumes.clear()

    def listing() -> str:
        return "".join(f"{n} {i}\n" for n, i in containers.items())

    async def run(cmd, *args, **kwargs):
        nonlocal listed
        if cmd == ds_module.DOCKER_PS_ALL_NAMES_IDS_CMD:
            stdout = listing()
            if not listed:
                listed = True
                containers[name] = _REPLACEMENT_ID
            return _listing(stdout)
        if cmd.startswith("/usr/bin/docker volume rm "):
            remove_volume_unless_mounted()
            return _listing("")
        assert cmd.startswith("/usr/bin/docker rm -fv "), cmd
        targets = shlex.split(cmd.split(">/dev/null")[0])[3:]
        events.append(" ".join(targets))
        gone = [n for n, i in containers.items() if n in targets or i in targets]
        for n in gone:
            del containers[n]
        rm_exit = 0 if len(gone) == len(targets) else 1
        if "/usr/bin/docker volume rm " in cmd:
            remove_volume_unless_mounted()
        if "printf 'RM" not in cmd:
            return _listing("", exit_status=rm_exit)
        names = "".join(f"NAME\t{line}\n" for line in listing().splitlines())
        return _listing(f"RM\t{rm_exit}\n{names}PS\t0\n")

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=run)
    ssh_client.containers = containers
    ssh_client.volumes = volumes
    return ssh_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "removals", "left_on_host", "survivor_events"),
    [
        # a filler retry's new container: the customer's node keeps no filler, so it goes by its own ID
        # (the confirmation before that removal still reports it)
        ("filler_x", [_LISTED_ID, _REPLACEMENT_ID], {}, 1),
        # a same-name pod created since the listing is not the stale one: it stays
        ("pod_old", [_LISTED_ID], {"pod_old": _REPLACEMENT_ID}, 0),
    ],
    ids=["replacement_filler", "replacement_pod"],
)
async def test_customer_removal_removes_the_listed_container_not_a_same_name_replacement(
    docker_service, caplog, name, removals, left_on_host, survivor_events
):
    events: list[str] = []
    ssh_client = _host_where_a_same_name_container_replaces_the_listed_one(name, events)

    await _clean_for_customer(docker_service, ssh_client, active_volume_names=["volume_x", "volume_old"])

    assert events == removals
    assert ssh_client.containers == left_on_host
    assert len(_events(caplog)) == survivor_events


@pytest.mark.asyncio
async def test_a_replacement_fillers_volume_is_removed_once_the_replacement_is_gone(docker_service):
    # the replacement mounts its filler's volume_<id>: a volume rm before its removal finds it in use
    ssh_client = _host_where_a_same_name_container_replaces_the_listed_one("filler_x", [])

    await _clean_for_customer(docker_service, ssh_client)

    assert ssh_client.containers == {}
    assert ssh_client.volumes == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hung_command", "hung_send", "removals"),
    [
        (ds_module._docker_rm_command([_REPLACEMENT_ID]), 1, [_LISTED_ID]),
        # the replacement's rm answers; the confirming listing after it hangs
        (ds_module.DOCKER_PS_ALL_NAMES_IDS_CMD, 2, [_LISTED_ID, _REPLACEMENT_ID]),
    ],
    ids=["replacement_rm", "listing_after_it"],
)
async def test_a_replacement_filler_removal_that_hangs_fails_the_cleanup(
    docker_service, monkeypatch, hung_command, hung_send, removals
):
    events: list[str] = []
    ssh_client = _host_where_a_same_name_container_replaces_the_listed_one("filler_x", events)
    answer_like_the_host = ssh_client.run.side_effect
    sends = 0

    async def run(cmd, *args, **kwargs):
        nonlocal sends
        sends += cmd == hung_command
        if cmd == hung_command and sends == hung_send:
            await asyncio.Event().wait()  # a hung dockerd
        return await answer_like_the_host(cmd, *args, **kwargs)

    ssh_client.run = AsyncMock(side_effect=run)
    monkeypatch.setattr(ds_module, "_CUSTOMER_CONTAINER_REMOVAL_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(Exception, match="replacement filler.* did not finish"):
        await asyncio.wait_for(
            _clean_for_customer(docker_service, ssh_client, active_volume_names=["volume_x"]), 5
        )
    assert events == removals


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answers_before_the_hang", "hung_command"),
    [
        # the retry of the surviving filler hangs
        (1, "/usr/bin/docker rm -fv "),
        # the retry and its re-listing answer; the volume rm after them hangs
        (3, "/usr/bin/docker volume rm "),
    ],
    ids=["survivor_retry", "fallback_volume_rm"],
)
async def test_a_survivor_retry_that_hangs_fails_the_cleanup_within_the_bound(
    docker_service, monkeypatch, answers_before_the_hang, hung_command
):
    # the rm exits 1 with the filler still listed; a step after it hangs on a wedged dockerd
    replies = [_listing("pod_target\nfiller_busy\n"), _removal("filler_busy", rm_exit=1)]
    replies += [_listing("")] * (answers_before_the_hang - 1)
    hung: list[str] = []

    async def run(cmd, *args, **kwargs):
        if replies:
            return replies.pop(0)
        hung.append(cmd)
        await asyncio.Event().wait()  # a hung dockerd

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=run)
    monkeypatch.setattr(ds_module, "_CUSTOMER_CONTAINER_REMOVAL_TIMEOUT_SECONDS", 0.05)

    with pytest.raises(Exception, match="after a failed docker rm -fv did not finish"):
        await asyncio.wait_for(_clean_for_customer(docker_service, ssh_client), 2)
    assert [cmd.startswith(hung_command) for cmd in hung] == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("workload_kind", "every_filler"),
    [(WorkloadKind.CUSTOMER_RENTAL, True), (WorkloadKind.FILLER, False)],
)
async def test_create_path_asks_for_every_filler_on_a_customer_create_only(
    svc, monkeypatch, workload_kind, every_filler
):
    ssh_client = _ssh_client(inspect_exit=0)
    _patch_happy(svc, monkeypatch, ssh_client)

    result = await _run(svc, _payload(workload_kind=workload_kind))

    assert isinstance(result, ContainerCreated)
    assert svc.clean_existing_containers.await_args.kwargs["remove_every_filler"] is every_filler


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
    assert result.failure_step == "final_filler_check"
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
