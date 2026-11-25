"""DAH-5392: a delayed create's stale sweep never removes a newer create's `pod_*` on a split node.

Create A's active_container_names are built by the backend before it sends A; if A waits (the per-node
create exclusion) and create B runs first, B's pod is not on A's list. The sweep removes an unlisted
`pod_*` (and its volume) only when the backend confirms its rental closed; anything else keeps it.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, Mock

import pytest

import services.docker_service as ds_module
from protocol.vc_protocol.compute_requests import PodRentalActiveResponse
from services.docker_service import DockerService, own_sweep_removals

NEWER = "11111111-1111-1111-1111-111111111111"
LEFTOVER = "22222222-2222-2222-2222-222222222222"
CLOSED = PodRentalActiveResponse(active=False)
ACTIVE = PodRentalActiveResponse(active=True, status="PENDING")


@pytest.fixture(autouse=True)
def _no_sweeps_from_other_tests():
    own_sweep_removals.clear()


@pytest.fixture
def retry_ssh_mock(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(ds_module, "retry_ssh_command", mock)
    return mock


async def _sweep(answers: dict, retry_ssh_mock, backend: bool = True) -> tuple[list[str], str]:
    async def rental_active(pod_id):
        answer = answers[pod_id]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    backend_client = Mock(get_pod_rental_active=AsyncMock(side_effect=rental_active)) if backend else None
    svc = DockerService(
        ssh_service=Mock(),
        redis_service=Mock(get=AsyncMock(return_value=None)),
        attestation_service=Mock(),
        backend_client=backend_client,
    )
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout=f"pod_target\npod_{NEWER}\npod_{LEFTOVER}\n", stderr=""))
    removed = await svc.clean_existing_containers(
        ssh_client=ssh_client, default_extra={}, pod_name="pod_target", active_container_names=[]
    )
    if backend_client is not None:
        # the create's own container is never asked about: its handling is unchanged
        assert sorted(call.args[0] for call in backend_client.get_pod_rental_active.await_args_list) == [NEWER, LEFTOVER]
    return removed, " ".join(call.args[1] for call in retry_ssh_mock.await_args_list)


@pytest.mark.asyncio
async def test_an_active_pod_is_kept_and_a_closed_leftover_removed_with_its_volume(retry_ssh_mock):
    removed, sent = await _sweep({NEWER: ACTIVE, LEFTOVER: CLOSED}, retry_ssh_mock)

    assert removed == ["pod_target", f"pod_{LEFTOVER}"]
    assert f"pod_{NEWER}" not in sent and f"volume_{NEWER}" not in sent
    assert f"volume_{LEFTOVER}" in sent


@pytest.mark.parametrize(
    "answer",
    [None, ACTIVE, RuntimeError("backend down"), TimeoutError()],
    ids=["no-answer", "active", "error", "timeout"],
)
@pytest.mark.asyncio
async def test_a_pod_whose_rental_is_not_confirmed_closed_is_kept_with_its_volume(retry_ssh_mock, answer):
    removed, sent = await _sweep({NEWER: answer, LEFTOVER: answer}, retry_ssh_mock)

    assert removed == ["pod_target"]
    assert f"volume_{NEWER}" not in sent and f"volume_{LEFTOVER}" not in sent


@pytest.mark.asyncio
async def test_no_backend_client_keeps_every_unlisted_pod(retry_ssh_mock):
    removed, sent = await _sweep({}, retry_ssh_mock, backend=False)

    assert removed == ["pod_target"]
    assert "volume_1" not in sent and "volume_2" not in sent


@pytest.mark.asyncio
async def test_a_name_that_is_no_backend_pod_is_removed_without_asking(retry_ssh_mock):
    backend_client = Mock(get_pod_rental_active=AsyncMock())
    svc = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock(), backend_client=backend_client)
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout="pod_stale\npod_stale__prev\n", stderr=""))

    removed = await svc.clean_existing_containers(ssh_client=ssh_client, default_extra={}, pod_name="pod_target")

    assert removed == ["pod_stale", "pod_stale__prev"]
    backend_client.get_pod_rental_active.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_hung_backend_answer_is_bounded_and_keeps_the_pod(retry_ssh_mock, monkeypatch):
    monkeypatch.setattr(ds_module, "_POD_RENTAL_ACTIVE_TIMEOUT_SECONDS", 0.01)
    never = asyncio.get_running_loop().create_future()

    async def hang(_pod_id):
        return await never

    svc = DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        backend_client=Mock(get_pod_rental_active=hang),
    )
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout=f"pod_{NEWER}\n", stderr=""))

    assert await svc.clean_existing_containers(ssh_client=ssh_client, default_extra={}, pod_name="pod_target") == []


@pytest.mark.asyncio
async def test_lookups_are_capped_per_sweep_and_unchecked_pods_kept(retry_ssh_mock):
    """A provider can leave thousands of stopped pod_<uuid> containers: one sweep asks about at most the cap."""
    cap = ds_module.MAX_UNLISTED_POD_LOOKUPS
    ids = [f"{i:08d}-0000-0000-0000-000000000000" for i in range(cap + 5)]
    backend_client = Mock(get_pod_rental_active=AsyncMock(return_value=CLOSED))
    svc = DockerService(
        ssh_service=Mock(),
        redis_service=Mock(get=AsyncMock(return_value=None)),
        attestation_service=Mock(),
        backend_client=backend_client,
    )
    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(return_value=Mock(exit_status=0, stdout="".join(f"pod_{i}\n" for i in ids), stderr=""))

    removed = await svc.clean_existing_containers(
        ssh_client=ssh_client, default_extra={}, pod_name="pod_target", active_container_names=[]
    )

    asked = {call.args[0] for call in backend_client.get_pod_rental_active.await_args_list}
    assert len(asked) == backend_client.get_pod_rental_active.await_count == cap
    assert sorted(removed) == sorted(f"pod_{i}" for i in asked)
    sent = " ".join(call.args[1] for call in retry_ssh_mock.await_args_list)
    for unchecked in set(ids) - asked:
        assert f"pod_{unchecked}" not in removed and f"pod_{unchecked}" not in sent


@pytest.mark.asyncio
async def test_many_listed_pods_are_read_a_bounded_number_at_a_time(monkeypatch):
    monkeypatch.setattr(ds_module, "MAX_UNLISTED_POD_LOOKUPS", 500)
    # a provider's docker ps can list thousands of pod_<uuid> names; one backend read per name, never all at once
    running = peak = 0

    async def rental_active(_pod_id):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0)
        running -= 1
        return CLOSED

    backend_client = Mock(get_pod_rental_active=AsyncMock(side_effect=rental_active))
    svc = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock(), backend_client=backend_client)
    names = [f"pod_{uuid.uuid4()}" for _ in range(500)]

    closed = await svc._pods_with_closed_rental(names, {})

    assert closed == set(names)
    assert backend_client.get_pod_rental_active.await_count == 500
    assert peak == ds_module._POD_RENTAL_READS_AT_ONCE


@pytest.mark.asyncio
async def test_a_stalled_backend_holds_a_long_sweep_one_timeout_window_not_one_per_batch(monkeypatch):
    monkeypatch.setattr(ds_module, "_POD_RENTAL_ACTIVE_TIMEOUT_SECONDS", 0.05)

    calls = 0

    async def hang(_pod_id):
        nonlocal calls
        calls += 1
        await asyncio.sleep(3600)

    svc = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock(), backend_client=Mock(get_pod_rental_active=hang))
    names = [f"pod_{uuid.uuid4()}" for _ in range(500)]
    loop = asyncio.get_running_loop()
    started = loop.time()

    closed = await svc._pods_with_closed_rental(names, {})

    assert closed == set()
    # 32 batches of 16 would wait 32 windows, and no read starts once the deadline has passed
    assert loop.time() - started < 0.05 * 4
    assert calls == ds_module._POD_RENTAL_READS_AT_ONCE


@pytest.mark.asyncio
async def test_a_long_list_runs_a_bounded_number_of_tasks():
    tasks_seen = 0

    async def rental_active(_pod_id):
        nonlocal tasks_seen
        tasks_seen = max(tasks_seen, len(asyncio.all_tasks()))
        await asyncio.sleep(0)
        return CLOSED

    svc = DockerService(
        ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock(),
        backend_client=Mock(get_pod_rental_active=AsyncMock(side_effect=rental_active)),
    )

    await svc._pods_with_closed_rental([f"pod_{uuid.uuid4()}" for _ in range(500)], {})

    # the test's own task plus one per worker, never one per name
    assert tasks_seen <= ds_module._POD_RENTAL_READS_AT_ONCE + 1


OWN_ID = "a" * 64
OWNER = "miner-a:exec-a"


POD_ID = "target"
# the create attempt running the sweep (redis_service.renter_create_id); an earlier attempt of the pod had another
ATTEMPT = "attempt-now"
EARLIER = "attempt-done"


# the backend's create command ids (io.lium.create_command_id): a failed create the retry names as superseded, and
# the create that made the renter's live pod
FAILED = "cmd-failed"
LIVE = "cmd-live"


@pytest.mark.parametrize(
    ("status", "label", "pod", "attempt", "command", "current", "superseded", "removal"),
    [
        # lio#2394 (25 Nov, coordinator): the backend's retry after a failed create names that create's command id;
        # its stopped container (this registration, this pod, that id) is freed before disk admission, not forced
        ("exited", OWNER, POD_ID, EARLIER, FAILED, ATTEMPT, FAILED, "rm -v"),
        ("created", OWNER, POD_ID, EARLIER, FAILED, ATTEMPT, FAILED, "rm -v"),
        ("dead", OWNER, POD_ID, EARLIER, FAILED, ATTEMPT, FAILED, "rm -v"),
        # a renter-stopped pod is a live rental: its create is not the superseded one, so it is kept
        ("exited", OWNER, POD_ID, EARLIER, LIVE, ATTEMPT, FAILED, None),
        # no superseded id (an older backend, or no retry): every stopped container is kept, the create fails on the name
        ("exited", OWNER, POD_ID, EARLIER, LIVE, ATTEMPT, "", None),
        ("exited", OWNER, POD_ID, EARLIER, FAILED, ATTEMPT, "", None),
        ("exited", OWNER, POD_ID, EARLIER, "", ATTEMPT, FAILED, None),  # made before the command label
        ("created", "", "", "", "", ATTEMPT, FAILED, None),  # made before the labels
        ("exited", OWNER, "", "", FAILED, ATTEMPT, FAILED, None),  # this registration's label without the pod's
        ("exited", OWNER, "other", EARLIER, FAILED, ATTEMPT, FAILED, None),  # another pod's
        ("exited", "miner-b:exec-b", POD_ID, EARLIER, FAILED, ATTEMPT, FAILED, None),  # another registration's
        # the superseded create's container running (started meanwhile, or never failed): the running rule keeps it
        ("running", OWNER, POD_ID, EARLIER, FAILED, ATTEMPT, FAILED, None),
        ("exited", OWNER, POD_ID, ATTEMPT, LIVE, ATTEMPT, "", "rm -v -f"),  # this very attempt's own container
        # lio#2394 (25 Nov): a create re-sent after a connector restart resumes its cut-off attempt (same attempt id)
        # and meets the container that attempt already started: provably its own, so it goes (forced) and is made again
        ("running", OWNER, POD_ID, ATTEMPT, LIVE, ATTEMPT, "", "rm -v -f"),
        # lio#2394 r450 H1: the pod's live container from an earlier create that finished (dedupe expired, a new
        # command id, a validator restart, a delayed create after a repointed registration): never removed
        ("running", OWNER, POD_ID, EARLIER, LIVE, ATTEMPT, "", None),
        ("running", OWNER, POD_ID, EARLIER, LIVE, ATTEMPT, LIVE, None),  # even when named superseded: it runs
        ("running", OWNER, POD_ID, "", "", ATTEMPT, "", None),  # the pod's live container made before the attempt label
        ("running", OWNER, POD_ID, "", "", "", "", None),  # a create with no attempt id proves nothing about it
        ("running", OWNER, "", ATTEMPT, "", ATTEMPT, "", None),  # this registration's label without the pod's
        ("running", OWNER, "other", ATTEMPT, "", ATTEMPT, "", None),  # another pod's
        ("running", "", "", "", "", ATTEMPT, "", None),  # an unlabelled running replacement on a shared daemon
        ("running", "miner-b:exec-b", POD_ID, ATTEMPT, "", ATTEMPT, "", None),  # another registration's replacement
        (None, None, None, None, None, ATTEMPT, FAILED, None),  # the inspect failed
    ],
)
def test_only_this_registrations_leftover_is_removed_by_its_id_forced_only_when_it_is_this_attempts_own(
    tmp_path, status, label, pod, attempt, command, current, superseded, removal
):
    import subprocess

    from services.docker_service import _own_leftover_rm_command, _own_stopped_leftover_check

    log = tmp_path / "calls"
    stub = _docker_stub(tmp_path, log, None if status is None else (status, label, pod, attempt, command))
    check = _own_stopped_leftover_check(OWN_ID, OWNER, POD_ID, current, superseded).replace("/usr/bin/docker", str(stub))

    done = subprocess.run(["sh", "-c", check], capture_output=True, text=True)
    # the sweep sends the removal only after a passing check, forced only for this attempt's own labelled container
    if done.returncode in (0, 4):
        rm = _own_leftover_rm_command(OWN_ID, force=done.returncode == 4).replace("/usr/bin/docker", str(stub))
        subprocess.run(["sh", "-c", rm], check=True)

    calls = log.read_text().splitlines()
    assert [c for c in calls if c.startswith("rm")] == ([f"{removal} {OWN_ID}"] if removal else [])


def _docker_stub(tmp_path, log, inspected: tuple | None):
    """A docker CLI that logs each call and answers rm with 0. `inspected` is (state, registration, pod, attempt,
    create command id) (None: the inspect fails); inspect renders the -f template from it, so a check reading fewer
    labels sees fewer."""
    import json
    import shlex
    import sys

    from services.const import (
        LIUM_CREATE_ATTEMPT_LABEL,
        LIUM_CREATE_COMMAND_LABEL,
        LIUM_EXECUTOR_ID_LABEL,
        LIUM_POD_ID_LABEL,
    )

    container = None
    if inspected is not None:
        state, registration, pod, attempt, command = inspected
        container = {
            "state": state,
            "labels": {
                LIUM_EXECUTOR_ID_LABEL: registration,
                LIUM_POD_ID_LABEL: pod,
                LIUM_CREATE_ATTEMPT_LABEL: attempt,
                LIUM_CREATE_COMMAND_LABEL: command,
            },
        }
    render = tmp_path / "render.py"
    render.write_text(
        "import json, re, sys\n"
        f"c = json.loads({json.dumps(json.dumps(container))})\n"
        "if c is None: sys.exit(1)\n"
        "t = sys.argv[sys.argv.index('-f') + 1]\n"
        "t = t.replace('{{.State.Status}}', c['state'])\n"
        "print(re.sub(r'\\{\\{index \\.Config\\.Labels \"([^\"]+)\"\\}\\}', lambda m: c['labels'].get(m.group(1), ''), t))\n"
    )
    stub = tmp_path / "docker"
    stub.write_text(
        f'#!/bin/sh\necho "$*" >> {shlex.quote(str(log))}\n'
        f'case "$1" in inspect) exec {shlex.quote(sys.executable)} {shlex.quote(str(render))} "$@";; rm) exit 0;; esac\n'
    )
    stub.chmod(0o755)
    return stub


def _host_shell_ssh(stub):
    """An SSH double that runs each command in a local shell with `stub` as docker: the real check decides."""
    import subprocess

    async def answer(command, **_):
        done = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", str(stub))], capture_output=True, text=True)
        return Mock(exit_status=done.returncode, stdout=done.stdout, stderr=done.stderr)

    async def create_process(command, *args, **kwargs):
        return Mock(wait=lambda: answer(command), wait_closed=AsyncMock(), close=Mock())

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=answer)
    ssh_client.create_process = create_process
    return ssh_client


@pytest.mark.parametrize(
    ("state", "container_attempt", "command", "superseded", "removal"),
    [
        # the create re-sent after a connector restart resumes its cut-off attempt: the container that attempt
        # started is this attempt's own, so it goes (forced) and the create makes it again
        ("running", ATTEMPT, LIVE, "", "rm -v -f"),
        # lio#2394 r450 H1: create C1 finished and the renter uses the pod; a later create of the pod on this
        # registration (dedupe answer expired after 900 s, dedupe failed open, a new command id, a validator restart
        # that emptied the in-process registry, a delayed create after a repointed registration) is the only create
        # running now, but the live container is C1's: it stays and the create fails on the name instead
        ("running", EARLIER, LIVE, "", None),
        ("running", "", "", "", None),  # made before the attempt label
        # lio#2394 (25 Nov, coordinator): a renter-stopped pod is a live rental, kept whatever the create names
        ("exited", EARLIER, LIVE, FAILED, None),
        ("exited", EARLIER, LIVE, "", None),
        # the backend's retry after a failed create names it: that create's stopped container goes, not forced
        ("exited", EARLIER, FAILED, FAILED, "rm -v"),
        # an older backend names nothing: the failed create's stopped container is kept, the create fails on the name
        ("exited", EARLIER, FAILED, "", None),
        ("exited", "", "", FAILED, None),  # made before the labels
    ],
    ids=[
        "this-attempts-own",
        "finished-earlier-create-live-pod",
        "unlabelled-live-pod",
        "renter-stopped-live-pod-kept",
        "renter-stopped-live-pod-kept-no-superseded",
        "superseded-failed-attempt-removed",
        "failed-attempt-kept-without-superseded-id",
        "unlabelled-stopped-kept",
    ],
)
@pytest.mark.asyncio
async def test_the_only_create_of_the_pod_force_removes_only_its_own_attempts_running_container(
    tmp_path, retry_ssh_mock, state, container_attempt, command, superseded, removal
):
    from services.redis_service import renter_create_id

    own_sweep_removals.clear()
    log = tmp_path / "calls"
    stub = _docker_stub(tmp_path, log, (state, OWNER, POD_ID, container_attempt, command))
    ssh_client = _host_shell_ssh(stub)
    ssh_client.run.side_effect = _with_listing(ssh_client.run.side_effect, f"pod_target {OWN_ID}\n")
    token = renter_create_id.set(ATTEMPT)
    try:
        with ds_module.inflight_creates.track("target", ("miner-a", "exec-a")):
            result = await _sweep_service().clean_existing_containers(
                ssh_client=ssh_client,
                default_extra={},
                pod_name="pod_target",
                active_container_names=[],
                pod_owner=OWNER,
                superseded_command_id=superseded,
            )
    finally:
        renter_create_id.reset(token)

    rms = [c for c in log.read_text().splitlines() if c.startswith("rm")]
    removed = removal is not None
    assert rms == ([f"{removal} {OWN_ID}"] if removed else [])
    assert (result == ["pod_target"]) is removed and bool(own_sweep_removals.sent_rm_for(OWN_ID)) is removed
    assert retry_ssh_mock.await_args_list == []  # no sweep by name, no volume rm
    own_sweep_removals.clear()


def _with_listing(answer, listing: str):
    """The sweep's container listing answered with `listing`; every other command goes to `answer`."""

    async def run(command, **kwargs):
        if "docker inspect" in command or "docker rm" in command:
            return await answer(command, **kwargs)
        return Mock(exit_status=0, stdout=listing, stderr="")

    return run


def _sweep_ssh(listing: str, answers: dict):
    """An SSH double for the pod create's sweep: the listing through run(), the leftover's check and removal as host
    bounded commands. `answers` maps "check"/"rm" to an exit status, or to an Event the command waits on (a stalled
    daemon behind a live transport); `seen` records each command and whether the leftover was marked at that time."""
    seen = []

    async def answer(command, **_):
        kind = "check" if "docker inspect" in command else "rm" if "docker rm -v" in command else "list"
        seen.append((kind, own_sweep_removals.sent_rm_for(OWN_ID)))
        reply = answers.get(kind, 0)
        if isinstance(reply, asyncio.Event):
            await reply.wait()
        return Mock(exit_status=reply if isinstance(reply, int) else 0, stdout=listing if kind == "list" else "", stderr="")

    async def create_process(command, *args, **kwargs):
        # the channel opens at once; the command's answer is what may never come
        return Mock(wait=lambda: answer(command), wait_closed=AsyncMock(), close=Mock())

    ssh_client = AsyncMock()
    ssh_client.run = AsyncMock(side_effect=answer)
    ssh_client.create_process = create_process
    return ssh_client, seen


def _sweep_service():
    return DockerService(
        ssh_service=Mock(), redis_service=Mock(get=AsyncMock(return_value=None)), attestation_service=Mock()
    )


@pytest.mark.parametrize(("check_exit", "rm_exit"), [(0, 0), (0, 1), (3, 0)])
@pytest.mark.parametrize("listing", [f"pod_target {OWN_ID}\n", "pod_target\n"], ids=["with-id", "without-id"])
@pytest.mark.asyncio
async def test_a_pod_creates_sweep_never_force_removes_the_pods_own_name_or_its_volume(
    retry_ssh_mock, listing, check_exit, rm_exit
):
    """DAH-6531: one pod keeps one name on every registration; on a shared daemon it may be another registration's
    running replacement, labelled or made before the labels. Only the checked leftover may be removed, and a removal
    is recorded as this sweep's only once the check passed: one Docker refused leaves no record, one the host timeout
    cut short keeps it (Docker may still complete it) and ends the sweep (see the host-timeout test)."""
    own_sweep_removals.clear()
    ssh_client, seen = _sweep_ssh(listing, {"check": check_exit, "rm": rm_exit})

    removed = await _sweep_service().clean_existing_containers(
        ssh_client=ssh_client, default_extra={}, pod_name="pod_target", active_container_names=[], pod_owner=OWNER
    )

    listed = "aaaa" in listing
    assert retry_ssh_mock.await_args_list == []  # no forced rm, no volume rm
    expected = [("list", False)] + (
        [("check", False)] + ([("rm", True)] if check_exit == 0 else []) if listed else []
    )
    assert seen == expected  # nothing is marked before the check passes
    assert own_sweep_removals.sent_rm_for(OWN_ID) is (listed and check_exit == 0 and rm_exit == 0)
    assert removed == (["pod_target"] if listed and check_exit == 0 and rm_exit == 0 else [])
    own_sweep_removals.clear()


@pytest.mark.parametrize("stalled", ["check", "rm"])
@pytest.mark.asyncio
async def test_a_stalled_docker_behind_a_live_ssh_session_ends_the_sweep_at_its_deadline(
    retry_ssh_mock, monkeypatch, stalled
):
    # the SSH transport stays open while the daemon never answers the check or the removal
    monkeypatch.setattr(ds_module, "REMOVE_CONFIRM_INSPECT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(ds_module, "_CUSTOMER_CONTAINER_REMOVAL_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(ds_module, "_HOST_TIMEOUT_VALIDATOR_MARGIN_SECONDS", 0.05)
    own_sweep_removals.clear()
    ssh_client, _ = _sweep_ssh(f"pod_target {OWN_ID}\n", {stalled: asyncio.Event()})

    sweep_task = asyncio.ensure_future(
        _sweep_service().clean_existing_containers(
            ssh_client=ssh_client, default_extra={}, pod_name="pod_target", active_container_names=[], pod_owner=OWNER
        )
    )
    done, _ = await asyncio.wait({sweep_task}, timeout=5)

    assert sweep_task in done
    with pytest.raises(TimeoutError):
        sweep_task.result()
    own_sweep_removals.clear()


@pytest.mark.parametrize("status", [124, 137])
@pytest.mark.parametrize("stalled", ["check", "rm"])
@pytest.mark.asyncio
async def test_a_host_timeout_of_the_check_or_removal_ends_the_sweep_before_any_volume_step(
    retry_ssh_mock, stalled, status
):
    # the host's own timeout answers first: the sweep must not go on to the volume listing, which has no deadline
    own_sweep_removals.clear()
    ssh_client, seen = _sweep_ssh(f"pod_target {OWN_ID}\n", {stalled: status})

    with pytest.raises(TimeoutError):
        await _sweep_service().clean_existing_containers(
            ssh_client=ssh_client,
            default_extra={},
            pod_name="pod_target",
            active_container_names=[],
            pod_owner=OWNER,
            clear_volume=True,
        )

    assert [kind for kind, _ in seen] == ["list", "check"] + (["rm"] if stalled == "rm" else [])
    assert retry_ssh_mock.await_args_list == []
    assert own_sweep_removals.sent_rm_for(OWN_ID) is (stalled == "rm")  # a removal that may still complete stays marked
    own_sweep_removals.clear()


@pytest.mark.parametrize(
    "others",
    [
        [("miner-b", "exec-b")],  # another registration's create of the pod
        [("miner-a", "exec-a"), ("miner-a", "exec-a")],  # this registration's create, registered twice
    ],
    ids=["other-registration", "same-registration-twice"],
)
@pytest.mark.parametrize("caller_tracked", [True, False])
@pytest.mark.asyncio
async def test_the_sweep_keeps_the_pods_leftover_while_another_create_of_the_pod_runs(
    retry_ssh_mock, others, caller_tracked
):
    # stand check 2394-19: the leftover is named by the pod alone, so another live create of the pod may be about to
    # take it; only the one create of the pod (the caller's, if tracked) may remove it
    import contextlib
    import contextvars

    registry = ds_module.inflight_creates
    own_sweep_removals.clear()
    ssh_client, seen = _sweep_ssh(f"pod_target {OWN_ID}\n", {})
    tracked = [(contextvars.Context(), registry.track("target", executor)) for executor in others]
    for ctx, cm in tracked:  # each in its own context, as separate creates are
        ctx.run(cm.__enter__)
    try:
        with registry.track("target", ("miner-a", "exec-a")) if caller_tracked else contextlib.nullcontext():
            removed = await _sweep_service().clean_existing_containers(
                ssh_client=ssh_client, default_extra={}, pod_name="pod_target", active_container_names=[], pod_owner=OWNER
            )
    finally:
        for ctx, cm in tracked:
            ctx.run(cm.__exit__, None, None, None)

    assert "rm" not in [kind for kind, _ in seen]
    assert not own_sweep_removals.sent_rm_for(OWN_ID)
    assert removed == []
    own_sweep_removals.clear()


@pytest.mark.asyncio
async def test_the_sweep_of_the_only_create_of_the_pod_removes_its_leftover(retry_ssh_mock):
    own_sweep_removals.clear()
    ssh_client, seen = _sweep_ssh(f"pod_target {OWN_ID}\n", {})
    with ds_module.inflight_creates.track("target", ("miner-a", "exec-a")):
        removed = await _sweep_service().clean_existing_containers(
            ssh_client=ssh_client, default_extra={}, pod_name="pod_target", active_container_names=[], pod_owner=OWNER
        )

    assert removed == ["pod_target"] and own_sweep_removals.sent_rm_for(OWN_ID)
    own_sweep_removals.clear()
