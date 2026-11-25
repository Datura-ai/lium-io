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


@pytest.mark.parametrize(
    ("status", "label", "pod", "removal"),
    [
        ("exited", OWNER, "", "rm -v"),  # this registration's earlier attempt: its layer is freed before disk admission
        ("created", "", "", "rm -v"),  # made before the labels
        ("dead", OWNER, "", "rm -v"),
        ("exited", OWNER, POD_ID, "rm -v -f"),  # the pod's own container on this registration, in any state
        # lio#2394 (25 Nov): a re-sent create meets the container its first attempt already started, labelled with
        # this pod and this registration: it is this create's own, so it goes (forced) and the create makes it again
        ("running", OWNER, POD_ID, "rm -v -f"),
        ("running", OWNER, "", None),  # this registration's label without the pod's: not proven this pod's
        ("running", OWNER, "other", None),  # another pod's
        ("running", "", "", None),  # an unlabelled running replacement on a shared daemon
        ("running", "miner-b:exec-b", POD_ID, None),  # another registration's running replacement of the pod
        ("exited", "miner-b:exec-b", POD_ID, None),  # another registration's
        (None, None, None, None),  # the inspect failed
    ],
)
def test_only_this_registrations_leftover_is_removed_by_its_id_forced_only_when_it_is_the_pods_own(
    tmp_path, status, label, pod, removal
):
    import shlex
    import subprocess

    from services.docker_service import _own_leftover_rm_command, _own_stopped_leftover_check

    log = tmp_path / "calls"
    inspect = "exit 1" if status is None else f"echo {shlex.quote(f'{status} {label} {pod}')}"
    stub = tmp_path / "docker"
    stub.write_text(
        f'#!/bin/sh\necho "$*" >> {shlex.quote(str(log))}\ncase "$1" in inspect) {inspect};; rm) exit 0;; esac\n'
    )
    stub.chmod(0o755)
    check = _own_stopped_leftover_check(OWN_ID, OWNER, POD_ID).replace("/usr/bin/docker", str(stub))

    done = subprocess.run(["sh", "-c", check], capture_output=True, text=True)
    # the sweep sends the removal only after a passing check, forced only for the pod's own labelled container
    if done.returncode in (0, 4):
        rm = _own_leftover_rm_command(OWN_ID, force=done.returncode == 4).replace("/usr/bin/docker", str(stub))
        subprocess.run(["sh", "-c", rm], check=True)

    calls = log.read_text().splitlines()
    assert [c for c in calls if c.startswith("rm")] == ([f"{removal} {OWN_ID}"] if removal else [])


@pytest.mark.asyncio
async def test_a_resent_create_removes_the_container_its_first_attempt_started(retry_ssh_mock):
    """lio#2394 (25 Nov): a create the backend re-sends after a connector restart finds the pod's container that the
    first attempt started. The check names it the pod's own (exit 4), so the only create of the pod removes it, forced,
    by its ID, and makes it again; before, it was kept and the re-sent create failed on the name for good."""
    own_sweep_removals.clear()
    ssh_client, seen = _sweep_ssh(f"pod_target {OWN_ID}\n", {"check": 4})
    with ds_module.inflight_creates.track("target", ("miner-a", "exec-a")):
        removed = await _sweep_service().clean_existing_containers(
            ssh_client=ssh_client, default_extra={}, pod_name="pod_target", active_container_names=[], pod_owner=OWNER
        )

    assert [kind for kind, _ in seen] == ["list", "check", "rm"]
    assert f"rm -v -f {OWN_ID}" in ssh_client.commands[-1]
    assert removed == ["pod_target"] and own_sweep_removals.sent_rm_for(OWN_ID)
    assert retry_ssh_mock.await_args_list == []  # no sweep by name, no volume rm
    own_sweep_removals.clear()


def _sweep_ssh(listing: str, answers: dict):
    """An SSH double for the pod create's sweep: the listing through run(), the leftover's check and removal as host
    bounded commands. `answers` maps "check"/"rm" to an exit status, or to an Event the command waits on (a stalled
    daemon behind a live transport); `seen` records each command and whether the leftover was marked at that time."""
    seen = []
    commands = []

    async def answer(command, **_):
        commands.append(command)
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
    ssh_client.commands = commands
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
