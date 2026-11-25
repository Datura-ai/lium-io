"""DAH-6531: the delete for the node a moved pod left runs only on the host that node booted as. Its
registration can point at the replacement's host by the time the delete is sent, so the boot id check and the
removal run as one shell on one connection, before any of the validator's per-pod create state is touched."""

# ruff: noqa: F811 — the docker_service fixture is imported from test_docker_service
from __future__ import annotations

import asyncio
import contextvars
import shlex
import subprocess
from pathlib import Path

import pytest
from payload_models.payloads import (
    BaseServerRequest,
    ContainerDeleted,
    ContainerDeleteOnNodeLeftRequest,
    ContainerDeleteRequest,
    FailedContainerErrorCodes,
    FailedContainerRequest,
)
import services.docker_service as docker_service_module
from services.docker_service import (
    _node_left_delete_command,
    inflight_creates,
)

from tests.test_delete_drops_dirty_volume import _delete, _delete_payload, _wire_delete
from tests.test_docker_service import (  # noqa: F401
    _make_ssh_command_result,
    docker_service,
    mock_dependencies,
)

NODE_LEFT_BOOT = "11111111-2222-4333-8444-555555555555"
HOST_BOOT = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
# the boot the backend read the pod's current placement on when it sent the delete: another node's
PLACED_ELSEWHERE_BOOT = "99999999-8888-4777-8666-555555555555"


def _payload(boot: str = NODE_LEFT_BOOT, placed_on: str = PLACED_ELSEWHERE_BOOT) -> ContainerDeleteOnNodeLeftRequest:
    delete = _delete_payload()
    return ContainerDeleteOnNodeLeftRequest(
        **delete.model_dump(exclude={"message_type", "container_name"}),
        container_name=f"pod_{delete.pod_id}",
        expected_boot_id=boot,
        create_command_id=LEFT_PLACEMENT,
        pod_placed_on_boot_id=placed_on,
    )


def _host_exits(status: int):
    async def answer(command: str):
        return _make_ssh_command_result(exit_status=status)

    return answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (97, FailedContainerErrorCodes.NotTheNodeLeft),
        (0, None),
        (1, FailedContainerErrorCodes.UnknownError),
    ],
)
async def test_one_host_command_decides_and_no_sdk_or_create_state_is_touched(
    docker_service, monkeypatch, status, expected
):
    order = _wire_delete(docker_service, monkeypatch, _host_exits(status))
    payload = _payload()

    result = await _delete(docker_service, payload)

    assert (
        len(order) == 1 and order[0][0] == "ssh"
    )  # the check and the removal: one command, no SDK call
    if expected is None:
        assert isinstance(result, ContainerDeleted)
    else:
        assert isinstance(result, FailedContainerRequest) and result.error_code is expected


LEFT = "left-executor"
LEFT_MINER = "miner-a"
LEFT_LABEL = f"{LEFT_MINER}:{LEFT}"
# the create command id of the placement on the node left; the pod placed back on that boot later has another
LEFT_PLACEMENT = "placement-left"
LATER_PLACEMENT = "placement-later"
# an edit of the pod on the node left: its earlier containers carry the create id from before the edit
EDITED_PLACEMENT = "placement-before-the-edit"


def _daemon_stub(
    tmp_path: Path,
    container_owner: str | None,
    sticky: bool = False,
    sticky_kept: bool = False,
    inspect_fails: bool = False,
    pod_id: str = "",
    container_pod: str | None = None,
    prev_owner: str | None = None,
    sticky_prev: bool = False,
    kept_pod: str | None = None,
    placement: str = LEFT_PLACEMENT,
    kept_placement: str = LEFT_PLACEMENT,
    prev_placement: str = LEFT_PLACEMENT,
) -> Path:
    """A docker stub. Each file under containers/ is a container named as the file, holding "<id> <registration
    label> <pod label> <placement label>": the pod's container (`cid`, labelled `container_owner`, `container_pod`,
    default `pod_id`, and `placement`) and its parked edit container `<name>__prev` (`pid`, labelled `prev_owner` and
    `prev_placement`); inspect prints the placement only when the format asks for it. The kept container `kept1` (left registration, pod `kept_pod`, default `pod_id`, `kept_placement`) exists while its file does, listed when every label filter
    matches. A removal deletes each unless sticky. `inspect_fails` makes inspect fail as a daemon
    error would, with the containers still there. Each file under volumes/ is a volume whose lines are its labels
    (`inuse` marks one a container still uses), inspected as "<pod id> <registration>" and removed by name."""
    log, containers, kept, volumes = (
        tmp_path / "calls", tmp_path / "containers", tmp_path / "kept", tmp_path / "volumes"
    )
    volumes.mkdir(exist_ok=True)
    containers.mkdir(exist_ok=True)
    kept.write_text(
        f"io.lium.executor_id={LEFT_LABEL} io.lium.pod_id={pod_id if kept_pod is None else kept_pod} "
        f"io.lium.create_command_id={kept_placement}"
    )
    name = f"pod_{pod_id}"
    for container_name, cid, owner, pod, made_by in [
        (name, "cid", container_owner, pod_id if container_pod is None else container_pod, placement),
        (f"{name}__prev", "pid", prev_owner, pod_id, prev_placement),
    ]:
        if owner is not None:
            (containers / container_name).write_text(f"{cid} {owner} {pod} {made_by}")
    cs, k, vols = (shlex.quote(str(path)) for path in (containers, kept, volumes))
    sticky_ids = " ".join(cid for cid, on in [("cid", sticky), ("pid", sticky_prev)] if on) or "-"
    rm_kept = "true" if sticky_kept else f"rm -f {k}"
    stub = tmp_path / "docker"
    stub.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {shlex.quote(str(log))}\n'
        'eval "last=\\${$#}"\n'
        'case "$1" in\n'
        f'  ps) case "$*" in *label=*) [ -e {k} ] || exit 0; for a in "$@"; do case "$a" in label=*) '
        f'case " $(cat {k}) " in *" ${{a#label=}} "*) ;; *) exit 0;; esac;; esac; done; echo kept1;; '
        f'*name=*) n=${{last#name=^}}; n=${{n%\\$}}; [ -e {cs}/"$n" ] && cut -d" " -f1 {cs}/"$n";; '
        f'*) for f in {cs}/*; do [ -e "$f" ] && cut -d" " -f1 "$f"; done; [ -e {k} ] && echo kept1;; esac; exit 0;;\n'
        + ("  inspect) exit 1;;\n" if inspect_fails else "")
        + f'  inspect) [ -e {cs}/"$last" ] || exit 1; case "$*" in *create_command_id*) cat {cs}/"$last";; '
        f'*) cut -d" " -f1-3 {cs}/"$last";; esac;;\n'
        f'  rm) for a in "$@"; do case "$a" in kept1) {rm_kept};; -*) ;; '
        f'*) case " {sticky_ids} " in *" $a "*) ;; *) for f in {cs}/*; do '
        f'[ -e "$f" ] && [ "$(cut -d" " -f1 "$f")" = "$a" ] && rm -f "$f"; done;; esac;; esac; done;;\n'
        '  volume) f=' + vols + '/$last; [ -e "$f" ] || exit 1; case "$2" in\n'
        '    inspect) pod=$(sed -n "s/^io.lium.pod_id=//p" "$f"); ex=$(sed -n "s/^io.lium.executor_id=//p" "$f"); '
        'echo "$pod $ex";;\n'
        '    rm) grep -qx inuse "$f" && exit 1; rm -f "$f";;\n'
        "  esac;;\n"
        "esac\n"
    )
    stub.chmod(0o755)
    return stub


def _volume(tmp_path: Path, name: str, *labels: str) -> Path:
    path = tmp_path / "volumes" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(f"{label}\n" for label in labels))
    return path


def _run(tmp_path: Path, payload, container_owner: str | None = LEFT_LABEL, **stub) -> tuple[int, str]:
    stub = _daemon_stub(tmp_path, container_owner, pod_id=payload.pod_id, **stub)
    payload = payload.model_copy(update={"executor_id": LEFT, "miner_hotkey": LEFT_MINER})
    command = _node_left_delete_command(payload).replace("/usr/bin/docker", str(stub))
    done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
    log = tmp_path / "calls"
    return done.returncode, log.read_text() if log.exists() else ""


def test_another_boot_runs_no_docker_command(tmp_path):
    status, calls = _run(tmp_path, _payload(NODE_LEFT_BOOT))

    assert (status, calls) == (97, "")


@pytest.mark.parametrize("read", ["unreadable", "empty"])
def test_a_host_boot_id_that_cannot_be_read_never_matches(tmp_path, read):
    # even against an empty expected value (refused before the command is built): a failed read is no boot identity
    boot_file = tmp_path / "boot_id"
    if read == "empty":
        boot_file.write_text("")
    stub = _daemon_stub(tmp_path, LEFT_LABEL, False)
    payload = _payload(HOST_BOOT).model_copy(update={"executor_id": LEFT, "miner_hotkey": LEFT_MINER})
    command = (
        _node_left_delete_command(payload)
        .replace("/usr/bin/docker", str(stub))
        .replace("/proc/sys/kernel/random/boot_id", str(boot_file))
        .replace(f"= {HOST_BOOT} ]", "= '' ]")
    )
    assert "= '' ]" in command

    done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)

    assert done.returncode == 97
    assert not (tmp_path / "calls").exists()


@pytest.mark.parametrize(
    "labels",
    [
        lambda pod: [pod, f"io.lium.executor_id={LEFT_LABEL}"],  # the left registration's
        # a replacement made on this daemon by another registration, or another miner's registration of this executor
        lambda pod: [pod, "io.lium.executor_id=miner-b:replacement-executor"],
        lambda pod: [pod, f"io.lium.executor_id=miner-b:{LEFT}"],
        lambda pod: [],  # made before the labels
    ],
)
@pytest.mark.parametrize("container_owner", [LEFT_LABEL, None])
def test_the_node_left_removes_its_container_and_kept_containers_and_never_the_pods_volume(
    tmp_path, labels, container_owner
):
    # one pod keeps one volume name on every registration: on a shared daemon it may be the replacement's
    payload = _payload(HOST_BOOT)
    volume = _volume(tmp_path, f"volume_{payload.pod_id}", *labels(f"io.lium.pod_id={payload.pod_id}"))

    status, calls = _run(tmp_path, payload, container_owner=container_owner)

    assert status == 0
    assert ("rm -fv cid kept1" in calls) is (container_owner is not None)  # the checked container, by its ID
    assert f"label=io.lium.executor_id={LEFT_LABEL}" in calls  # kept containers of the left registration only
    assert volume.exists()
    assert "volume" not in calls


@pytest.mark.parametrize("owner", [f"{LEFT_MINER}:replacement-executor", f"miner-b:{LEFT}", LEFT, ""])
def test_a_container_made_for_another_registration_on_this_host_is_never_touched(tmp_path, owner):
    # a replacement created on the left host through a re-pointed registration, or by another miner's registration
    # of the same executor on a shared daemon, carries its own pair; a container made before the label carries none
    status, calls = _run(tmp_path, _payload(HOST_BOOT), container_owner=owner)

    assert status == 97
    assert "rm" not in calls.replace("inspect", "")


@pytest.mark.parametrize("stub", [{"sticky": True}, {"sticky_kept": True}])
def test_a_container_still_present_fails_the_delete(tmp_path, stub):
    status, calls = _run(tmp_path, _payload(HOST_BOOT), **stub)

    assert status == 1
    assert "volume" not in calls


def test_a_failed_inspect_of_a_container_still_there_removes_nothing_and_fails(tmp_path):
    # only a listing that succeeded and shows nothing under the name proves the container absent
    status, calls = _run(tmp_path, _payload(HOST_BOOT), inspect_fails=True)

    assert status == 1
    assert "rm " not in calls


def test_a_delete_naming_another_pods_container_is_refused():
    # 2394-12: the name is the pod's own; another pod's container (and its kept containers) is never selected
    payload = _payload(HOST_BOOT)
    with pytest.raises(ValueError):
        _node_left_delete_command(payload.model_copy(update={"container_name": "pod_another"}))


def test_a_container_under_the_pods_name_labelled_for_another_pod_is_never_touched(tmp_path):
    # 2394-12: the left registration's label alone does not make it this pod's
    status, calls = _run(tmp_path, _payload(HOST_BOOT), container_pod="another-pod")

    assert status == 97
    assert "rm" not in calls.replace("inspect", "")


def test_a_kept_container_of_another_pod_under_the_pods_kept_name_is_never_touched(tmp_path):
    # 2394-16: the left registration's label and the kept name's prefix alone do not make it this pod's
    status, calls = _run(tmp_path, _payload(HOST_BOOT), kept_pod="another-pod")

    assert status == 0
    assert "kept1" not in "".join(line for line in calls.splitlines(True) if line.startswith("rm"))
    assert (tmp_path / "kept").exists()


@pytest.mark.parametrize("placement", [LATER_PLACEMENT, ""], ids=["placed-back-later", "made-before-the-label"])
def test_the_pods_container_from_another_placement_on_the_same_boot_is_never_touched(tmp_path, placement):
    # lio#2394 M1: P moves E1 -> E2, E2's create fails and the backend places P back on E1 on the same boot before
    # the node-left delete for the first placement runs: the new live container carries the left registration, the
    # pod and the boot, but its own create command id, so the delete removes nothing
    status, calls = _run(tmp_path, _payload(HOST_BOOT), placement=placement)

    assert status == 97
    assert "rm" not in calls.replace("inspect", "")


@pytest.mark.parametrize(
    "other_placement", [EDITED_PLACEMENT, LATER_PLACEMENT, ""], ids=["edited", "placed-back-later", "made-before-the-label"]
)
@pytest.mark.parametrize("container_owner", [LEFT_LABEL, None], ids=["live", "no-live-container"])
def test_parked_and_kept_containers_of_another_placement_are_kept(tmp_path, other_placement, container_owner):
    # coordinator r460: only a container the left placement's create made goes. One made under another create id (an
    # edit's, the pod placed back on this boot later, or none) may be a later placement's and is kept, whether or not
    # a live container is there to tell
    status, calls = _run(
        tmp_path,
        _payload(HOST_BOOT),
        container_owner=container_owner,
        prev_owner=LEFT_LABEL,
        prev_placement=other_placement,
        kept_placement=other_placement,
    )

    assert status == 0
    removals = [line for line in calls.splitlines() if line.startswith("rm")]
    assert removals == (["rm -fv cid"] if container_owner else [])
    assert f"label=io.lium.create_command_id={LEFT_PLACEMENT}" in calls  # the kept listing names the placement
    assert (tmp_path / "containers" / f"pod_{_run_pod(calls)}__prev").exists() and (tmp_path / "kept").exists()


def _run_pod(calls: str) -> str:
    # the pod id, as the kept listing names it
    return calls.split("label=io.lium.pod_id=")[1].split()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("placed_on", [HOST_BOOT, ""], ids=["placed-back-on-this-boot", "cannot-be-told"])
async def test_a_delete_that_does_not_prove_the_pod_left_this_boot_removes_nothing(
    docker_service, monkeypatch, tmp_path, placed_on
):
    # coordinator r460: the pod placed back on the same boot, no live container, the left placement's parked and kept
    # edit containers there: only the delete itself can prove the pod left, never a live container's absence
    payload = _payload(HOST_BOOT, placed_on=placed_on).model_copy(update={"executor_id": LEFT, "miner_hotkey": LEFT_MINER})
    stub = _daemon_stub(tmp_path, None, pod_id=payload.pod_id, prev_owner=LEFT_LABEL)

    async def the_host_runs_it(command: str):
        done = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", str(stub))], capture_output=True)
        return _make_ssh_command_result(exit_status=done.returncode)

    order = _wire_delete(docker_service, monkeypatch, the_host_runs_it)

    result = await _delete(docker_service, payload)

    assert isinstance(result, FailedContainerRequest) and order == []
    assert (tmp_path / "containers" / f"pod_{payload.pod_id}__prev").exists() and (tmp_path / "kept").exists()
    assert not (tmp_path / "calls").exists()


@pytest.mark.parametrize("container_owner", [LEFT_LABEL, None])
def test_the_left_registrations_parked_edit_container_goes_with_it(tmp_path, container_owner):
    # 2394-13: an edit parks the pod's container as <name>__prev; success means it is gone too
    status, calls = _run(tmp_path, _payload(HOST_BOOT), container_owner=container_owner, prev_owner=LEFT_LABEL)

    assert status == 0
    assert ("rm -fv cid pid kept1" if container_owner else "rm -fv pid kept1") in calls
    assert not any((tmp_path / "containers").iterdir())


def test_a_parked_edit_container_still_present_fails_the_delete(tmp_path):
    status, _ = _run(tmp_path, _payload(HOST_BOOT), prev_owner=LEFT_LABEL, sticky_prev=True)

    assert status == 1


@pytest.mark.parametrize("prev_owner", [f"miner-b:{LEFT}", ""])
def test_a_parked_edit_container_of_another_registration_keeps_everything(tmp_path, prev_owner):
    status, calls = _run(tmp_path, _payload(HOST_BOOT), prev_owner=prev_owner)

    assert status == 97
    assert "rm" not in calls.replace("inspect", "")


@pytest.mark.parametrize(
    "missing", ["executor_id", "miner_hotkey", "expected_boot_id", "create_command_id", "pod_placed_on_boot_id"]
)
def test_a_delete_naming_no_registration_or_boot_id_is_refused(missing):
    with pytest.raises(ValueError):
        _node_left_delete_command(_payload(HOST_BOOT).model_copy(update={missing: ""}))


# coordinator r475: only /proc's own spelling (lowercase 8-4-4-4-12) is a boot id; any other compares unequal to the
# host's boot and would pass as proof that the pod left it
NOT_A_BOOT_ID = {
    "blank": " ",
    "upper-case": HOST_BOOT.upper(),
    "trailing-newline": HOST_BOOT + "\n",
    "padded": f" {HOST_BOOT} ",
    "no-dashes": HOST_BOOT.replace("-", ""),
    "braces": "{" + HOST_BOOT + "}",
    "urn": f"urn:uuid:{HOST_BOOT}",
    "garbage": "garbage",
}


@pytest.mark.parametrize("field", ["expected_boot_id", "pod_placed_on_boot_id"])
@pytest.mark.parametrize("spelling", NOT_A_BOOT_ID.values(), ids=NOT_A_BOOT_ID.keys())
def test_a_boot_id_not_in_canonical_form_removes_nothing(field, spelling):
    payload = _payload(HOST_BOOT).model_copy(update={field: spelling})

    with pytest.raises(ValueError):
        _node_left_delete_command(payload)


def test_the_backend_message_parses_to_the_checked_delete():
    raw = _payload().model_dump_json()

    assert type(BaseServerRequest.parse(raw)) is ContainerDeleteOnNodeLeftRequest


@pytest.mark.asyncio
async def test_the_removal_waits_for_every_create_of_the_pod_and_cancels_none(docker_service, monkeypatch):
    # a late create of the pod from the node left could make its container after the removal; the delete is sent
    # once, so it waits for the pod's creates instead of answering a retry
    order = _wire_delete(docker_service, monkeypatch, _host_exits(0))
    payload = _payload()
    replacement = ("miner-b", "replacement-executor")
    tracked = inflight_creates.track(payload.pod_id, replacement)
    tracked.__enter__()

    delete = asyncio.create_task(_delete(docker_service, payload))
    await asyncio.sleep(0.05)
    assert order == [] and not delete.done()
    assert not inflight_creates.is_cancelled(payload.pod_id, replacement)
    tracked.__exit__(None, None, None)
    result = await asyncio.wait_for(delete, 5)

    assert isinstance(result, ContainerDeleted) and len(order) == 1


@pytest.mark.asyncio
async def test_a_create_of_the_pod_that_registers_during_the_removal_is_removed_by_a_rerun(docker_service, monkeypatch):
    # gateway review on lium-io#2394 at 8f7f513: the left registration's create registers while the removal runs and
    # makes its container after it. The backend sends this delete once, so the validator waits for that create and
    # reruns the identity-checked removal, which removes the second container; only then is it answered as deleted
    payload = _payload()
    tracked = inflight_creates.track(payload.pod_id, (payload.miner_hotkey, payload.executor_id))
    context = contextvars.Context()
    containers = []

    def the_create_makes_its_container_and_ends():
        containers.append("pod")
        context.run(tracked.__exit__, None, None, None)

    async def removal(command: str):
        containers.clear()  # the command removes the pod's container
        if len(order) == 1:  # the delayed create registers during the first removal and makes its container after it
            context.run(tracked.__enter__)
            asyncio.get_running_loop().call_later(0.05, the_create_makes_its_container_and_ends)
        return _make_ssh_command_result(exit_status=0)

    order = _wire_delete(docker_service, monkeypatch, removal)
    result = await asyncio.wait_for(_delete(docker_service, payload), 5)

    assert len(order) == 2  # the second command ran after the create ended
    assert containers == []  # and removed the container the create made
    assert isinstance(result, ContainerDeleted)


@pytest.mark.asyncio
async def test_a_create_of_the_pod_registered_during_the_removal_and_still_running_at_the_deadline_is_never_answered_as_deleted(
    docker_service, monkeypatch
):
    # stand check 2394-10: a create still running when the time runs out may make its container after the removal
    _shrink_the_answer_deadline(monkeypatch)
    payload = _payload()
    tracked = inflight_creates.track(payload.pod_id, (payload.miner_hotkey, payload.executor_id))
    context = contextvars.Context()

    async def a_create_registers_during_the_command(command: str):
        context.run(tracked.__enter__)
        return _make_ssh_command_result(exit_status=0)

    order = _wire_delete(docker_service, monkeypatch, a_create_registers_during_the_command)
    try:
        result = await _delete(docker_service, payload)
    finally:
        context.run(tracked.__exit__, None, None, None)

    assert len(order) == 1  # the removal ran
    assert isinstance(result, FailedContainerRequest)


@pytest.mark.asyncio
async def test_a_create_of_the_pod_registered_right_after_the_final_wait_is_never_answered_as_deleted(
    docker_service, monkeypatch
):
    _shrink_the_answer_deadline(monkeypatch)
    payload = _payload()
    order = _wire_delete(docker_service, monkeypatch, _host_exits(0))
    tracked = inflight_creates.track(payload.pod_id, ("miner-b", "replacement-executor"))
    context = contextvars.Context()
    wait = inflight_creates.wait_for_no_create
    waits = 0

    async def the_final_wait_passes_then_a_create_registers(pod_id, timeout):
        nonlocal waits
        waits += 1
        passed = await wait(pod_id, timeout)
        if waits == 2:
            context.run(tracked.__enter__)
        return passed

    monkeypatch.setattr(inflight_creates, "wait_for_no_create", the_final_wait_passes_then_a_create_registers)
    try:
        result = await _delete(docker_service, payload)
    finally:
        context.run(tracked.__exit__, None, None, None)

    assert waits >= 2
    assert isinstance(result, FailedContainerRequest)
    assert len(order) <= 1


# the answer deadline the test shrinks it to, and the bounds the host command and the dial keep inside it
_ANSWER_BY = 1.0
_RESERVED = 0.4


def _shrink_the_answer_deadline(monkeypatch) -> None:
    monkeypatch.setattr(docker_service_module, "NODE_LEFT_DELETE_ANSWER_SECONDS", _ANSWER_BY)
    monkeypatch.setattr(docker_service_module, "_PARK_COMMAND_TIMEOUT_SECONDS", _RESERVED / 2)
    monkeypatch.setattr(docker_service_module, "_OWN_SSH_CONNECTION_MARGIN_SEC", _RESERVED / 2)


@pytest.mark.asyncio
async def test_a_create_still_running_at_the_end_of_the_wait_deletes_nothing(docker_service, monkeypatch):
    _shrink_the_answer_deadline(monkeypatch)
    order = _wire_delete(docker_service, monkeypatch, _host_exits(0))
    payload = _payload()

    with inflight_creates.track(payload.pod_id, ("miner-b", "replacement-executor")):
        result = await _delete(docker_service, payload)

    assert isinstance(result, FailedContainerRequest) and order == []


@pytest.mark.asyncio
@pytest.mark.parametrize("create_runs", [0.2, 10.0])
async def test_the_delete_is_answered_within_the_deadline_the_backend_holds_for(docker_service, monkeypatch, create_runs):
    # review on lium-io#2394: the backend holds the replacement's name-held failure for NODE_LEFT_DELETE_ANSWER_SECONDS
    # after sending this delete. Both waits for the pod's creates (before the dial and under the lock) and the host
    # command share that one deadline, so the answer comes inside it: never after two full waits and the command
    _shrink_the_answer_deadline(monkeypatch)
    payload = _payload()
    host_commands: list[str] = []

    async def the_host_stalls(command: str):
        host_commands.append(command)
        await asyncio.sleep(10)

    _wire_delete(docker_service, monkeypatch, the_host_stalls)
    prepare = docker_service._prepare_known_hosts_policy

    async def dial_while_a_create_runs(*args, **kwargs):
        registered = asyncio.Event()

        async def create():  # of the pod, registered after the first wait passed; its own task and context
            with inflight_creates.track(payload.pod_id, ("miner-b", "replacement-executor")):
                registered.set()
                await asyncio.sleep(create_runs)

        asyncio.get_running_loop().create_task(create(), context=contextvars.Context())
        await registered.wait()
        return await prepare(*args, **kwargs)

    docker_service._prepare_known_hosts_policy = dial_while_a_create_runs
    started = asyncio.get_running_loop().time()
    result = await asyncio.wait_for(_delete(docker_service, payload), 5)

    assert isinstance(result, FailedContainerRequest)
    assert asyncio.get_running_loop().time() - started < _ANSWER_BY + 0.2
    # a create that ended in time leaves the command its own bound; one still running sends nothing to the host
    assert len(host_commands) == (1 if create_runs < _ANSWER_BY - _RESERVED else 0)


@pytest.mark.asyncio
async def test_the_answer_deadline_counts_from_the_requests_arrival_not_from_the_delete(docker_service, monkeypatch):
    # review on lium-io#2394 (F-57): the backend counts its hold from the send, so a slow miner handshake before the
    # delete starts must come out of the validator's deadline too, or the answer lands after the backend gave up
    _shrink_the_answer_deadline(monkeypatch)
    _wire_delete(docker_service, monkeypatch, _host_exits(0))
    payload = _payload()
    received = asyncio.get_running_loop().time()
    await asyncio.sleep(_ANSWER_BY / 2)  # the miner handshake

    with inflight_creates.track(payload.pod_id, ("miner-b", "replacement-executor")):
        result = await asyncio.wait_for(_delete(docker_service, payload), 5)

    assert isinstance(result, FailedContainerRequest)
    assert asyncio.get_running_loop().time() - received < _ANSWER_BY + 0.1


def test_the_answer_deadline_is_the_protocols():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lium_protocol"))
    from lium_protocol import backend_to_validator

    assert docker_service_module.NODE_LEFT_DELETE_ANSWER_SECONDS == backend_to_validator.NODE_LEFT_DELETE_ANSWER_SECONDS


@pytest.mark.asyncio
async def test_the_removal_never_touches_the_rented_machine_record_the_replacement_shares(docker_service, monkeypatch):
    # stand check 2394-08: the record is keyed by endpoint, and the container name is the pod's on both registrations,
    # so a replacement at the same endpoint would lose its entry
    from fakeredis import FakeServer
    from fakeredis.aioredis import FakeRedis
    from services.redis_service import RedisService

    from tests.test_delete_drops_dirty_volume import _executor_info

    _wire_delete(docker_service, monkeypatch, _host_exits(0))
    redis = RedisService.__new__(RedisService)
    redis.redis = FakeRedis(server=FakeServer())
    docker_service.redis_service = redis
    payload = _payload()
    executor = _executor_info(payload)
    await redis.add_rented_pod(executor, payload.pod_id, payload.container_name)
    await redis.add_rented_pod(executor, "other-pod", "pod_other")

    result = await _delete(docker_service, payload)

    assert isinstance(result, ContainerDeleted)
    names = [container["name"] for container in (await redis.get_rented_machine(executor))["containers"]]
    assert names == [payload.container_name, "pod_other"]


def test_a_node_left_delete_is_never_declined_as_renting_in_progress():
    # sent once: the source create's pending flag outlives its run while its receipt is processed
    from services.miner_service import _bypasses_renting_in_progress

    payload = _payload()
    assert not inflight_creates.is_running(payload.pod_id, (payload.miner_hotkey, payload.executor_id))
    assert _bypasses_renting_in_progress(payload)
    assert not _bypasses_renting_in_progress(ContainerDeleteRequest(**_delete_payload().model_dump(exclude={"message_type"})))
