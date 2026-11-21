"""DAH-6531: the delete for the node a moved pod left runs only on the host that node booted as. Its
registration can point at the replacement's host by the time the delete is sent, so the boot id check and the
removal run as one shell on one connection, before any of the validator's per-pod create state is touched."""

# ruff: noqa: F811 — the docker_service fixture is imported from test_docker_service
from __future__ import annotations

import asyncio
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


def _payload(boot: str = NODE_LEFT_BOOT) -> ContainerDeleteOnNodeLeftRequest:
    return ContainerDeleteOnNodeLeftRequest(
        **_delete_payload().model_dump(exclude={"message_type"}), expected_boot_id=boot
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


def _daemon_stub(
    tmp_path: Path,
    container_owner: str | None,
    sticky: bool = False,
    sticky_kept: bool = False,
    inspect_fails: bool = False,
) -> Path:
    """A docker stub: the container `cid` exists while its owner file does and the kept container `kept1` while its
    file does; a removal deletes each unless sticky. `inspect_fails` makes inspect fail as a daemon error would, with
    the container still there. Each file under volumes/ is a volume whose lines are its labels (`inuse` marks one a
    container still uses), inspected as "<pod id> <registration>" and removed by name."""
    log, container, kept, volumes = tmp_path / "calls", tmp_path / "container", tmp_path / "kept", tmp_path / "volumes"
    volumes.mkdir(exist_ok=True)
    kept.write_text("")
    if container_owner is not None:
        container.write_text(container_owner)
    c, k, vols = (shlex.quote(str(path)) for path in (container, kept, volumes))
    rm_container = "true" if sticky else f"rm -f {c}"
    rm_kept = "true" if sticky_kept else f"rm -f {k}"
    stub = tmp_path / "docker"
    stub.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {shlex.quote(str(log))}\n'
        'case "$1" in\n'
        f'  ps) case "$*" in *label=*) [ -e {k} ] && echo kept1;; *name=*) [ -e {c} ] && echo cid;; '
        f"*) [ -e {c} ] && echo cid; [ -e {k} ] && echo kept1;; esac; exit 0;;\n"
        + ("  inspect) exit 1;;\n" if inspect_fails else "")
        + f'  inspect) [ -e {c} ] || exit 1; case "$*" in *-f*) printf "cid "; cat {c};; esac;;\n'
        f'  rm) for a in "$@"; do case "$a" in cid) {rm_container};; kept1) {rm_kept};; esac; done;;\n'
        '  volume) eval "v=\\${$#}"; f=' + vols + '/$v; [ -e "$f" ] || exit 1; case "$2" in\n'
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
    stub = _daemon_stub(tmp_path, container_owner, **stub)
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


@pytest.mark.parametrize("missing", ["executor_id", "miner_hotkey", "expected_boot_id"])
def test_a_delete_naming_no_registration_or_boot_id_is_refused(missing):
    with pytest.raises(ValueError):
        _node_left_delete_command(_payload(HOST_BOOT).model_copy(update={missing: ""}))


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
async def test_a_create_still_running_at_the_end_of_the_wait_deletes_nothing(docker_service, monkeypatch):
    monkeypatch.setattr(docker_service_module, "_NODE_LEFT_CREATE_WAIT_SECONDS", 0.05)
    order = _wire_delete(docker_service, monkeypatch, _host_exits(0))
    payload = _payload()

    with inflight_creates.track(payload.pod_id, ("miner-b", "replacement-executor")):
        result = await _delete(docker_service, payload)

    assert isinstance(result, FailedContainerRequest) and order == []


def test_a_node_left_delete_is_never_declined_as_renting_in_progress():
    # sent once: the source create's pending flag outlives its run while its receipt is processed
    from services.miner_service import _bypasses_renting_in_progress

    payload = _payload()
    assert not inflight_creates.is_running(payload.pod_id, (payload.miner_hotkey, payload.executor_id))
    assert _bypasses_renting_in_progress(payload)
    assert not _bypasses_renting_in_progress(ContainerDeleteRequest(**_delete_payload().model_dump(exclude={"message_type"})))
