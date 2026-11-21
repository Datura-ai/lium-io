"""DAH-6531: the delete for the node a moved pod left runs only on the host that node booted as. Its
registration can point at the replacement's host by the time the delete is sent, so the boot id check and the
removal run as one shell on one connection, before any of the validator's per-pod create state is touched."""

# ruff: noqa: F811 — the docker_service fixture is imported from test_docker_service
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest
from payload_models.payloads import (
    BaseServerRequest,
    ContainerDeleted,
    ContainerDeleteOnNodeLeftRequest,
    FailedContainerErrorCodes,
    FailedContainerRequest,
)
from services.docker_service import _node_left_delete_command, executor_key, inflight_creates

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

    with inflight_creates.track(payload.pod_id, executor_key(payload)):
        result = await _delete(docker_service, payload)
        assert not inflight_creates.is_cancelled(payload.pod_id, executor_key(payload))

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


def _run(
    tmp_path: Path, payload, container_owner: str | None = LEFT_LABEL, sticky: bool = False
) -> tuple[int, str]:
    """Runs the real command against a docker stub: the container exists while its owner file does, a removal
    deletes it unless sticky; the volume answers the left registration's label."""
    log, container, volume = tmp_path / "calls", tmp_path / "container", tmp_path / "volume"
    if container_owner is not None:
        container.write_text(container_owner)
    volume.write_text(LEFT_LABEL)
    stub = tmp_path / "docker"
    keep = "true" if sticky else f"rm -f {shlex.quote(str(container))}"
    stub.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {shlex.quote(str(log))}\n'
        'case "$1" in\n'
        "  ps) echo kept1;;\n"
        f'  inspect) [ -e {shlex.quote(str(container))} ] || exit 1; case "$*" in *-f*) printf "cid "; cat {shlex.quote(str(container))};; esac;;\n'
        f"  rm) {keep};;\n"
        f'  volume) if [ "$2" = inspect ]; then cat {shlex.quote(str(volume))}; fi;;\n'
        "esac\n"
    )
    stub.chmod(0o755)
    payload = payload.model_copy(update={"executor_id": LEFT, "miner_hotkey": LEFT_MINER})
    command = _node_left_delete_command(payload).replace("/usr/bin/docker", str(stub))
    done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
    return done.returncode, log.read_text() if log.exists() else ""


def test_another_boot_runs_no_docker_command(tmp_path):
    status, calls = _run(tmp_path, _payload(NODE_LEFT_BOOT))

    assert (status, calls) == (97, "")


def test_the_node_left_removes_the_container_and_its_kept_containers_never_the_volume(tmp_path):
    # a replacement on the same daemon reuses volume_<pod id> with the left registration's label still on it
    status, calls = _run(tmp_path, _payload(HOST_BOOT))

    assert status == 0
    assert "rm -fv cid kept1" in calls  # the checked container, by its ID
    assert f"label=io.lium.executor_id={LEFT_LABEL}" in calls  # kept containers of the left registration only
    assert "volume" not in calls


@pytest.mark.parametrize("owner", [f"{LEFT_MINER}:replacement-executor", f"miner-b:{LEFT}", LEFT, ""])
def test_a_container_made_for_another_registration_on_this_host_is_never_touched(tmp_path, owner):
    # a replacement created on the left host through a re-pointed registration, or by another miner's registration
    # of the same executor on a shared daemon, carries its own pair; a container made before the label carries none
    status, calls = _run(tmp_path, _payload(HOST_BOOT), container_owner=owner)

    assert status == 97
    assert "rm" not in calls.replace("inspect", "")


def test_with_no_container_left_nothing_is_removed(tmp_path):
    status, calls = _run(tmp_path, _payload(HOST_BOOT), container_owner=None)

    assert status == 0
    assert "volume" not in calls and "rm -fv cid" not in calls


def test_a_container_still_present_fails(tmp_path):
    status, calls = _run(tmp_path, _payload(HOST_BOOT), sticky=True)

    assert status == 1


@pytest.mark.parametrize("missing", ["executor_id", "miner_hotkey"])
def test_a_delete_naming_no_registration_is_refused(missing):
    with pytest.raises(ValueError):
        _node_left_delete_command(_payload(HOST_BOOT).model_copy(update={missing: ""}))


def test_the_backend_message_parses_to_the_checked_delete():
    raw = _payload().model_dump_json()

    assert type(BaseServerRequest.parse(raw)) is ContainerDeleteOnNodeLeftRequest
