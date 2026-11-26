"""DAH-6531: the delete for the node a moved pod left removes, by the inspected ID, only the pod's container that the
left placement's create made (labelled with the pod and its create command id)."""

# ruff: noqa: F811 — the docker_service fixture is imported from test_docker_service
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from payload_models.payloads import (
    BaseServerRequest,
    ContainerDeleted,
    ContainerDeleteOnNodeLeftRequest,
    ContainerDeleteRequest,
    FailedContainerErrorCodes,
    FailedContainerRequest,
)
from services.docker_service import _node_left_delete_command, inflight_creates

from tests.test_delete_drops_dirty_volume import _delete, _delete_payload, _wire_delete
from tests.test_docker_service import (  # noqa: F401
    _make_ssh_command_result,
    docker_service,
    mock_dependencies,
)

LEFT_PLACEMENT = "placement-left"


def _payload() -> ContainerDeleteOnNodeLeftRequest:
    delete = _delete_payload()
    return ContainerDeleteOnNodeLeftRequest(
        **delete.model_dump(exclude={"message_type", "container_name"}),
        container_name=f"pod_{delete.pod_id}",
        create_command_id=LEFT_PLACEMENT,
    )


def _daemon(tmp_path: Path, containers: dict[str, str], sticky: bool = False) -> Path:
    """A docker stub: each file under containers/ is a container named as the file, holding "<id> <pod label>
    <create command id label>". inspect prints it, ps lists ids (by exact name, or all), rm deletes by id."""
    store = tmp_path / "containers"
    store.mkdir()
    for name, fields in containers.items():
        (store / name).write_text(fields)
    cs, log = shlex.quote(str(store)), shlex.quote(str(tmp_path / "calls"))
    stub = tmp_path / "docker"
    stub.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {log}\n'
        'eval "last=\\${$#}"\n'
        'case "$1" in\n'
        f'  inspect) [ -e {cs}/"$last" ] && cat {cs}/"$last";;\n'
        f'  ps) case "$last" in name=*) n=${{last#name=^}}; n=${{n%\\$}}; [ -e {cs}/"$n" ] && cut -d" " -f1 {cs}/"$n";; '
        f'*) for f in {cs}/*; do [ -e "$f" ] && cut -d" " -f1 "$f"; done;; esac; exit 0;;\n'
        + ("  rm) ;;\n" if sticky else
           f'  rm) for f in {cs}/*; do [ "$(cut -d" " -f1 "$f")" = "$last" ] && rm -f "$f"; done; exit 0;;\n')
        + "esac\n"
    )
    stub.chmod(0o755)
    return stub


def _run(tmp_path: Path, payload, containers: dict[str, str], **stub) -> tuple[int, set[str]]:
    docker = _daemon(tmp_path, containers, **stub)
    command = _node_left_delete_command(payload).replace("/usr/bin/docker", str(docker))
    done = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
    return done.returncode, {path.name for path in (tmp_path / "containers").iterdir()}


def test_only_the_left_placements_container_goes_and_edit_containers_stay(tmp_path):
    payload = _payload()
    name = payload.container_name
    edit_copies = {f"{name}__prev": f"pid {payload.pod_id} {LEFT_PLACEMENT}",
                   f"kept_unmounted_{payload.pod_id}_1": f"kid {payload.pod_id} {LEFT_PLACEMENT}"}

    status, left = _run(tmp_path, payload, {name: f"cid {payload.pod_id} {LEFT_PLACEMENT}", **edit_copies})

    assert (status, left) == (0, set(edit_copies))


@pytest.mark.parametrize(
    "labels",
    ["{pod} placement-later", "{pod} ", "other-pod placement-left", " "],
    ids=["placed-back-later", "made-before-the-label", "another-pods", "unlabelled"],
)
def test_any_other_container_under_the_name_is_kept_and_answered_not_the_node_left(tmp_path, labels):
    payload = _payload()
    containers = {payload.container_name: "cid " + labels.format(pod=payload.pod_id)}

    assert _run(tmp_path, payload, containers) == (97, {payload.container_name})


def test_no_container_under_the_name_is_done(tmp_path):
    assert _run(tmp_path, _payload(), {}) == (0, set())


def test_a_container_still_there_after_the_removal_fails_the_delete(tmp_path):
    payload = _payload()

    assert _run(tmp_path, payload, {payload.container_name: f"cid {payload.pod_id} {LEFT_PLACEMENT}"}, sticky=True)[0] == 1


@pytest.mark.parametrize(
    "update",
    [
        {"create_command_id": ""},
        {"container_name": "pod_another-pod"},
    ],
)
def test_a_delete_that_names_no_placement_or_another_pod_is_refused(update):
    with pytest.raises(ValueError):
        _node_left_delete_command(_payload().model_copy(update=update))


def test_the_backend_message_parses_to_the_checked_delete():
    assert type(BaseServerRequest.parse(_payload().model_dump_json())) is ContainerDeleteOnNodeLeftRequest


def _host_exits(status: int):
    async def answer(command: str):
        return _make_ssh_command_result(exit_status=status)

    return answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [(97, FailedContainerErrorCodes.NotTheNodeLeft), (0, None), (1, FailedContainerErrorCodes.UnknownError)],
)
async def test_one_host_command_decides_and_no_sdk_call_is_made(docker_service, monkeypatch, status, expected):
    order = _wire_delete(docker_service, monkeypatch, _host_exits(status))

    result = await _delete(docker_service, _payload())

    assert len(order) == 1 and order[0][0] == "ssh"
    if expected is None:
        assert isinstance(result, ContainerDeleted)
    else:
        assert isinstance(result, FailedContainerRequest) and result.error_code is expected


@pytest.mark.asyncio
async def test_a_create_of_the_pod_running_on_any_registration_refuses_the_delete_and_is_not_cancelled(
    docker_service, monkeypatch
):
    order = _wire_delete(docker_service, monkeypatch, _host_exits(0))
    payload = _payload()
    replacement = ("miner-b", "replacement-executor")

    with inflight_creates.track(payload.pod_id, replacement):
        result = await _delete(docker_service, payload)

    assert isinstance(result, FailedContainerRequest) and order == []
    assert not inflight_creates.is_cancelled(payload.pod_id, replacement)


@pytest.mark.asyncio
async def test_a_create_of_the_pod_registered_during_the_removal_is_never_answered_as_deleted(docker_service, monkeypatch):
    payload = _payload()
    tracked = inflight_creates.track(payload.pod_id, ("miner-b", "replacement-executor"))

    async def registers_meanwhile(command: str):
        tracked.__enter__()
        return _make_ssh_command_result(exit_status=0)

    _wire_delete(docker_service, monkeypatch, registers_meanwhile)
    try:
        result = await _delete(docker_service, payload)
    finally:
        tracked.__exit__(None, None, None)

    assert isinstance(result, FailedContainerRequest)


@pytest.mark.asyncio
async def test_the_removal_never_touches_the_rented_machine_record_the_replacement_shares(docker_service, monkeypatch):
    # the record is keyed by endpoint, and the container name is the pod's on both registrations
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

    assert isinstance(await _delete(docker_service, payload), ContainerDeleted)
    assert [c["name"] for c in (await redis.get_rented_machine(executor))["containers"]] == [payload.container_name]


@pytest.mark.asyncio
async def test_a_reported_host_key_that_builds_no_policy_never_connects(docker_service, monkeypatch):
    # asyncssh takes known_hosts=None as "verify nothing": the peer answering the boot and labels could be any host
    from services.attestation_service import HostPolicyResult
    from services.docker_service import DockerService

    order = _wire_delete(docker_service, monkeypatch, _host_exits(0))
    monkeypatch.setattr(docker_service, "_prepare_known_hosts_policy", DockerService._prepare_known_hosts_policy.__get__(docker_service))
    docker_service.attestation_service.prepare_host_policy = AsyncMock(return_value=HostPolicyResult(known_hosts=None))
    monkeypatch.setattr("services.docker_service.require_rental_docker_ssh_host_key", lambda executor_info: "ssh-ed25519 AAAA")

    result = await _delete(docker_service, _payload())

    assert isinstance(result, FailedContainerRequest) and order == []


def test_a_node_left_delete_is_never_declined_as_renting_in_progress():
    # the source create's pending flag outlives its run while its receipt is processed
    from services.miner_service import _bypasses_renting_in_progress

    assert _bypasses_renting_in_progress(_payload())
    assert not _bypasses_renting_in_progress(ContainerDeleteRequest(**_delete_payload().model_dump(exclude={"message_type"})))


@pytest.mark.asyncio
async def test_a_plain_delete_leaves_the_build_artifacts_while_another_registrations_create_of_the_pod_runs(
    docker_service, monkeypatch
):
    # named by the pod alone: a moved pod's create on a daemon both registrations share may be building under them
    _wire_delete(docker_service, monkeypatch, _host_exits(0))
    monkeypatch.setattr(docker_service, "_cleanup_custom_build_artifacts", cleaned := AsyncMock())
    payload = _delete_payload()

    with inflight_creates.track(payload.pod_id, ("miner-b", "replacement-executor")):
        await _delete(docker_service, payload)
    cleaned.assert_not_awaited()
    await _delete(docker_service, payload)
    cleaned.assert_awaited_once()
