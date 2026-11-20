"""A create re-sent under the same command ID after a connector restart (its first answer lost) finds the container
its first run finished on the node and reports that run, instead of starting a second container."""

from __future__ import annotations

import shlex
from unittest.mock import AsyncMock, Mock

import pytest
from payload_models.payloads import ContainerCreated
from services.const import LIUM_CREATE_COMMAND_ID_LABEL
from services.docker_service import DockerService, _CREATE_RECORD_STATE_MARK, _create_record_path
from test_deploy_optimizations import _executor_info, _patch_happy, _payload, _ssh_client, _ssh_result


@pytest.fixture
def svc():
    # the same services test_probe_pod_no_restart_policy builds
    return DockerService(ssh_service=Mock(), redis_service=Mock(get=AsyncMock(return_value=None), mget=AsyncMock(side_effect=lambda keys: [None] * len(keys)), set=AsyncMock(), write_atomically=AsyncMock(), get_rented_machine=AsyncMock(return_value=None), rented_machines_synced=AsyncMock(return_value=True)), attestation_service=Mock())


class _Node:
    """The executor host's record files and the pod container's running state and label."""

    def __init__(self):
        self.files: dict[str, str] = {}
        self.container: tuple[bool, str] | None = None

    def wrap(self, ssh_client):
        inner = ssh_client.run.side_effect

        def run(cmd, *args, **kwargs):
            if cmd.startswith("mkdir -p /var/lib/lium/creates"):
                parts = shlex.split(cmd)
                self.files[parts[-1]] = parts[parts.index("printf") + 2]
                return _ssh_result(exit_status=0)
            if _CREATE_RECORD_STATE_MARK in cmd:
                path = next(p for p in self.files) if self.files else ""
                state = f"{str(self.container[0]).lower()} {self.container[1]}" if self.container else ""
                return _ssh_result(exit_status=0, stdout=f"{self.files.get(path, '')}\n{_CREATE_RECORD_STATE_MARK}\n{state}\n")
            return inner(cmd, *args, **kwargs)

        ssh_client.run = AsyncMock(side_effect=run)
        return ssh_client


async def _create(svc, payload, ssh_client):
    return await svc.create_container(
        payload=payload, executor_info=_executor_info(payload), keypair=Mock(ss58_address="validator-hotkey"),
        private_key="encrypted",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["cmd-1", "cmd-other"])
async def test_a_create_rerun_after_a_restart_reports_its_own_running_container(svc, monkeypatch, label):
    node = _Node()
    ssh_client = node.wrap(_ssh_client(inspect_exit=0))
    _patch_happy(svc, monkeypatch, ssh_client)
    specs = []
    build = svc._build_rental_container_run_spec
    monkeypatch.setattr(svc, "_build_rental_container_run_spec", lambda **kw: specs.append(build(**kw)) or specs[-1])
    payload = _payload(command_id="cmd-1")

    first = await _create(svc, payload, ssh_client)
    assert isinstance(first, ContainerCreated)
    assert list(node.files) == [_create_record_path(payload.pod_id)]
    runs = specs
    assert [spec.labels.get(LIUM_CREATE_COMMAND_ID_LABEL) for spec in runs] == ["cmd-1"]
    node.container = (True, label)   # the first run's container is up; its answer was lost to a restart

    again = await _create(svc, payload, ssh_client)

    if label == "cmd-1":
        assert again == first
        assert len(specs) == 1   # no second container
    else:   # not this command's container: the create runs as usual
        assert len(specs) == 2



@pytest.mark.asyncio
@pytest.mark.parametrize("gap", ["cut_off_before_record", "record_write_failed", "unreadable_record", "stopped"])
async def test_a_rerun_never_removes_its_own_container_without_a_readable_reply(svc, monkeypatch, gap):
    """The first run's container is on the node with this command's label, but its reply cannot be replayed: the
    connector died between `docker run` and the record write, the write failed, the file is unreadable, or the
    container is stopped. The rerun answers nothing and never reaches the stale-container cleanup."""
    node = _Node()
    ssh_client = node.wrap(_ssh_client(inspect_exit=0))
    _patch_happy(svc, monkeypatch, ssh_client)
    specs = []
    build = svc._build_rental_container_run_spec
    monkeypatch.setattr(svc, "_build_rental_container_run_spec", lambda **kw: specs.append(build(**kw)) or specs[-1])
    payload = _payload(command_id="cmd-1")
    first = await _create(svc, payload, ssh_client)
    assert isinstance(first, ContainerCreated)
    path = _create_record_path(payload.pod_id)
    if gap in ("cut_off_before_record", "record_write_failed"):
        node.files.clear()
    elif gap == "unreadable_record":
        node.files[path] = "{trunc"
    node.container = (gap != "stopped", "cmd-1")
    cleanup = AsyncMock()
    monkeypatch.setattr(svc, "clean_existing_containers", cleanup)

    again = await _create(svc, payload, ssh_client)

    assert again is None
    assert len(specs) == 1   # no second container
    cleanup.assert_not_awaited()
