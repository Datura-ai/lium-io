"""DAH-2991 — a container dockerd cannot kill is killed directly instead of failing every delete.

ticket-0287: four backend deletes of pod_11655dc5… failed with dockerd's "could not kill: tried to
kill container, but did not receive an exit event"; the backend then dropped the pod and the
orphan held 8 of 10 rental ports, INSUFFICIENT_PORTS, score 0 for 5 h until a host reboot.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
import services.docker_service as ds_module
from core.docker_utils import DockerCommand
from payload_models.payloads import ContainerDeleteRequest
from services.docker_service import DockerService, _is_docker_could_not_kill_error

PROD_ERROR = (
    "Docker SDK remove container failed: 500 Server Error for "
    "http+docker://ssh/v1.44/containers/pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7?v=True&link=False&force=True: "
    'Internal Server Error ("cannot remove container "/pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7": '
    'could not kill: tried to kill container, but did not receive an exit event")'
)


def test_could_not_kill_error_is_recognised_from_the_prod_text():
    assert _is_docker_could_not_kill_error(RuntimeError(PROD_ERROR))
    assert not _is_docker_could_not_kill_error(RuntimeError("404 Client Error: No such container: pod_x"))
    assert not _is_docker_could_not_kill_error(RuntimeError("409 Conflict: removal of container is already in progress"))


def test_kill_command_targets_init_and_shim_of_that_container_only():
    cmd = DockerCommand.kill_container_processes("pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7")
    assert "docker inspect -f '{{.State.Pid}}' pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7" in cmd
    assert "kill -9 $pid" in cmd and "kill -9 $shim" in cmd
    assert "containerd-shim" in cmd  # the parent is killed only if it really is the shim
    assert "restart" not in cmd  # never the whole daemon: other tenants live on the node


@pytest.mark.asyncio
async def test_delete_kills_processes_directly_and_retries_when_docker_cannot_kill(monkeypatch):
    """First force-remove raises could-not-kill -> kill over ssh -> second force-remove succeeds -> deleted."""
    svc = DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())
    payload = ContainerDeleteRequest(
        miner_hotkey="miner", executor_id="d51b8008-7338-4c49-a5ae-d37e876ff79f",
        pod_id="11655dc5-53ba-4a8d-a341-fe6c9d12bda7",
        container_name="pod_11655dc5-53ba-4a8d-a341-fe6c9d12bda7",
    )
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=Mock(exit_status=0, stdout="killed pid=4242 shim=4141", stderr=""))
    attempts: list[str] = []

    async def force_remove(docker_client, p, log):
        attempts.append(p.container_name)
        if len(attempts) == 1:
            raise RuntimeError(PROD_ERROR)
        return None

    monkeypatch.setattr(svc, "_force_remove_container", force_remove)

    outcome = await svc._force_remove_or_kill(Mock(), payload, ssh, ds_module._BoundLog({}))

    assert outcome is None
    assert attempts == [payload.container_name] * 2
    kill_cmd = ssh.run.await_args.args[0]
    assert ".State.Pid" in kill_cmd and payload.container_name in kill_cmd
