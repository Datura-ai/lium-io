"""A rented container Docker reports restarting is named in its per-pod verdict; the node's verdict is not changed."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from neurons.validators.src.services.task.checks.rented_machine import (
    RENTED_POD_RESTARTING,
    PodRunningAndAuthorizedKeys,
    TenantEnforcementCheck,
    _check_pod_running_and_read_authorized_keys,
)

from tests.test_gpu_usage_check import _scored_ctx, _snapshot_with_pod

RESTARTING = 'Error response from daemon: Container abc123 is restarting, wait until the container is running'
PAUSED = "Error response from daemon: Container abc123 is paused, unpause the container before exec"


class FakeSSH:
    def __init__(self, exec_stderr, listed="cid", restarting="false"):
        self.exec_stderr, self.listed, self.restarting = exec_stderr, listed, restarting

    async def run(self, command):
        if "docker ps" in command:
            return SimpleNamespace(stdout=self.listed, stderr="", exit_status=0)
        if "docker inspect" in command:
            return SimpleNamespace(stdout=self.restarting + "\n", stderr="", exit_status=0)
        return SimpleNamespace(stdout="", stderr=self.exec_stderr, exit_status=1)


@pytest.mark.asyncio
async def test_a_restarting_container_is_read_as_listed_and_restarting():
    result = await _check_pod_running_and_read_authorized_keys(FakeSSH(RESTARTING, restarting="true"), "pod_x")
    assert (result.running, result.authorized_keys, result.restarting) == (True, None, True)


@pytest.mark.asyncio
async def test_a_running_container_whose_read_fails_is_not_restarting():
    # the refusal text alone does not decide it: Docker's own state reads running, not restarting
    result = await _check_pod_running_and_read_authorized_keys(FakeSSH(RESTARTING), "pod_x")
    assert (result.running, result.authorized_keys, result.restarting) == (True, None, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ssh",
    [FakeSSH(PAUSED), FakeSSH(RESTARTING, listed="", restarting="true"), FakeSSH("", restarting="true")],
)
async def test_paused_unlisted_or_readable_is_not_restarting(ssh):
    assert (await _check_pod_running_and_read_authorized_keys(ssh, "pod_x")).restarting is False


async def _rented_cycle(ctx, read: PodRunningAndAuthorizedKeys):
    rented_machine = "neurons.validators.src.services.task.checks.rented_machine"
    with (
        patch(f"{rented_machine}._check_pod_running_and_read_authorized_keys", AsyncMock(return_value=read)),
        patch(f"{rented_machine}._restore_missing_volume_mount", AsyncMock(return_value=True)),
    ):
        return await TenantEnforcementCheck().run(ctx)


def _pod_id(ctx) -> str:
    return next(iter(ctx.state.rented_data.executors.values())).pods[0].pod_id


@pytest.mark.asyncio
async def test_a_restarting_pod_gets_the_fault_and_the_same_verdict(context_factory):
    restarting_ctx = _scored_ctx(context_factory, rented_data=_snapshot_with_pod(last_verifyx_download_speed=800.0))
    plain_ctx = _scored_ctx(context_factory, rented_data=_snapshot_with_pod(last_verifyx_download_speed=800.0))

    restarting = await _rented_cycle(
        restarting_ctx, PodRunningAndAuthorizedKeys(running=True, authorized_keys=None, restarting=True)
    )
    plain = await _rented_cycle(plain_ctx, PodRunningAndAuthorizedKeys(running=True, authorized_keys=[]))

    assert restarting.updates["state"].rented_pod_faults == {_pod_id(restarting_ctx): [RENTED_POD_RESTARTING]}
    assert not plain.updates["state"].rented_pod_faults
    # detection only: pass, halt, rented and the event are what a running pod gets
    assert (restarting.passed, restarting.halt, restarting.updates["rented"]) == (plain.passed, plain.halt, True)
    assert restarting.event.reason_code == plain.event.reason_code
