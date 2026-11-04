"""DAH-5618: a create or reboot replies ContainerCreated only once the pod's mapped SSH port answers."""

from unittest.mock import AsyncMock

import pytest

from protocol.vc_protocol.validator_requests import PodSshResult
from services import docker_service as docker_service_module
from services.task.checks.rented_pod_ssh import ConnectOutcome

B, R, N, T = PodSshResult.BANNER, PodSshResult.REFUSED, PodSshResult.NO_BANNER, PodSshResult.TIMEOUT


@pytest.mark.real_pod_ssh_check
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcomes, answers",
    [
        ([B], True),
        ([R, N, B], True),  # an image's own sshd that starts a little late
        ([N] * 50, False),  # a host that accepts and drops the session (F-1679)
        ([R] * 50, False),
        ([T] * 50, False),
    ],
)
async def test_the_pod_must_send_an_ssh_line_before_the_create_succeeds(monkeypatch, outcomes, answers):
    probe = AsyncMock(side_effect=[ConnectOutcome(o) for o in outcomes])
    monkeypatch.setattr(docker_service_module, "tcp_connect_fault", probe)
    monkeypatch.setattr(docker_service_module, "POD_SSH_ANSWER_DEADLINE_S", 0.05)
    monkeypatch.setattr(docker_service_module, "POD_SSH_ANSWER_RETRY_DELAY_S", 0.001)

    if answers:
        await docker_service_module._raise_unless_pod_ssh_answers("203.0.113.7", 20001)
        assert probe.await_count == len(outcomes)
    else:
        with pytest.raises(RuntimeError, match=f"SSH port 20001 did not answer.*{outcomes[0].value}"):
            await docker_service_module._raise_unless_pod_ssh_answers("203.0.113.7", 20001)
    assert all(call.args[:2] == ("203.0.113.7", 20001) for call in probe.await_args_list)
