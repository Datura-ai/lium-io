from __future__ import annotations

from types import SimpleNamespace

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.task.checks.rented_mid_cycle import RentedMidCycleCheck
from services.task.messages import TenantEnforcementMessages as Msg
from services.task.pipeline_factory import PipelineFactory

from tests.helpers import build_services, build_state, make_context


class PodRunningSSH:
    """Answers the rented check's reads: the pod's container is up, no keys."""

    def __init__(self):
        self.commands: list[str] = []

    async def run(self, command: str):
        self.commands.append(command)
        return SimpleNamespace(stdout="container_id" if "docker ps" in command else "", exit_status=0)


def rented_now() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="pod-1", container_name="tenant-1", rented_ports=[])],
            )
        },
        banned_guids=[],
    )


def make_ctx(fresh: RentedExecutorsResponse | None, ssh=None):
    services = build_services()
    services.backend.get_rented_executors_now.return_value = fresh
    # the cycle-start snapshot saw the node unrented
    state = build_state(rented_data=RentedExecutorsResponse(executors={}, banned_guids=[]))
    return make_context(services=services, state=state, ssh=ssh)


@pytest.mark.asyncio
async def test_node_rented_since_cycle_start_halts_as_rented():
    ssh = PodRunningSSH()
    result = await RentedMidCycleCheck().run(make_ctx(rented_now(), ssh=ssh))

    assert result.passed and result.halt
    assert result.event.reason_code == Msg.ALREADY_RENTED.reason
    assert result.event.check_id == RentedMidCycleCheck.check_id
    assert result.updates["rented"] is True
    assert any("docker ps" in cmd for cmd in ssh.commands)


@pytest.mark.parametrize("fresh", [RentedExecutorsResponse(executors={}, banned_guids=[]), None])
@pytest.mark.asyncio
async def test_node_still_unrented_or_list_unread_goes_on(fresh):
    result = await RentedMidCycleCheck().run(make_ctx(fresh))

    assert result.passed and not result.halt
    assert result.event.reason_code == Msg.NOT_RENTED.reason
    assert not result.updates


@pytest.mark.parametrize("fast_path", [False, True])
def test_rechecked_right_before_the_gpu_and_network_tests(fast_path):
    names = [type(check).__name__ for check in PipelineFactory.build_checks(fast_path=fast_path)]
    before = [names[i + 1] for i, name in enumerate(names) if name == "RentedMidCycleCheck"]
    assert before == ["LocalVerifyCheck", "ParallelStage" if fast_path else "CapabilityCheck"]
    assert names.index("TenantEnforcementCheck") < names.index("RentedMidCycleCheck")
