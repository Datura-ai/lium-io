from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.executor_rollout import ExecutorRolloutTracker
from services.task.checks.rented_mid_cycle import RentedMidCycleCheck
from services.task.messages import TenantEnforcementMessages as Msg
from services.task.pipeline_factory import PipelineFactory

from tests.helpers import build_services, build_state, make_context
from test_executor_rollout_grace import _failed, _scored, _validator_process


def rented(*pods: RentedPod) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=list(pods),
            )
        },
        banned_guids=[],
    )


UNRENTED = RentedExecutorsResponse(executors={}, banned_guids=[])
# created a moment ago: the backend lists it, its container is not up and not named yet
STARTING = RentedPod(pod_id="pod-new", container_name="", rented_ports=[], status="PENDING")
OLD = RentedPod(pod_id="pod-old", container_name="tenant-old", rented_ports=[])


class NoSSH:
    """The check must not read the host: a starting pod is not judged."""

    async def run(self, command: str):
        raise AssertionError(f"unexpected SSH call: {command}")


def make_ctx(fresh: RentedExecutorsResponse | None, at_start: RentedExecutorsResponse = UNRENTED):
    services = build_services()
    services.backend.get_rented_executors_now.return_value = fresh
    return make_context(services=services, state=build_state(rented_data=at_start), ssh=NoSSH())


@pytest.mark.asyncio
async def test_a_pod_still_starting_at_the_recheck_ends_the_run_with_no_verdict():
    result = await RentedMidCycleCheck().run(make_ctx(rented(STARTING)))

    assert not result.passed and not result.halt
    assert result.event.reason_code == Msg.RENTED_SINCE_CYCLE_START.reason
    assert result.event.reason_code != Msg.POD_NOT_RUNNING.reason
    # no score, no rented flag, no reset of the verified job
    assert not result.updates


@pytest.mark.parametrize(
    "fresh, at_start",
    [
        (UNRENTED, UNRENTED),
        (None, UNRENTED),
        # already listed at start: TenantEnforcementCheck judged it, nothing new here
        (rented(OLD), rented(OLD)),
    ],
)
@pytest.mark.asyncio
async def test_no_new_pod_goes_on(fresh, at_start):
    result = await RentedMidCycleCheck().run(make_ctx(fresh, at_start))

    assert result.passed and not result.halt
    assert result.event.reason_code == Msg.NOT_RENTED.reason
    assert not result.updates


@pytest.mark.asyncio
async def test_the_run_ended_by_a_new_pod_gets_no_verdict_this_cycle():
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.fetch_executor_digest_or_none = AsyncMock(return_value=None)
    validator_process.observe_executor_rollout = AsyncMock(return_value=MagicMock(covers=lambda _: False))
    held = _failed("node-1", Msg.RENTED_SINCE_CYCLE_START.reason)
    healthy = _scored("node-2")

    kept, withheld = await validator_process.withhold_verdicts_for_rollout(
        {"5Miner": [held, healthy]}, 7500, "batch-1", MagicMock()
    )

    assert kept == {"5Miner": [healthy]}
    assert [w.result.executor_info.uuid for w in withheld] == ["node-1"]


@pytest.mark.parametrize("fast_path", [False, True])
def test_rechecked_right_before_the_gpu_and_network_tests(fast_path):
    names = [type(check).__name__ for check in PipelineFactory.build_checks(fast_path=fast_path)]
    before = [names[i + 1] for i, name in enumerate(names) if name == "RentedMidCycleCheck"]
    assert before == ["LocalVerifyCheck", "ParallelStage" if fast_path else "CapabilityCheck"]
    assert names.index("TenantEnforcementCheck") < names.index("RentedMidCycleCheck")
