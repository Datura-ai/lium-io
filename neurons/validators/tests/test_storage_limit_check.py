"""StorageLimitCheck: a node whose Docker cannot cap a container's disk (18 Oct 2026, serdkey node 4837fec9).

A renter filled the disk of a node without a per-container storage limit and took the host down. An unrented
node without the limit scores 0; a rented one only gets a warning.
"""

from __future__ import annotations

import pytest
from neurons.validators.tests.helpers import build_state
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse
from services.task.checks.rented_machine import TenantEnforcementCheck
from services.task.checks.storage_limit import StorageLimitCheck
from services.task.messages import StorageLimitMessages as Msg
from services.task.pipeline_factory import PipelineFactory


async def run(context_factory, specs: dict, *, rented: bool = False):
    rented_data = None
    if rented:
        executor = RentedExecutor.model_validate(
            {
                "miner_hotkey": "hk",
                "executor_ip_address": "1.2.3.4",
                "executor_ip_port": "8080",
                "pods": [{"pod_id": "pod-1", "container_name": "c", "gpu_count": 1}],
            }
        )
        rented_data = RentedExecutorsResponse(executors={context_factory().executor.uuid: executor})
    ctx = context_factory(state=build_state(specs=specs, rented_data=rented_data))
    return await StorageLimitCheck().run(ctx)


@pytest.mark.asyncio
async def test_a_node_with_the_limit_passes(context_factory):
    result = await run(context_factory, {"storage_limit_supported": True})

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("specs", [{"storage_limit_supported": False}, {}])
async def test_an_unrented_node_without_the_limit_scores_zero(context_factory, specs):
    result = await run(context_factory, specs)

    assert result.passed is False
    assert result.event.reason_code == Msg.REQUIRED.reason
    assert result.updates["score"] == 0.0


@pytest.mark.asyncio
async def test_a_rented_node_without_the_limit_only_warns(context_factory):
    result = await run(
        context_factory,
        {"storage_limit_supported": False, "storage_limit_scrape_error": "pquota missing"},
        rented=True,
    )

    assert result.passed is True
    assert result.event.reason_code == Msg.MISSING.reason
    assert result.event.what_we_saw["detail"] == "pquota missing"
    assert not result.updates


@pytest.mark.parametrize(
    "build",
    [
        PipelineFactory.build_checks,
        PipelineFactory.build_fast_path_checks,
        PipelineFactory.build_dry_run_checks,
    ],
)
def test_the_check_runs_before_the_rented_halt_in_every_pipeline(build):
    ids = [check.check_id for check in build()]

    assert ids.index(StorageLimitCheck.check_id) < ids.index(TenantEnforcementCheck.check_id)
