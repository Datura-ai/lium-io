from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    RentedExecutorsResponse,
)
from neurons.validators.src.services.task.checks.gpu_usage import GpuUsageCheck
from neurons.validators.src.services.task.messages import GpuUsageMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


@contextmanager
def foreign_gate(*, enforce: bool = True, check_enabled: bool = True):
    """DAH-2735: the ownership gate ships shadow-first, like every money-withholding gate."""
    with patch("neurons.validators.src.services.task.checks.gpu_usage.settings") as s:
        s.FOREIGN_GPU_WORKLOAD_CHECK_ENABLED = check_enabled
        s.FOREIGN_GPU_WORKLOAD_ENFORCEMENT_ENABLED = enforce
        yield s


@pytest.mark.parametrize(
    "gpu_details,gpu_processes,expected_pass,expected_reason",
    [
        # No processes - should pass
        (
            [{"gpu_utilization": 10, "memory_utilization": 10}],
            [],
            True,
            Msg.USAGE_OK.reason,
        ),
        # Usage within limits, owner unreadable — inability to measure is not a violation
        (
            [{"gpu_utilization": 3, "memory_utilization": 4}],
            [{"pid": 1234, "name": "test"}],
            True,
            Msg.USAGE_OK.reason,
        ),
        # GPU utilization at limit (>= 5%) - should fail
        (
            [{"gpu_utilization": 5, "memory_utilization": 3}],
            [{"pid": 1234, "name": "test"}],
            False,
            Msg.USAGE_HIGH.reason,
        ),
        # GPU utilization exceeds limit - should fail
        (
            [{"gpu_utilization": 10, "memory_utilization": 3}],
            [{"pid": 1234, "name": "test"}],
            False,
            Msg.USAGE_HIGH.reason,
        ),
        # Memory utilization exceeds limit (> 5%) - should fail
        (
            [{"gpu_utilization": 3, "memory_utilization": 6}],
            [{"pid": 1234, "name": "test"}],
            False,
            Msg.USAGE_HIGH.reason,
        ),
        # Both exceed limits - should fail
        (
            [{"gpu_utilization": 10, "memory_utilization": 10}],
            [{"pid": 1234, "name": "test"}, {"pid": 5678, "name": "test2"}],
            False,
            Msg.USAGE_HIGH.reason,
        ),
    ],
)
@pytest.mark.asyncio
async def test_gpu_usage_check(
    gpu_details,
    gpu_processes,
    expected_pass,
    expected_reason,
    context_factory,
):
    services = build_services()
    config = build_context_config()
    state = build_state(gpu_details=gpu_details, gpu_processes=gpu_processes)

    ctx = context_factory(services=services, config=config, state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    # Verify the what_we_saw field contains process count
    if gpu_processes:
        assert result.event.what_we_saw.get("process_count") == len(gpu_processes)


# --- A rental that ends while the run is inside it is teardown, not an orphan ---
# The scrape reads the GPU processes at the start of the run. TenantEnforcementCheck then finds the
# pod the snapshot listed no longer running, the backend calls its rental inactive, and the run goes
# on as unrented, with the stopping pod container still in the process list.


# --- DAH-2427: ghost-GPU detection (util pinned, no memory, no processes) ---
# Stateless by design (review feedback): see the signature -> cure immediately (the CUDA
# context cycle is harmless) -> verdict from re-sampling the live card. Cured = node stays.


def _ssh_result(exit_code: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(
        exit_code=exit_code, stdout=stdout, stderr=stderr, error_message=None, success=exit_code == 0
    )


# --- DAH-2735: foreign GPU workloads (Nodexo/SN106) that idle below the percentage gates ---
# A competitor's rental holds 22.4 GB of VRAM at 0% reported load; its GPU workers run as bare
# host processes outside any container. Ownership decides, and the expected owners come from
# the backend, so `docker rename` buys nothing.

EXECUTOR_CONTAINER_ID = "58b5771305ac0f2d1a1f0c8f7c2b9d2e"
FILLER = "filler_5703f4c9-c2f4-4fae-a652-3dee4753030a"


def _specs(*extra_containers: str) -> dict[str, object]:
    containers = [{"container_id": EXECUTOR_CONTAINER_ID, "name": "executor-executor-1"}]
    containers += [{"container_id": f"id-{name}", "name": name} for name in extra_containers]
    return {"docker": {"container_id": EXECUTOR_CONTAINER_ID, "containers": containers}}


def _idle_state(gpu_details, gpu_processes, *, fillers: list[str] | None = None) -> object:
    return build_state(
        gpu_details=gpu_details,
        gpu_processes=gpu_processes,
        specs=_specs(*(fillers or [])),
        rented_data=RentedExecutorsResponse(
            executors={},
            all_filler_containers_by_executor={"executor-123": fillers} if fillers else {},
        ),
    )


@pytest.mark.asyncio
async def test_foreign_container_at_idle_utilization_fails(context_factory):
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 22400}],
        [{"pid": 4242, "container_name": "nodexo-rental-1cd1ba2b"}],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.FOREIGN_PROCESS.reason
    assert result.event.what_we_saw["foreign_processes"][0].container_name == "nodexo-rental-1cd1ba2b"


@pytest.mark.asyncio
async def test_container_renamed_to_look_like_a_filler_still_fails(context_factory):
    # The ticket's core requirement: a verdict `docker rename` cannot defeat. The node's real
    # filler set comes from the backend, so an unknown `filler_*` name is still foreign.
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 22400}],
        [{"pid": 4242, "container_name": "filler_not-ours"}],
        fillers=[FILLER],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.FOREIGN_PROCESS.reason


@pytest.mark.asyncio
async def test_host_process_escaping_the_cgroup_namespace_still_fails(context_factory):
    # Every real GPU process on the prod fleet escapes the scrape's namespace (`0::/../…`);
    # Nodexo's bare host workers are exactly that shape.
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 686}],
        [{"pid": 2844137, "info": "0::/../../user.slice/user-1000.slice/session-1.scope", "container_name": None}],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.FOREIGN_PROCESS.reason


