import pytest

from neurons.validators.src.services.const import POD_CONTAINER_PREFIX
from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.task.checks.gpu_usage import GpuUsageCheck
from neurons.validators.src.services.task.messages import GpuUsageMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


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
        # Usage within limits - should pass
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


@pytest.mark.asyncio
async def test_gpu_usage_orphaned_container(context_factory):
    """Test detection of orphaned rental containers."""
    services = build_services()
    config = build_context_config()

    # GPU usage exceeds limits with orphaned rental container
    pod_id = "5703f4c9-c2f4-4fae-a652-3dee4753030a"
    container_name = f"{POD_CONTAINER_PREFIX}{pod_id}"
    gpu_details = [{"gpu_utilization": 100, "memory_utilization": 61}]
    gpu_processes = [
        {
            "pid": 3217038,
            "info": "0::/../df2b545dac1b4caa3642d0db98ca054a0d923a1d0a3e470b60852c5aac81301f/init.scope",
            "container_name": container_name,
        }
    ]

    state = build_state(gpu_details=gpu_details, gpu_processes=gpu_processes)
    ctx = context_factory(services=services, config=config, state=state, rented=False)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.ORPHANED_CONTAINER.reason
    assert result.event.what_we_saw.get("orphaned_container") == container_name
    assert result.event.what_we_saw.get("rental_status") == "ended"
    assert result.event.what_we_saw.get("container_status") == "still running"
    assert f"docker stop {container_name}" in result.event.remediation


@pytest.mark.asyncio
async def test_gpu_usage_allows_exact_mapped_filler_container(context_factory):
    filler_container = "filler_5703f4c9-c2f4-4fae-a652-3dee4753030a"
    state = build_state(
        gpu_details=[{"gpu_utilization": 100, "memory_utilization": 61}],
        gpu_processes=[{"pid": 3217038, "container_name": filler_container}],
        rented_data=RentedExecutorsResponse(
            executors={},
            filler_containers_by_executor={"executor-123": filler_container},
        ),
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason
    assert result.event.what_we_saw["filler_container"] == filler_container


@pytest.mark.asyncio
async def test_gpu_usage_rejects_unmapped_filler_container(context_factory):
    state = build_state(
        gpu_details=[{"gpu_utilization": 100, "memory_utilization": 61}],
        gpu_processes=[{"pid": 3217038, "container_name": "filler_other"}],
        rented_data=RentedExecutorsResponse(
            executors={},
            filler_containers_by_executor={"executor-123": "filler_active"},
        ),
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.USAGE_HIGH.reason
