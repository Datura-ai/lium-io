from unittest.mock import AsyncMock, MagicMock

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.const import POD_CONTAINER_PREFIX
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
        # Usage within limits but the process belongs to no Lium container — foreign (DAH-2735)
        (
            [{"gpu_utilization": 3, "memory_utilization": 4}],
            [{"pid": 1234, "name": "test"}],
            False,
            Msg.FOREIGN_PROCESS.reason,
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
            all_filler_containers_by_executor={"executor-123": [filler_container]},
        ),
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason
    assert result.event.what_we_saw["filler_containers"] == [filler_container]


@pytest.mark.asyncio
async def test_gpu_usage_rejects_unmapped_filler_container(context_factory):
    state = build_state(
        gpu_details=[{"gpu_utilization": 100, "memory_utilization": 61}],
        gpu_processes=[{"pid": 3217038, "container_name": "filler_other"}],
        rented_data=RentedExecutorsResponse(
            executors={},
            all_filler_containers_by_executor={"executor-123": ["filler_active"]},
        ),
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.USAGE_HIGH.reason


# --- DAH-2427: ghost-GPU detection (util pinned, no memory, no processes) ---
# Stateless by design (review feedback): see the signature -> cure immediately (the CUDA
# context cycle is harmless) -> verdict from re-sampling the live card. Cured = node stays.

WEDGED_UUID = "GPU-bdf72357-83a7-09d3-9809-729c734aa80a"
HEALTHY_UUID = "GPU-dde887f6-488b-085f-a7b0-a71557f3e330"

GPU_QUERY_PREFIX = "nvidia-smi --query-gpu="
COMPUTE_APPS_PREFIX = "nvidia-smi --query-compute-apps="
CURE_PREFIX = "CUDA_VISIBLE_DEVICES="


def _wedged_detail(uuid: str = WEDGED_UUID) -> dict:
    return {"uuid": uuid, "gpu_utilization": 100, "memory_utilization": 0}


def _ssh_result(exit_code: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(exit_code=exit_code, stdout=stdout, stderr=stderr, error_message=None)


def _mock_runner(requery_gpu_csv: str = "", cure_exit_code: int = 0) -> MagicMock:
    """Route the three commands the ghost path can issue: cure, GPU re-query, compute apps."""

    async def route(command: str, **_kwargs) -> MagicMock:
        if command.startswith(CURE_PREFIX):
            return _ssh_result(exit_code=cure_exit_code, stdout="ctx open/close OK" if cure_exit_code == 0 else "AssertionError")
        if command.startswith(GPU_QUERY_PREFIX):
            return _ssh_result(stdout=requery_gpu_csv)
        if command.startswith(COMPUTE_APPS_PREFIX):
            return _ssh_result(stdout="")
        raise AssertionError(f"unexpected command: {command}")

    runner = MagicMock()
    runner.run = AsyncMock(side_effect=route)
    return runner


def _commands(runner: MagicMock) -> list[str]:
    return [call.args[0] for call in runner.run.await_args_list]


@pytest.mark.asyncio
async def test_ghost_cured_in_place_passes(context_factory):
    runner = _mock_runner(requery_gpu_csv=f"{WEDGED_UUID}, 0, 0\n")
    state = build_state(gpu_details=[_wedged_detail()], gpu_processes=[])
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state, runner=runner)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.GPU_WEDGE_CURED.reason
    assert any(cmd.startswith(CURE_PREFIX) and WEDGED_UUID in cmd for cmd in _commands(runner))


@pytest.mark.asyncio
async def test_ghost_still_latched_after_cure_fails_the_check(context_factory):
    runner = _mock_runner(requery_gpu_csv=f"{WEDGED_UUID}, 100, 0\n")
    state = build_state(gpu_details=[_wedged_detail()], gpu_processes=[])
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state, runner=runner)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_WEDGED.reason
    assert result.event.what_we_saw["still_wedged"] == [WEDGED_UUID]


@pytest.mark.asyncio
async def test_verdict_comes_from_the_card_not_the_cure_exit_code(context_factory):
    """A failed cure command with a clean re-query still passes — the card is the truth."""
    runner = _mock_runner(requery_gpu_csv=f"{WEDGED_UUID}, 0, 0\n", cure_exit_code=1)
    state = build_state(gpu_details=[_wedged_detail()], gpu_processes=[])
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state, runner=runner)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.GPU_WEDGE_CURED.reason
    assert result.event.what_we_saw["cure_results"][0].exit_code == 1


@pytest.mark.asyncio
async def test_no_commands_when_no_ghost_signature(context_factory):
    runner = _mock_runner()
    state = build_state(
        gpu_details=[{"uuid": HEALTHY_UUID, "gpu_utilization": 0, "memory_utilization": 0}],
        gpu_processes=[],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state, runner=runner)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason
    runner.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_wedge_signature_with_processes_keeps_legacy_path(context_factory):
    runner = _mock_runner()
    state = build_state(gpu_details=[_wedged_detail()], gpu_processes=[{"pid": 1234, "name": "python"}])
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state, runner=runner)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.USAGE_HIGH.reason
    runner.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_the_ghost_uuid_is_cured_on_a_mixed_host(context_factory):
    runner = _mock_runner(requery_gpu_csv=f"{WEDGED_UUID}, 0, 0\n{HEALTHY_UUID}, 0, 0\n")
    state = build_state(
        gpu_details=[
            _wedged_detail(),
            {"uuid": HEALTHY_UUID, "gpu_utilization": 0, "memory_utilization": 0},
        ],
        gpu_processes=[],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state, runner=runner)

    await GpuUsageCheck().run(ctx)

    cure_commands = [cmd for cmd in _commands(runner) if cmd.startswith(CURE_PREFIX)]
    assert len(cure_commands) == 1
    assert WEDGED_UUID in cure_commands[0]
    assert not any(HEALTHY_UUID in cmd for cmd in cure_commands)


@pytest.mark.asyncio
async def test_dry_run_detects_ghost_without_curing(context_factory):
    runner = _mock_runner()
    state = build_state(gpu_details=[_wedged_detail()], gpu_processes=[])
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state, runner=runner)

    result = await GpuUsageCheck(dry_run=True).run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_WEDGED.reason
    assert result.event.what_we_saw["cure_attempted"] is False
    runner.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_unexpected_error_falls_back_to_the_usage_guard(context_factory):
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=RuntimeError("ssh exploded"))
    state = build_state(gpu_details=[_wedged_detail()], gpu_processes=[])
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state, runner=runner)

    result = await GpuUsageCheck().run(ctx)

    # Detection is an addition to the process guard — its failure must not break validation.
    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason


@pytest.mark.asyncio
async def test_ghost_event_payload_serializes_to_json(context_factory):
    runner = _mock_runner(requery_gpu_csv=f"{WEDGED_UUID}, 0, 0\n")
    state = build_state(gpu_details=[_wedged_detail()], gpu_processes=[])
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state, runner=runner)

    result = await GpuUsageCheck().run(ctx)

    dumped = result.event.model_dump(mode="json")["what_we_saw"]
    assert dumped["wedged_gpus"][0]["gpu_uuid"] == WEDGED_UUID
    assert dumped["cure_results"][0]["gpu_uuid"] == WEDGED_UUID
    assert dumped["still_wedged"] == []


@pytest.mark.asyncio
async def test_gpu_usage_allows_multiple_filler_bundles_on_split_node(context_factory):
    # DAH-2465: a GPU-split node runs one filler per VRAM bundle; processes from ANY of the node's
    # fillers must read as clean, not "GPU busy outside validator" (which zeroed the score).
    bundle_a = "filler_5703f4c9-c2f4-4fae-a652-3dee4753030a"
    bundle_b = "filler_9622d623-dcb3-27dc-52c6-ef6c937df3ae"
    state = build_state(
        gpu_details=[{"gpu_utilization": 98, "memory_utilization": 61}],
        gpu_processes=[
            {"pid": 3217038, "container_name": bundle_a},
            {"pid": 3217099, "container_name": bundle_b},
        ],
        rented_data=RentedExecutorsResponse(
            executors={},
            all_filler_containers_by_executor={"executor-123": [bundle_a, bundle_b]},
        ),
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason
    assert result.event.what_we_saw["filler_containers"] == sorted([bundle_a, bundle_b])


# --- DAH-2735: foreign GPU workloads (Nodexo/SN106) that idle below the percentage gates ---
# A competitor's rental holds 22.4 GB of VRAM at 0% reported load; its GPU workers run as bare
# host processes outside any container. Ownership, not utilization, is the signal.


@pytest.mark.asyncio
async def test_foreign_container_at_idle_utilization_fails(context_factory):
    state = build_state(
        gpu_details=[{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 22400}],
        gpu_processes=[{"pid": 4242, "container_name": "nodexo-rental-1cd1ba2b"}],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.FOREIGN_PROCESS.reason
    assert result.event.what_we_saw["foreign_processes"][0]["container_name"] == "nodexo-rental-1cd1ba2b"


@pytest.mark.asyncio
async def test_bare_host_gpu_process_fails(context_factory):
    # PM2-supervised gpu_worker.py from a plain user session: no container at all.
    state = build_state(
        gpu_details=[{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 686}],
        gpu_processes=[{"pid": 2844137, "info": "0::/user.slice/user-1000.slice/session-1.scope", "container_name": None}],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.FOREIGN_PROCESS.reason


@pytest.mark.asyncio
async def test_held_vram_with_no_visible_processes_fails(context_factory):
    state = build_state(
        gpu_details=[{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 22400}],
        gpu_processes=[],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.VRAM_HELD.reason


@pytest.mark.asyncio
async def test_driver_reserve_vram_with_no_processes_passes(context_factory):
    # NVML counts the driver-reserved block (up to ~728 MB measured on B200) — not a workload.
    state = build_state(
        gpu_details=[{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 728}],
        gpu_processes=[],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason


@pytest.mark.asyncio
async def test_filler_holding_vram_passes(context_factory):
    # Dolphin fillers legitimately hold most of the card; ownership makes them clean.
    filler_container = "filler_5703f4c9-c2f4-4fae-a652-3dee4753030a"
    state = build_state(
        gpu_details=[{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 69000}],
        gpu_processes=[{"pid": 3217038, "container_name": filler_container}],
        rented_data=RentedExecutorsResponse(
            executors={},
            all_filler_containers_by_executor={"executor-123": [filler_container]},
        ),
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason
