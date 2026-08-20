from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.const import POD_CONTAINER_PREFIX
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
async def test_bare_host_gpu_process_fails(context_factory):
    # PM2-supervised gpu_worker.py from a plain user session: no container at all.
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 686}],
        [{"pid": 2844137, "info": "0::/../../user.slice/user-1000.slice/session-1.scope", "container_name": None}],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.FOREIGN_PROCESS.reason


@pytest.mark.asyncio
async def test_held_vram_with_no_visible_processes_fails(context_factory):
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 22400}],
        [],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.VRAM_HELD.reason
    assert result.event.what_we_saw["held_vram"][0].memory_used_mb == 22400


@pytest.mark.asyncio
async def test_driver_reserve_vram_with_no_processes_passes(context_factory):
    # NVML counts the driver-reserved block (up to ~728 MB measured on B200) — not a workload.
    state = _idle_state([{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 728}], [])
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason


@pytest.mark.asyncio
async def test_the_nodes_own_filler_holding_vram_passes(context_factory):
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 69000}],
        [{"pid": 3217038, "container_name": FILLER}],
        fillers=[FILLER],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason


@pytest.mark.asyncio
async def test_validator_own_gpu_work_in_the_executor_container_passes(context_factory):
    # VerifyX runs over SSH inside the executor container, so its cgroup is the executor's.
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 3000}],
        [{"pid": 91011, "container_name": "executor-executor-1"}],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason


@pytest.mark.asyncio
async def test_held_vram_passes_when_the_node_runs_a_filler(context_factory):
    # Seen in prod: NVML returned no processes on a node whose filler was working at 34%.
    state = _idle_state(
        [{"gpu_utilization": 34, "memory_utilization": 2, "memory_used_mb": 2371}],
        [],
        fillers=[FILLER],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason


@pytest.mark.asyncio
async def test_foreign_process_only_warns_while_enforcement_is_off(context_factory):
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 22400}],
        [{"pid": 4242, "container_name": "nodexo-rental-1cd1ba2b"}],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate(enforce=False):
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.FOREIGN_PROCESS.reason
    assert result.event.severity == "warning"


@pytest.mark.asyncio
async def test_disabled_check_leaves_the_legacy_verdict_untouched(context_factory):
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 22400}],
        [{"pid": 4242, "container_name": "nodexo-rental-1cd1ba2b"}],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate(check_enabled=False):
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason


@pytest.mark.asyncio
async def test_missing_backend_truth_never_judges(context_factory):
    # No rented_data means no allowlist — an inability to measure, not a violation.
    state = build_state(
        gpu_details=[{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 22400}],
        gpu_processes=[{"pid": 4242, "container_name": "nodexo-rental-1cd1ba2b"}],
        specs=_specs(),
        rented_data=None,
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason


@pytest.mark.asyncio
async def test_foreign_event_payload_serializes_to_json(context_factory):
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 22400}],
        [{"pid": 4242, "info": "0::/../pm2-lichsl.service", "container_name": None}],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    dumped = result.event.model_dump(mode="json")["what_we_saw"]
    assert dumped["foreign_processes"][0]["cgroup"] == "0::/../pm2-lichsl.service"
    assert dumped["foreign_processes"][0]["pid"] == 4242


@pytest.mark.asyncio
async def test_validator_own_gpu_work_in_the_executors_cgroup_passes(context_factory):
    # A VerifyX run that outlived its SSH command timeout shares the scrape's cgroup namespace,
    # so it carries no container id and no name — it must not read as a competitor's workload.
    state = _idle_state(
        [{"gpu_utilization": 0, "memory_utilization": 0, "memory_used_mb": 3000}],
        [{"pid": 91011, "info": "0::/init.scope", "container_name": None}],
    )
    ctx = context_factory(services=build_services(), config=build_context_config(), state=state)

    with foreign_gate():
        result = await GpuUsageCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.USAGE_OK.reason


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
