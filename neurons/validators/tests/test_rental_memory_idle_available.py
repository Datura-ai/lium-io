"""DAH-6066: a rental's memory limit is also capped by the host's idle MemAvailable, never raised by it."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from services.docker_service import DockerService
from services.redis_service import IDLE_MEM_AVAILABLE_PREFIX, IDLE_MEM_AVAILABLE_TTL_SECONDS

GIB_KIB = 1024 * 1024
HOST_TOTAL_KIB = 1007 * GIB_KIB
IDLE_AVAILABLE_KIB = 960 * GIB_KIB
KEY = f"{IDLE_MEM_AVAILABLE_PREFIX}:exec-1"


def _service(stored: str | None = None) -> DockerService:
    redis = Mock(set=AsyncMock(), get=AsyncMock(return_value=stored))
    return DockerService(ssh_service=Mock(), redis_service=redis, attestation_service=Mock())


def _ssh_listing(*names: str, available_kib: int = IDLE_AVAILABLE_KIB) -> Mock:
    stdout = f"MemAvailable:   {available_kib} kB\n" + "".join(f"{name} {i:064x}\n" for i, name in enumerate(names))
    return Mock(
        run=AsyncMock(return_value=SimpleNamespace(exit_status=0, stdout=stdout, stderr=""))
    )


@pytest.mark.asyncio
async def test_idle_node_uses_live_available_and_stores_it():
    # Arrange
    service = _service()
    ssh_client = _ssh_listing("lium-executor")

    # Act
    result = await service._idle_mem_available_kib(ssh_client, "exec-1")

    # Assert
    assert result == (IDLE_AVAILABLE_KIB, "live")
    ssh_client.run.assert_awaited_once()  # value and listing from one command: the same instant
    service.redis_service.set.assert_awaited_once_with(
        KEY, str(IDLE_AVAILABLE_KIB), ex=IDLE_MEM_AVAILABLE_TTL_SECONDS
    )


@pytest.mark.asyncio
async def test_busy_node_uses_stored_baseline():
    # Arrange
    service = _service(stored=str(IDLE_AVAILABLE_KIB))
    ssh_client = _ssh_listing("lium-executor", "pod_other", "filler_x", available_kib=100 * GIB_KIB)

    # Act
    result = await service._idle_mem_available_kib(ssh_client, "exec-1")

    # Assert
    assert result == (IDLE_AVAILABLE_KIB, "stored")
    service.redis_service.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_node_without_baseline_keeps_todays_limit():
    # Arrange
    service = _service(stored=None)
    ssh_client = _ssh_listing("filler_x", available_kib=100 * GIB_KIB)

    # Act
    idle_kib, source = await service._idle_mem_available_kib(ssh_client, "exec-1")

    # Assert
    assert (idle_kib, source) == (None, "none")
    assert DockerService._rental_memory_gb(
        0, HOST_TOTAL_KIB, 1.0, idle_kib
    ) == DockerService._rental_memory_gb(0, HOST_TOTAL_KIB, 1.0)


def test_idle_available_lowers_the_limit_by_its_gpu_share():
    # Act
    memory_gb = DockerService._rental_memory_gb(0, HOST_TOTAL_KIB, 0.5, IDLE_AVAILABLE_KIB)

    # Assert: (960 - 4 GiB margin) * 0.5, below today's (1007 - 30.2) * 0.5 = 488
    assert memory_gb == 478


def test_idle_available_never_raises_the_limit_above_the_total_clamp():
    # Arrange
    old_limit = DockerService._rental_memory_gb(900, HOST_TOTAL_KIB, 1.0)

    # Act
    memory_gb = DockerService._rental_memory_gb(900, HOST_TOTAL_KIB, 1.0, HOST_TOTAL_KIB * 2)

    # Assert
    assert memory_gb == old_limit == 900


@pytest.mark.asyncio
@pytest.mark.parametrize("stored,expected", [(str(IDLE_AVAILABLE_KIB), IDLE_AVAILABLE_KIB - 4 * GIB_KIB), (None, None)])
async def test_published_specs_carry_the_host_ceiling_from_the_stored_idle_baseline(context_factory, stored, expected):
    # Arrange: the backend lists the capped RAM from ram.tenant_ceiling; no baseline sends no key
    from neurons.validators.src.services.task.result_handler import ResultHandler
    from tests.helpers import build_state

    state = build_state(specs={"gpu": {"count": 1, "details": []}, "ram": {"total": HOST_TOTAL_KIB}})
    ctx = context_factory(state=state)
    redis = Mock(get=AsyncMock(return_value=stored))

    # Act
    result = await ResultHandler(redis_service=redis, dry_run=True).handle_result(
        context=ctx,
        miner_info=SimpleNamespace(miner_hotkey="hk", job_batch_id="batch"),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="",
        success=True,
    )

    # Assert
    assert result.spec["ram"].get("tenant_ceiling") == expected
    assert result.spec["ram"]["total"] == HOST_TOTAL_KIB
    redis.get.assert_awaited_once_with(f"{IDLE_MEM_AVAILABLE_PREFIX}:{ctx.executor.uuid}")
