"""DAH-6066: a rental's memory limit is also capped by the host's idle MemAvailable, never raised by it."""

import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from services.docker_service import _CONTAINER_UNRECLAIMABLE_MEMORY_CMD, DockerService
from services.redis_service import IDLE_MEM_AVAILABLE_PREFIX, IDLE_MEM_AVAILABLE_TTL_SECONDS

GIB_KIB = 1024 * 1024
HOST_TOTAL_KIB = 1007 * GIB_KIB
IDLE_AVAILABLE_KIB = 960 * GIB_KIB
KEY = f"{IDLE_MEM_AVAILABLE_PREFIX}:exec-1"


def _service(stored: str | None = None) -> DockerService:
    redis = Mock(set=AsyncMock(), get=AsyncMock(return_value=stored))
    return DockerService(ssh_service=Mock(), redis_service=redis, attestation_service=Mock())


def _ssh_stats(
    *rows: tuple[str, str], available_kib: int = IDLE_AVAILABLE_KIB, exit_status: int = 0
) -> Mock:
    stdout = f"MemAvailable:   {available_kib} kB\n" + "".join(
        f"{name} {usage}\n" for name, usage in rows
    )
    return Mock(
        run=AsyncMock(
            return_value=SimpleNamespace(exit_status=exit_status, stdout=stdout, stderr="")
        )
    )


@pytest.mark.asyncio
async def test_idle_node_uses_live_available_and_stores_it():
    # Arrange
    service = _service()
    ssh_client = _ssh_stats(("lium-executor", "314572800"))

    # Act
    result = await service._idle_mem_available_kib(ssh_client, "exec-1")

    # Assert
    assert result == (IDLE_AVAILABLE_KIB, "live")
    ssh_client.run.assert_awaited_once()  # value and usage from one command: the same instant
    service.redis_service.set.assert_awaited_once_with(
        KEY, str(IDLE_AVAILABLE_KIB), ex=IDLE_MEM_AVAILABLE_TTL_SECONDS
    )


@pytest.mark.asyncio
async def test_busy_node_adds_back_our_containers_live_usage_not_a_stale_sample():
    # Arrange: a stored sample from an older idle create; the provider's own workload grew since
    service = _service(stored=str(IDLE_AVAILABLE_KIB))
    ssh_client = _ssh_stats(
        ("lium-executor", "-"),
        ("pod_other", str(300 * GIB_KIB * 1024)),
        ("filler_x", str(500 * GIB_KIB * 1024)),
        available_kib=100 * GIB_KIB,
    )

    # Act
    result = await service._idle_mem_available_kib(ssh_client, "exec-1")

    # Assert: 100 free + 800 held by our containers, the provider's 1 GiB not added back
    assert result == (900 * GIB_KIB, "live")
    service.redis_service.set.assert_awaited_once_with(
        KEY, str(900 * GIB_KIB), ex=IDLE_MEM_AVAILABLE_TTL_SECONDS
    )


@pytest.mark.asyncio
async def test_failed_live_read_falls_back_to_the_stored_value():
    # Arrange
    service = _service(stored=str(IDLE_AVAILABLE_KIB))
    ssh_client = _ssh_stats(("pod_other", str(300 * GIB_KIB * 1024)), exit_status=1)

    # Act
    result = await service._idle_mem_available_kib(ssh_client, "exec-1")

    # Assert
    assert result == (IDLE_AVAILABLE_KIB, "stored")
    service.redis_service.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_read_without_stored_value_keeps_todays_limit():
    # Arrange
    service = _service(stored=None)
    ssh_client = _ssh_stats(("filler_x", "-"))  # its cgroup memory.stat was unreadable

    # Act
    idle_kib, source = await service._idle_mem_available_kib(ssh_client, "exec-1")

    # Assert
    assert (idle_kib, source) == (None, "none")
    assert DockerService._rental_memory_gb(
        0, HOST_TOTAL_KIB, 1.0, idle_kib
    ) == DockerService._rental_memory_gb(0, HOST_TOTAL_KIB, 1.0)


@pytest.mark.parametrize(
    "path,stat,expected",
    [
        (
            "sys/fs/cgroup/system.slice/docker-{id}.scope",
            "anon 1000\nfile 9000\nkernel 300\nslab_reclaimable 100\n",
            "1200",
        ),
        (
            "sys/fs/cgroup/docker/{id}",
            "anon 1000\nfile 9000\nslab_unreclaimable 50\nkernel_stack 20\n",
            "1070",
        ),
        (
            "sys/fs/cgroup/memory/docker/{id}",
            "cache 9000\nrss 1000\ntotal_cache 9000\ntotal_rss 1500\n",
            "1500",
        ),
        ("sys/fs/cgroup/elsewhere/{id}", "anon 1000\n", "-"),
    ],
    ids=["v2-systemd", "v2-cgroupfs-old-kernel", "v1", "unreadable"],
)
def test_container_memory_read_leaves_out_page_cache(tmp_path, path, stat, expected):
    # Arrange: a host cgroup tree under a fake /proc/1/root and a fake docker that lists one pod
    container_id = "a" * 64
    stat_file = tmp_path / path.format(id=container_id) / "memory.stat"
    stat_file.parent.mkdir(parents=True)
    stat_file.write_text(stat)
    command = _CONTAINER_UNRECLAIMABLE_MEMORY_CMD.replace("/proc/1/root/", f"{tmp_path}/").replace(
        "/usr/bin/docker ps --no-trunc --format '{{.ID}} {{.Names}}'",
        f"echo '{container_id} pod_x'",
    )

    # Act
    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True)

    # Assert
    assert result.stdout == f"pod_x {expected}\n"


def test_idle_available_lowers_the_limit_by_its_gpu_share():
    # Act
    memory_gb = DockerService._rental_memory_gb(0, HOST_TOTAL_KIB, 0.5, IDLE_AVAILABLE_KIB)

    # Assert: 960 * 97% margin * 0.5, below today's (1007 - 30.2) * 0.5 = 488
    assert memory_gb == 465


def test_idle_available_never_raises_the_limit_above_the_total_clamp():
    # Arrange
    old_limit = DockerService._rental_memory_gb(900, HOST_TOTAL_KIB, 1.0)

    # Act
    memory_gb = DockerService._rental_memory_gb(900, HOST_TOTAL_KIB, 1.0, HOST_TOTAL_KIB * 2)

    # Assert
    assert memory_gb == old_limit == 900


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored,expected", [(str(IDLE_AVAILABLE_KIB), IDLE_AVAILABLE_KIB * 97 // 100), (None, None)]
)
async def test_published_specs_carry_the_host_ceiling_from_the_stored_idle_baseline(
    context_factory, stored, expected
):
    # Arrange: the backend lists the capped RAM from ram.tenant_ceiling; no baseline sends no key
    from neurons.validators.src.services.task.result_handler import ResultHandler
    from tests.helpers import build_state

    state = build_state(
        specs={"gpu": {"count": 1, "details": []}, "ram": {"total": HOST_TOTAL_KIB}}
    )
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
