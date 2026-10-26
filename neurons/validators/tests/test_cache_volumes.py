from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from payload_models.payloads import CacheVolume, ContainerCreateRequest, WorkloadKind
from services.docker_service import DockerService


@pytest.fixture
def docker_service() -> DockerService:
    # _build_rental_container_run_spec is pure (no I/O), so mocked dependencies suffice.
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=Mock(),
    )


def _make_payload(
    *,
    workload_kind: WorkloadKind,
    cache_volumes: list[CacheVolume] | None,
    active_container_names: list[str] | None = None,
) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        workload_kind=workload_kind,
        cache_volumes=cache_volumes,
        active_container_names=active_container_names or [],
    )


class FakeSshClient:
    """Records commands and replays a canned `docker volume ls` listing."""

    def __init__(self, listed_volumes: list[str]):
        self._listing = "\n".join(listed_volumes)
        self.commands: list[str] = []

    async def run(self, command: str, *args, **kwargs):
        self.commands.append(command)
        stdout = self._listing if "volume ls" in command else ""
        return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")


class HostSshClient(FakeSshClient):
    """Answers the three questions the affordability decision asks: which cache volumes exist, where
    the docker root is, and how much space is free on it."""

    def __init__(self, listed_volumes: list[str], free_gb: float):
        super().__init__(listed_volumes)
        self._free_bytes = int(free_gb * 1024**3)

    async def run(self, command: str, *args, **kwargs):
        self.commands.append(command)
        if "docker info" in command:
            return SimpleNamespace(exit_status=0, stdout="/var/lib/docker\n", stderr="")
        if "df -P -B1" in command:
            body = f"Filesystem 1B-blocks Used Available Capacity\n/dev/sda1 1 1 {self._free_bytes} 1% /hostfs"
            return SimpleNamespace(exit_status=0, stdout=body, stderr="")
        stdout = self._listing if "volume ls" in command else ""
        return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")


def _removed_volumes(ssh_client: FakeSshClient) -> list[str]:
    return sorted(
        name
        for command in ssh_client.commands
        if "volume rm" in command
        for name in command.split("volume rm ", 1)[1].split(" 2>/dev/null")[0].split()
    )


# ---------------------------------------------------------------- reclaim (levels 2 and 3)


@pytest.mark.asyncio
@pytest.mark.parametrize(("free_gb", "granted"), [(5000, False), (5300, True)])
async def test_an_engy_checkpoint_is_cached_only_when_it_is_at_most_30pct_of_free_disk(
    docker_service, free_gb, granted
):
    # DAH-5014: the size comes from the name, not DPHN's 40 GB. 1561 GB is 31% of 5000 GB, 29% of 5300 GB.
    ssh_client = HostSshClient(["volume_pod"], free_gb=free_gb)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="engy_cache_hf_1561gb_kimi-k3-1a2b3c4d", target="/root/.cache/huggingface")
        ],
    )

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})

    assert bool(affordable) is granted


@pytest.mark.asyncio
@pytest.mark.parametrize(("free_gb", "granted"), [(220, False), (230, True)])
async def test_an_engy_checkpoint_and_the_pod_volume_must_leave_the_listing_floor_and_margin(
    docker_service, free_gb, granted
):
    # DAH-5014: real free disk minus the 31 GB checkpoint and the 40 GB pod volume created after this
    # check must stay over the 100 GB floor + 50 GB margin: 220 leaves 149, 230 leaves 159.
    ssh_client = HostSshClient(["volume_pod"], free_gb=free_gb)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="engy_cache_hf_31gb_qwen3-8-27b-1a2b3c4d", target="/root/.cache/huggingface")
        ],
    )
    payload.volume_limit_gb = 40

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})

    assert bool(affordable) is granted


@pytest.mark.asyncio
async def test_an_engy_checkpoint_of_unknown_size_is_never_downloaded_into_a_cache(docker_service):
    ssh_client = HostSshClient(["volume_pod"], free_gb=100_000)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="engy_cache_hf_kimi-k3-1a2b3c4d", target="/root/.cache/huggingface")
        ],
    )

    assert await docker_service.select_affordable_cache_volumes(ssh_client, payload, {}) == []


@pytest.mark.asyncio
async def test_rental_evicts_the_oldest_cache_first_and_keeps_the_rest_once_it_fits(docker_service, monkeypatch):
    # DAH-5014: the Engy checkpoint (~1.5 TB) gives way to a renter too, but only as far as the rental
    # needs: the older cache goes first and the free-disk re-check stops the eviction there.
    class InspectingSshClient(FakeSshClient):
        async def run(self, command: str, *args, **kwargs):
            if "volume inspect" in command:
                self.commands.append(command)
                body = "engy_cache_hf_new 2026-10-22T10:00:00Z\nengy_cache_hf_old 2026-10-01T10:00:00Z"
                return SimpleNamespace(exit_status=0, stdout=body, stderr="")
            return await super().run(command, *args, **kwargs)

    ssh_client = InspectingSshClient(["engy_cache_hf_new", "engy_cache_hf_old", "volume_pod"])
    monkeypatch.setattr(docker_service, "get_docker_root_dir", AsyncMock(return_value="/var/lib/docker"))
    monkeypatch.setattr(
        docker_service, "_get_fs_available_bytes", AsyncMock(side_effect=[50 * 1024**3, 900 * 1024**3])
    )
    payload = _make_payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL, cache_volumes=None)
    payload.volume_limit_gb = 500

    await docker_service.reclaim_dphn_cache_for_rental(ssh_client, payload, {})

    assert _removed_volumes(ssh_client) == ["engy_cache_hf_old"]
