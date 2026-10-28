"""DAH-4300: the validator fills an Engy checkpoint cache in a helper container beside whatever the node runs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from payload_models.payloads import CacheVolume, ContainerCreateRequest, WorkloadKind
from services.const import ENGY_CACHE_PREFETCH_CONTAINER_NAME
from services.docker_service import DockerService

_KIMI = "engy_cache_hf_1561gb_kimi-k3-1a2b3c4d"


@pytest.fixture
def docker_service() -> DockerService:
    return DockerService(
        ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock(), rental_docker_client_factory=Mock()
    )


class Host:
    """A node: its containers, its Engy volumes, its free disk and what `du` reads in an existing volume."""

    def __init__(self, containers=(), volumes=(), free_gb=5000.0, held_gb=0.0):
        self.containers, self.volumes = list(containers), list(volumes)
        self.free_gb, self.held_gb = free_gb, held_gb
        self.commands: list[str] = []

    async def run(self, command: str, *args, **kwargs):
        self.commands.append(command)
        stdout = ""
        if "docker ps" in command:
            stdout = "\n".join(f"{name}\tid-{name}" for name in self.containers)
        elif "volume ls" in command:
            stdout = "\n".join(self.volumes)
        elif "du -sk" in command:
            stdout = f"{int(self.held_gb * 10**9 / 1024)}\t/cache\n"
        elif "docker info" in command:
            stdout = "/var/lib/docker\n"
        elif "df -P -B1" in command:
            stdout = f"Filesystem 1B-blocks Used Available Capacity\n/dev/sda1 1 1 {int(self.free_gb * 1024**3)} 1% /h"
        return SimpleNamespace(exit_status=0, stdout=stdout, stderr="")

    def started(self) -> list[str]:
        return [c for c in self.commands if "docker run -d" in c and ENGY_CACHE_PREFETCH_CONTAINER_NAME in c]


async def _prefetch(docker_service, host, volume=_KIMI, local_dir="kimi-k3"):
    return await docker_service.start_engy_cache_prefetch(host, volume, "moonshotai/Kimi-K3", local_dir, {})


@pytest.mark.asyncio
async def test_a_free_node_downloads_into_the_cache_volume_without_a_gpu(docker_service):
    host = Host(containers=["filler_pearl"], free_gb=5000)

    assert await _prefetch(docker_service, host) == "started"

    [run] = host.started()
    assert f"-v {_KIMI}:/cache " in run and run.count(" -v ") == 1  # only the cache volume
    assert "--gpus" not in run and "--cpus" in run and "--memory" in run
    assert "hf download moonshotai/Kimi-K3 --local-dir /cache/kimi-k3" in run


@pytest.mark.asyncio
async def test_a_partial_cache_resumes_when_the_node_affords_only_what_is_missing(docker_service):
    # 1561 GB from scratch on 2500 GB free is refused (over 30% and under 2000 GB beside it); 161 GB missing is not
    assert await _prefetch(docker_service, Host(free_gb=2500)) == "unaffordable"
    host = Host(volumes=[_KIMI], free_gb=2500, held_gb=1400)

    assert await _prefetch(docker_service, host) == "started"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("host", "outcome"),
    [
        (Host(containers=["pod_abc", "filler_x"]), "rented"),
        (Host(containers=[ENGY_CACHE_PREFETCH_CONTAINER_NAME]), "running"),
        (Host(volumes=[_KIMI], held_gb=1561), "complete"),
        (Host(free_gb=1700), "unaffordable"),  # the checkpoint does not fit beside the floor
    ],
)
async def test_no_download_on_a_rented_busy_complete_or_full_node(docker_service, host, outcome):
    assert await _prefetch(docker_service, host) == outcome
    assert host.started() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("volume", "local_dir"),
    [
        ("volume_renter", "x"),
        ("dphn_cache_hf_x", "x"),
        ("engy_cache_hf_x", "x"),  # no size in the name
        (_KIMI, "../etc"),
        (_KIMI, "/root"),
        (_KIMI, ""),
    ],
)
async def test_only_a_sized_engy_cache_volume_and_a_relative_dir_are_accepted(docker_service, volume, local_dir):
    host = Host()

    assert await _prefetch(docker_service, host, volume, local_dir) == "refused"
    assert host.commands == []


@pytest.mark.asyncio
async def test_a_customer_rental_stops_the_download_before_reclaiming_disk(docker_service):
    host = Host(volumes=[_KIMI], free_gb=5000)
    rental = ContainerCreateRequest(
        miner_hotkey="hk", executor_id="ex", pod_id="pod", docker_image="img:tag", gpu_uuids=["g0"],
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
    )
    rental.volume_limit_gb = 100

    await docker_service.reclaim_dphn_cache_for_rental(host, rental, {})

    assert f"docker rm -f {ENGY_CACHE_PREFETCH_CONTAINER_NAME}" in host.commands[0]


@pytest.mark.asyncio
async def test_an_engy_run_takes_over_the_volume_the_download_was_filling(docker_service):
    host = Host(volumes=[_KIMI], free_gb=5000, held_gb=1561)
    engy = ContainerCreateRequest(
        miner_hotkey="hk", executor_id="ex", pod_id="pod", docker_image="img:tag", gpu_uuids=["g0"],
        workload_kind=WorkloadKind.FILLER, cache_volumes=[CacheVolume(name=_KIMI, target="/root/models")],
    )

    assert await docker_service.select_affordable_cache_volumes(host, engy, {}) == engy.cache_volumes
    assert f"docker rm -f {ENGY_CACHE_PREFETCH_CONTAINER_NAME}" in host.commands[0]
