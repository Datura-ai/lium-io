from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from payload_models.payloads import CacheVolume, ContainerCreateRequest, WorkloadKind
from services.docker_service import DockerService, _filler_run_volume_gb


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
    # A 1500 GB run volume could not hold it, so the share rule alone decides.
    ssh_client = HostSshClient(["volume_pod"], free_gb=free_gb)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="engy_cache_hf_1561gb_kimi-k3-1a2b3c4d", target="/root/.cache/huggingface")
        ],
    )
    payload.volume_limit_gb = 1500

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})

    assert bool(affordable) is granted


@pytest.mark.asyncio
@pytest.mark.parametrize(("volume_limit_gb", "granted"), [(1500, False), (1900, True)])
async def test_an_engy_checkpoint_over_30pct_is_cached_when_the_runs_own_volume_could_hold_it(
    docker_service, volume_limit_gb, granted
):
    # 1561 GB is 41% of 3800 GB free. Refused, it would download into a 1900 GB run volume anyway, on every
    # launch; 3800 - 1561 - 1900 = 339 still clears the floor. A 1500 GB volume could not hold it: refused.
    ssh_client = HostSshClient(["volume_pod"], free_gb=3800)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[CacheVolume(name="engy_cache_hf_1561gb_kimi-k3-1a2b3c4d", target="/root/models")],
    )
    payload.volume_limit_gb = volume_limit_gb

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


@pytest.mark.parametrize(
    ("volume_limit_gb", "cache_names", "expected_gb"),
    [
        (1900, ["engy_cache_hf_1561gb_kimi-k3-1a2b3c4d"], 100),  # checkpoint in the cache: no second copy
        (1900, [], 1900),  # cache refused: the run volume still has to hold the checkpoint
        (40, ["engy_cache_hf_31gb_qwen3-8-27b-1a2b3c4d"], 40),  # never grows a smaller request
        (80, ["dphn_cache_hf_x"], 80),  # Dolphin keeps its size
        (None, ["engy_cache_hf_1561gb_kimi-k3-1a2b3c4d"], None),  # no storage limit: nothing to size
    ],
)
def test_the_filler_run_volume_drops_the_checkpoint_once_the_engy_cache_holds_it(
    volume_limit_gb, cache_names, expected_gb
):
    caches = [CacheVolume(name=name, target="/root/models") for name in cache_names]

    assert _filler_run_volume_gb(volume_limit_gb, caches) == expected_gb


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
async def test_an_engy_create_sweeps_the_old_dolphin_cache_but_a_dolphin_create_keeps_the_engy_one(docker_service):
    # DAH-5192: a node moved to Engy no longer runs Dolphin; the reverse would drop a TB checkpoint
    engy = [CacheVolume(name="engy_cache_hf_new", target="/c")]
    ssh_client = FakeSshClient(["engy_cache_hf_new", "dphn_cache_hf_old", "volume_pod"])
    payload = _make_payload(workload_kind=WorkloadKind.FILLER, cache_volumes=engy)

    assert await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {}) == ["dphn_cache_hf_old"]

    dolphin = [CacheVolume(name="dphn_cache_hf_new", target="/c")]
    ssh_client = FakeSshClient(["dphn_cache_hf_new", "engy_cache_hf_old"])
    payload = _make_payload(workload_kind=WorkloadKind.FILLER, cache_volumes=dolphin)

    assert await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {}) == []


# ---------------------------------------------------------------- DAH-5192: partial caches, thresholds, rentals

_KIMI = "engy_cache_hf_1561gb_kimi-k3-1a2b3c4d"


class DuHostSshClient(HostSshClient):
    """A host whose existing cache volume holds `held_kb` (the helper's `du -sk` answer)."""

    def __init__(self, listed_volumes: list[str], free_gb: float, du_stdout: str):
        super().__init__(listed_volumes, free_gb)
        self._du_stdout = du_stdout

    async def run(self, command: str, *args, **kwargs):
        if "du -sk" in command:
            self.commands.append(command)
            return SimpleNamespace(exit_status=0, stdout=self._du_stdout, stderr="")
        return await super().run(command, *args, **kwargs)


def _kimi_filler(volume_limit_gb: int | None = 1900) -> ContainerCreateRequest:
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER, cache_volumes=[CacheVolume(name=_KIMI, target="/root/models")]
    )
    payload.volume_limit_gb = volume_limit_gb
    return payload


def _kb(gb: float) -> str:
    return f"{int(gb * 10**9 / 1024)}\t/cache\n"


@pytest.mark.asyncio
async def test_a_complete_cache_is_mounted_without_measuring_free_disk(docker_service):
    ssh_client = DuHostSshClient([_KIMI], free_gb=10, du_stdout=_kb(1560))

    assert await docker_service.select_affordable_cache_volumes(ssh_client, _kimi_filler(), {}) == _kimi_filler().cache_volumes
    assert not any("df -P" in command for command in ssh_client.commands)


@pytest.mark.asyncio
@pytest.mark.parametrize(("free_gb", "granted"), [(3800, True), (1200, False)])
async def test_an_interrupted_download_is_resumed_when_the_node_can_afford_what_it_still_lacks(
    docker_service, free_gb, granted
):
    # A rental removed the filler at 400 of 1561 GB. 3800 free: the 1161 GB left fits, the volume is remounted and the
    # next run resumes it. 1200 free: 1200 - 1161 - 100 is under the floor, so the partial is not mounted.
    ssh_client = DuHostSshClient([_KIMI], free_gb=free_gb, du_stdout=_kb(400))

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, _kimi_filler(), {})

    assert bool(affordable) is granted
    assert any("df -P" in command for command in ssh_client.commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("du_stdout", ["", "du: /cache: I/O error\n", "garbage"])
async def test_an_unmeasurable_cache_is_mounted_as_before(docker_service, du_stdout):
    ssh_client = DuHostSshClient([_KIMI], free_gb=10, du_stdout=du_stdout)

    assert await docker_service.select_affordable_cache_volumes(ssh_client, _kimi_filler(), {}) != []


@pytest.mark.asyncio
@pytest.mark.parametrize("volume_limit_gb", [1900, None])
@pytest.mark.parametrize(("free_gb", "granted"), [(2600, False), (3500, False), (3600, True), (5100, True)])
async def test_a_big_checkpoint_is_cached_only_when_the_node_still_fits_kimi_k3_beside_it(
    docker_service, free_gb, granted, volume_limit_gb
):
    # The cache stays used, and the backend gives kimi-k3 only to a node with 2000 GB free. 3500 - 1561 = 1939 GB:
    # cached, the node would never fit kimi-k3 again, so the run downloads into its own volume. 3600 leaves 2039 GB.
    # 3700-5100 GB B300s were refused by the 30% share in prod (26 Oct) and re-downloaded on every run.
    ssh_client = HostSshClient(["volume_pod"], free_gb=free_gb)

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, _kimi_filler(volume_limit_gb), {})

    assert bool(affordable) is granted


def _rental(volume_limit_gb: int | None) -> ContainerCreateRequest:
    payload = _make_payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL, cache_volumes=None)
    payload.volume_limit_gb = volume_limit_gb
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize("volume_limit_gb", [None, 2000])
async def test_a_rental_that_fits_beside_the_cache_keeps_it(docker_service, volume_limit_gb):
    # listed beside the cache: 2200 GB free covers a 2000 GB volume; no size (no --storage-opt) gets the disk beside it
    ssh_client = HostSshClient([_KIMI, "volume_pod"], free_gb=2200)

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, _rental(volume_limit_gb), {}) is False
    assert _removed_volumes(ssh_client) == []


@pytest.mark.asyncio
async def test_rental_evicts_the_oldest_cache_first_and_keeps_the_rest_once_it_fits(docker_service, monkeypatch):
    # A stale listing (the download kept writing until the filler was removed) must not fail a paid create: the
    # older cache goes first and the free-disk re-check stops the eviction there.
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
        docker_service, "_get_fs_available_bytes", AsyncMock(side_effect=[50 * 1024**3, 60 * 1024**3, 900 * 1024**3])
    )  # before, after the idle filler images went, after the older cache went

    assert await docker_service.reclaim_dphn_cache_for_rental(ssh_client, _rental(500), {}) is True
    assert _removed_volumes(ssh_client) == ["engy_cache_hf_old"]
