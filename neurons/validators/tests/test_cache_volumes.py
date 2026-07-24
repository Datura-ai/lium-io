from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from payload_models.payloads import CacheVolume, ContainerCreateRequest, CustomOptions, WorkloadKind
from services.docker_service import DockerService, _build_cache_volume_mounts, _validate_cache_volume
from services.rental_docker_sdk import GpuDockerConfig


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


def test_build_cache_volume_mounts_filler_appends_all_entries():
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache"),
            CacheVolume(name="dphn_cache_dp_v1", target="/opt/dolphinpod"),
        ],
    )

    mounts = _build_cache_volume_mounts(payload, {"/root"})

    assert [(m.source, m.target) for m in mounts] == [
        ("dphn_cache_hf_v1", "/root/.cache"),
        ("dphn_cache_dp_v1", "/opt/dolphinpod"),
    ]


def test_build_cache_volume_mounts_ignored_for_customer_rental():
    payload = _make_payload(
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache")],
    )

    mounts = _build_cache_volume_mounts(payload, {"/root"})

    assert mounts == []


def test_build_cache_volume_mounts_no_field_is_empty():
    payload = _make_payload(workload_kind=WorkloadKind.FILLER, cache_volumes=None)

    mounts = _build_cache_volume_mounts(payload, {"/root"})

    assert mounts == []


def test_build_cache_volume_mounts_skips_target_colliding_with_local_volume():
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_v1", target="/root")],
    )

    mounts = _build_cache_volume_mounts(payload, {"/root"})

    assert mounts == []


def test_build_cache_volume_mounts_skips_target_colliding_with_mnt():
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="dphn_cache_dp_v1", target="/mnt"),
            CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache"),
        ],
    )

    mounts = _build_cache_volume_mounts(payload, {"/root", "/mnt"})

    assert [(m.source, m.target) for m in mounts] == [("dphn_cache_hf_v1", "/root/.cache")]


def test_validate_cache_volume_rejects_host_bind_name():
    with pytest.raises(ValueError):
        _validate_cache_volume(CacheVolume(name="/etc", target="/root/.cache"))


def test_validate_cache_volume_rejects_ephemeral_prefix():
    with pytest.raises(ValueError):
        _validate_cache_volume(CacheVolume(name="volume_abc", target="/root/.cache"))


@pytest.mark.parametrize("bad_target", ["relative/path", "/", "/opt/../etc", "/root:/x"])
def test_validate_cache_volume_rejects_bad_target(bad_target: str):
    with pytest.raises(ValueError):
        _validate_cache_volume(CacheVolume(name="dphn_cache_hf_v1", target=bad_target))


def test_run_spec_includes_filler_cache_mounts(docker_service):
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache"),
            CacheVolume(name="dphn_cache_dp_v1", target="/opt/dolphinpod"),
        ],
    )

    run_spec = docker_service._build_rental_container_run_spec(
        payload=payload,
        container_name="filler_pod",
        custom_options=CustomOptions(),
        port_maps=[],
        local_volume="volume_pod",
        local_volume_path="/root",
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
    )

    assert [(m.source, m.target) for m in run_spec.volumes] == [
        ("volume_pod", "/root"),
        ("dphn_cache_hf_v1", "/root/.cache"),
        ("dphn_cache_dp_v1", "/opt/dolphinpod"),
    ]


def test_run_spec_omits_cache_mounts_for_customer_rental(docker_service):
    payload = _make_payload(
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache")],
    )

    run_spec = docker_service._build_rental_container_run_spec(
        payload=payload,
        container_name="pod",
        custom_options=CustomOptions(),
        port_maps=[],
        local_volume="volume_pod",
        local_volume_path="/root",
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
    )

    assert [(m.source, m.target) for m in run_spec.volumes] == [("volume_pod", "/root")]


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


@pytest.mark.asyncio
async def test_an_existing_cache_is_mounted_without_charging_for_it_again(docker_service):
    # THE bug this moved here to fix. The backend used to subtract the cache size on every launch, so a
    # node granted the cache once fell below its own threshold by exactly that download and was refused
    # it forever after. Mounting a volume that already exists costs zero disk, so no check applies —
    # 170 GB free is under floor+margin+size (190) yet the cache must still be mounted.
    ssh_client = HostSshClient(["dphn_cache_hf_v1", "dphn_cache_dp_v1"], free_gb=170)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache"),
            CacheVolume(name="dphn_cache_dp_v1", target="/opt/dolphinpod"),
        ],
    )

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})

    assert [volume.name for volume in affordable] == ["dphn_cache_hf_v1", "dphn_cache_dp_v1"]


@pytest.mark.asyncio
async def test_a_missing_cache_is_refused_when_the_download_would_delist_the_node(docker_service):
    # Nothing on the host yet, so the mount really does cost a download. Below floor+margin+size the
    # node would drop out of the rental listing, where neither renters nor fillers can reach it.
    ssh_client = HostSshClient(["volume_pod"], free_gb=170)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache")],
    )

    assert await docker_service.select_affordable_cache_volumes(ssh_client, payload, {}) == []


@pytest.mark.asyncio
async def test_a_missing_cache_is_granted_when_the_node_can_afford_the_download(docker_service):
    ssh_client = HostSshClient(["volume_pod"], free_gb=250)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache")],
    )

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})

    assert [volume.name for volume in affordable] == ["dphn_cache_hf_v1"]


@pytest.mark.asyncio
async def test_a_tight_node_keeps_the_half_of_its_cache_that_already_exists(docker_service):
    # A model bump renames one volume. The surviving one costs nothing, so it is still mounted; only
    # the renamed one needs a download the node cannot afford.
    ssh_client = HostSshClient(["dphn_cache_hf_v1"], free_gb=170)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache"),
            CacheVolume(name="dphn_cache_dp_v2", target="/opt/dolphinpod"),
        ],
    )

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})

    assert [volume.name for volume in affordable] == ["dphn_cache_hf_v1"]


@pytest.mark.asyncio
async def test_a_customer_rental_never_gets_cache_volumes(docker_service):
    ssh_client = HostSshClient([], free_gb=1000)
    payload = _make_payload(
        workload_kind=WorkloadKind.CUSTOMER_RENTAL,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache")],
    )

    assert await docker_service.select_affordable_cache_volumes(ssh_client, payload, {}) == []


@pytest.mark.asyncio
async def test_an_unreadable_disk_grants_only_what_already_exists(docker_service):
    # Fail closed: without a reading we cannot promise the download fits, but what is already on the
    # host is free to mount either way.
    class BlindSshClient(HostSshClient):
        async def run(self, command: str, *args, **kwargs):
            if "df -P -B1" in command:
                raise ConnectionResetError("ssh dropped while measuring the disk")
            return await super().run(command, *args, **kwargs)

    ssh_client = BlindSshClient(["dphn_cache_hf_v1"], free_gb=1000)
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="dphn_cache_hf_v1", target="/root/.cache"),
            CacheVolume(name="dphn_cache_dp_v2", target="/opt/dolphinpod"),
        ],
    )

    affordable = await docker_service.select_affordable_cache_volumes(ssh_client, payload, {})

    assert [volume.name for volume in affordable] == ["dphn_cache_hf_v1"]


def _removed_volumes(ssh_client: FakeSshClient) -> list[str]:
    removals = [command for command in ssh_client.commands if "volume rm" in command]
    if not removals:
        return []
    return sorted(removals[0].split("volume rm ", 1)[1].split(" 2>/dev/null")[0].split())


@pytest.mark.asyncio
async def test_sweep_removes_the_previous_model_cache(docker_service):
    # A Dolphin model update renames the cache volume; the set left behind by the old model must go in
    # the same create, otherwise every update leaks another ~37 GB onto the host.
    ssh_client = FakeSshClient(["dphn_cache_hf_old", "dphn_cache_hf_new", "volume_pod", "some_other_volume"])
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_new", target="/root/.cache")],
    )

    await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {})

    assert _removed_volumes(ssh_client) == ["dphn_cache_hf_old"]


@pytest.mark.asyncio
async def test_sweep_removes_the_old_cache_even_when_no_requested_volume_exists_yet(docker_service):
    # The strand scenario: a model+tag bump renames BOTH volumes on a tight node. The old set is dead
    # weight that keeps the node under the affordability gate, so the sweep must run against the
    # REQUESTED names before affordability is judged — narrowing first would empty the request and the
    # sweep would misread it as "no cache wanted", leaving the dead 37GB in place forever.
    ssh_client = FakeSshClient(["dphn_cache_hf_old", "dphn_cache_dp_old", "volume_pod"])
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="dphn_cache_hf_new", target="/root/.cache"),
            CacheVolume(name="dphn_cache_dp_new", target="/opt/dolphinpod"),
        ],
    )

    await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {})

    assert _removed_volumes(ssh_client) == ["dphn_cache_dp_old", "dphn_cache_hf_old"]


@pytest.mark.asyncio
async def test_a_failing_sweep_never_fails_the_launch(docker_service):
    # The sweep runs inside create_container's try. Housekeeping that raises would surface as a
    # container-create failure, mark the filler FAILED and cost the node a backoff strike — the exact
    # failure class DAH-2475 exists to remove. The rental-side reclaim already swallows its errors.
    class BrokenSshClient(FakeSshClient):
        async def run(self, command: str, *args, **kwargs):
            if "volume rm" in command:
                raise ConnectionResetError("ssh connection reset during housekeeping")
            return await super().run(command, *args, **kwargs)

    ssh_client = BrokenSshClient(["dphn_cache_hf_old", "dphn_cache_hf_new"])
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_new", target="/root/.cache")],
    )

    await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {})


@pytest.mark.asyncio
async def test_sweep_keeps_the_current_cache_and_unrelated_volumes(docker_service):
    ssh_client = FakeSshClient(["dphn_cache_hf_new", "dphn_cache_dp_new", "volume_pod"])
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[
            CacheVolume(name="dphn_cache_hf_new", target="/root/.cache"),
            CacheVolume(name="dphn_cache_dp_new", target="/opt/dolphinpod"),
        ],
    )

    await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {})

    assert _removed_volumes(ssh_client) == []


@pytest.mark.asyncio
async def test_sweep_leaves_the_cache_alone_when_no_cache_is_requested(docker_service):
    # A node denied the cache (too little free disk) must NOT have its existing cache deleted here:
    # its free disk is low BECAUSE the cache is downloaded, so deleting would raise free disk, the
    # next cycle would grant the cache again, and the node would re-download ~37 GB forever.
    # Reclaiming belongs to the rental/disk-pressure path.
    ssh_client = FakeSshClient(["dphn_cache_hf_old", "dphn_cache_dp_old", "volume_pod"])
    payload = _make_payload(workload_kind=WorkloadKind.FILLER, cache_volumes=None)

    await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {})

    assert ssh_client.commands == []


@pytest.mark.asyncio
async def test_sweep_never_touches_a_customer_rental_host(docker_service):
    ssh_client = FakeSshClient(["dphn_cache_hf_old"])
    payload = _make_payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL, cache_volumes=None)

    await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {})

    assert ssh_client.commands == []


@pytest.mark.asyncio
async def test_rented_pod_cache_failure_does_not_fail_the_create(docker_service):
    # DAH-2475/B2: the rented-machine hash is a cache the backend rebuilds every ~10 min, so a Redis
    # blip at the last step must NOT propagate — raising here used to trip the cleanup path and
    # destroy a container that was already built and running (throwing away a ~40 min DPHN download).
    docker_service.redis_service.add_rented_pod = AsyncMock(side_effect=Exception("Timeout connecting to server"))

    await docker_service._cache_rented_pod_best_effort(
        executor_info=Mock(),
        pod_id="pod",
        container_name="filler_pod",
        default_extra={},
    )

    docker_service.redis_service.add_rented_pod.assert_awaited_once()


@pytest.mark.asyncio
async def test_sweep_ignores_its_own_container_when_looking_for_siblings(docker_service):
    # The run being created is already STARTING before the backend builds this request, so its OWN
    # container name arrives in active_container_names. Counting it made the sibling guard always
    # true and the sweep never ran on a real node (found on staging 2026-07-23).
    ssh_client = FakeSshClient(["dphn_cache_hf_old", "dphn_cache_hf_new"])
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_new", target="/root/.cache")],
        active_container_names=["filler_pod"],  # == f"filler_{payload.pod_id}"
    )

    await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {})

    assert _removed_volumes(ssh_client) == ["dphn_cache_hf_old"]


@pytest.mark.asyncio
async def test_sweep_is_skipped_while_a_live_filler_sibling_holds_the_cache(docker_service):
    # Docker cannot remove a volume a sibling still mounts, so listing volumes there is a guaranteed
    # no-op — skipping it keeps an extra SSH exec off every bundle create while a node fills.
    ssh_client = FakeSshClient(["dphn_cache_hf_old", "dphn_cache_hf_new"])
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_new", target="/root/.cache")],
        active_container_names=["filler_sibling", "filler_pod"],
    )

    await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {})

    assert ssh_client.commands == []


# ---------------------------------------------------------------- reclaim (levels 2 and 3)


@pytest.mark.asyncio
async def test_rental_reclaims_the_cache_when_the_requested_volume_does_not_fit(docker_service, monkeypatch):
    # The disk belongs to the renter: if what they asked for does not fit next to the ~37GB filler
    # cache, the cache goes. Reclaiming HERE (not at filler stop) is what stops the re-download loop.
    ssh_client = FakeSshClient(["dphn_cache_hf_x", "dphn_cache_dp_x", "volume_pod"])
    monkeypatch.setattr(docker_service, "get_docker_root_dir", AsyncMock(return_value="/var/lib/docker"))
    monkeypatch.setattr(docker_service, "_get_fs_available_bytes", AsyncMock(return_value=50 * 1024**3))
    payload = _make_payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL, cache_volumes=None)
    payload.volume_limit_gb = 100

    await docker_service.reclaim_dphn_cache_for_rental(ssh_client, payload, {})

    assert _removed_volumes(ssh_client) == ["dphn_cache_dp_x", "dphn_cache_hf_x"]


@pytest.mark.asyncio
async def test_rental_keeps_the_cache_when_the_volume_already_fits(docker_service, monkeypatch):
    ssh_client = FakeSshClient(["dphn_cache_hf_x", "volume_pod"])
    monkeypatch.setattr(docker_service, "get_docker_root_dir", AsyncMock(return_value="/var/lib/docker"))
    monkeypatch.setattr(docker_service, "_get_fs_available_bytes", AsyncMock(return_value=900 * 1024**3))
    payload = _make_payload(workload_kind=WorkloadKind.CUSTOMER_RENTAL, cache_volumes=None)
    payload.volume_limit_gb = 100

    await docker_service.reclaim_dphn_cache_for_rental(ssh_client, payload, {})

    assert _removed_volumes(ssh_client) == []


@pytest.mark.asyncio
async def test_filler_create_never_triggers_the_rental_reclaim(docker_service):
    # A filler create must not reclaim its own cache — that is the thrash loop we designed out.
    ssh_client = FakeSshClient(["dphn_cache_hf_x"])
    payload = _make_payload(workload_kind=WorkloadKind.FILLER, cache_volumes=None)
    payload.volume_limit_gb = 100

    await docker_service.reclaim_dphn_cache_for_rental(ssh_client, payload, {})

    assert ssh_client.commands == []
