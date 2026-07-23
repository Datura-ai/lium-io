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
async def test_sweep_is_skipped_while_a_live_filler_sibling_holds_the_cache(docker_service):
    # Docker cannot remove a volume a sibling still mounts, so listing volumes there is a guaranteed
    # no-op — skipping it keeps an extra SSH exec off every bundle create while a node fills.
    ssh_client = FakeSshClient(["dphn_cache_hf_old", "dphn_cache_hf_new"])
    payload = _make_payload(
        workload_kind=WorkloadKind.FILLER,
        cache_volumes=[CacheVolume(name="dphn_cache_hf_new", target="/root/.cache")],
        active_container_names=["filler_sibling"],
    )

    await docker_service.sweep_stale_cache_volumes(ssh_client, payload, {})

    assert ssh_client.commands == []
