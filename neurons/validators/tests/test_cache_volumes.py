from unittest.mock import Mock

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


def _make_payload(*, workload_kind: WorkloadKind, cache_volumes: list[CacheVolume] | None) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        workload_kind=workload_kind,
        cache_volumes=cache_volumes,
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
