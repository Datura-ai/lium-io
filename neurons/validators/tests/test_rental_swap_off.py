"""A rental container runs with swap off and a renter-first OOM score."""

import pytest
from pydantic import ValidationError

from core.config import Settings
from services.rental_docker_sdk import ContainerRunSpec, _build_host_config_kwargs
from tests.test_deploy_optimizations import svc  # noqa: F401  (pytest fixture)


def test_host_config_sets_memory_swap_equal_to_memory_and_the_oom_score():
    kwargs = _build_host_config_kwargs(
        ContainerRunSpec(image="img", name="pod_x", memory_gb=30, memory_swap_gb=30, oom_score_adj=500)
    )

    assert kwargs["mem_limit"] == "30g"
    assert kwargs["memswap_limit"] == "30g"
    assert kwargs["oom_score_adj"] == 500


def test_host_config_without_swap_off_is_the_flags_of_before():
    kwargs = _build_host_config_kwargs(ContainerRunSpec(image="img", name="pod_x", memory_gb=30))

    assert kwargs["mem_limit"] == "30g"
    assert "memswap_limit" not in kwargs
    assert "oom_score_adj" not in kwargs


def test_host_config_never_sends_memory_swap_without_memory():
    kwargs = _build_host_config_kwargs(ContainerRunSpec(image="img", name="pod_x", memory_swap_gb=8))

    assert "mem_limit" not in kwargs and "memswap_limit" not in kwargs


def _run_spec(docker_service, memory_gb):
    from payload_models.payloads import ContainerCreateRequest, CustomOptions
    from services.rental_docker_sdk import GpuDockerConfig

    payload = ContainerCreateRequest(
        miner_hotkey="hk",
        executor_id="ex",
        pod_id="pod",
        docker_image="img:tag",
        gpu_uuids=["g0"],
        memory_gb=memory_gb,
    )
    return docker_service._build_rental_container_run_spec(
        payload=payload,
        container_name="pod_test",
        custom_options=CustomOptions(),
        port_maps=[(22, 30022, 40022)],
        local_volume="volume_pod",
        local_volume_path="/root",
        encrypted_local_volume=False,
        external_volume_name=None,
        gpu_devices=GpuDockerConfig(),
        effective_storage_limit_gb=None,
        cpu_count=None,
    )


@pytest.mark.parametrize(
    "memory_gb, expected",
    [(30, (30, 30, 500)), (0, (None, None, None))],
)
def test_run_spec_swap_off_follows_the_memory_limit(svc, memory_gb, expected):  # noqa: F811
    run_spec = _run_spec(svc, memory_gb)

    assert (run_spec.memory_gb or None, run_spec.memory_swap_gb, run_spec.oom_score_adj) == expected


@pytest.mark.parametrize("value", ["0", "-1", "1001"])
def test_an_oom_score_outside_1_to_1000_fails_startup(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("RENTAL_CONTAINER_OOM_SCORE_ADJ", value)

    with pytest.raises(ValidationError, match="RENTAL_CONTAINER_OOM_SCORE_ADJ"):
        Settings(_env_file=None)


@pytest.mark.parametrize("value", ["1", "1000"])
def test_an_oom_score_at_the_bounds_is_accepted(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("RENTAL_CONTAINER_OOM_SCORE_ADJ", value)

    assert Settings(_env_file=None).RENTAL_CONTAINER_OOM_SCORE_ADJ == int(value)
