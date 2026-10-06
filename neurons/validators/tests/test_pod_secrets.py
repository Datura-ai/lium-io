"""DAH-1482: renter secrets reach the pod as files on a tmpfs, never as env or on the disk layer."""

from unittest.mock import Mock

import pytest
from core.config import settings
from payload_models.payloads import ContainerCreateRequest, CustomOptions
from services.docker_service import DockerService
from services.rental_docker_sdk import (
    POD_SECRETS_DIR,
    GpuDockerConfig,
    _build_host_config_kwargs,
    build_pod_secret_exec_specs,
)

SECRETS = {"HF_TOKEN": "hf_value_1", "API_KEY": "value_2"}


def _payload(secrets: dict[str, str] | None) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey="hk", executor_id="ex", pod_id="pod", docker_image="img:tag", gpu_uuids=["g0"], secrets=secrets
    )


def _host_config(payload: ContainerCreateRequest) -> dict:
    docker_service = DockerService(
        ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock(), rental_docker_client_factory=Mock()
    )
    run_spec = docker_service._build_rental_container_run_spec(
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
    assert not set(SECRETS.values()) & set(run_spec.environment.values())
    return _build_host_config_kwargs(run_spec)


@pytest.mark.parametrize(
    ("enabled", "secrets", "mounted"),
    [(True, SECRETS, True), (False, SECRETS, False), (True, None, False)],
    ids=["on", "flag-off", "no-secrets"],
)
def test_the_tmpfs_is_mounted_only_with_the_flag_on_and_secrets_sent(monkeypatch, enabled, secrets, mounted) -> None:
    monkeypatch.setattr(settings, "POD_SECRETS_TMPFS_ENABLED", enabled)

    host_config = _host_config(_payload(secrets))

    assert (POD_SECRETS_DIR in (host_config.get("tmpfs") or {})) is mounted


def test_each_value_goes_on_stdin_as_the_image_user_and_ready_comes_last() -> None:
    specs = build_pod_secret_exec_specs(container_name="pod_test", secrets=SECRETS)

    assert [spec.stdin for spec in specs] == ["hf_value_1", "value_2", None]
    assert all(spec.user == "" for spec in specs)
    assert all("hf_value_1" not in part and "value_2" not in part for spec in specs for part in spec.argv)
    assert specs[-1].argv[-1].endswith(f"date -u > {POD_SECRETS_DIR}/.ready")


@pytest.mark.parametrize("name", ["../../tmp/HF_TOKEN", "HF_TOKEN=hf_pasted_value", "1KEY", "A; rm -rf /"])
def test_a_bad_name_is_refused_without_echoing_it(name) -> None:
    with pytest.raises(ValueError) as refused:
        build_pod_secret_exec_specs(container_name="pod_test", secrets={name: "value"})

    assert name not in str(refused.value)


def test_the_request_neither_logs_nor_reserializes_the_values() -> None:
    payload = _payload(SECRETS)

    assert "hf_value_1" not in str(payload)
    assert "hf_value_1" not in payload.model_dump_json()
