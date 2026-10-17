"""DAH-3240 rental volume fast path: a failed loopback plugin enable never leaks host text into errors or logs."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from services.docker_service import (
    DockerService,
    LoopbackPluginDisabledError,
    VolumeHostProbe,
)
from test_docker_service import _FakeRentalDockerFactory


@pytest.fixture
def docker_service():
    return DockerService(
        ssh_service=Mock(),
        redis_service=Mock(),
        attestation_service=Mock(),
        rental_docker_client_factory=_FakeRentalDockerFactory(),
    )


# ---------------------------------------------------------------------------
# create_local_volume: skip the Docker Hub round trip when the plugin is enabled
# ---------------------------------------------------------------------------


async def _create_volume(docker_service, ssh_client, probe):
    docker_service.stream_log = AsyncMock()
    docker_client = docker_service.rental_docker_client_factory.client
    await docker_service.create_local_volume(
        ssh_client=ssh_client,
        docker_client=docker_client,
        local_volume="volume_test",
        log_tag="tag",
        log_text="Creating docker volume volume_test",
        log_extra={},
        limit=40,
        timeout=10,
        host_probe=probe,
    )
    return docker_client.created_volumes


def _disabled_plugin_probe() -> VolumeHostProbe:
    return VolumeHostProbe(
        docker_root_dir="/var/lib/docker",
        df_avail_bytes=None,
        vloopback_volume_names=[],
        loopback_plugin_enabled=False,
        loopback_plugin_installed=True,
    )


@pytest.mark.asyncio
async def test_create_local_volume_disabled_plugin_enable_error_logs_only_its_type(
    docker_service, monkeypatch
):
    from services import docker_service as docker_service_module

    secret_text = "ssh ubuntu@10.1.2.3: token=ghp_leakme at /home/provider/.ssh/id_rsa"
    ssh_client = Mock()
    ssh_client.run = AsyncMock(side_effect=OSError(secret_text))
    warning = Mock()
    monkeypatch.setattr(docker_service_module.logger, "warning", warning)

    with pytest.raises(LoopbackPluginDisabledError) as exc_info:
        await _create_volume(docker_service, ssh_client, _disabled_plugin_probe())

    assert "enable error: OSError" in str(exc_info.value)
    assert secret_text not in str(exc_info.value)
    (logged,), _ = warning.call_args
    assert str(logged).startswith("Loopback plugin enable failed")
    assert logged.extra["error_type"] == "OSError"
    assert logged.extra["loopback_plugin"] == "vloopback"
    assert "error" not in logged.extra
    full = logged.to_full_string()
    assert "10.1.2.3" not in full and "ghp_leakme" not in full and "id_rsa" not in full
