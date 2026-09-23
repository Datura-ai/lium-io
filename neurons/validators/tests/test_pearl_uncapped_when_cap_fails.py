"""DAH-3630: a PEARL filler whose GPU power cap does not take.

``apply_filler_gpu_power_limits`` is all-or-nothing and undoes its own partial work, so after a
False the host is exactly as it was. The create then fails (flag off, as before) or starts the
filler at the host's own power limit (``ENABLE_PEARL_UNCAPPED_WHEN_CAP_FAILS``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from core.config import settings
from payload_models.payloads import FailedContainerRequest, GpuPowerLimit, WorkloadKind
from services import docker_service as ds_module
from services.docker_service import DockerService
from test_deploy_optimizations import (
    _payload as _deploy_payload,
    _run as _run_create_container,
    _ssh_client as _deploy_ssh_client,
)
from test_prerun_host_probe import _probe, _wire


@pytest.fixture
def svc() -> DockerService:
    return DockerService(ssh_service=Mock(), redis_service=Mock(), attestation_service=Mock())


def _pearl_payload():
    return _deploy_payload(
        workload_kind=WorkloadKind.FILLER,
        gpu_power_limits=[GpuPowerLimit(gpu_uuid="GPU-test", watts=300)],
    )


def _wire_failed_cap(svc: DockerService, monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(settings, "RENTAL_PRERUN_HOST_PROBE_ENABLED", True)
    _wire(svc, monkeypatch, _deploy_ssh_client(), probe_result=_probe())
    monkeypatch.setattr(
        "services.docker_service.apply_filler_gpu_power_limits", AsyncMock(return_value=False)
    )
    mock_logger = Mock()
    monkeypatch.setattr(ds_module, "logger", mock_logger)
    return mock_logger


def _uncapped_start_warnings(mock_logger: Mock) -> list:
    return [
        call
        for call in mock_logger.warning.call_args_list
        if call.args and (getattr(call.args[0], "extra", None) or {}).get("reason") == "pearl_started_uncapped"
    ]


@pytest.mark.asyncio
async def test_failed_cap_still_refuses_the_filler_while_the_flag_is_off(svc, monkeypatch) -> None:
    """Regression: the runtime change leaking out of its flag. While the backend guard still bans a
    PEARL run at default power, an uncapped start costs the node 12 h instead of a retry."""
    monkeypatch.setattr(settings, "ENABLE_PEARL_UNCAPPED_WHEN_CAP_FAILS", False)
    mock_logger = _wire_failed_cap(svc, monkeypatch)

    result = await _run_create_container(svc, _pearl_payload())

    assert isinstance(result, FailedContainerRequest)
    svc._run_rental_docker_create_with_port_retry.assert_not_awaited()
    assert _uncapped_start_warnings(mock_logger) == []


@pytest.mark.asyncio
async def test_failed_cap_starts_the_filler_at_the_hosts_own_limit_with_the_flag_on(svc, monkeypatch) -> None:
    """Regression: PEARL still refused on a host whose cap does not take, leaving the GPU idle while
    it draws the unrented incentive. The start is logged at WARNING with the GPUs it left uncapped,
    and the customer-path restore/raise (for containers with no cap of their own) does not run."""
    monkeypatch.setattr(settings, "ENABLE_PEARL_UNCAPPED_WHEN_CAP_FAILS", True)
    mock_logger = _wire_failed_cap(svc, monkeypatch)

    result = await _run_create_container(svc, _pearl_payload())

    assert type(result).__name__ == "ContainerCreated", getattr(result, "msg", "")
    svc._run_rental_docker_create_with_port_retry.assert_awaited_once()
    warnings = _uncapped_start_warnings(mock_logger)
    assert len(warnings) == 1
    assert warnings[0].args[0].extra["gpu_uuids"] == ["GPU-test"]
    ds_module.raise_low_power_limits_to_default.assert_not_awaited()
    ds_module.restore_tracked_gpu_power_limits.assert_not_awaited()
