"""lium mine preflight must not refuse a node the validator passes (DAH-6616)."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from preflight.base import CheckStatus
from preflight.checks.gpu_check import GPUCheck
from preflight.checks.matrix_check import MatrixValidationCheck
from preflight.checks.verifyx_check import VerifyXCheck
from test_verifyx_capacity_probe import (
    DOWNLOAD_RATE_LIMITED_ERROR,
    DOWNLOAD_TIMEOUT_ERROR,
    DOWNLOAD_TRANSPORT_ERROR,
    _cloudflare_unreachable_payload,
    _probe_payload,
)

PASSING = {"memory_execution": {"success": True}, "storage_execution": {}}


def _errors(net: dict) -> list[str]:
    return VerifyXCheck()._collect_errors({"response_data": {**PASSING, "network_execution": net}})


@pytest.mark.parametrize("error", [DOWNLOAD_TIMEOUT_ERROR, DOWNLOAD_RATE_LIMITED_ERROR, DOWNLOAD_TRANSPORT_ERROR])
def test_a_cloudflare_side_fault_with_the_package_downloaded_passes(error):
    assert _errors(_cloudflare_unreachable_payload(error=error)) == []


def test_a_finished_probe_passes():
    assert _errors(_probe_payload()) == []


@pytest.mark.parametrize("error", ["speed.cloudflare.com blocked by host firewall", "HTTP 503 Service Unavailable"])
def test_a_host_caused_network_failure_still_fails(error):
    assert _errors(_cloudflare_unreachable_payload(error=error)) == [f"Network: {error}"]


def test_a_cloudflare_fault_with_the_package_download_failed_still_fails():
    net = _cloudflare_unreachable_payload(error=DOWNLOAD_TIMEOUT_ERROR)
    net["download"] = {**net["download"], "status": "failed"}
    assert _errors(net) == [f"Network: {DOWNLOAD_TIMEOUT_ERROR}"]


def test_an_nvml_failure_reports_the_driver_error_not_no_gpus():
    def init():
        raise RuntimeError("NVML_ERROR_DRIVER_NOT_LOADED")

    nvml = SimpleNamespace(nvmlInit=init, nvmlShutdown=lambda: None)
    with patch.dict(sys.modules, {"pynvml": nvml}):
        result = asyncio.run(GPUCheck().run({"gpu": {"max_gpu_count": 8}}))
    assert result.status == CheckStatus.FAILED
    assert result.message == "NVIDIA driver query (NVML) failed: RuntimeError: NVML_ERROR_DRIVER_NOT_LOADED"


@pytest.mark.parametrize("check", [MatrixValidationCheck, VerifyXCheck])
def test_the_gpu_using_checks_report_the_nvml_error_too(check):
    def init():
        raise RuntimeError("NVML_ERROR_DRIVER_NOT_LOADED")

    with patch.dict(sys.modules, {"pynvml": SimpleNamespace(nvmlInit=init, nvmlShutdown=lambda: None)}):
        result = asyncio.run(check().run({}))
    assert result.status == CheckStatus.FAILED
    assert "NVML_ERROR_DRIVER_NOT_LOADED" in result.message
    assert "No GPUs detected" not in result.message


def test_no_devices_still_reads_as_no_gpus():
    nvml = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None, nvmlDeviceGetCount=lambda: 0)
    with patch.dict(sys.modules, {"pynvml": nvml}):
        result = asyncio.run(GPUCheck().run({"gpu": {"max_gpu_count": 8}}))
    assert result.status == CheckStatus.FAILED
    assert result.message.startswith("No GPUs detected")
