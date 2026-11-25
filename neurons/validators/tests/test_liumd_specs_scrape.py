"""DAH-4689: the liumd machine-scrape document, with a Python fallback in specs mode."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from helpers import build_context_config, build_services, build_state, make_context
from services.liumd_exec_client import LiumdRefusal
from services.local_verify_client import LocalVerifyAnswer, LocalVerifyUnavailable
from services.task import liumd_shadow
from services.task.checks import machine_spec_scrape
from services.task.checks.machine_spec_scrape import MachineSpecScrapeCheck, normalize_liumd_specs
from services.task.messages import MachineSpecMessages as Msg
from test_machine_spec_scrape_check import (
    FERNET_TOKEN,
    RAW_SPECS,
    DummySSHCommandRunner,
    DummySSHService,
    make_command_result,
)

LIUMD_SPECS = {
    "gpu": {
        "count": 2,
        "details": [
            {"name": "NVIDIA A10", "uuid": "GPU-1", "capacity": 24576},
            {"name": "NVIDIA A10", "uuid": "GPU-2", "capacity": 24576},
        ],
    },
    "gpu_processes": [],
    "sysbox_runtime": True,
    "storage_limit_supported": True,
    "vloopback_check": {"verdict": "supported"},
}


def _answer(specs):
    return LocalVerifyAnswer(
        nonce="nonce",
        executor_uuid="executor-123",
        executor_version="liumd/0.1.0",
        elapsed_ms=10,
        deadline_hit=False,
        steps={},
        round_trip_ms=20,
        specs=specs,
    )


def _ctx(runner):
    return make_context(
        services=build_services(ssh=DummySSHService(decrypted_data=RAW_SPECS)),
        config=build_context_config(
            machine_scrape_source="scrape source",
            machine_scrape_filename="scrape.sh",
            obfuscation_keys={},
        ),
        state=build_state(remote_dir="/remote"),
        runner=runner,
        encrypt_key="test-encrypt-key",
    )


def _start_reading(monkeypatch, result):
    async def done():
        return result

    task = asyncio.ensure_future(done())
    calls = []

    def start(ctx):
        calls.append(ctx)
        return task

    monkeypatch.setattr(liumd_shadow, "start_facts_after_scrape", start)
    return calls


def _liumd_on(monkeypatch):
    monkeypatch.setattr(machine_spec_scrape.settings, "VALIDATOR_LIUMD_SPECS", True)
    monkeypatch.setattr(machine_spec_scrape.settings, "DRY_RUN", False)


@pytest.mark.asyncio
async def test_the_liumd_document_replaces_the_python_scrape(monkeypatch):
    _liumd_on(monkeypatch)
    start = _start_reading(monkeypatch, _answer(LIUMD_SPECS))
    runner = DummySSHCommandRunner(result=make_command_result(success=False))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is True
    assert result.event.reason_code == Msg.SCRAPE_OK.reason
    assert result.event.what_we_saw["delivery"] == "liumd"
    state = result.updates["state"]
    assert state.specs["gpu"]["count"] == 2
    assert state.gpu_count == 2
    assert [detail["uuid"] for detail in state.gpu_details] == ["GPU-1", "GPU-2"]
    assert "vloopback_check" not in state.specs
    assert state.scrape_specs is None
    assert state.liumd_shadow_facts_attempted is True
    assert state.liumd_shadow_facts_task is not None
    assert len(start) == 1
    assert runner.calls == []


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (None, "no_early_read"),
        (LocalVerifyUnavailable("not_supported", "exit 127: no liumd on this host"), "not_supported"),
        (_answer({}), "malformed"),
        (_answer({"cpu": {}}), "malformed"),
    ],
    ids=["no-early-read", "not-supported", "empty-specs", "no-gpu-key"],
)
@pytest.mark.asyncio
async def test_an_unusable_liumd_document_falls_back_to_the_python_scrape(
    monkeypatch, outcome, reason
):
    _liumd_on(monkeypatch)
    start = _start_reading(monkeypatch, outcome)
    runner = DummySSHCommandRunner(result=make_command_result(success=True, stdout=FERNET_TOKEN))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is True
    assert result.event.reason_code == Msg.SCRAPE_OK.reason
    assert result.event.what_we_saw["delivery"] == "stdin"
    assert result.event.what_we_saw["liumd_fallback_reason"] == reason
    state = result.updates["state"]
    assert state.specs["gpu"]["count"] == 1
    assert state.scrape_specs is not None
    assert state.liumd_shadow_facts_task is None
    assert len(runner.calls) == 1
    assert len(start) == 1


@pytest.mark.asyncio
async def test_a_failed_python_fallback_reports_the_liumd_reason(monkeypatch):
    _liumd_on(monkeypatch)
    _start_reading(monkeypatch, LocalVerifyUnavailable("binary_refresh_failed", "curl exit None"))
    runner = DummySSHCommandRunner(result=make_command_result(success=False))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is False
    assert result.event.what_we_saw["delivery"] == "stdin"
    assert result.event.what_we_saw["liumd_fallback_reason"] == "binary_refresh_failed"
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_a_liumd_refusal_keeps_its_error_and_detail_as_the_fallback_reason(monkeypatch):
    _liumd_on(monkeypatch)
    refusal = LiumdRefusal(
        exit_status=3, error="docker_unavailable", detail="cannot reach the docker daemon",
        echoed=True, round_trip_ms=15,
    )
    _start_reading(monkeypatch, refusal)
    runner = DummySSHCommandRunner(result=make_command_result(success=True, stdout=FERNET_TOKEN))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is True
    assert result.event.what_we_saw["liumd_fallback_reason"] == "docker_unavailable"
    assert result.event.what_we_saw["liumd_fallback_detail"] == "cannot reach the docker daemon"


@pytest.mark.asyncio
async def test_a_slow_liumd_read_times_out_and_falls_back(monkeypatch):
    _liumd_on(monkeypatch)
    monkeypatch.setattr(machine_spec_scrape, "LIUMD_SPECS_MIN_READ_SECONDS", 0.01)
    monkeypatch.setattr(machine_spec_scrape, "LIUMD_SPECS_READ_TIMEOUT_SECONDS", 0.01)

    async def slow():
        await asyncio.sleep(1)
        return _answer(LIUMD_SPECS)

    task = asyncio.ensure_future(slow())
    monkeypatch.setattr(liumd_shadow, "start_facts_after_scrape", lambda ctx: task)
    runner = DummySSHCommandRunner(result=make_command_result(success=True, stdout=FERNET_TOKEN))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is True
    assert result.event.what_we_saw["liumd_fallback_reason"] == "timeout"
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_an_unexpected_liumd_read_error_falls_back(monkeypatch):
    _liumd_on(monkeypatch)

    async def boom():
        raise RuntimeError("shadow bug")

    task = asyncio.ensure_future(boom())
    monkeypatch.setattr(liumd_shadow, "start_facts_after_scrape", lambda ctx: task)
    runner = DummySSHCommandRunner(result=make_command_result(success=True, stdout=FERNET_TOKEN))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is True
    assert result.event.what_we_saw["liumd_fallback_reason"] == "error"
    assert len(runner.calls) == 1


@pytest.mark.asyncio
async def test_a_liumd_document_without_gpus_keeps_the_scrape_failure_handling(monkeypatch):
    _liumd_on(monkeypatch)
    _start_reading(monkeypatch, _answer({"gpu": {"count": 0, "details": []}}))
    runner = DummySSHCommandRunner(result=make_command_result(success=False))
    reset = AsyncMock(return_value=False)
    loss = AsyncMock(return_value=[])
    monkeypatch.setattr(machine_spec_scrape, "missing_gpus_under_reset", reset)
    monkeypatch.setattr(machine_spec_scrape, "_report_rented_gpu_loss", loss)

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is False
    assert result.event.reason_code == Msg.SCRAPE_FAILED_NO_GPU.reason
    assert result.event.what_we_saw["delivery"] == "liumd"
    reset.assert_awaited_once()
    loss.assert_awaited_once()
    assert loss.await_args.args[1]["gpu"] == {"count": 0, "details": []}
    assert runner.calls == []


@pytest.mark.asyncio
async def test_a_liumd_document_without_gpus_during_a_renter_reset_is_held(monkeypatch):
    _liumd_on(monkeypatch)
    _start_reading(monkeypatch, _answer({"gpu": {"count": 0, "details": []}}))
    runner = DummySSHCommandRunner(result=make_command_result(success=False))
    loss = AsyncMock(return_value=[])
    monkeypatch.setattr(machine_spec_scrape, "missing_gpus_under_reset", AsyncMock(return_value=True))
    monkeypatch.setattr(machine_spec_scrape, "_report_rented_gpu_loss", loss)

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is False
    assert result.event.reason_code == Msg.SCRAPE_GPU_RESET_IN_PROGRESS.reason
    loss.assert_not_awaited()
    assert runner.calls == []


@pytest.mark.asyncio
async def test_a_liumd_nvml_error_is_reported_as_a_driver_failure(monkeypatch):
    _liumd_on(monkeypatch)
    _start_reading(
        monkeypatch,
        _answer({"gpu": {"count": 0, "details": []}, "gpu_scrape_error": "NVMLError(16)"}),
    )
    runner = DummySSHCommandRunner(result=make_command_result(success=False))
    monkeypatch.setattr(machine_spec_scrape, "missing_gpus_under_reset", AsyncMock(return_value=False))
    monkeypatch.setattr(machine_spec_scrape, "_report_rented_gpu_loss", AsyncMock(return_value=[]))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is False
    assert result.event.reason_code == Msg.SCRAPE_FAILED_DRIVER.reason
    assert result.event.what_we_saw["gpu_scrape_error"] == "NVMLError(16)"
    assert runner.calls == []


def test_normalization_maps_the_liumd_contract_without_mutating_the_snapshot():
    raw = {
        "data_gpu": {
            "gpu_count": 1,
            "gpu_details": [{"gpu.name": "NVIDIA A10", "gpu.uuid": "GPU-1", "gpu.capacity": 24576.0}],
        },
        "data_processes": [{"processes_pid": 7, "processes_info": "info", "processes_container_name": None}],
        "data_ram": {"ram_total": 1024.0, "ram_free": 512.0, "ram_used": 512.0, "ram_available": 900.0, "ram_utilization": 12.5},
        "data_os": "Ubuntu 24.04 LTS",
        "data_network": {},
        "data_vloopback_check": {"vc_verdict": "pass"},
    }
    snapshot = json.loads(json.dumps(raw))

    normalized = normalize_liumd_specs(raw)

    assert "data_gpu" not in normalized
    assert normalized["gpu"]["count"] == 1
    assert normalized["gpu"]["details"][0] == {"name": "NVIDIA A10", "uuid": "GPU-1", "capacity": 24576.0}
    assert normalized["gpu_processes"][0]["pid"] == 7
    assert normalized["gpu_processes"][0]["container_name"] is None
    assert normalized["ram"]["total"] == 1024.0
    assert normalized["os"] == "Ubuntu 24.04 LTS"
    assert "vloopback_check" not in normalized
    assert raw == snapshot
