"""DAH-4689: the liumd machine-scrape document, with a Python fallback in specs mode."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from unittest.mock import AsyncMock

import pytest
from helpers import build_context_config, build_services, build_state, make_context
from services.liumd_exec_client import LiumdRefusal
from services.local_verify_client import LocalVerifyAnswer, LocalVerifyUnavailable
from services.task import liumd_shadow
from services.task.checks import gpu_identity, machine_spec_scrape
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
    "gpu_identity": {
        "cards": [
            {
                "index": 0,
                "uuid": "GPU-1",
                "pci": "0000:31:00.0",
                "model": "NVIDIA A10",
                "vram_mb": 24576,
                "driver": "595.112.07",
                "dev_node_present": True,
                "minor": 0,
                "sources": {
                    "nvml": {
                        "status": "agree",
                        "uuid": "GPU-1",
                        "pci": "0000:31:00.0",
                        "model": "NVIDIA A10",
                        "vram_mb": 24576,
                        "error": None,
                    },
                    "proc": {
                        "status": "ok",
                        "uuid": "GPU-1",
                        "pci": "0000:31:00.0",
                        "model": "NVIDIA A10",
                        "irq": "0",
                    },
                    "sysfs": {
                        "status": "ok",
                        "vendor": "0x10de",
                        "device": "0x2236",
                        "class": None,
                        "subsystem_vendor": "0x10de",
                        "subsystem_device": "0x145f",
                        "link_speed": "16.0 GT/s PCIe",
                        "link_width": "16",
                    },
                },
                "mismatches": [],
            },
            {
                "index": 1,
                "uuid": "GPU-2",
                "pci": "0000:32:00.0",
                "model": "NVIDIA A10",
                "vram_mb": 24576,
                "driver": "595.112.07",
                "dev_node_present": True,
                "minor": 1,
                "sources": {
                    "nvml": {
                        "status": "agree",
                        "uuid": "GPU-2",
                        "pci": "0000:32:00.0",
                        "model": "NVIDIA A10",
                        "vram_mb": 24576,
                        "error": None,
                    },
                    "proc": {
                        "status": "ok",
                        "uuid": "GPU-2",
                        "pci": "0000:32:00.0",
                        "model": "NVIDIA A10",
                        "irq": "0",
                    },
                    "sysfs": {
                        "status": "ok",
                        "vendor": "0x10de",
                        "device": "0x2236",
                        "class": None,
                        "subsystem_vendor": "0x10de",
                        "subsystem_device": "0x145f",
                        "link_speed": "16.0 GT/s PCIe",
                        "link_width": "16",
                    },
                },
                "mismatches": [],
            },
        ],
        "nvidia_module_present": True,
        "proc_driver_present": True,
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


def _with_identity(mutation=None):
    specs = copy.deepcopy(LIUMD_SPECS)
    if mutation is not None:
        mutation(specs["gpu_identity"])
    return specs


@pytest.mark.asyncio
async def test_shadow_logs_a_mismatch_and_passes(monkeypatch, caplog):
    _liumd_on(monkeypatch)
    monkeypatch.setattr(machine_spec_scrape.settings, "VALIDATOR_LIUMD_EVIDENCE_MODE", "shadow")
    specs = _with_identity(lambda identity: identity["cards"][1].update(uuid="GPU-1"))
    _start_reading(monkeypatch, _answer(specs))
    runner = DummySSHCommandRunner(result=make_command_result(success=False))

    with caplog.at_level(logging.WARNING, logger=gpu_identity.__name__):
        result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is True
    [line] = [r for r in caplog.records if r.getMessage() == gpu_identity.GPU_IDENTITY_EVENT]
    extra = line.msg.extra
    assert extra["mode"] == "shadow" and extra["would_fail"] is True
    assert "duplicate_uuid" in extra["reasons"]
    assert any("duplicate uuid" in d for d in extra["details"])
    assert extra["claimed_count"] == extra["evidence_count"] == 2


@pytest.mark.asyncio
async def test_shadow_logs_missing_identity_and_passes(monkeypatch, caplog):
    _liumd_on(monkeypatch)
    monkeypatch.setattr(machine_spec_scrape.settings, "VALIDATOR_LIUMD_EVIDENCE_MODE", "shadow")
    specs = _with_identity()
    specs.pop("gpu_identity")
    _start_reading(monkeypatch, _answer(specs))
    runner = DummySSHCommandRunner(result=make_command_result(success=False))

    with caplog.at_level(logging.WARNING, logger=gpu_identity.__name__):
        result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is True
    [line] = [r for r in caplog.records if r.getMessage() == gpu_identity.GPU_IDENTITY_EVENT]
    assert line.msg.extra["reasons"] == ["identity_missing"]
    assert line.msg.extra["would_fail"] is True


@pytest.mark.asyncio
async def test_enforce_passes_a_clean_identity_and_drops_the_block(monkeypatch):
    _liumd_on(monkeypatch)
    monkeypatch.setattr(machine_spec_scrape.settings, "VALIDATOR_LIUMD_EVIDENCE_MODE", "enforce")
    specs = _with_identity()
    specs["gpu_identity"]["wall_clock_ms"] = 1
    specs["gpu_identity"]["first_result_spread_ms"] = 999_999
    specs["gpu_identity"]["cards"][0]["tflops"] = 0.01
    _start_reading(monkeypatch, _answer(specs))
    runner = DummySSHCommandRunner(result=make_command_result(success=False))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is True
    assert result.event.reason_code == Msg.SCRAPE_OK.reason
    assert "gpu_identity" not in result.updates["state"].specs
    assert result.updates["state"].gpu_identity["cards"]


@pytest.mark.asyncio
async def test_enforce_fails_a_duplicate_card(monkeypatch):
    _liumd_on(monkeypatch)
    monkeypatch.setattr(machine_spec_scrape.settings, "VALIDATOR_LIUMD_EVIDENCE_MODE", "enforce")
    specs = _with_identity(lambda identity: identity["cards"][1].update(pci="0000:31:00.0"))
    _start_reading(monkeypatch, _answer(specs))
    runner = DummySSHCommandRunner(result=make_command_result(success=False))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_IDENTITY_MISMATCH.reason
    assert "duplicate_pci" in result.event.what_we_saw["reasons"]
    assert result.event.what_we_saw["delivery"] == "liumd"
    assert MachineSpecScrapeCheck.fatal is True


@pytest.mark.asyncio
async def test_enforce_fails_when_the_binary_sends_no_identity_block(monkeypatch):
    _liumd_on(monkeypatch)
    monkeypatch.setattr(machine_spec_scrape.settings, "VALIDATOR_LIUMD_EVIDENCE_MODE", "enforce")
    specs = _with_identity()
    specs.pop("gpu_identity")
    _start_reading(monkeypatch, _answer(specs))
    runner = DummySSHCommandRunner(result=make_command_result(success=False))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_IDENTITY_MISMATCH.reason
    assert result.event.what_we_saw["reasons"] == ["identity_missing"]


@pytest.mark.asyncio
async def test_enforce_refuses_the_python_fallback(monkeypatch):
    _liumd_on(monkeypatch)
    monkeypatch.setattr(machine_spec_scrape.settings, "VALIDATOR_LIUMD_EVIDENCE_MODE", "enforce")
    _start_reading(monkeypatch, LocalVerifyUnavailable("not_supported", "exit 127: no liumd"))
    runner = DummySSHCommandRunner(result=make_command_result(success=True, stdout=FERNET_TOKEN))

    result = await MachineSpecScrapeCheck().run(_ctx(runner))

    assert result.passed is False
    assert result.event.what_we_saw["delivery"] == "liumd"
    assert result.event.what_we_saw["liumd_fallback_refused"] == "not_supported"
    assert runner.calls == []
