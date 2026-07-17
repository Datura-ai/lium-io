"""DAH-2427: post-teardown wedged-GPU sweep in DockerService.delete_container."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neurons.validators.src.services.docker_service import (
    _sweep_wedged_gpus_after_teardown,
)
from neurons.validators.src.services.gpu_wedge import parse_wedged_gpu_uuids

WEDGED_UUID = "GPU-bdf72357-83a7-09d3-9809-729c734aa80a"
HEALTHY_UUID = "GPU-dde887f6-488b-085f-a7b0-a71557f3e330"


def test_parse_picks_only_the_wedge_signature():
    gpu_csv = f"{WEDGED_UUID}, 100, 0\n{HEALTHY_UUID}, 0, 0\nGPU-aaaa, 100, 40000\n"

    assert parse_wedged_gpu_uuids(gpu_csv, "") == [WEDGED_UUID]


def test_parse_skips_host_with_live_compute_apps():
    gpu_csv = f"{WEDGED_UUID}, 100, 0\n"

    assert parse_wedged_gpu_uuids(gpu_csv, "12345\n") == []


def test_parse_tolerates_garbage_lines():
    gpu_csv = "not-a-gpu-line\nGPU-bbbb, N/A, N/A\n"

    assert parse_wedged_gpu_uuids(gpu_csv, "") == []


def _ssh_result(stdout: str) -> MagicMock:
    return MagicMock(stdout=stdout, stderr="", exit_status=0)


@pytest.mark.asyncio
async def test_sweep_cures_wedged_gpu(monkeypatch):
    monkeypatch.setattr(
        "neurons.validators.src.services.docker_service.GPU_WEDGE_SWEEP_SETTLE_SECONDS", 0
    )
    responses = {
        "nvidia-smi --query-gpu": _ssh_result(f"{WEDGED_UUID}, 100, 0\n{HEALTHY_UUID}, 0, 0\n"),
        "nvidia-smi --query-compute-apps": _ssh_result(""),
        "CUDA_VISIBLE_DEVICES=": _ssh_result("ctx open/close OK"),
    }

    async def fake_run(cmd: str):
        for prefix, result in responses.items():
            if cmd.startswith(prefix):
                return result
        raise AssertionError(f"unexpected command: {cmd}")

    ssh_client = MagicMock()
    ssh_client.run = AsyncMock(side_effect=fake_run)
    log = MagicMock()

    await _sweep_wedged_gpus_after_teardown(ssh_client, log)

    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert not any("pkill" in cmd or "nvidia-smi -i" in cmd for cmd in commands)
    assert any(f"CUDA_VISIBLE_DEVICES={WEDGED_UUID}" in cmd and "cuCtxCreate" in cmd for cmd in commands)
    assert not any(f"CUDA_VISIBLE_DEVICES={HEALTHY_UUID}" in cmd for cmd in commands)


@pytest.mark.asyncio
async def test_sweep_is_a_noop_on_healthy_gpus(monkeypatch):
    monkeypatch.setattr(
        "neurons.validators.src.services.docker_service.GPU_WEDGE_SWEEP_SETTLE_SECONDS", 0
    )

    async def fake_run(cmd: str):
        if cmd.startswith("nvidia-smi --query-gpu"):
            return _ssh_result(f"{HEALTHY_UUID}, 0, 0\n")
        if cmd.startswith("nvidia-smi --query-compute-apps"):
            return _ssh_result("")
        raise AssertionError(f"unexpected command: {cmd}")

    ssh_client = MagicMock()
    ssh_client.run = AsyncMock(side_effect=fake_run)

    await _sweep_wedged_gpus_after_teardown(ssh_client, MagicMock())

    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert not any("pkill" in cmd or "CUDA_VISIBLE_DEVICES" in cmd for cmd in commands)


@pytest.mark.asyncio
async def test_sweep_skips_the_settle_wait_when_nothing_looks_wedged():
    """The common healthy teardown must pay one cheap query pair and no settle wait."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    async def fake_run(cmd: str):
        if cmd.startswith("nvidia-smi --query-gpu"):
            return _ssh_result(f"{HEALTHY_UUID}, 0, 0\n")
        if cmd.startswith("nvidia-smi --query-compute-apps"):
            return _ssh_result("")
        raise AssertionError(f"unexpected command: {cmd}")

    ssh_client = MagicMock()
    ssh_client.run = AsyncMock(side_effect=fake_run)

    with patch("neurons.validators.src.services.docker_service.asyncio.sleep", fake_sleep):
        await _sweep_wedged_gpus_after_teardown(ssh_client, MagicMock())

    assert slept == []
    assert ssh_client.run.await_count == 2


@pytest.mark.asyncio
async def test_sweep_does_not_reset_a_card_that_settles_by_itself(monkeypatch):
    """A workload caught mid-exit looks wedged at t=0; the second sample must clear it."""
    monkeypatch.setattr(
        "neurons.validators.src.services.docker_service.GPU_WEDGE_SWEEP_SETTLE_SECONDS", 0
    )
    gpu_query_results = iter(
        [
            _ssh_result(f"{WEDGED_UUID}, 100, 0\n"),
            _ssh_result(f"{WEDGED_UUID}, 0, 0\n"),
        ]
    )

    async def fake_run(cmd: str):
        if cmd.startswith("nvidia-smi --query-gpu"):
            return next(gpu_query_results)
        if cmd.startswith("nvidia-smi --query-compute-apps"):
            return _ssh_result("")
        raise AssertionError(f"unexpected command: {cmd}")

    ssh_client = MagicMock()
    ssh_client.run = AsyncMock(side_effect=fake_run)

    await _sweep_wedged_gpus_after_teardown(ssh_client, MagicMock())

    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert not any("pkill" in cmd or "CUDA_VISIBLE_DEVICES" in cmd for cmd in commands)
