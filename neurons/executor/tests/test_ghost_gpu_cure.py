"""Ghost GPU auto-cure (DAH-2431): detection signature and watchdog behavior.

A "ghost" GPU reads 100% utilization with ~0 MiB used and zero compute
processes (Blackwell GSP latch). The watchdog must cure only a sustained
ghost signature and must never touch a GPU that owns processes or memory.
"""

import asyncio
from types import SimpleNamespace

import gpu_ghost_cure
import gpus_utility

GHOST_UUID = "GPU-8c6b85ff-2c93-6b26-8802-8b3e93a9b1f5"
BUSY_UUID = "GPU-1f6f4c6e-9b6a-4a0e-b1c2-3d4e5f607182"
IDLE_UUID = "GPU-aa11bb22-cc33-dd44-ee55-ff6677889900"


def _fake_smi(gpu_rows, compute_app_rows):
    def fake(args):
        if args[0].startswith("--query-gpu"):
            return gpu_rows
        return compute_app_rows

    return fake


def test_detect_ghosts_flags_pinned_idle_gpu(monkeypatch):
    # Arrange: one latched GPU, one genuinely idle GPU, no compute processes
    gpu_rows = [[GHOST_UUID, "100", "1"], [IDLE_UUID, "0", "1"]]
    monkeypatch.setattr(gpu_ghost_cure, "_smi", _fake_smi(gpu_rows, []))

    # Act
    ghosts = gpu_ghost_cure.detect_ghosts()

    # Assert
    assert ghosts == [GHOST_UUID]


def test_detect_ghosts_ignores_gpu_with_compute_processes(monkeypatch):
    # Arrange: 100% util and low memory, but a compute process owns the GPU
    gpu_rows = [[BUSY_UUID, "100", "2"]]
    monkeypatch.setattr(gpu_ghost_cure, "_smi", _fake_smi(gpu_rows, [[BUSY_UUID]]))

    # Act
    ghosts = gpu_ghost_cure.detect_ghosts()

    # Assert
    assert ghosts == []


def test_detect_ghosts_ignores_gpu_with_memory_in_use(monkeypatch):
    # Arrange: 100% util but real memory allocated (a working GPU, not a ghost)
    gpu_rows = [[BUSY_UUID, "100", "40321"]]
    monkeypatch.setattr(gpu_ghost_cure, "_smi", _fake_smi(gpu_rows, []))

    # Act
    ghosts = gpu_ghost_cure.detect_ghosts()

    # Assert
    assert ghosts == []


def test_spawn_cure_reports_failure_when_child_exits_nonzero(monkeypatch):
    # Arrange: the cure child process fails
    monkeypatch.setattr(
        gpu_ghost_cure.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="cuInit failed"),
    )

    # Act
    cured = gpu_ghost_cure.spawn_cure(GHOST_UUID)

    # Assert
    assert cured is False


def test_spawn_cure_rechecks_ghost_signature_after_child_success(monkeypatch):
    # Arrange: child succeeds, re-check no longer sees the ghost
    captured = {}

    def fake_run(cmd, env=None, **kwargs):
        captured["uuid"] = env["GHOST_CURE_UUID"]
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(gpu_ghost_cure.subprocess, "run", fake_run)
    monkeypatch.setattr(gpu_ghost_cure.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(gpu_ghost_cure, "detect_ghosts", lambda: [])

    # Act
    cured = gpu_ghost_cure.spawn_cure(GHOST_UUID)

    # Assert
    assert cured is True
    assert captured["uuid"] == GHOST_UUID


def _drive_watchdog(monkeypatch, tick_results, ticks_wanted):
    # run ghost_watchdog with interval=0, feeding detect_ghosts one canned result per
    # tick (last result repeats), until ticks_wanted samples were consumed;
    # returns cure calls as (uuid, tick_number)
    ticks = []
    cures = []

    def fake_detect():
        result = tick_results[min(len(ticks), len(tick_results) - 1)]
        ticks.append(result)
        return result

    def fake_cure(uuid):
        cures.append((uuid, len(ticks)))
        return True

    monkeypatch.setattr(gpus_utility.gpu_ghost_cure, "detect_ghosts", fake_detect)
    monkeypatch.setattr(gpus_utility.gpu_ghost_cure, "spawn_cure", fake_cure)

    async def run():
        task = asyncio.create_task(gpus_utility.ghost_watchdog(interval=0))
        while len(ticks) < ticks_wanted:
            await asyncio.sleep(0.01)
        task.cancel()

    asyncio.run(run())
    return cures


def test_ghost_watchdog_cures_only_after_sustained_signature(monkeypatch):
    # Arrange: the ghost signature holds for 3 consecutive samples, then the cure clears it
    tick_results = [[GHOST_UUID], [GHOST_UUID], [GHOST_UUID], []]

    # Act
    cures = _drive_watchdog(monkeypatch, tick_results, ticks_wanted=6)

    # Assert: cure fired exactly once, on the third consecutive ghost sample
    assert cures == [(GHOST_UUID, gpus_utility.GHOST_CONSECUTIVE_SAMPLES)]


def test_ghost_watchdog_resets_streak_when_signature_clears(monkeypatch):
    # Arrange: the signature never holds for 3 consecutive samples
    tick_results = [[GHOST_UUID], [GHOST_UUID], [], [GHOST_UUID], [GHOST_UUID], []]

    # Act
    cures = _drive_watchdog(monkeypatch, tick_results, ticks_wanted=6)

    # Assert
    assert cures == []
