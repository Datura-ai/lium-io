"""DAH-2671 item 3 — full-matmul-all-cards work-proof.

RED→GREEN: before this change the work-proof ran ONE process with no device selection, so cards
1..N-1 were never touched and a 1-GPU host advertising 8 passed. Now every claimed card gets its
own challenge, pinned with CUDA_VISIBLE_DEVICES=<index>, timed in aggregate on the validator:
  - a 1-real-card host under an 8-GPU claim fails device selection → fails under enforcement;
  - an honest 8-card host passes;
  - an aggregate wall-clock over threshold fails under enforcement, only emits under shadow;
  - a single-card (legacy) executor is never judged by the all-cards path (probe returns None).
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import services.matrix_validation_service as mvs
from services.gpu_spec_table import GPU_VRAM_SIZES_MB, VRAM_FLOOR_RATIO, matmul_probe_vram_mb


@pytest.fixture
def svc(monkeypatch):
    """Real ValidationService with a fully mocked native wrapper (no .so on the test host)."""
    wrapper = MagicMock(name="DMCompVerifyWrapper")
    wrapper.DMCompVerify_new.return_value = "ptr"
    wrapper.getCipherText.return_value = "deadbeef"
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *a, **k: wrapper)
    return mvs.ValidationService()


def _executor():
    return SimpleNamespace(root_dir="/root/app", python_path="/usr/bin/python")


def _spec(count, model="NVIDIA H100 80GB HBM3"):
    return {
        "gpu": {
            "count": count,
            "details": [
                {"name": model, "uuid": f"GPU-{i}", "capacity": 81920} for i in range(count)
            ],
        }
    }


def _card_index(cmd: str) -> int:
    return int(cmd.split("CUDA_VISIBLE_DEVICES=")[1].split(" ")[0])


def _ssh(real_cards: int):
    # a host with `real_cards` genuine GPUs: indices past that fail device selection (invalid ordinal)
    async def run(cmd, *args, **kwargs):
        if "CUDA_VISIBLE_DEVICES=" in cmd:
            if _card_index(cmd) < real_cards:
                return SimpleNamespace(exit_status=0, stdout="RESULT_JSON: {}", stderr="")
            return SimpleNamespace(exit_status=1, stdout="", stderr="invalid device ordinal")
        # single-card fallback path (no device prefix) — benign output.
        return SimpleNamespace(exit_status=0, stdout="UUID: x", stderr="")

    return SimpleNamespace(run=run)


def _flags(monkeypatch, *, enforce):
    monkeypatch.setattr(mvs.settings, "MATMUL_ALLCARDS_CHECK_ENABLED", True, raising=False)
    monkeypatch.setattr(
        mvs.settings, "MATMUL_ALLCARDS_ENFORCEMENT_ENABLED", enforce, raising=False
    )


@pytest.mark.asyncio
async def test_spoof_one_card_under_eight_claim_fails_under_enforcement(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)
    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=_ssh(real_cards=1),
        executor_info=_executor(),
        default_extra={},
        machine_spec=_spec(8),
    )
    assert result.success is False
    assert result.error_message.startswith("All-cards GPU work-proof failed")


@pytest.mark.asyncio
async def test_honest_eight_cards_probe_passes(svc):
    probe = await svc._probe_all_claimed_cards(_ssh(real_cards=8), _executor(), {}, _spec(8))
    assert probe is not None
    assert probe.passed is True
    assert len(probe.per_card) == 8
    assert all(card["ok"] for card in probe.per_card)


@pytest.mark.asyncio
async def test_wall_clock_over_threshold_fails_under_enforcement(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)
    # -1s threshold: any real elapsed trips it even though every card ran fine (models serialisation).
    monkeypatch.setattr(mvs, "MATMUL_ALLCARDS_WALL_CLOCK_SECONDS", -1, raising=False)
    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=_ssh(real_cards=8),
        executor_info=_executor(),
        default_extra={},
        machine_spec=_spec(8),
    )
    assert result.success is False
    assert "wall-clock" in result.error_message


@pytest.mark.asyncio
async def test_wall_clock_over_threshold_only_emits_under_shadow(svc, monkeypatch, caplog):
    _flags(monkeypatch, enforce=False)
    monkeypatch.setattr(mvs, "MATMUL_ALLCARDS_WALL_CLOCK_SECONDS", -1, raising=False)
    with caplog.at_level("WARNING"):
        result = await svc.validate_gpu_model_and_process_job(
            ssh_client=_ssh(real_cards=8),
            executor_info=_executor(),
            default_extra={},
            machine_spec=_spec(8),
        )
    # shadow: the probe never turns into a returned all-cards failure.
    assert not result.error_message.startswith("All-cards GPU work-proof failed")
    # but the observation is emitted.
    assert any("All-cards GPU work-proof" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_legacy_single_card_is_not_judged_by_all_cards(svc):
    # gpu_count <= 1 → probe returns None, so the executor is judged only by the unchanged
    # single-card path and a not-yet-upgraded / single-GPU fleet is never zeroed by this check.
    probe = await svc._probe_all_claimed_cards(_ssh(real_cards=1), _executor(), {}, _spec(1))
    assert probe is None


@pytest.mark.asyncio
async def test_unknown_model_skips_probe(svc):
    # a model with no registry VRAM size cannot be sized safely → probe returns None (fail-open).
    probe = await svc._probe_all_claimed_cards(
        _ssh(real_cards=8), _executor(), {}, _spec(8, model="NVIDIA MADE-UP 9000")
    )
    assert probe is None


@pytest.mark.asyncio
async def test_native_cipher_failure_skips_probe(svc, monkeypatch):
    # Our own .so failing must not read as the host failing: an empty cipher would go out as a bare
    # `--cipher_text ` and every card would come back ok=False. Skip the probe instead.
    monkeypatch.setattr(
        svc, "_generate_card_cipher", lambda machine_info, params: "", raising=False
    )
    probe = await svc._probe_all_claimed_cards(_ssh(real_cards=8), _executor(), {}, _spec(8))
    assert probe is None


def test_matmul_probe_vram_mb_floors_below_observed():
    # Finding 1: size each card's challenge from the 0.90 floor, NOT the raw nominal, so an honest
    # B200/L40S is never asked to allocate past its usable (NVML-reported) VRAM and OOM.
    b200 = matmul_probe_vram_mb("NVIDIA B200")
    assert b200 == round(196608 * VRAM_FLOOR_RATIO)
    assert b200 < 183359                                   # below observed usable total (registry note)
    assert b200 < GPU_VRAM_SIZES_MB["NVIDIA B200"][0]      # strictly below the raw nominal
    # multi-variant: smallest nominal, then floored
    assert matmul_probe_vram_mb("NVIDIA GeForce RTX 3060") == round(8192 * VRAM_FLOOR_RATIO)
    assert matmul_probe_vram_mb("NVIDIA MADE-UP 9000") is None


@pytest.mark.asyncio
async def test_card_fan_out_stays_under_sshd_max_sessions(svc):
    # Finding: one SSH channel per card on a single connection; OpenSSH defaults to MaxSessions 10,
    # so a 14-card host must not open 14 at once or the tail channels read as failed work-proofs.
    in_flight = 0
    peak = 0

    async def run(cmd, *a, **k):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return SimpleNamespace(exit_status=0, stdout="RESULT_JSON: {}", stderr="")

    probe = await svc._probe_all_claimed_cards(SimpleNamespace(run=run), _executor(), {}, _spec(14))
    assert probe is not None and probe.passed is True
    assert peak <= mvs.MATMUL_ALLCARDS_MAX_CONCURRENT_CARDS < 10


@pytest.mark.asyncio
async def test_timeout_sweeps_orphan_matmul(svc, monkeypatch):
    # Finding 2: a timed-out per-card run must trigger _kill_remote_matmul, or the orphaned remote
    # process keeps holding VRAM into the fatal single-card matmul that runs next in the same call.
    _flags(monkeypatch, enforce=False)
    killed = AsyncMock()
    monkeypatch.setattr(svc, "_kill_remote_matmul", killed)

    async def run(cmd, *a, **k):
        if "CUDA_VISIBLE_DEVICES=0" in cmd:
            raise TimeoutError
        return SimpleNamespace(exit_status=0, stdout="RESULT_JSON: {}", stderr="")

    probe = await svc._probe_all_claimed_cards(SimpleNamespace(run=run), _executor(), {}, _spec(2))
    assert probe is not None and probe.passed is False
    killed.assert_awaited_once()


@pytest.mark.asyncio
async def test_ssh_error_sweeps_orphan_matmul(svc, monkeypatch):
    # A channel that dies mid-run leaves the remote matmul untouched, exactly like a timeout — and
    # the fatal single-card matmul that follows lands on device 0 and OOMs against it.
    _flags(monkeypatch, enforce=False)
    killed = AsyncMock()
    monkeypatch.setattr(svc, "_kill_remote_matmul", killed)

    async def run(cmd, *a, **k):
        if "CUDA_VISIBLE_DEVICES=0" in cmd:
            raise OSError("channel closed")
        return SimpleNamespace(exit_status=0, stdout="RESULT_JSON: {}", stderr="")

    probe = await svc._probe_all_claimed_cards(SimpleNamespace(run=run), _executor(), {}, _spec(2))
    assert probe is not None and probe.passed is False
    killed.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_plain_card_failure_does_not_sweep(svc, monkeypatch):
    # A card that answered (non-zero exit) has no process left to kill; sweeping there would pkill
    # the fatal single-card matmul of a concurrent validator run on the same host for nothing.
    _flags(monkeypatch, enforce=False)
    killed = AsyncMock()
    monkeypatch.setattr(svc, "_kill_remote_matmul", killed)

    async def run(cmd, *a, **k):
        if "CUDA_VISIBLE_DEVICES=0" in cmd:
            return SimpleNamespace(exit_status=1, stdout="", stderr="invalid device ordinal")
        return SimpleNamespace(exit_status=0, stdout="RESULT_JSON: {}", stderr="")

    probe = await svc._probe_all_claimed_cards(SimpleNamespace(run=run), _executor(), {}, _spec(2))
    assert probe is not None and probe.passed is False
    killed.assert_not_awaited()
