"""DAH-2671 item 3 — full-matmul-all-cards work-proof.

RED→GREEN: before this change the work-proof ran ONE process with no device selection, so cards
1..N-1 were never touched and a 1-GPU host advertising 8 passed. Now every claimed card gets its
own challenge, pinned with CUDA_VISIBLE_DEVICES=<card UUID>, timed in aggregate on the validator:
  - a 1-real-card host under an 8-GPU claim fails device selection → fails under enforcement;
  - a claimed card CUDA cannot see exits 0 with `UUID:  None` → that card fails (no exit-status pass);
  - an honest 8-card host passes;
  - an aggregate wall-clock over threshold fails under enforcement, only emits under shadow;
  - a single-card (legacy) executor is never judged by the all-cards path (probe returns None);
  - a multi-card claim the probe cannot run (unsized model, no details, our cipher) fails closed.
"""
import re
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import services.matrix_validation_service as mvs


@pytest.fixture
def svc(monkeypatch):
    """Real ValidationService with a fully mocked native wrapper (no .so on the test host)."""
    wrapper = MagicMock(name="DMCompVerifyWrapper")
    wrapper.DMCompVerify_new.return_value = "ptr"
    wrapper.getCipherText.return_value = "deadbeef"
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *a, **k: wrapper)
    service = mvs.ValidationService()
    real_evaluate = service.evaluate_matmul_output

    def evaluate(challenge, *, stdout, stderr=""):
        # a real card's right answer to its own challenge; anything else is judged for real
        if stdout == ANSWER:
            return mvs.ValidationResult(success=True, sealed=True)
        return real_evaluate(challenge, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(service, "evaluate_matmul_output", evaluate)
    return service


ANSWER = "RIGHT_ANSWER"


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
    # each card is pinned by its UUID (GPU-<i> in _spec), or by its index when it has none
    pin = cmd.split("CUDA_VISIBLE_DEVICES=")[1].split(" ")[0]
    return int(pin.removeprefix("GPU-"))


def _ssh(real_cards: int):
    # a host with `real_cards` genuine GPUs: indices past that fail device selection (invalid ordinal)
    async def run(cmd, *args, **kwargs):
        if "CUDA_VISIBLE_DEVICES=" in cmd:
            if _card_index(cmd) < real_cards:
                return SimpleNamespace(exit_status=0, stdout=ANSWER, stderr="")
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
async def test_a_filler_sample_skips_the_all_cards_probe(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)
    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=_ssh(real_cards=1),
        executor_info=_executor(),
        default_extra={},
        machine_spec=_spec(8),
        probe_all_cards=False,
    )
    assert not result.error_message.startswith("All-cards GPU work-proof failed")


@pytest.mark.parametrize(
    ("model", "capacity", "challenge_vram_mb"),
    [
        ("NVIDIA H100 80GB HBM3", 81920, 73728),
        # memory-modded cards arrive under the base card's raw name; each card is sized to the modded memory
        ("NVIDIA GeForce RTX 4090", 49140, 44237),
        ("NVIDIA GeForce RTX 4080", 32760, 29491),
        ("NVIDIA GeForce RTX 4080 SUPER", 32760, 29491),
    ],
)
@pytest.mark.asyncio
async def test_honest_eight_cards_probe_passes(svc, monkeypatch, model, capacity, challenge_vram_mb):
    spec = _spec(8, model)
    for detail in spec["gpu"]["details"]:
        detail["capacity"] = capacity
    calls = MagicMock(wraps=svc._card_challenge)
    monkeypatch.setattr(svc, "_card_challenge", calls)

    probe = await svc.probe_all_claimed_cards(_ssh(real_cards=8), _executor(), {}, spec)

    assert probe is not None
    assert probe.passed is True
    assert len(probe.per_card) == 8
    assert all(card["ok"] for card in probe.per_card)
    assert [call.args[1] for call in calls.call_args_list] == [challenge_vram_mb] * 8




@pytest.mark.asyncio
async def test_each_card_answers_a_challenge_keyed_to_it_alone_and_its_sealed_metrics_come_back(
    svc, monkeypatch
):
    # DAH-6241: every card's own cc and SM count, read from its own sealed answer
    _flags(monkeypatch, enforce=False)
    keyed_to: list[str] = []
    svc.wrapper.generateChallenge.side_effect = lambda ptr, seed, machine_info, uuid: keyed_to.append(
        machine_info
    )

    def sealed_answer(challenge, *, stdout, stderr=""):
        card = challenge.log_extra.get("card_uuid")  # None: the capability run itself
        sm_count = 108 if card == "GPU-3" else 132
        return mvs.ValidationResult(
            success=True, sealed=True, metrics={"cc": "9.0", "sm_count": sm_count, "card": card}
        )

    monkeypatch.setattr(svc, "evaluate_matmul_output", sealed_answer)

    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=_ssh(real_cards=8),
        executor_info=_executor(),
        default_extra={},
        machine_spec=_spec(8),
    )

    assert sorted(result.card_metrics) == list(range(8))
    assert result.card_metrics[3]["sm_count"] == 108
    assert all(result.card_metrics[i]["card"] == f"GPU-{i}" for i in range(8))
    per_card_keys = [json.loads(info) for info in keyed_to if json.loads(info)["gpu_count"] == 1]
    assert [key["uuids"] for key in per_card_keys] == [f"GPU-{i}" for i in range(8)]


@pytest.mark.asyncio
async def test_a_card_whose_answer_is_not_sealed_gives_no_metrics_and_still_passes_the_work_proof(
    svc, monkeypatch
):
    _flags(monkeypatch, enforce=True)
    monkeypatch.setattr(
        svc,
        "evaluate_matmul_output",
        lambda challenge, *, stdout, stderr="": mvs.ValidationResult(
            success=True, sealed=False, metrics={"cc": "8.0", "sm_count": 108}
        ),
    )

    probe = await svc.probe_all_claimed_cards(_ssh(real_cards=8), _executor(), {}, _spec(8))

    assert probe.passed is True
    assert all(card["seal_ok"] is False and "metrics" not in card for card in probe.per_card)


def _ssh_cuda_blind(real_cards: int):
    # a run pinned to a card CUDA cannot see still exits 0 and prints `UUID:  None`
    async def run(cmd, *args, **kwargs):
        if "CUDA_VISIBLE_DEVICES=" in cmd and _card_index(cmd) < real_cards:
            return SimpleNamespace(exit_status=0, stdout=ANSWER, stderr="")
        return SimpleNamespace(
            exit_status=0, stdout="UUID:  None", stderr="CUDA error: no CUDA-capable device is detected"
        )

    return SimpleNamespace(run=run)


@pytest.mark.asyncio
async def test_a_claimed_card_cuda_cannot_see_fails_even_though_its_run_exits_zero(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)

    probe = await svc.probe_all_claimed_cards(_ssh_cuda_blind(real_cards=1), _executor(), {}, _spec(8))

    assert probe.passed is False
    assert probe.failure_reason == "7/8 card(s) failed device work-proof"
    assert [c["reason"] for c in probe.per_card] == [""] + ["no_answer"] * 7
    # card 0 answered, so the others' silence is theirs: no live-workload waiver (lio#2305 review)
    assert probe.no_card_answered is False

    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=_ssh_cuda_blind(real_cards=1), executor_info=_executor(), default_extra={}, machine_spec=_spec(8)
    )
    assert result.success is False
    assert result.error_message.startswith("All-cards GPU work-proof failed")
    assert result.all_cards_no_answer is False


@pytest.mark.asyncio
async def test_a_wrong_answer_on_one_card_is_never_a_no_answer(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)

    async def run(cmd, *args, **kwargs):
        if _card_index(cmd) == 5:
            return SimpleNamespace(exit_status=0, stdout="UUID: someone-elses", stderr="")
        return SimpleNamespace(exit_status=0, stdout=ANSWER, stderr="")

    probe = await svc.probe_all_claimed_cards(SimpleNamespace(run=run), _executor(), {}, _spec(8))

    assert probe.passed is False
    assert probe.per_card[5]["reason"] == "wrong_answer"
    assert probe.no_card_answered is False


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (_spec(8, model="NVIDIA Imaginary 9000"), "no work-proof size"),
        ({"gpu": {"count": 8, "details": []}}, "no card details"),
    ],
)
@pytest.mark.asyncio
async def test_a_multi_card_claim_the_probe_cannot_run_fails_closed(svc, monkeypatch, spec, reason):
    _flags(monkeypatch, enforce=True)
    for detail in spec["gpu"]["details"]:
        detail["capacity"] = 1

    probe = await svc.probe_all_claimed_cards(_ssh(real_cards=8), _executor(), {}, spec)

    assert probe is not None and probe.passed is False
    assert reason in probe.failure_reason
    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=_ssh(real_cards=8), executor_info=_executor(), default_extra={}, machine_spec=spec
    )
    assert result.error_message.startswith("All-cards GPU work-proof failed: not probed")


@pytest.mark.asyncio
async def test_our_own_cipher_failure_fails_the_probe_rather_than_skipping_it(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)
    svc.wrapper.getCipherText.return_value = ""

    probe = await svc.probe_all_claimed_cards(_ssh(real_cards=8), _executor(), {}, _spec(8))

    assert probe.passed is False
    assert "cipher" in probe.failure_reason


@pytest.mark.asyncio
async def test_a_single_card_host_is_proven_by_the_single_card_run(svc):
    assert await svc.probe_all_claimed_cards(_ssh(real_cards=1), _executor(), {}, _spec(1)) is None


@pytest.mark.parametrize(("cards", "threshold"), [(8, 90), (12, 180), (14, 180)])
@pytest.mark.parametrize("over", [False, True])
@pytest.mark.asyncio
async def test_the_wall_clock_is_the_flat_90_s_per_wave(svc, monkeypatch, cards, threshold, over):
    _flags(monkeypatch, enforce=True)
    # over: the first run and its retry are both over the limit
    elapsed = threshold + (1.0 if over else -1.0)
    ticks = iter([0.0, elapsed, 0.0, elapsed])
    monkeypatch.setattr(mvs.time, "perf_counter", lambda: next(ticks))

    probe = await svc.probe_all_claimed_cards(_ssh(real_cards=cards), _executor(), {}, _spec(cards))

    assert probe.over_wall_clock is over
    assert probe.passed is not over


def test_both_flags_default_off():
    # validators on the public image pull a release at once: they must not enforce before prod's shadow step
    from core.config import Settings

    fields = Settings.model_fields if hasattr(Settings, "model_fields") else Settings.__fields__
    assert fields["MATMUL_ALLCARDS_CHECK_ENABLED"].default is False
    assert fields["MATMUL_ALLCARDS_ENFORCEMENT_ENABLED"].default is False


def test_every_scored_model_has_a_work_proof_size():
    # an unsized model fails the probe closed, so a model added to the rates needs a size too
    from services.const import GPU_MODEL_RATES

    assert [m for m in GPU_MODEL_RATES if not mvs.matmul_probe_vram_mb(m)] == []


def test_only_an_all_cards_failure_with_no_card_answering_is_open_to_the_live_workload_waiver():
    from services.task.checks.capability import _probe_gave_no_answer

    failed = "All-cards GPU work-proof failed: 7/8 card(s) failed device work-proof"
    assert _probe_gave_no_answer(mvs.ValidationResult(success=False, error_message=failed, all_cards_no_answer=True))
    assert not _probe_gave_no_answer(mvs.ValidationResult(success=False, error_message=failed))


def _ssh_scripted(answers_by_card: dict[int, list[str]]):
    # each card in `answers_by_card` answers its pinned runs from its list in turn ("timeout" raises the
    # run's timeout); every other card answers right; kills are recorded
    calls = {"runs": {card: 0 for card in answers_by_card}, "kills": [], "matmuls": []}

    async def run(cmd, *args, **kwargs):
        if "CUDA_VISIBLE_DEVICES=" in cmd:
            calls["matmuls"].append(cmd)
        if "pkill -9" in cmd:   # the tagged kill (executor_payload.kill_runs)
            calls["kills"].append(cmd)
            return SimpleNamespace(exit_status=0, stdout="", stderr="")
        if "CUDA_VISIBLE_DEVICES=" in cmd and _card_index(cmd) in answers_by_card:
            card = _card_index(cmd)
            answer = answers_by_card[card][calls["runs"][card]]
            calls["runs"][card] += 1
            if answer == "timeout":
                raise TimeoutError()
            return SimpleNamespace(exit_status=0, stdout=answer, stderr="")
        if "CUDA_VISIBLE_DEVICES=" in cmd:
            return SimpleNamespace(exit_status=0, stdout=ANSWER, stderr="")
        return SimpleNamespace(exit_status=0, stdout="UUID: x", stderr="")

    return SimpleNamespace(run=run), calls


@pytest.mark.asyncio
async def test_a_timeout_retries_every_card_together_and_a_right_answer_then_passes(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)
    ssh, calls = _ssh_scripted({0: ["timeout", ANSWER], 7: [ANSWER, ANSWER]})

    probe = await svc.probe_all_claimed_cards(ssh, _executor(), {}, _spec(8))

    assert probe.passed is True
    assert all(card["ok"] and card["retried"] for card in probe.per_card)
    # the whole run again (card 7 answered right the first time and still ran again), after one sweep
    assert calls["runs"] == {0: 2, 7: 2}
    assert len(calls["kills"]) == 1
    # the retry runs under a fresh tag: the kill's cancel marker would stop every run started under the first one
    killed = re.search(r"matrix\.([0-9a-f]{16})-cancelled", calls["kills"][0]).group(1)
    tags = [re.search(r"matrix\.([0-9a-f]{16})-cancelled", cmd).group(1) for cmd in calls["matmuls"]]
    assert len(set(tags[:8])) == 1 and len(set(tags[8:])) == 1 and tags[0] == killed and tags[8] != killed


@pytest.mark.asyncio
async def test_one_card_that_times_out_again_while_the_others_answer_fails_with_no_waiver(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)
    ssh, calls = _ssh_scripted({0: ["timeout", "timeout"]})

    probe = await svc.probe_all_claimed_cards(ssh, _executor(), {}, _spec(8))

    assert probe.passed is False
    assert probe.per_card[0]["reason"] == "timeout_after_retry"
    assert probe.no_card_answered is False
    assert calls["runs"] == {0: 2}
    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=_ssh_scripted({0: ["timeout", "timeout"]})[0],
        executor_info=_executor(),
        default_extra={},
        machine_spec=_spec(8),
    )
    assert result.success is False
    assert result.error_message.startswith("All-cards GPU work-proof failed")
    assert result.all_cards_no_answer is False


@pytest.mark.asyncio
async def test_every_card_timing_out_twice_is_no_answer_left_to_the_live_workload_waiver(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)
    ssh, _ = _ssh_scripted({card: ["timeout", "timeout"] for card in range(8)})

    probe = await svc.probe_all_claimed_cards(ssh, _executor(), {}, _spec(8))

    assert probe.passed is False
    assert {card["reason"] for card in probe.per_card} == {"timeout_after_retry"}
    assert probe.no_card_answered is True


@pytest.mark.asyncio
async def test_a_timeout_then_a_wrong_answer_fails(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)
    ssh, _ = _ssh_scripted({0: ["timeout", "UUID: someone-elses"]})

    probe = await svc.probe_all_claimed_cards(ssh, _executor(), {}, _spec(8))

    assert probe.passed is False
    assert probe.per_card[0]["reason"] == "wrong_answer"
    assert probe.no_card_answered is False


@pytest.mark.asyncio
async def test_a_wrong_answer_anywhere_fails_with_no_retry_even_beside_a_timeout(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)
    ssh, calls = _ssh_scripted({0: ["UUID: someone-elses", ANSWER], 1: ["timeout", ANSWER]})

    probe = await svc.probe_all_claimed_cards(ssh, _executor(), {}, _spec(8))

    assert probe.passed is False
    assert probe.per_card[0]["reason"] == "wrong_answer"
    assert "retried" not in probe.per_card[0]
    assert calls["runs"] == {0: 1, 1: 1}


@pytest.mark.asyncio
async def test_the_retry_run_is_timed_on_its_own_against_the_same_wall_clock(svc, monkeypatch):
    _flags(monkeypatch, enforce=True)
    # first run 0 -> 130 s (timed out), retry 130 -> 221 s: 91 s, over the 90 s of one wave
    ticks = iter([0.0, 130.0, 130.0, 221.0])
    monkeypatch.setattr(mvs.time, "perf_counter", lambda: next(ticks))
    ssh, _ = _ssh_scripted({0: ["timeout", ANSWER]})

    probe = await svc.probe_all_claimed_cards(ssh, _executor(), {}, _spec(8))

    assert probe.over_wall_clock is True and probe.passed is False


@pytest.mark.parametrize(("retry_elapsed", "passed"), [(85.0, True), (91.0, False)])
@pytest.mark.asyncio
async def test_a_run_over_the_wall_clock_with_every_card_right_is_run_again_and_the_retry_decides(
    svc, monkeypatch, retry_elapsed, passed
):
    _flags(monkeypatch, enforce=True)
    ticks = iter([0.0, 92.0, 0.0, retry_elapsed])
    monkeypatch.setattr(mvs.time, "perf_counter", lambda: next(ticks))
    ssh, calls = _ssh_scripted({0: [ANSWER, ANSWER]})

    probe = await svc.probe_all_claimed_cards(ssh, _executor(), {}, _spec(8))

    assert calls["runs"] == {0: 2}
    assert all(card["ok"] and card["retried"] for card in probe.per_card)
    assert probe.passed is passed and probe.over_wall_clock is not passed


@pytest.mark.asyncio
async def test_a_cancelled_all_cards_run_kills_the_remote_matmuls_and_stays_cancelled(svc, monkeypatch):
    """Review (17 Nov): a cancel (cycle deadline, fast path) left every card's remote matmul running."""
    import asyncio

    _flags(monkeypatch, enforce=True)
    started, kills = asyncio.Event(), []

    async def run(cmd, *args, **kwargs):
        if "pkill -9" in cmd:   # the tagged kill (executor_payload.kill_runs)
            kills.append(cmd)
            return SimpleNamespace(exit_status=0, stdout="", stderr="")
        started.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(svc.probe_all_claimed_cards(SimpleNamespace(run=run), _executor(), {}, _spec(8)))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(kills) == 1
