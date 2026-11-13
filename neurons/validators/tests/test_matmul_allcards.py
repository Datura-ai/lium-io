"""DAH-2671 item 3 — full-matmul-all-cards work-proof.

RED→GREEN: before this change the work-proof ran ONE process with no device selection, so cards
1..N-1 were never touched and a 1-GPU host advertising 8 passed. Now every claimed card gets its
own challenge, pinned with CUDA_VISIBLE_DEVICES=<card UUID>, timed in aggregate on the validator:
  - a 1-real-card host under an 8-GPU claim fails device selection → fails under enforcement;
  - an honest 8-card host passes;
  - an aggregate wall-clock over threshold fails under enforcement, only emits under shadow;
  - a single-card (legacy) executor is never judged by the all-cards path (probe returns None).
"""
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
    # each card is pinned by its UUID (GPU-<i> in _spec), or by its index when it has none
    pin = cmd.split("CUDA_VISIBLE_DEVICES=")[1].split(" ")[0]
    return int(pin.removeprefix("GPU-"))


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
    ("model", "capacity", "budget"),
    [
        ("NVIDIA H100 80GB HBM3", 81920, 73728),
        # memory-modded cards arrive under the base card's raw name; each card is sized to the modded memory
        ("NVIDIA GeForce RTX 4090", 49140, 44237),
        ("NVIDIA GeForce RTX 4080", 32760, 29491),
        ("NVIDIA GeForce RTX 4080 SUPER", 32760, 29491),
    ],
)
@pytest.mark.asyncio
async def test_honest_eight_cards_probe_passes(svc, monkeypatch, model, capacity, budget):
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
    assert [call.args[1] for call in calls.call_args_list] == [budget] * 8




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
