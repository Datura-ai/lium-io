"""DAH-6353: a memory-modded model is paid only after a sealed memory proof on every card."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from helpers import FakeRedis, build_services, build_state
from neurons.validators.src.services import matrix_validation_service as mvs
from neurons.validators.src.services.matrix_validation_service import AllCardsProbe, ValidationResult
from neurons.validators.src.services.task.checks import capability
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.messages import CapabilityMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

RTX_4090_48GB = "NVIDIA GeForce RTX 4090 48GB"
RTX_4090 = "NVIDIA GeForce RTX 4090"


def _specs(model: str, count: int = 2) -> dict:
    return {
        "gpu": {
            "count": count,
            "details": [{"name": model, "uuid": f"GPU-{i}", "capacity": 49140} for i in range(count)],
        }
    }


def _probe(passed: bool = True, sealed: list[bool] | None = None) -> AllCardsProbe:
    sealed = [True, True] if sealed is None else sealed
    return AllCardsProbe(
        passed=passed,
        failure_reason="",
        elapsed_seconds=1.0,
        over_wall_clock=False,
        per_card=[{"card_index": i, "ok": passed, "seal_ok": seal} for i, seal in enumerate(sealed)],
    )


@pytest.mark.parametrize(
    ("probe", "proven"),
    [
        (_probe(), True),
        (_probe(sealed=[True, False]), False),
        (_probe(passed=False), False),
        (_probe(sealed=[]), False),
    ],
)
def test_the_probe_proves_memory_only_with_a_sealed_pass_on_every_card(probe, proven):
    assert probe.proves_memory_of_every_card is proven


@pytest.fixture
def svc(monkeypatch):
    wrapper = MagicMock(name="DMCompVerifyWrapper")
    wrapper.DMCompVerify_new.return_value = "ptr"
    wrapper.getCipherText.return_value = "deadbeef"
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *a, **k: wrapper)
    return mvs.ValidationService()


def _ssh():
    async def run(cmd, *args, **kwargs):
        return SimpleNamespace(exit_status=0, stdout="RESULT_JSON: {}", stderr="")

    return SimpleNamespace(run=run)


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "probed"), [(RTX_4090_48GB, True), (RTX_4090, False)])
async def test_the_all_card_probe_runs_on_a_single_modded_card(svc, model, probed):
    probe = await svc.probe_all_claimed_cards(
        _ssh(), SimpleNamespace(root_dir="/root/app", python_path="/usr/bin/python"), {}, _specs(model, 1)
    )

    assert (probe is not None) is probed


@pytest.mark.asyncio
@pytest.mark.parametrize(("enabled", "proven"), [(True, True), (False, False)])
async def test_the_result_says_whether_every_card_memory_is_proven(svc, monkeypatch, enabled, proven):
    monkeypatch.setattr(mvs.settings, "MATMUL_ALLCARDS_CHECK_ENABLED", enabled, raising=False)
    monkeypatch.setattr(svc, "probe_all_claimed_cards", AsyncMock(return_value=_probe()))

    result = await svc.validate_gpu_model_and_process_job(
        ssh_client=_ssh(),
        executor_info=SimpleNamespace(root_dir="/root/app", python_path="/usr/bin/python"),
        default_extra={},
        machine_spec=_specs(RTX_4090_48GB),
    )

    assert result.memory_proven_on_every_card is proven


class StubValidation:
    def __init__(self, result: ValidationResult):
        self.result = result

    async def validate_gpu_model_and_process_job(self, **_):
        return self.result


def _ok(proven: bool) -> ValidationResult:
    return ValidationResult(
        success=True, metrics={"cc": "8.9", "sm_count": 128}, memory_proven_on_every_card=proven
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "proven", "passed"),
    [
        (RTX_4090_48GB, False, False),
        (RTX_4090_48GB, True, True),
        (RTX_4090, False, True),  # a factory card is unchanged
    ],
)
async def test_a_modded_card_passes_only_with_its_memory_proven(context_factory, model, proven, passed):
    redis = FakeRedis()
    ctx = context_factory(
        services=build_services(validation=StubValidation(_ok(proven)), redis=redis),
        state=build_state(specs=_specs(model)),
    )

    result = await CapabilityCheck().run(ctx)

    assert result.passed is passed
    if not passed:
        assert result.event.reason_code == Msg.MODDED_MEMORY_UNPROVEN.reason
    assert bool(redis.store.get(capability.modded_memory_proof_key(ctx))) is (model == RTX_4090_48GB and proven)


def _filler_ctx(context_factory, model: str, redis: FakeRedis):
    state = build_state(
        specs=_specs(model),
        rented_data=RentedExecutorsResponse(
            executors={}, banned_guids=[], filler_containers_by_executor={"executor-123": "filler_active"}
        ),
    )
    return context_factory(services=build_services(validation=StubValidation(_ok(False)), redis=redis), state=state)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "proof_cards", "passed"),
    [
        (RTX_4090_48GB, None, False),  # never proven
        (RTX_4090_48GB, ["GPU-0"], False),  # proven on other cards
        (RTX_4090_48GB, ["GPU-0", "GPU-1"], True),
        (RTX_4090, None, True),  # a factory card is unchanged
    ],
)
async def test_a_filler_node_keeps_a_modded_model_only_with_a_stored_proof(
    context_factory, monkeypatch, model, proof_cards, passed
):
    monkeypatch.setattr(capability.settings, "FILLER_MATMUL_SAMPLE_EVERY_N_CYCLES", 0)
    redis = FakeRedis()
    ctx = _filler_ctx(context_factory, model, redis)
    if proof_cards is not None:
        redis.store[capability.modded_memory_proof_key(ctx)] = json.dumps(proof_cards)

    result = await CapabilityCheck().run(ctx)

    assert result.passed is passed


@pytest.mark.asyncio
async def test_a_redis_outage_never_proves_a_modded_card(context_factory, monkeypatch):
    monkeypatch.setattr(capability.settings, "FILLER_MATMUL_SAMPLE_EVERY_N_CYCLES", 0)
    ctx = _filler_ctx(context_factory, RTX_4090_48GB, FakeRedis(failing=True))

    result = await CapabilityCheck().run(ctx)

    assert result.passed is False
