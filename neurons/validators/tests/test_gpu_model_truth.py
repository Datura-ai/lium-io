"""DAH-6241: a node whose card is not the model it reports earns nothing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neurons.validators.src.services.matrix_validation_service import UUID_MISMATCH_ERROR_PREFIX, ValidationResult
from neurons.validators.src.services.task.availability import availability_errors
from neurons.validators.src.services.task.checks import capability
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.checks.gpu_model_verdict import GpuModelVerdictCheck
from neurons.validators.src.services.const import GPU_MODEL_RATES
from neurons.validators.src.services.task.checks.gpu_model_truth import (
    EXPECTED_CC_AND_SM,
    EXPECTED_CC_ONLY,
    gpu_model_mismatch,
)
from neurons.validators.src.services.task.messages import CapabilityMessages as Msg

from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from tests.helpers import FakeRedis, build_context_config, build_services, build_state
from tests.test_capability_check import DummyValidationService
from tests.test_filler_matmul_sample import SamplingValidationService


def _specs(model: str) -> dict:
    return {"gpu": {"count": 1, "details": [{"name": model}]}}


# prod, 12 Nov 2026: the three nodes of one hotkey that report a model their card is not
FAKED = [
    ("NVIDIA H100 80GB HBM3", "8.0", 108),  # an A100
    ("NVIDIA GeForce RTX 5070 Ti", "8.6", 48),  # an Ampere card
    ("NVIDIA RTX 6000 Ada Generation", "8.6", 84),  # an RTX A6000
]


@pytest.mark.parametrize("model", sorted(EXPECTED_CC_AND_SM))
def test_every_model_on_the_fleet_passes_with_its_own_card(model):
    # the table is what the sealed matmul read on the fleet's honest nodes of each name
    cc, sm_count = EXPECTED_CC_AND_SM[model]
    assert gpu_model_mismatch(_specs(model), {"cc": cc, "sm_count": sm_count}) is None


@pytest.mark.parametrize("model, cc, sm_count", FAKED)
def test_a_card_that_is_not_the_reported_model_fails(model, cc, sm_count):
    mismatch = gpu_model_mismatch(_specs(model), {"cc": cc, "sm_count": sm_count, "device": "x"})

    assert mismatch is not None
    assert mismatch.measured_compute_capability == cc
    assert mismatch.measured_sm_count == sm_count
    assert mismatch.measured_device_name == "x"
    assert (mismatch.expected_compute_capability, mismatch.expected_sm_count) == (
        EXPECTED_CC_AND_SM[model]
    )


@pytest.mark.parametrize(
    "model, cc, sm_count",
    [
        ("NVIDIA H100 PCIe", "9.0", 132),  # prod: an SXM H100 listed as the PCIe card
        ("NVIDIA GeForce RTX 4090", "12.0", 170),  # a newer, bigger card under an older name
    ],
)
def test_a_card_bigger_than_the_reported_model_passes(model, cc, sm_count):
    assert gpu_model_mismatch(_specs(model), {"cc": cc, "sm_count": sm_count}) is None


def test_a_newer_generation_with_fewer_sms_fails():
    # a 5070 (12.0, 48 SM) under a 4090 name: newer, but a smaller card
    assert gpu_model_mismatch(_specs("NVIDIA GeForce RTX 4090"), {"cc": "12.0", "sm_count": 48})


def test_the_card_judged_is_the_one_the_matmul_ran_on():
    specs = {"gpu": {"count": 2, "details": [{"name": "NVIDIA Some Future GPU"}, {"name": "NVIDIA H100 80GB HBM3"}]}}
    metrics = {"cc": "8.0", "sm_count": 108}

    assert gpu_model_mismatch(specs, metrics) is None
    assert gpu_model_mismatch(specs, metrics, card_index=1) is not None
    assert gpu_model_mismatch(specs, metrics, card_index=2) is None


def test_the_same_generation_with_fewer_sms_fails():
    # a cut-down card of the same generation is still not the reported model
    assert gpu_model_mismatch(_specs("NVIDIA GeForce RTX 4090"), {"cc": "8.9", "sm_count": 76})


@pytest.mark.parametrize(
    "specs, metrics",
    [
        (_specs("NVIDIA Some Future GPU"), {"cc": "8.0", "sm_count": 108}),
        (_specs("NVIDIA H100 80GB HBM3"), None),
        (_specs("NVIDIA H100 80GB HBM3"), {}),
        (_specs("NVIDIA H100 80GB HBM3"), {"cc": "8.0"}),
        (_specs("NVIDIA H100 80GB HBM3"), {"cc": "", "sm_count": 108}),
        (_specs("NVIDIA H100 80GB HBM3"), {"cc": "8.0", "sm_count": "108"}),
        (_specs("NVIDIA H100 80GB HBM3"), {"cc": "eight", "sm_count": 108}),
        ({"gpu": {"count": 1}}, {"cc": "8.0", "sm_count": 108}),
    ],
    ids=["unknown_model", "no_metrics", "empty_metrics", "no_sm", "empty_cc", "sm_text", "cc_text", "no_name"],
)
def test_an_unknown_model_or_a_run_without_both_numbers_never_fails(specs, metrics):
    assert gpu_model_mismatch(specs, metrics) is None


async def _run(context_factory, model: str, cc: str, sm_count: int, redis=None):
    service = DummyValidationService(success=True, metrics={"cc": cc, "sm_count": sm_count})
    ctx = context_factory(
        services=build_services(validation=service, redis=redis or FakeRedis()),
        config=build_context_config(),
        state=build_state(specs=_specs(model)),
    )
    return await CapabilityCheck().run(ctx)


@pytest.mark.asyncio
async def test_the_capability_check_fails_a_faked_model_with_its_own_reason(context_factory):
    result = await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108)

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_MODEL_MISMATCH.reason
    assert "gpu_metrics" not in str(result.updates)


@pytest.mark.asyncio
async def test_the_capability_check_passes_the_real_model(context_factory):
    result = await _run(context_factory, "NVIDIA H100 80GB HBM3", "9.0", 132)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFY_OK.reason


# --- a node that runs a filler skips the full matmul: its last verdict and the filler sample judge it

FAKE_H100 = ("NVIDIA H100 80GB HBM3", {"cc": "8.0", "sm_count": 108})
REAL_H100 = ("NVIDIA H100 80GB HBM3", {"cc": "9.0", "sm_count": 132})


NO_ROOM = ValidationResult(  # a busy card: the filler holds the VRAM
    success=False,
    returned_uuid="None",
    stderr="Failed to allocate d_A: out of memory",
    error_message=f"{UUID_MISMATCH_ERROR_PREFIX}: expected 'nonce', got 'None'",
)
TIMED_OUT = ValidationResult(success=False, timed_out=True, error_message="timed out")
WRONG_ANSWER = ValidationResult(success=False, error_message="unexpected stdout")


def _filler_ctx(context_factory, redis, metrics, *, success=True, specs=None, failure=NO_ROOM):
    service = SamplingValidationService(
        result=ValidationResult(success=True, metrics=metrics) if success else failure
    )
    state = build_state(
        specs=specs or _specs(FAKE_H100[0]),
        rented_data=RentedExecutorsResponse(
            executors={}, banned_guids=[], filler_containers_by_executor={"executor-123": "filler_active"}
        ),
    )
    ctx = context_factory(services=build_services(validation=service, redis=redis), state=state)
    return ctx, service


@pytest.fixture
def sample_never_due(monkeypatch):
    monkeypatch.setattr(capability, "_filler_matmul_sample_due", lambda ctx, cycle: False)


@pytest.mark.asyncio
async def test_a_faked_model_found_by_the_full_matmul_still_fails_once_a_filler_runs(
    context_factory, sample_never_due
):
    redis = FakeRedis()
    full = await _run(context_factory, *FAKE_H100[:1], "8.0", 108, redis=redis)
    assert full.event.reason_code == Msg.GPU_MODEL_MISMATCH.reason

    ctx, service = _filler_ctx(context_factory, redis, FAKE_H100[1])
    result = await CapabilityCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_MODEL_MISMATCH.reason
    assert service.budgets == []  # judged by the stored verdict, no sample out of turn
    assert redis.ttl[_verdict_key()] == capability.GPU_MODEL_VERDICT_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_filler_node_with_no_verdict_is_sampled_now_and_a_faked_model_fails(
    context_factory, sample_never_due
):
    redis = FakeRedis()
    ctx, service = _filler_ctx(context_factory, redis, FAKE_H100[1])

    result = await CapabilityCheck().run(ctx)

    assert len(service.budgets) == 1
    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_MODEL_MISMATCH.reason
    # the next filler cycle fails on the stored verdict without another sample
    ctx, service = _filler_ctx(context_factory, redis, REAL_H100[1])
    assert (await CapabilityCheck().run(ctx)).passed is False
    assert service.budgets == []


@pytest.mark.asyncio
async def test_a_filler_sample_of_the_real_model_clears_a_stored_mismatch(context_factory, monkeypatch):
    redis = FakeRedis()
    await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108, redis=redis)
    monkeypatch.setattr(capability, "_filler_matmul_sample_due", lambda ctx, cycle: True)

    ctx, _ = _filler_ctx(context_factory, redis, REAL_H100[1])
    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_SKIPPED.reason


@pytest.mark.asyncio
async def test_a_failed_sample_keeps_a_stored_mismatch(context_factory, monkeypatch):
    redis = FakeRedis()
    await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108, redis=redis)
    monkeypatch.setattr(capability, "_filler_matmul_sample_due", lambda ctx, cycle: True)

    ctx, _ = _filler_ctx(context_factory, redis, None, success=False)

    assert (await CapabilityCheck().run(ctx)).passed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [NO_ROOM, TIMED_OUT], ids=["no_memory", "timeout"])
async def test_a_first_sample_with_no_room_or_a_timeout_is_not_retried_every_cycle(
    context_factory, sample_never_due, failure
):
    redis = FakeRedis()
    ctx, service = _filler_ctx(context_factory, redis, None, success=False, failure=failure)
    assert (await CapabilityCheck().run(ctx)).passed is True
    assert redis.ttl[_verdict_key()] == capability.GPU_MODEL_NO_VERDICT_TTL_SECONDS

    ctx, service = _filler_ctx(context_factory, redis, None, success=False, failure=failure)
    assert (await CapabilityCheck().run(ctx)).passed is True
    assert service.budgets == []


def _lium_filler(ctx, owner="lium"):
    ctx.state.rented_data.all_filler_containers_by_executor = {"executor-123": ["filler_active"]}
    ctx.state.rented_data.default_job_owner_by_executor = {"executor-123": owner}
    ssh = AsyncMock()
    ssh.run.return_value = SimpleNamespace(exit_status=0, stdout="filler_active", stderr="")
    return ctx.model_copy(update={"ssh": ssh})


VALIDATOR_ERROR = ValidationResult(success=False, validator_error=True, error_message="Failed to execute SSH command: x")
WRONG_UUID = ValidationResult(
    success=False, returned_uuid="GPU-other", error_message=f"{UUID_MISMATCH_ERROR_PREFIX}: expected 'nonce', got 'GPU-other'"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "full_check, passed, verdict_ttl",
    [
        (ValidationResult(success=True, metrics=REAL_H100[1]), True, "week"),
        (WRONG_ANSWER, False, "week"),
        (WRONG_UUID, False, "week"),
        (NO_ROOM, True, "day"),
        (TIMED_OUT, True, "day"),
        (VALIDATOR_ERROR, True, "day"),
    ],
    ids=["answers", "wrong_answer", "wrong_uuid", "no_answer", "timeout", "validator_error"],
)
async def test_after_the_filler_stop_only_a_wrong_answer_fails_the_node(
    context_factory, sample_never_due, full_check, passed, verdict_ttl
):
    # DAH-6332: the cycle after a wrong sample stops Lium's filler and runs the full check once; no answer is no verdict
    redis = FakeRedis()
    ctx, _ = _filler_ctx(context_factory, redis, None, success=False, failure=WRONG_ANSWER)
    assert (await CapabilityCheck().run(_lium_filler(ctx))).passed is True
    assert _verdict_key() not in redis.ttl

    ctx, service = _filler_ctx(context_factory, redis, None, success=False, failure=full_check)
    result = await CapabilityCheck().run(_lium_filler(ctx))
    assert service.budgets == [None]  # the full check, not a sample
    assert result.event.what_we_saw["stopped_fillers_for_full_check"] == ["filler_active"]
    assert result.passed is passed
    assert (result.event.reason_code == Msg.FULL_CHECK_INCONCLUSIVE.reason) is (passed and verdict_ttl == "day")
    assert redis.ttl[_verdict_key()] == (
        capability.GPU_MODEL_VERDICT_TTL_SECONDS if verdict_ttl == "week" else capability.GPU_MODEL_NO_VERDICT_TTL_SECONDS
    )

    # the filler is back: the node is judged (or waits a day), so it is neither stopped nor sampled again;
    # a proven wrong answer stands
    ctx, service = _filler_ctx(context_factory, redis, None, success=False, failure=WRONG_ANSWER)
    result = await CapabilityCheck().run(_lium_filler(ctx))
    assert result.passed is passed
    assert service.budgets == []
    assert "stopped_fillers_for_full_check" not in result.event.what_we_saw


@pytest.mark.asyncio
async def test_a_full_check_that_raises_after_the_stop_is_inconclusive(context_factory, sample_never_due):
    redis = FakeRedis()
    ctx, service = _filler_ctx(context_factory, redis, None)
    await redis.set(capability._full_check_due_key(ctx), "0")
    service.raises = True

    result = await CapabilityCheck().run(_lium_filler(ctx))

    assert result.passed is True
    assert result.event.reason_code == Msg.FULL_CHECK_INCONCLUSIVE.reason
    assert redis.ttl[_verdict_key()] == capability.GPU_MODEL_NO_VERDICT_TTL_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_card, passed", [(True, False), (False, True)], ids=["wrong_answer", "no_answer"])
async def test_an_all_cards_failure_after_the_stop_fails_only_on_a_wrong_answer(
    context_factory, sample_never_due, wrong_card, passed
):
    failed = ValidationResult(
        success=False,
        error_message="All-cards GPU work-proof failed: 1/2 card(s) failed device work-proof",
        all_cards_wrong_answer=wrong_card,
    )
    redis = FakeRedis()
    ctx, _ = _filler_ctx(context_factory, redis, None, success=False, failure=failed)
    await redis.set(capability._full_check_due_key(ctx), "0")

    assert (await CapabilityCheck().run(_lium_filler(ctx))).passed is passed


@pytest.mark.asyncio
async def test_another_owners_filler_is_never_stopped_for_the_full_check(context_factory, sample_never_due):
    redis = FakeRedis()
    budgets = []
    for _ in range(2):
        ctx, service = _filler_ctx(context_factory, redis, None, success=False, failure=WRONG_ANSWER)
        ctx = _lium_filler(ctx, owner="miner")
        result = await CapabilityCheck().run(ctx)
        assert result.passed is True
        budgets.append(len(service.budgets))
        ctx.ssh.run.assert_not_awaited()

    # review: the one-day entry is stored as before, so the node is not sampled every cycle
    assert budgets == [1, 0]
    assert redis.ttl[_verdict_key()] == capability.GPU_MODEL_NO_VERDICT_TTL_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize("enforced", [True, False], ids=["enforced", "enforcement_off"])
@pytest.mark.parametrize("full_check", [REAL_H100[1], None], ids=["every_card_answers", "a_card_missing"])
async def test_a_judged_filler_node_gets_the_all_cards_proof_on_its_sample_turn(
    context_factory, monkeypatch, enforced, full_check
):
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_CHECK_ENABLED", enforced)
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_ENFORCEMENT_ENABLED", enforced)
    redis = FakeRedis()
    await _run(context_factory, *REAL_H100[:1], "9.0", 132, redis=redis)  # a stored passing verdict
    monkeypatch.setattr(capability, "_filler_matmul_sample_due", lambda ctx, cycle: True)

    ctx, service = _filler_ctx(context_factory, redis, full_check, success=full_check is not None, failure=WRONG_ANSWER)
    ctx = _lium_filler(ctx)
    result = await CapabilityCheck().run(ctx)

    if not enforced:
        # the proof would decide nothing: the filler keeps running and the one-card sample runs as before
        assert service.budgets == [capability.settings.FILLER_MATMUL_SAMPLE_VRAM_MB]
        ctx.ssh.run.assert_not_awaited()
        return
    assert service.budgets == [None]  # the full check, not a sample
    assert result.event.what_we_saw["stopped_fillers_for_full_check"] == ["filler_active"]
    # a failed proof is the node's own: the stopped filler the backend still lists waives nothing
    assert result.passed is (full_check is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize("card_back", [False, True], ids=["card_still_fails", "card_answers_again"])
async def test_a_failed_all_cards_proof_keeps_failing_the_node_once_the_filler_is_back(
    context_factory, monkeypatch, card_back
):
    # review: the failed proof left the earlier passing verdict standing, so the next filler turn passed the node
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_CHECK_ENABLED", True)
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_ENFORCEMENT_ENABLED", True)
    redis = FakeRedis()
    await _run(context_factory, *REAL_H100[:1], "9.0", 132, redis=redis)  # a stored passing verdict
    monkeypatch.setattr(capability, "_filler_matmul_sample_due", lambda ctx, cycle: True)
    ctx, _ = _filler_ctx(context_factory, redis, None, success=False, failure=WRONG_ANSWER)
    assert (await CapabilityCheck().run(_lium_filler(ctx))).passed is False

    # the next cycle is off the sample turn and the filler runs again
    monkeypatch.setattr(capability, "_filler_matmul_sample_due", lambda ctx, cycle: False)
    ctx, service = _filler_ctx(
        context_factory,
        redis,
        REAL_H100[1] if card_back else None,
        success=card_back,
        failure=WRONG_ANSWER,
    )
    result = await CapabilityCheck().run(_lium_filler(ctx))
    # DAH-6332: only a passing full check clears the stored failure; no filler sample re-runs the card meanwhile
    assert service.budgets == []
    assert result.passed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("card_refused, passed", [(True, False), (False, True)], ids=["card_no_answer", "card_timeout"])
async def test_a_card_that_does_not_answer_on_the_sample_turn_fails_the_node_and_is_stored(
    context_factory, monkeypatch, card_refused, passed
):
    # review of lio#2511: Lium's filler was stopped for the sample turn, so the cards are idle; a card that did not
    # answer is the node's own failure, stored so the node keeps failing once the filler is back. A card that only
    # timed out (a slow model) proves nothing either way.
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_CHECK_ENABLED", True)
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_ENFORCEMENT_ENABLED", True)
    redis = FakeRedis()
    await _run(context_factory, *REAL_H100[:1], "9.0", 132, redis=redis)  # a stored passing verdict
    monkeypatch.setattr(capability, "_filler_matmul_sample_due", lambda ctx, cycle: True)
    failed = ValidationResult(
        success=False,
        error_message="All-cards GPU work-proof failed: 1/2 card(s) failed device work-proof",
        all_cards_card_refused=card_refused,
    )
    ctx, _ = _filler_ctx(context_factory, redis, None, success=False, failure=failed)
    result = await CapabilityCheck().run(_lium_filler(ctx))
    assert result.event.what_we_saw["stopped_fillers_for_full_check"] == ["filler_active"]
    assert result.passed is passed
    assert _stored_kinds(redis) == ([] if passed else [(0, "full_check_failed")])

    # the next cycle is off the sample turn and the filler runs again
    monkeypatch.setattr(capability, "_filler_matmul_sample_due", lambda ctx, cycle: False)
    ctx, _ = _filler_ctx(context_factory, redis, REAL_H100[1])
    assert (await CapabilityCheck().run(_lium_filler(ctx))).passed is passed


@pytest.mark.asyncio
async def test_a_redis_outage_never_stops_the_filler_for_the_all_cards_proof(context_factory, monkeypatch):
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_CHECK_ENABLED", True)
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_ENFORCEMENT_ENABLED", True)
    monkeypatch.setattr(capability, "_filler_matmul_sample_due", lambda ctx, cycle: True)

    ctx, service = _filler_ctx(context_factory, FakeRedis(failing=True), REAL_H100[1])
    ctx = _lium_filler(ctx)
    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert "stopped_fillers_for_full_check" not in result.event.what_we_saw
    ctx.ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_judged_filler_node_off_its_sample_turn_keeps_its_filler(context_factory, monkeypatch, sample_never_due):
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_CHECK_ENABLED", True)
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_ENFORCEMENT_ENABLED", True)
    redis = FakeRedis()
    await _run(context_factory, *REAL_H100[:1], "9.0", 132, redis=redis)

    ctx, service = _filler_ctx(context_factory, redis, REAL_H100[1])
    ctx = _lium_filler(ctx)
    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert service.budgets == []
    ctx.ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_redis_outage_neither_fails_nor_samples_a_filler_node(context_factory, sample_never_due):
    ctx, service = _filler_ctx(context_factory, FakeRedis(failing=True), FAKE_H100[1])

    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert service.budgets == []


def _verdict_key() -> str:
    return capability.GPU_MODEL_VERDICT_KEY.format(miner_hotkey="miner-hotkey", executor_uuid="executor-123")


# --- every card, a rented node, and a model the node stops reporting


def _two_card_specs(model: str = "NVIDIA H100 80GB HBM3") -> dict:
    return {"gpu": {"count": 2, "details": [{"name": model, "uuid": "GPU-0"}, {"name": model, "uuid": "GPU-1"}]}}


@pytest.mark.asyncio
async def test_a_faked_card_after_card_0_fails_from_its_own_sealed_metrics(context_factory):
    redis = FakeRedis()
    service = DummyValidationService(success=True, metrics={"cc": "9.0", "sm_count": 132})
    real_validate = service.validate_gpu_model_and_process_job

    async def with_cards(**kwargs):
        result = await real_validate(**kwargs)
        result.card_metrics = {0: {"cc": "9.0", "sm_count": 132}, 1: {"cc": "8.0", "sm_count": 108}}
        return result

    service.validate_gpu_model_and_process_job = with_cards
    ctx = context_factory(
        services=build_services(validation=service, redis=redis),
        config=build_context_config(),
        state=build_state(specs=_two_card_specs()),
    )

    result = await CapabilityCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_MODEL_MISMATCH.reason
    assert [m["card_index"] for m in json.loads(redis.store[_verdict_key()])["mismatches"]] == [1]


def _rented(specs: dict, redis) -> object:
    state = build_state(
        specs=specs,
        rented_data=RentedExecutorsResponse(
            executors={
                "executor-123": RentedExecutor(
                    miner_hotkey="miner-hotkey",
                    executor_ip_address="1.2.3.4",
                    executor_ip_port="40022",
                    pods=[RentedPod(pod_id="pod-1", container_name="pod_renter")],
                )
            },
            banned_guids=[],
        ),
    )
    return state


async def _verdict_check(context_factory, redis, state, validation=None):
    ctx = context_factory(services=build_services(validation=validation, redis=redis), state=state)
    return await GpuModelVerdictCheck().run(ctx)


@pytest.mark.asyncio
async def test_a_rented_node_with_a_stored_mismatch_fails_before_the_rented_halt(context_factory):
    redis = FakeRedis()
    await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108, redis=redis)
    no_matmul = DummyValidationService(success=True, metrics=REAL_H100[1])

    result = await _verdict_check(
        context_factory, redis, _rented(_specs("NVIDIA H100 80GB HBM3"), redis), validation=no_matmul
    )

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_MODEL_MISMATCH.reason
    # Rustam 12 Nov: the renter's pod keeps running (not fatal), score 0, hidden from new rentals
    assert result.fatal is False
    assert result.updates == {"gpu_model_passed": False, "clear_verified_job_info": True}
    assert [e.reason_code for e in availability_errors([result.event])] == [Msg.GPU_MODEL_MISMATCH.reason]
    assert no_matmul.called_with is None  # nothing runs beside the renter's pod


@pytest.mark.parametrize("stored", ["none", "pass"])
@pytest.mark.asyncio
async def test_a_rented_node_with_no_mismatch_stored_is_untouched(context_factory, stored):
    redis = FakeRedis()
    if stored == "pass":
        await _run(context_factory, "NVIDIA H100 80GB HBM3", "9.0", 132, redis=redis)

    result = await _verdict_check(context_factory, redis, _rented(_specs("NVIDIA H100 80GB HBM3"), redis))

    assert result.passed is True
    assert result.event.reason_code == Msg.GPU_MODEL_VERDICT_OK.reason


@pytest.mark.asyncio
async def test_an_idle_node_is_left_to_the_capability_check(context_factory):
    # the matmul runs later in the same cycle and writes a fresh verdict
    redis = FakeRedis()
    await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108, redis=redis)

    result = await _verdict_check(context_factory, redis, build_state(specs=_specs("NVIDIA H100 80GB HBM3")))

    assert result.passed is True


@pytest.mark.asyncio
async def test_a_stored_mismatch_for_a_model_the_node_no_longer_reports_is_ignored(context_factory):
    redis = FakeRedis()
    await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108, redis=redis)

    result = await _verdict_check(context_factory, redis, _rented(_specs("NVIDIA A100 80GB PCIe"), redis))

    assert result.passed is True


@pytest.mark.asyncio
async def test_a_filler_node_that_now_reports_another_model_is_sampled_again(context_factory, sample_never_due):
    redis = FakeRedis()
    await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108, redis=redis)
    ctx, service = _filler_ctx(context_factory, redis, {"cc": "8.0", "sm_count": 108})
    ctx.state.specs["gpu"]["details"][0]["name"] = "NVIDIA A100 80GB PCIe"

    result = await CapabilityCheck().run(ctx)

    assert len(service.budgets) == 1  # no verdict for the new model: judged now
    assert result.passed is True  # an A100 is not in the table, so it is not judged


@pytest.mark.asyncio
async def test_a_verdict_stored_under_another_miner_does_not_judge_this_node(context_factory):
    redis = FakeRedis()
    other = DummyValidationService(success=True, metrics=FAKE_H100[1])
    ctx = context_factory(
        services=build_services(validation=other, redis=redis),
        state=build_state(specs=_specs(FAKE_H100[0])),
        miner_hotkey="other-miner",
    )
    assert (await CapabilityCheck().run(ctx)).passed is False

    result = await _verdict_check(context_factory, redis, _rented(_specs(FAKE_H100[0]), redis))

    assert result.passed is True


@pytest.mark.asyncio
async def test_a_filler_sample_passing_on_another_card_keeps_a_cards_stored_mismatch(
    context_factory, monkeypatch
):
    redis = FakeRedis()
    service = DummyValidationService(success=True, metrics=FAKE_H100[1])
    ctx = context_factory(
        services=build_services(validation=service, redis=redis),
        config=build_context_config(),
        state=build_state(specs=_two_card_specs()),
    )
    assert (await CapabilityCheck().run(ctx)).passed is False  # card 0 is the faked one
    monkeypatch.setattr(capability, "_filler_matmul_sample_due", lambda ctx, cycle: True)
    monkeypatch.setattr(capability, "_filler_sample_offset", lambda ctx: 0)
    monkeypatch.setattr(capability, "_filler_cycle_index", lambda ctx: 1)  # card 1's turn
    monkeypatch.setattr(capability.settings, "FILLER_MATMUL_SAMPLE_EVERY_N_CYCLES", 1)

    ctx, sampler = _filler_ctx(context_factory, redis, REAL_H100[1], specs=_two_card_specs())
    result = await CapabilityCheck().run(ctx)

    assert sampler.gpu_uuids == ["GPU-1"]
    assert result.passed is False
    assert result.event.what_we_saw["card_index"] == 0


@pytest.mark.asyncio
async def test_an_idle_faked_model_fails_fatally_and_is_hidden_from_new_rentals(context_factory):
    result = await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108)

    assert result.fatal is True
    assert result.updates == {"gpu_model_passed": False, "clear_verified_job_info": True}
    (entry,) = availability_errors([result.event])
    assert (entry.reach_source, entry.reach_target) == ("validator", "gpu")


def test_a_gpu_model_mismatch_is_the_nodes_own_fault_never_our_outage():
    from neurons.validators.src.services.task.availability import (
        NODE_GPU_FAULT_CODES,
        AvailabilityErrorCode,
        _is_the_nodes_own_fault,
    )

    assert AvailabilityErrorCode(Msg.GPU_MODEL_MISMATCH.reason) in NODE_GPU_FAULT_CODES
    assert _is_the_nodes_own_fault({"reason_code": Msg.GPU_MODEL_MISMATCH.reason}) is True


def test_an_nvml_alias_of_a_listed_model_is_judged():
    assert gpu_model_mismatch(_specs("Tesla H100 80GB HBM3"), {"cc": "8.0", "sm_count": 108}) is not None


def test_every_rated_model_is_judged_by_its_compute_capability():
    # unjudged until a node of the name shows its numbers (no source for its value)
    assert sorted(set(GPU_MODEL_RATES) - set(EXPECTED_CC_AND_SM) - set(EXPECTED_CC_ONLY)) == ["NVIDIA RTX 6000D"]
    assert sorted(set(EXPECTED_CC_ONLY) & set(EXPECTED_CC_AND_SM)) == []


class _PinnedCardService(DummyValidationService):
    """The idle matmul runs on card 0 only (all-cards off); a run pinned to a card UUID answers that
    card's own numbers."""

    def __init__(self, card_0: dict, pinned: dict[str, dict]):
        super().__init__(success=True, metrics=card_0)
        self.card_0, self.pinned, self.gpu_uuids = card_0, pinned, []

    async def validate_gpu_model_and_process_job(self, *, gpu_uuid=None, vram_budget_mb=None, probe_all_cards=True, **kwargs):
        if gpu_uuid:
            self.gpu_uuids.append(gpu_uuid)
        self.metrics = self.pinned[gpu_uuid] if gpu_uuid else self.card_0
        return await super().validate_gpu_model_and_process_job(**kwargs)


async def _idle_run(context_factory, redis, service):
    ctx = context_factory(
        services=build_services(validation=service, redis=redis),
        config=build_context_config(),
        state=build_state(specs=_two_card_specs()),
    )
    return await CapabilityCheck().run(ctx)


def _store_card_1_mismatch(redis) -> None:
    mismatch = gpu_model_mismatch(_two_card_specs(), FAKE_H100[1], 1)
    redis.store[_verdict_key()] = json.dumps({"mismatches": [mismatch.as_dict()]})


@pytest.mark.asyncio
async def test_a_stored_mismatch_on_card_1_is_measured_again_on_that_card_and_clears_when_real(context_factory):
    redis = FakeRedis()
    _store_card_1_mismatch(redis)
    service = _PinnedCardService(REAL_H100[1], {"GPU-1": REAL_H100[1]})

    result = await _idle_run(context_factory, redis, service)

    assert service.gpu_uuids == ["GPU-1"]
    assert result.passed is True
    assert json.loads(redis.store[_verdict_key()])["mismatches"] == []


@pytest.mark.asyncio
async def test_a_stored_mismatch_on_card_1_stands_when_that_card_is_still_faked(context_factory):
    redis = FakeRedis()
    _store_card_1_mismatch(redis)
    service = _PinnedCardService(REAL_H100[1], {"GPU-1": FAKE_H100[1]})

    result = await _idle_run(context_factory, redis, service)

    assert service.gpu_uuids == ["GPU-1"]
    assert result.passed is False
    assert result.event.what_we_saw["card_index"] == 1


# --- review at f1c3f85c: a judged card with no numbers, cc-only names, and the verdict store


async def _no_numbers(context_factory, redis):
    return await _run_metrics(context_factory, redis, {})


async def _run_metrics(context_factory, redis, metrics, model="NVIDIA H100 80GB HBM3"):
    service = DummyValidationService(success=True, metrics=metrics)
    ctx = context_factory(
        services=build_services(validation=service, redis=redis),
        config=build_context_config(),
        state=build_state(specs=_specs(model)),
    )
    return await CapabilityCheck().run(ctx)


@pytest.mark.asyncio
async def test_a_judged_card_with_no_numbers_and_no_measured_pass_fails_as_not_measured(context_factory):
    result = await _no_numbers(context_factory, FakeRedis())

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_MODEL_NOT_MEASURED.reason


@pytest.mark.asyncio
async def test_one_miss_after_a_measured_pass_is_tolerated_then_fails_until_numbers_return(context_factory):
    redis = FakeRedis()
    assert (await _run_metrics(context_factory, redis, REAL_H100[1])).passed is True

    assert (await _no_numbers(context_factory, redis)).passed is True
    second = await _no_numbers(context_factory, redis)
    assert second.passed is False
    assert second.event.reason_code == Msg.GPU_MODEL_NOT_MEASURED.reason

    assert (await _run_metrics(context_factory, redis, REAL_H100[1])).passed is True
    assert json.loads(redis.store[_verdict_key()]) == {"mismatches": [], "measured_ok": [0]}


@pytest.mark.parametrize(
    "model, cc, sm_count, passed",
    [
        ("NVIDIA RTX PRO 2000 Blackwell", "8.6", 48, False),  # prod: an Ampere card under a Blackwell name
        ("NVIDIA CMP 170HX", "8.0", 70, True),
        ("NVIDIA H100 NVL", "9.0", 114, True),  # cc-only now: its SM count varies by SKU
    ],
)
def test_a_cc_only_name_is_judged_by_its_compute_capability_alone(model, cc, sm_count, passed):
    assert (gpu_model_mismatch(_specs(model), {"cc": cc, "sm_count": sm_count}) is None) is passed


@pytest.mark.parametrize(
    "model", ["NVIDIA A100-PCIE-40GB", "NVIDIA A800 80GB PCIe", "NVIDIA H100 NVL", "NVIDIA H800 80GB HBM3", "NVIDIA H800 PCIe"]
)
def test_names_with_more_than_one_sm_count_are_cc_only(model):
    assert model not in EXPECTED_CC_AND_SM and model in EXPECTED_CC_ONLY


class _ReadFailsRedis(FakeRedis):
    async def get(self, key: str):
        raise ConnectionError("read failed")


@pytest.mark.asyncio
async def test_a_failed_verdict_read_writes_nothing(context_factory):
    redis = _ReadFailsRedis()

    await _run_metrics(context_factory, redis, FAKE_H100[1])

    assert _verdict_key() not in redis.store


@pytest.mark.asyncio
async def test_a_rented_read_of_a_standing_mismatch_refreshes_its_ttl(context_factory):
    redis = FakeRedis()
    await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108, redis=redis)
    redis.ttl[_verdict_key()] = 5

    await _verdict_check(context_factory, redis, _rented(_specs("NVIDIA H100 80GB HBM3"), redis))

    assert redis.ttl[_verdict_key()] == capability.GPU_MODEL_VERDICT_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_dry_run_stores_no_verdict(context_factory, monkeypatch):
    monkeypatch.setattr(capability.settings, "DRY_RUN", True)
    redis = FakeRedis()

    await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108, redis=redis)

    assert _verdict_key() not in redis.store


# --- review at 9ff05368: not measured fails only on card 0 of the idle full matmul; elsewhere it is logged


@pytest.mark.asyncio
async def test_a_filler_sample_with_no_numbers_is_logged_and_does_not_fail(context_factory, sample_never_due):
    redis = FakeRedis()
    ctx, _ = _filler_ctx(context_factory, redis, {})

    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["matmul_sample"]["no_numbers"] is True
    assert json.loads(redis.store[_verdict_key()])["mismatches"] == []
    assert redis.ttl[_verdict_key()] == capability.GPU_MODEL_VERDICT_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_carried_remeasure_with_no_numbers_keeps_the_stored_mismatch_and_adds_none(context_factory):
    redis = FakeRedis()
    _store_card_1_mismatch(redis)
    service = _PinnedCardService(REAL_H100[1], {"GPU-1": {}})

    result = await _idle_run(context_factory, redis, service)

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_MODEL_MISMATCH.reason
    assert [m["kind"] for m in json.loads(redis.store[_verdict_key()])["mismatches"]] == ["below_model"]


@pytest.mark.asyncio
async def test_an_all_cards_answer_with_no_numbers_after_card_0_is_logged_and_does_not_fail(context_factory):
    redis = FakeRedis()
    service = DummyValidationService(success=True, metrics=REAL_H100[1])
    real_validate = service.validate_gpu_model_and_process_job

    async def with_cards(**kwargs):
        result = await real_validate(**kwargs)
        result.card_metrics = {0: REAL_H100[1], 1: {}}
        return result

    service.validate_gpu_model_and_process_job = with_cards
    ctx = context_factory(
        services=build_services(validation=service, redis=redis),
        config=build_context_config(),
        state=build_state(specs=_two_card_specs()),
    )

    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["no_numbers_cards"] == [1]


@pytest.mark.asyncio
async def test_a_stored_mismatch_stands_when_the_card_is_renamed_to_an_alias_of_its_model(context_factory):
    redis = FakeRedis()
    await _run(context_factory, "NVIDIA H100 80GB HBM3", "8.0", 108, redis=redis)

    result = await _verdict_check(context_factory, redis, _rented(_specs("Tesla H100 80GB HBM3"), redis))

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_MODEL_MISMATCH.reason


@pytest.mark.parametrize("owner, removed", [("lium", True), ("miner", False)])
@pytest.mark.asyncio
async def test_a_faked_model_on_a_filler_node_removes_lium_filler_only(context_factory, sample_never_due, owner, removed):
    redis = FakeRedis()
    ctx, _ = _filler_ctx(context_factory, redis, FAKE_H100[1])
    ctx.state.rented_data.all_filler_containers_by_executor = {"executor-123": ["filler_active"]}
    ctx.state.rented_data.default_job_owner_by_executor = {"executor-123": owner}
    ssh = AsyncMock()
    ssh.run.return_value = SimpleNamespace(exit_status=0, stdout="", stderr="")
    ctx = ctx.model_copy(update={"ssh": ssh})

    result = await CapabilityCheck().run(ctx)

    assert result.event.reason_code == Msg.GPU_MODEL_MISMATCH.reason
    assert result.event.what_we_saw.get("removed_fillers") == (["filler_active"] if removed else None)


def test_a_bare_nvidia_h100_with_the_sxm_sm_count_passes_as_the_pcie_card():
    assert gpu_model_mismatch(_specs("NVIDIA H100"), {"cc": "9.0", "sm_count": 132}) is None
    assert gpu_model_mismatch(_specs("NVIDIA H100"), {"cc": "8.0", "sm_count": 108}) is not None


@pytest.mark.asyncio
async def test_a_carried_work_proof_failure_clears_on_a_right_answer_with_no_numbers(context_factory, monkeypatch):
    """Review (17 Nov): a filler-sample work-proof failure on card 1, the filler then removed. The normal path's pinned
    remeasure of card 1 gets a right answer without numbers: that clears the failure, as the filler sample does."""
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_CHECK_ENABLED", True)
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_ENFORCEMENT_ENABLED", True)
    redis = FakeRedis()
    failed = capability.work_proof_failed(_two_card_specs(), 1)
    redis.store[_verdict_key()] = json.dumps({"mismatches": [failed.as_dict()]})
    service = _PinnedCardService(REAL_H100[1], {"GPU-1": {}})

    result = await _idle_run(context_factory, redis, service)

    assert service.gpu_uuids == ["GPU-1"]
    assert result.passed is True
    assert json.loads(redis.store[_verdict_key()])["mismatches"] == []


@pytest.mark.asyncio
async def test_an_out_of_turn_retry_moves_to_the_next_card_each_cycle(context_factory, monkeypatch):
    # lio#2456 review: a node with no verdict is retried every cycle on the next card, its turn included
    picked = []

    async def card_sample(ctx, specs, card_index, sample):
        picked.append(card_index)  # no numbers and no answer: no verdict, so the next cycle retries

    monkeypatch.setattr(capability, "_card_matmul_sample", card_sample)
    monkeypatch.setattr(capability, "_filler_sample_offset", lambda ctx: 23)
    monkeypatch.setattr(capability.settings, "FILLER_MATMUL_SAMPLE_EVERY_N_CYCLES", 24)
    redis = FakeRedis()
    for cycle in (96, 97):  # 97 is the node's turn: (97 + 23) % 24 == 0
        monkeypatch.setattr(capability, "_filler_cycle_index", lambda ctx, cycle=cycle: cycle)
        ctx, _ = _filler_ctx(context_factory, redis, REAL_H100[1], specs=_two_card_specs())
        await CapabilityCheck().run(ctx)

    assert picked == [1, 0]  # a day-step index at the turn would pick card 1 again


TWO_H100 = {"gpu": {"count": 2, "details": [{"name": REAL_H100[0], "uuid": "GPU-0"}, {"name": REAL_H100[0], "uuid": "GPU-1"}]}}
H100_AND_ADA = {
    "gpu": {"count": 2, "details": [{"name": REAL_H100[0], "uuid": "GPU-0"}, {"name": FAKED[2][0], "uuid": "GPU-1"}]}
}


@pytest.mark.asyncio
@pytest.mark.parametrize("specs, passed", [(TWO_H100, True), (H100_AND_ADA, False)], ids=["real", "faked"])
async def test_the_card_whose_sample_failed_is_judged_after_the_stop(context_factory, sample_never_due, specs, passed):
    # review: with all-cards off the full check proves card 0 only, so the marked card is run on its own
    redis = FakeRedis()
    ctx, service = _filler_ctx(context_factory, redis, REAL_H100[1], specs=specs)
    await redis.set(capability._full_check_due_key(ctx), "1")

    result = await CapabilityCheck().run(_lium_filler(ctx))

    assert service.budgets == [None, capability.settings.FILLER_MATMUL_SAMPLE_VRAM_MB]
    assert service.gpu_uuids[1] == "GPU-1"
    assert result.passed is passed
    assert result.event.what_we_saw["stopped_fillers_for_full_check"] == ["filler_active"]
    assert redis.ttl[_verdict_key()] == capability.GPU_MODEL_VERDICT_TTL_SECONDS


class _DueCardService(SamplingValidationService):
    """Card 0 answers right in the full check; the run pinned to the due card returns `due_card`."""

    def __init__(self, due_card: ValidationResult):
        super().__init__(result=ValidationResult(success=True, metrics=REAL_H100[1]))
        self.due_card = due_card

    async def validate_gpu_model_and_process_job(self, *, gpu_uuid=None, **kwargs):
        result = await super().validate_gpu_model_and_process_job(gpu_uuid=gpu_uuid, **kwargs)
        return self.due_card if gpu_uuid == "GPU-1" else result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "due_card, passed",
    [(WRONG_UUID, False), (NO_ROOM, True), (TIMED_OUT, True), (VALIDATOR_ERROR, True)],
    ids=["wrong_uuid", "empty_uuid", "timeout", "validator_error"],
)
async def test_after_the_stop_only_a_wrong_answer_from_the_due_card_fails_the_node(
    context_factory, sample_never_due, due_card, passed
):
    # review: card 0 passes the full check; a wrong UUID from the marked card fails the node as on card 0
    redis = FakeRedis()
    service = _DueCardService(due_card)
    ctx = context_factory(
        services=build_services(validation=service, redis=redis),
        state=build_state(
            specs=TWO_H100,
            rented_data=RentedExecutorsResponse(
                executors={}, banned_guids=[], filler_containers_by_executor={"executor-123": "filler_active"}
            ),
        ),
    )
    await redis.set(capability._full_check_due_key(ctx), "1")

    result = await CapabilityCheck().run(_lium_filler(ctx))

    assert service.gpu_uuids == [None, "GPU-1"]
    assert result.passed is passed
    if not passed:
        assert result.event.reason_code == Msg.GPU_WORK_PROOF_FAILED.reason
    assert redis.ttl[_verdict_key()] == (
        capability.GPU_MODEL_NO_VERDICT_TTL_SECONDS if passed else capability.GPU_MODEL_VERDICT_TTL_SECONDS
    )


def _two_card_filler_ctx(context_factory, redis, service):
    state = build_state(
        specs=TWO_H100,
        rented_data=RentedExecutorsResponse(
            executors={}, banned_guids=[], filler_containers_by_executor={"executor-123": "filler_active"}
        ),
    )
    return _lium_filler(context_factory(services=build_services(validation=service, redis=redis), state=state))


def _stored_kinds(redis) -> list[tuple[int, str]]:
    return [(m["card_index"], m["kind"]) for m in json.loads(redis.store[_verdict_key()])["mismatches"]]


RIGHT_ANSWER = ValidationResult(success=True, metrics=REAL_H100[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("all_cards", [False, True], ids=["all_cards_off", "all_cards_on"])
@pytest.mark.parametrize("card", [0, 1], ids=["card_0", "due_card"])
@pytest.mark.parametrize("retry", [NO_ROOM, WRONG_UUID], ids=["no_answer", "wrong_uuid"])
async def test_a_wrong_uuid_after_the_stop_fails_the_node_and_removes_the_filler_until_a_full_check_passes(
    context_factory, sample_never_due, monkeypatch, all_cards, card, retry
):
    # review: a proven wrong answer is stored like any mismatch, under all-cards on and off
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_CHECK_ENABLED", all_cards)
    monkeypatch.setattr(capability.settings, "MATMUL_ALLCARDS_ENFORCEMENT_ENABLED", all_cards)
    redis = FakeRedis()

    def service_answering(answer: ValidationResult):
        return _DueCardService(answer) if card else SamplingValidationService(result=answer)

    def without_filler(service):
        return context_factory(services=build_services(validation=service, redis=redis), state=build_state(specs=TWO_H100))

    ctx = _two_card_filler_ctx(context_factory, redis, service_answering(WRONG_UUID))
    await redis.set(capability._full_check_due_key(ctx), str(card))
    result = await CapabilityCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_WORK_PROOF_FAILED.reason
    assert result.event.what_we_saw["stopped_fillers_for_full_check"] == ["filler_active"]
    assert _stored_kinds(redis) == [(card, "full_check_failed")]

    # the default job placed a filler again: the node fails and the filler is removed, as on any stored mismatch
    service = service_answering(RIGHT_ANSWER)
    result = await CapabilityCheck().run(_two_card_filler_ctx(context_factory, redis, service))
    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_WORK_PROOF_FAILED.reason
    assert result.event.what_we_saw["removed_fillers"] == ["filler_active"]
    assert service.budgets == []

    # no filler: the normal full check runs; a retry with no answer or a wrong answer clears nothing
    assert (await CapabilityCheck().run(without_filler(service_answering(retry)))).passed is False
    assert _stored_kinds(redis) == [(card, "full_check_failed")]

    # only a full check that passes clears it
    assert (await CapabilityCheck().run(without_filler(service_answering(RIGHT_ANSWER)))).passed is True
    assert _stored_kinds(redis) == []


@pytest.mark.asyncio
async def test_no_answer_after_the_stop_passes_and_the_node_is_sampled_again_a_day_later(context_factory, sample_never_due):
    redis = FakeRedis()
    ctx = _two_card_filler_ctx(context_factory, redis, SamplingValidationService(result=NO_ROOM))
    await redis.set(capability._full_check_due_key(ctx), "0")
    assert (await CapabilityCheck().run(ctx)).passed is True
    assert _stored_kinds(redis) == []

    ctx = _two_card_filler_ctx(context_factory, redis, SamplingValidationService(result=WRONG_UUID))
    assert (await CapabilityCheck().run(ctx)).passed is True  # the day entry: no stop, no sample
    ctx.ssh.run.assert_not_awaited()

    del redis.store[_verdict_key()]  # a day later the entry is gone: sampled again, then the full check is due
    service = SamplingValidationService(result=WRONG_UUID)
    ctx = _two_card_filler_ctx(context_factory, redis, service)
    assert (await CapabilityCheck().run(ctx)).passed is True
    assert service.budgets == [capability.settings.FILLER_MATMUL_SAMPLE_VRAM_MB]
    assert capability._full_check_due_key(ctx) in redis.store


@pytest.mark.asyncio
async def test_a_full_check_on_idle_cards_clears_a_stored_full_check_failure(context_factory):
    # no filler was placed again: the normal full check runs every cycle and its pass clears the failure
    redis = FakeRedis()
    ctx = _two_card_filler_ctx(context_factory, redis, _DueCardService(WRONG_UUID))
    await redis.set(capability._full_check_due_key(ctx), "1")
    assert (await CapabilityCheck().run(ctx)).passed is False

    service = _DueCardService(RIGHT_ANSWER)
    state = build_state(specs=TWO_H100)
    result = await CapabilityCheck().run(context_factory(services=build_services(validation=service, redis=redis), state=state))

    assert service.gpu_uuids == [None, "GPU-1"]
    assert result.passed is True
    assert _stored_kinds(redis) == []
