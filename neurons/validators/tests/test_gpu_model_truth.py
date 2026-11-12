"""DAH-6241: a node whose card is not the model it reports earns nothing."""

import json

import pytest

from neurons.validators.src.services.matrix_validation_service import ValidationResult
from neurons.validators.src.services.task.availability import availability_errors
from neurons.validators.src.services.task.checks import capability
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.checks.gpu_model_verdict import GpuModelVerdictCheck
from neurons.validators.src.services.task.checks.gpu_model_truth import (
    EXPECTED_CC_AND_SM,
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


def _filler_ctx(context_factory, redis, metrics, *, success=True, specs=None):
    service = SamplingValidationService(
        result=ValidationResult(success=success, error_message="" if success else "no memory", metrics=metrics)
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
async def test_a_failed_first_sample_is_not_retried_every_cycle(context_factory, sample_never_due):
    redis = FakeRedis()
    ctx, service = _filler_ctx(context_factory, redis, None, success=False)
    assert (await CapabilityCheck().run(ctx)).passed is True
    assert redis.ttl[_verdict_key()] == capability.GPU_MODEL_NO_VERDICT_TTL_SECONDS

    ctx, service = _filler_ctx(context_factory, redis, None, success=False)
    assert (await CapabilityCheck().run(ctx)).passed is True
    assert service.budgets == []


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
