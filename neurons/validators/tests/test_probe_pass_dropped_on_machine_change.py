"""A node is listed again on a rental probe of the machine it is now, not on a pass the last machine earned.

The probe confirms every listed card as a CUDA device of its model, but runs once per interval. A node whose
listed cards changed (the spec, or a re-anchored set) or whose claimed card could not run the all-cards
work-proof has its last pass revoked, so it is hidden until a probe passes again.
"""

import pytest

from neurons.validators.src.services.matrix_validation_service import AllCardsProbe, ValidationResult
from neurons.validators.src.services.redis_service import RENTAL_PROBE_OK_PREFIX
from neurons.validators.src.services.task.checks import rental_probe
from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.checks.gpu_fingerprint import GpuFingerprintCheck
from neurons.validators.src.services.task.checks.spec_change import SpecChangeCheck

from tests.helpers import build_context_config, build_services, build_state

STAMP = f"{RENTAL_PROBE_OK_PREFIX}:executor-123"


def _services(**kwargs):
    services = build_services(**kwargs)
    services.redis.store[STAMP] = "1764230400"
    return services


@pytest.mark.parametrize(
    "verified_spec,new_spec,dropped",
    [
        ("NVIDIA GeForce RTX 4090:8", "NVIDIA GeForce RTX 4090:8", False),
        ("NVIDIA GeForce RTX 4090:8", "NVIDIA GeForce RTX 4090:4", True),
        ("NVIDIA GeForce RTX 4090:8", "NVIDIA RTX A6000:8", True),
    ],
)
@pytest.mark.asyncio
async def test_a_spec_change_drops_the_probe_pass(verified_spec, new_spec, dropped, context_factory):
    services = _services()
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(gpu_model_count=new_spec),
        verified={"spec": verified_spec},
    )
    await SpecChangeCheck().run(ctx)
    assert (services.redis.store.get(STAMP) == rental_probe._PASS_REVOKED_VALUE) is dropped


@pytest.mark.parametrize(
    "prev_uuids,current_uuids,dropped",
    [
        ("gpu-001,gpu-002", "gpu-001,gpu-002", False),
        ("gpu-001,gpu-002", "gpu-001,gpu-003", True),
        ("gpu-001,gpu-002", "gpu-003,gpu-004", True),
    ],
)
@pytest.mark.asyncio
async def test_a_re_anchored_gpu_set_drops_the_probe_pass(prev_uuids, current_uuids, dropped, context_factory):
    services = _services()
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(gpu_uuids=current_uuids),
        verified={"uuids": prev_uuids},
    )
    result = await GpuFingerprintCheck().run(ctx)
    assert result.passed
    assert (services.redis.store.get(STAMP) == rental_probe._PASS_REVOKED_VALUE) is dropped


@pytest.mark.parametrize(
    "reasons,refused",
    [
        (["", ""], False),
        # a card listed but not there: CUDA cannot select it, or it answers wrong
        (["", "no_answer"], True),
        (["", "exit_status=1"], True),
        (["", "wrong_answer"], True),
        (["", "empty_output"], True),
        # slowness, the transport or our own error says nothing about the card
        (["", "timeout"], False),
        (["", "timeout_after_retry"], False),
        (["", "ssh_error: reset"], False),
        (["", "validator_error"], False),
        # another workload holds the card's memory (review on lium-io#2515): the card is there
        (["", "vram_busy"], False),
    ],
)
def test_a_card_the_node_cannot_run_the_proof_on_is_a_refusal(reasons, refused):
    probe = AllCardsProbe(
        passed=not any(reasons),
        failure_reason="",
        elapsed_seconds=1.0,
        over_wall_clock=False,
        per_card=[{"card_index": i, "ok": not r, "reason": r} for i, r in enumerate(reasons)],
    )
    assert probe.card_refused is refused


class _Validation:
    def __init__(self, card_refused: bool):
        self.card_refused = card_refused

    async def validate_gpu_model_and_process_job(self, *, ssh_client, executor_info, default_extra, machine_spec):
        return ValidationResult(
            success=True,
            expected_uuid="test-uuid-123",
            returned_uuid="test-uuid-123",
            stdout="UUID: test-uuid-123",
            stderr="",
            error_message="",
            all_cards_card_refused=self.card_refused,
        )


@pytest.mark.parametrize("card_refused", [False, True])
@pytest.mark.asyncio
async def test_an_unenforced_all_cards_refusal_drops_the_probe_pass(card_refused, context_factory):
    services = _services(validation=_Validation(card_refused))
    ctx = context_factory(services=services, config=build_context_config(), state=build_state(specs={"gpu": {"count": 2}}))
    result = await CapabilityCheck().run(ctx)
    # the run itself still passes: unenforced, the all-cards run fails no node
    assert result.passed
    assert (services.redis.store.get(STAMP) == rental_probe._PASS_REVOKED_VALUE) is card_refused


@pytest.mark.asyncio
async def test_a_repeated_refusal_keeps_the_probe_interval(monkeypatch, context_factory):
    """Review of lio#2515: unenforced, the same card is refused every cycle. The first refusal revokes the pass and
    the probe re-earns it; a repeat within the interval must not revoke it again (a probe container every cycle
    instead of every RENTAL_PROBE_INTERVAL_HOURS). After the interval a refusal revokes once more."""
    services = _services(validation=_Validation(True))
    ctx = context_factory(services=services, config=build_context_config(), state=build_state(specs={"gpu": {"count": 2}}))
    await CapabilityCheck().run(ctx)
    assert services.redis.store[STAMP] == rental_probe._PASS_REVOKED_VALUE
    refusal_key = f"{rental_probe._REDIS_CARD_REFUSAL_REVOKED_PREFIX}:executor-123"
    assert services.redis.ttl[refusal_key] == rental_probe.settings.RENTAL_PROBE_INTERVAL_HOURS * 3600

    services.redis.store[STAMP] = "1764240000"  # the probe passed on this machine
    await CapabilityCheck().run(ctx)
    assert services.redis.store[STAMP] == "1764240000"

    del services.redis.store[refusal_key]  # the interval ran out
    await CapabilityCheck().run(ctx)
    assert services.redis.store[STAMP] == rental_probe._PASS_REVOKED_VALUE


@pytest.mark.asyncio
async def test_a_dry_run_leaves_the_live_probe_pass(monkeypatch, context_factory):
    # a dry-run validator shares Redis with the live one
    monkeypatch.setattr(rental_probe.settings, "DRY_RUN", True)
    services = _services()
    ctx = context_factory(
        services=services,
        config=build_context_config(),
        state=build_state(gpu_model_count="NVIDIA RTX A6000:8"),
        verified={"spec": "NVIDIA GeForce RTX 4090:8"},
    )
    await SpecChangeCheck().run(ctx)
    assert services.redis.store.get(STAMP) == "1764230400"
