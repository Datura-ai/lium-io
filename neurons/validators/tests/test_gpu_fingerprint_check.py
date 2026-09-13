"""GpuFingerprintCheck: the anchored GPU UUID set of a listed node (DAH-3457).

Legacy mode (GPU_ANCHOR_HARD_ENABLED off) keeps today's verdict and adds the hard rule's would-be decision to
the event; hard mode splits a changed set into GPU_MISSING (subset) and a broken anchor (any card outside it),
and a broken anchor stays broken under the same executor id.
"""

from unittest.mock import patch

import pytest

from neurons.validators.src.services.task.checks.gpu_fingerprint import GpuFingerprintCheck
from neurons.validators.src.services.task.checks.spec_change import SpecChangeCheck
from neurons.validators.src.services.task.messages import GpuFingerprintMessages as Msg
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory

from tests.helpers import build_context_config, build_services, build_state

SETTINGS = "neurons.validators.src.services.task.checks.gpu_fingerprint.settings"


def _ctx(context_factory, *, prev_uuids: str, current_uuids: str, anchor_broken: bool = False):
    verified = {"uuids": prev_uuids} if prev_uuids else {}
    if anchor_broken:
        verified["anchor_broken"] = True
    return context_factory(
        services=build_services(),
        config=build_context_config(),
        state=build_state(gpu_uuids=current_uuids),
        verified=verified,
    )


async def _run(context_factory, *, hard: bool, **kwargs):
    with patch(SETTINGS) as s:
        s.GPU_ANCHOR_HARD_ENABLED = hard
        return await GpuFingerprintCheck().run(_ctx(context_factory, **kwargs))


@pytest.mark.parametrize(
    "prev_uuids,current_uuids,expected_pass,expected_reason,expect_clear",
    [
        ("", "", True, Msg.UUID_OK.reason, False),
        ("", "gpu-001", True, Msg.UUID_OK.reason, False),
        ("gpu-001", "", True, Msg.UUID_OK.reason, False),
        ("gpu-001,gpu-002", "gpu-001,gpu-002", True, Msg.UUID_OK.reason, False),
        ("gpu-002,gpu-001", "gpu-001,gpu-002", True, Msg.UUID_OK.reason, False),  # sorted comparison
        ("gpu-001,gpu-002", "gpu-001,gpu-003", False, Msg.UUID_CHANGED.reason, True),
        ("gpu-001", "gpu-001,gpu-002", False, Msg.UUID_CHANGED.reason, True),
    ],
)
@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.asyncio
async def test_same_set_passes_and_a_different_set_fails_in_both_modes(
    prev_uuids, current_uuids, expected_pass, expected_reason, expect_clear, hard, context_factory
):
    """Regression: a changed set no longer fails, or a first verification (no anchor yet) fails, in either mode."""
    result = await _run(context_factory, hard=hard, prev_uuids=prev_uuids, current_uuids=current_uuids)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason
    if expect_clear:
        assert result.updates.get("clear_verified_job_info") is True
    else:
        assert "clear_verified_job_info" not in result.updates


@pytest.mark.parametrize(
    "prev_uuids,current_uuids,would_be,missing,unexpected",
    [
        ("gpu-001,gpu-002", "gpu-001", "GPU_MISSING", ["gpu-002"], []),
        ("gpu-001,gpu-002", "gpu-001,gpu-003", "ANCHOR_BROKEN", ["gpu-002"], ["gpu-003"]),
        ("gpu-001", "gpu-001,gpu-002", "ANCHOR_BROKEN", [], ["gpu-002"]),
    ],
)
@pytest.mark.asyncio
async def test_legacy_mode_records_the_hard_decision_without_taking_it(
    prev_uuids, current_uuids, would_be, missing, unexpected, context_factory
):
    """Regression: the flag is off and the stored row no longer says what the hard rule would have done,
    so the week of warn-mode data the flip is decided on is blank; or the flag is off and the node is
    marked broken anyway."""
    result = await _run(context_factory, hard=False, prev_uuids=prev_uuids, current_uuids=current_uuids)

    assert result.passed is False
    assert result.event.reason_code == Msg.UUID_CHANGED.reason
    assert result.event.event == Msg.UUID_CHANGED.event
    assert result.event.what_we_saw["anchor_hard_would_be"] == would_be
    assert result.event.what_we_saw["missing"] == missing
    assert result.event.what_we_saw["unexpected"] == unexpected
    assert result.updates == {"clear_verified_job_info": True}


@pytest.mark.asyncio
async def test_hard_mode_subset_is_gpu_missing_and_does_not_break_the_anchor(context_factory):
    """Regression: a card that dropped off the bus is treated as a swap, and the node never scores again
    once the card is back."""
    result = await _run(context_factory, hard=True, prev_uuids="gpu-001,gpu-002", current_uuids="gpu-002")

    assert result.passed is False
    assert result.event.reason_code == Msg.GPU_MISSING.reason
    assert result.event.what_we_saw["missing"] == ["gpu-001"]
    assert result.updates == {"clear_verified_job_info": True}
    assert "gpu_anchor_broken" not in result.updates


@pytest.mark.parametrize(
    "current_uuids",
    [
        "gpu-003",  # the one card replaced (5GeGsD2M, 184 events in 7 d)
        "gpu-001,gpu-003",  # one of two cards replaced
        "gpu-001,gpu-002,gpu-003",  # a card added
        "gpu-001,gpu-001,gpu-002",  # every anchored card present, one listed twice: not the anchored set either
    ],
)
@pytest.mark.asyncio
async def test_hard_mode_any_uuid_outside_the_anchor_breaks_it(current_uuids, context_factory):
    """Regression: an added or replaced card is read as GPU_MISSING (transient) or passes, so a node
    re-anchors on a different set under the same executor id."""
    result = await _run(context_factory, hard=True, prev_uuids="gpu-001,gpu-002", current_uuids=current_uuids)

    assert result.passed is False
    assert result.event.reason_code == Msg.ANCHOR_BROKEN.reason == "GPU_UUID_CHANGED"
    assert result.event.event == Msg.ANCHOR_BROKEN.event
    assert result.event.what_we_saw["anchor_broken"] is True
    assert result.event.what_we_saw.get("duplicate_uuid", False) is (current_uuids == "gpu-001,gpu-001,gpu-002")
    assert result.updates == {"clear_verified_job_info": True, "gpu_anchor_broken": True}


@pytest.mark.parametrize("current_uuids", ["gpu-001,gpu-002", "gpu-003", ""])
@pytest.mark.asyncio
async def test_hard_mode_a_broken_record_fails_before_the_sets_are_compared(current_uuids, context_factory):
    """Regression: the broken mark is ignored, so a node that swaps back to its anchored set (or reports
    nothing) is re-verified under the same executor id."""
    result = await _run(
        context_factory,
        hard=True,
        prev_uuids="gpu-001,gpu-002",
        current_uuids=current_uuids,
        anchor_broken=True,
    )

    assert result.passed is False
    assert result.event.event == Msg.ANCHOR_BROKEN.event
    assert result.event.what_we_saw["anchor_broken"] is True
    # already reset on the cycle that broke the anchor; a second reset would publish a second penalty trigger
    assert result.updates == {}


def test_fingerprint_runs_before_spec_change_in_both_pipelines():
    """Regression: SpecChangeCheck (model:count) runs first again, so a missing card fails as SPEC_CHANGED one
    step earlier and GPU_MISSING is never produced (7 d of prod: every subset case was filed as SPEC_CHANGED)."""
    for checks in (PipelineFactory.build_checks(), PipelineFactory.build_dry_run_checks()):
        fingerprint_index = next(i for i, check in enumerate(checks) if isinstance(check, GpuFingerprintCheck))
        spec_index = next(i for i, check in enumerate(checks) if isinstance(check, SpecChangeCheck))
        assert fingerprint_index < spec_index


@pytest.mark.asyncio
async def test_legacy_mode_ignores_a_broken_mark_left_by_hard_mode(context_factory):
    """Regression: switching the flag off does not restore the legacy behaviour for nodes marked while it
    was on."""
    result = await _run(
        context_factory,
        hard=False,
        prev_uuids="gpu-001,gpu-002",
        current_uuids="gpu-001,gpu-002",
        anchor_broken=True,
    )

    assert result.passed is True
    assert result.event.reason_code == Msg.UUID_OK.reason
