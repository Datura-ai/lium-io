"""GpuFingerprintCheck: the anchored GPU UUID set of a listed node (DAH-3457).

A full set of other cards (same count) is re-anchored and passes in both modes; a set that lost or gained cards
resets verification (GPU_MISSING under the hard rule when only anchored cards are missing). Nothing breaks the
anchor for good.
"""

from unittest.mock import patch

import pytest

from neurons.validators.src.services.task.checks.gpu_fingerprint import GpuFingerprintCheck
from neurons.validators.src.services.task.messages import GpuFingerprintMessages as Msg

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
        ("gpu-001,gpu-002", "gpu-001,gpu-003", True, Msg.UUID_OK.reason, False),  # same count: re-anchored
        ("gpu-001,gpu-002", "gpu-003,gpu-004", True, Msg.UUID_OK.reason, False),  # whole set other cards
        ("gpu-001", "gpu-001,gpu-002", False, Msg.UUID_CHANGED.reason, True),  # a card added
        ("gpu-001,gpu-002", "gpu-003", False, Msg.UUID_CHANGED.reason, True),  # fewer cards, one foreign
    ],
)
@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.asyncio
async def test_a_full_set_change_passes_and_a_count_change_fails_in_both_modes(
    prev_uuids, current_uuids, expected_pass, expected_reason, expect_clear, hard, context_factory
):
    """Regression: a full set of other cards fails or is held instead of re-verifying, a set that lost or gained cards
    passes, or a first verification (no anchor yet) fails, in either mode."""
    result = await _run(context_factory, hard=hard, prev_uuids=prev_uuids, current_uuids=current_uuids)

    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason
    assert "gpu_anchor_broken" not in result.updates
    if expect_clear:
        assert result.updates.get("clear_verified_job_info") is True
    else:
        assert "clear_verified_job_info" not in result.updates
    changed = set(prev_uuids.split(",")) != set(current_uuids.split(","))
    reanchored = bool(prev_uuids and current_uuids) and expected_pass and changed
    assert result.updates.get("gpu_reanchor", False) is reanchored


@pytest.mark.asyncio
async def test_hard_mode_only_missing_cards_are_gpu_missing_and_an_old_broken_mark_is_ignored(context_factory):
    """Regression: a changed set marks the anchor broken for good, or a mark written by that retired rule keeps
    failing a node whose set reads clean."""
    missing = await _run(context_factory, hard=True, prev_uuids="gpu-001,gpu-002", current_uuids="gpu-001")
    assert missing.passed is False and missing.event.reason_code == Msg.GPU_MISSING.reason
    assert missing.updates["clear_verified_job_info"] is True and "gpu_anchor_broken" not in missing.updates

    old_mark = await _run(
        context_factory, hard=True, prev_uuids="gpu-001,gpu-002", current_uuids="gpu-001,gpu-002", anchor_broken=True
    )
    assert old_mark.passed is True and old_mark.event.reason_code == Msg.UUID_OK.reason
