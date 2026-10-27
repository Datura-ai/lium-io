"""GpuFingerprintCheck: the anchored GPU UUID set of a listed node (DAH-3457).

Legacy mode (GPU_ANCHOR_HARD_ENABLED off) keeps today's verdict and adds the hard rule's would-be decision to
the event; hard mode splits a changed set into GPU_MISSING (subset) and a broken anchor (any card outside it),
and a broken anchor stays broken under the same executor id.
"""

from unittest.mock import patch

import pytest

from neurons.validators.src.services.task.checks.gpu_fingerprint import GpuFingerprintCheck
from neurons.validators.src.services.task.messages import GpuFingerprintMessages as Msg, render_message
from neurons.validators.src.services.task.pipeline import CheckResult, Pipeline

from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from tests.helpers import build_context_config, build_services, build_state, default_executor

SETTINGS = "neurons.validators.src.services.task.checks.gpu_fingerprint.settings"


def _live_rental() -> RentedExecutorsResponse:
    executor = default_executor()
    return RentedExecutorsResponse(
        executors={
            executor.uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address=executor.address,
                executor_ip_port=str(executor.port),
                pods=[RentedPod(pod_id="pod-1", container_name="pod_p1")],
            )
        },
        banned_guids=[],
    )


def _ctx(context_factory, *, prev_uuids: str, current_uuids: str, anchor_broken: bool = False, rented=None):
    verified = {"uuids": prev_uuids} if prev_uuids else {}
    if anchor_broken:
        verified["anchor_broken"] = True
    return context_factory(
        services=build_services(),
        config=build_context_config(),
        state=build_state(gpu_uuids=current_uuids, rented_data=rented),
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
    "current_uuids",
    [
        "gpu-003",  # the one card replaced (5GeGsD2M, 184 events in 7 d)
        "gpu-001,gpu-003",  # one of two cards replaced
        "gpu-001,gpu-002,gpu-003",  # a card added
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
    assert "duplicate_uuid" not in result.event.what_we_saw
    assert result.updates == {"clear_verified_job_info": True, "gpu_anchor_broken": True}




@pytest.mark.parametrize(
    "hard,kwargs",
    [
        (False, {"prev_uuids": "gpu-001,gpu-002", "current_uuids": "gpu-001,gpu-003"}),
        (True, {"prev_uuids": "gpu-001,gpu-002", "current_uuids": "gpu-001"}),
        (True, {"prev_uuids": "gpu-001", "current_uuids": "gpu-002"}),
        (True, {"prev_uuids": "gpu-001", "current_uuids": "gpu-001", "anchor_broken": True}),
    ],
)
@pytest.mark.asyncio
async def test_a_changed_set_under_a_live_rental_fails_without_stopping_the_rented_checks(
    hard, kwargs, context_factory
):
    """Regression (27 Oct): a rented node back from a reboot under a new GPU UUID halted here every cycle, so the
    rented-pod checks never ran again while the backend still showed the pod RUNNING."""
    with patch(SETTINGS) as s:
        s.GPU_ANCHOR_HARD_ENABLED = hard
        idle = await GpuFingerprintCheck().run(_ctx(context_factory, **kwargs))
        rented = await GpuFingerprintCheck().run(_ctx(context_factory, rented=_live_rental(), **kwargs))

    assert (idle.passed, idle.fatal) == (False, None)
    assert "gpu_fingerprint_passed" not in idle.updates
    assert (rented.passed, rented.fatal) == (False, False)
    assert rented.updates == {**idle.updates, "gpu_fingerprint_passed": False}


class _RentedHalt:
    """Stands in for TenantEnforcementCheck: a rented node's run ends here with a rented score."""

    check_id = "executor.validate.rented_state"
    fatal = True

    async def run(self, ctx):
        event = render_message(Msg.UUID_OK, ctx=ctx, check_id=self.check_id, what={})
        return CheckResult(
            passed=True, event=event, halt=True, updates={"rented": True, "score": 1.0, "job_score": 1.0, "success": True}
        )


class _Sink:
    async def emit(self, event):
        pass


@pytest.mark.parametrize("rented", [False, True])
@pytest.mark.asyncio
async def test_the_run_still_ends_as_the_fingerprint_failure_after_the_rented_checks(rented, context_factory):
    ctx = _ctx(
        context_factory,
        prev_uuids="gpu-001,gpu-002",
        current_uuids="gpu-001,gpu-003",
        rented=_live_rental() if rented else None,
    )
    with patch(SETTINGS) as s:
        s.GPU_ANCHOR_HARD_ENABLED = False
        ok, events, end = await Pipeline([GpuFingerprintCheck(), _RentedHalt()], _Sink()).run(ctx)

    assert ok is False
    assert events[-1].reason_code == Msg.UUID_CHANGED.reason
    assert [e.check_id for e in events] == (
        ["executor.validate.rented_state", "gpu.validate.fingerprint"] if rented else ["gpu.validate.fingerprint"]
    )
    assert (end.rented, end.score, end.success, end.clear_verified_job_info) == (False, 0.0, False, True)
