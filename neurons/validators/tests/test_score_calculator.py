"""Tests for calculate_scores in score_calculator.py, focused on the EMA verifyx download
speed threshold. Other score_calculator paths (collateral, rental, price) are exercised
via integration in test_score_check.py and test_pipeline_default_scenarios.py.
"""
import pytest

from neurons.validators.src.services.task.score_calculator import (
    MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
    calculate_scores,
)
from helpers import build_context_config, build_services, build_state, default_executor, make_context


def _ctx(ema_verifyx_download_speed, price_per_gpu=None):
    """Build a minimal context with the given EMA verifyx download speed."""
    executor = default_executor()
    # Override price_per_gpu to None so the price check is not triggered
    executor = executor.model_copy(update={"price_per_gpu": price_per_gpu})
    state = build_state(
        specs={"network": {"ema_verifyx_download_speed": ema_verifyx_download_speed}}
    )
    return make_context(
        executor=executor,
        state=state,
        services=build_services(),
        config=build_context_config(),
        collateral_deposited=True,
        is_rental_succeed=True,
    )


@pytest.mark.parametrize(
    "ema_speed, scores_zeroed, warning_fragment",
    [
        # No EMA available (VerifyX disabled or no measurement) — no penalty
        (None, False, None),
        # Above threshold — no penalty
        (MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS, False, None),
        (120.0, False, None),
        (500.0, False, None),
        # Just below threshold — both scores zeroed with warning
        (MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS - 0.1, True, "EMA verifyx download speed too slow"),
        (99.9, True, "99.9 Mbps"),
        (50.0, True, "50.0 Mbps"),
        (0.0, True, "0.0 Mbps"),
    ],
)
def test_ema_verifyx_download_speed_scoring(ema_speed, scores_zeroed, warning_fragment):
    ctx = _ctx(ema_speed)
    actual_score, job_score, warning = calculate_scores(ctx, rented=False)

    if scores_zeroed:
        assert actual_score == 0.0
        assert job_score == 0.0
        assert warning_fragment in warning
    else:
        assert actual_score == 1.0
        assert job_score == 1.0
        assert warning == ""


def test_no_network_specs_does_not_penalise():
    """ctx.state.specs has no 'network' key at all — should not raise or penalise."""
    executor = default_executor()
    executor = executor.model_copy(update={"price_per_gpu": None})
    state = build_state(specs={})
    ctx = make_context(
        executor=executor,
        state=state,
        services=build_services(),
        config=build_context_config(),
        collateral_deposited=True,
        is_rental_succeed=True,
    )
    actual_score, _job_score, warning = calculate_scores(ctx, rented=False)
    assert actual_score == 1.0
    assert warning == ""


def test_none_specs_does_not_penalise():
    """ctx.state.specs is None — should not raise or penalise."""
    executor = default_executor()
    executor = executor.model_copy(update={"price_per_gpu": None})
    state = build_state(specs=None)
    ctx = make_context(
        executor=executor,
        state=state,
        services=build_services(),
        config=build_context_config(),
        collateral_deposited=True,
        is_rental_succeed=True,
    )
    actual_score, _job_score, warning = calculate_scores(ctx, rented=False)
    assert actual_score == 1.0
    assert warning == ""
