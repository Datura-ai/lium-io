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


def _ctx_without_specs(specs, price_per_gpu=None):
    executor = default_executor()
    executor = executor.model_copy(update={"price_per_gpu": price_per_gpu})
    state = build_state(specs=specs)
    return make_context(
        executor=executor,
        state=state,
        services=build_services(),
        config=build_context_config(),
        collateral_deposited=True,
        is_rental_succeed=True,
    )


def _ctx(ema_verifyx_download_speed, price_per_gpu=None):
    return _ctx_without_specs(
        {"network": {"ema_verifyx_download_speed": ema_verifyx_download_speed}},
        price_per_gpu=price_per_gpu,
    )


@pytest.mark.parametrize(
    "ema_speed, scores_zeroed, warning_fragment",
    [
        # No EMA available for a non-rented executor — zero (regression guard for DAH verifyx null fix)
        (None, True, "unavailable"),
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
def test_ema_verifyx_download_speed_scoring_unrented(ema_speed, scores_zeroed, warning_fragment):
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


def test_ema_download_missing_rented_does_not_zero():
    """Rented executor with missing EMA must not be zeroed — avoids killing emission on active rentals."""
    ctx = _ctx(None)
    actual_score, job_score, warning = calculate_scores(ctx, rented=True)
    assert actual_score == 1.0
    assert job_score == 1.0
    assert "unavailable" not in warning


def test_ema_download_below_threshold_rented_still_zeros_actual_score():
    """Rented executor below threshold: actual_score zeroed, final job_score forced to 1.0 by _format_return."""
    ctx = _ctx(50.0)
    actual_score, job_score, warning = calculate_scores(ctx, rented=True)
    assert actual_score == 0.0
    # job_score is forced to 1.0 for rented in _format_return
    assert job_score == 1.0
    assert "too slow" in warning


def test_no_network_specs_unrented_zeros():
    """ctx.state.specs has no 'network' key at all — non-rented should zero."""
    ctx = _ctx_without_specs({})
    actual_score, job_score, warning = calculate_scores(ctx, rented=False)
    assert actual_score == 0.0
    assert job_score == 0.0
    assert "unavailable" in warning


def test_no_network_specs_rented_does_not_penalise():
    """ctx.state.specs has no 'network' key — rented should not zero."""
    ctx = _ctx_without_specs({})
    actual_score, job_score, warning = calculate_scores(ctx, rented=True)
    assert actual_score == 1.0
    assert job_score == 1.0
    assert warning == ""


def test_none_specs_unrented_zeros():
    """ctx.state.specs is None — non-rented should zero."""
    ctx = _ctx_without_specs(None)
    actual_score, job_score, warning = calculate_scores(ctx, rented=False)
    assert actual_score == 0.0
    assert job_score == 0.0
    assert "unavailable" in warning


def test_none_specs_rented_does_not_penalise():
    """ctx.state.specs is None — rented should not zero."""
    ctx = _ctx_without_specs(None)
    actual_score, job_score, warning = calculate_scores(ctx, rented=True)
    assert actual_score == 1.0
    assert job_score == 1.0
    assert warning == ""
