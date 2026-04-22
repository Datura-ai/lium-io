"""Tests for calculate_scores in score_calculator.py, focused on the EMA verifyx download
speed threshold. Other score_calculator paths (collateral, rental, price) are exercised
via integration in test_score_check.py and test_pipeline_default_scenarios.py.
"""
import pytest

from neurons.validators.src.services.task.score_calculator import calculate_scores
from neurons.validators.src.services.task.checks.verifyx import MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS
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
        # No EMA available for a non-rented executor — zero (verifyx disabled or first-run edge case)
        (None, True, "unavailable"),
        # At or above threshold — no penalty (threshold enforcement is in VerifyXCheck, not here)
        (MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS, False, None),
        (120.0, False, None),
        (500.0, False, None),
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
