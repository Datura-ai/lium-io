"""Tests for calculate_scores in score_calculator.py, focused on the EMA verifyx download
speed threshold. Other score_calculator paths (collateral, rental, price) are exercised
via integration in test_score_check.py and test_pipeline_default_scenarios.py.
"""
import pytest
from helpers import (
    build_context_config,
    build_services,
    build_state,
    default_executor,
    make_context,
)
from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG
from neurons.validators.src.services.task.checks.verifyx import MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS
from neurons.validators.src.services.task.score_calculator import calculate_scores


class StubSharedConfigClient:
    def __init__(self, *, require_storage_limit_supported: bool):
        self.config = DEFAULT_SHARED_CONFIG.model_copy(
            update={"require_storage_limit_supported": require_storage_limit_supported}
        )


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


def _ctx_with_storage_policy(specs, *, required: bool, rented: bool = False):
    executor = default_executor().model_copy(update={"price_per_gpu": None})
    state = build_state(specs=specs)
    services = build_services(
        shared_config_client=StubSharedConfigClient(
            require_storage_limit_supported=required
        )
    )
    return make_context(
        executor=executor,
        state=state,
        services=services,
        config=build_context_config(),
        collateral_deposited=True,
        is_rental_succeed=rented,
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


@pytest.mark.parametrize("storage_value", [False, None])
def test_storage_limit_required_zeros_unrented_when_missing_or_false(storage_value):
    ctx = _ctx_with_storage_policy(
        {"network": {"ema_verifyx_download_speed": 120.0}, "storage_limit_supported": storage_value},
        required=True,
    )

    actual_score, job_score, warning = calculate_scores(ctx, rented=False)

    assert actual_score == 0.0
    assert job_score == 0.0
    assert "Storage limit support required" in warning


def test_storage_limit_required_zeros_unrented_when_key_missing():
    ctx = _ctx_with_storage_policy(
        {"network": {"ema_verifyx_download_speed": 120.0}},
        required=True,
    )

    actual_score, job_score, warning = calculate_scores(ctx, rented=False)

    assert actual_score == 0.0
    assert job_score == 0.0
    assert "Storage limit support required" in warning


def test_storage_limit_required_allows_compliant_unrented_executor():
    ctx = _ctx_with_storage_policy(
        {"network": {"ema_verifyx_download_speed": 120.0}, "storage_limit_supported": True},
        required=True,
    )

    actual_score, job_score, warning = calculate_scores(ctx, rented=False)

    assert actual_score == 1.0
    assert job_score == 1.0
    assert warning == ""


def test_storage_limit_disabled_does_not_affect_unrented_score():
    ctx = _ctx_with_storage_policy(
        {"network": {"ema_verifyx_download_speed": 120.0}, "storage_limit_supported": False},
        required=False,
    )

    actual_score, job_score, warning = calculate_scores(ctx, rented=False)

    assert actual_score == 1.0
    assert job_score == 1.0
    assert warning == ""


def test_storage_limit_required_does_not_zero_rented_executor():
    ctx = _ctx_with_storage_policy(
        {"storage_limit_supported": False},
        required=True,
        rented=True,
    )

    actual_score, job_score, warning = calculate_scores(ctx, rented=True)

    assert actual_score == 1.0
    assert job_score == 1.0
    assert "Storage limit support required" not in warning
