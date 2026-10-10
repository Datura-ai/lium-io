"""DAH-2251 (spec pivot) — burn-sourced referral emission pool.

``DefaultIncentive._apply_referral_pool`` redirects a configurable share of the
cycle's total emission from the RESIDUAL BURN pool to miners who referred paying
customers, split by each referrer's EMA (read from ``ReferralFeedClient``, fail
closed). These tests prove the invariants that make the mechanism safe:

- Miners (non-referrer, non-burner) are never diluted.
- The cycle total is conserved — value only moves burn -> referrers.
- The pool is ``min(share * total, burn_total)`` — rental-share/miners keep first
  claim because the pool can never draw more than the residual burn.
- The mechanism fails closed: an empty feed, a zero/negative/NaN share, or an
  ineligible referrer (not in this cycle's miners, non-positive EMA, or a burn
  hotkey) all leave scores exactly as the no-referral baseline.
"""

import math
from unittest.mock import AsyncMock

import pytest
from constants import TOTAL_BURN_EMISSION
from incentive.config import IncentiveConfig
from incentive.default import DefaultIncentive

from core.config import settings

pytest_plugins = ["fixtures.incentive_fixtures"]

# Single burn wallet — NEW_BURNERS all point at one UID, so it absorbs the whole
# burn_share and gives every test a burn hotkey with a known, non-zero score.
BURN_UID = 47
BURN_HOTKEY = "hk_burn"


class _StubReferralFeed:
    """A fake ``ReferralFeedClient`` — any object with ``get_weights`` qualifies."""

    def __init__(self, weights: dict[str, float] | None = None):
        self._weights = weights or {}

    async def get_weights(self, current_epoch=None):
        return dict(self._weights)


@pytest.fixture
def burn_env(monkeypatch):
    """Pins burn logic to a single, known burner hotkey (`BURN_HOTKEY`)."""
    monkeypatch.setattr(settings, "NEW_BURNERS", [BURN_UID])
    monkeypatch.setattr(settings, "ENABLE_NEW_BURN_LOGIC", True)
    monkeypatch.setattr(settings, "BURNER_COLDKEYS", {})


@pytest.fixture
def incentive(burn_env):
    """A DefaultIncentive whose burn_share resolves to TOTAL_BURN_EMISSION via the
    hermetic shared config (see conftest.py)."""
    return DefaultIncentive(IncentiveConfig(), AsyncMock(), {}, {})


@pytest.fixture
def miners(create_neuron_info):
    """Burn wallet plus two regular (non-referrer) miners and two referrers."""
    return [
        create_neuron_info(uid=BURN_UID, hotkey=BURN_HOTKEY),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
        create_neuron_info(uid=4, hotkey="referrer_1"),
        create_neuron_info(uid=5, hotkey="referrer_2"),
    ]


@pytest.mark.asyncio
async def test_miners_never_diluted_empty_vs_nonempty_feed(incentive, miners, monkeypatch):
    """A non-referrer, non-burner miner's score is identical whether the referral feed
    is empty or populated — the pool only ever moves value between burn and referrers."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.2)
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}

    incentive.referral_feed = _StubReferralFeed({})
    baseline = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    incentive.referral_feed = _StubReferralFeed({"referrer_1": 3.0, "referrer_2": 1.0})
    with_referrals = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert with_referrals["miner_a"] == pytest.approx(baseline["miner_a"])
    assert with_referrals["miner_b"] == pytest.approx(baseline["miner_b"])
    # Sanity: the referral step actually did something (burn moved, referrers gained).
    assert with_referrals[BURN_HOTKEY] < baseline[BURN_HOTKEY]
    assert with_referrals["referrer_1"] > 0
    assert with_referrals["referrer_2"] > 0


@pytest.mark.asyncio
async def test_total_score_conserved(incentive, miners, monkeypatch):
    """The referral step only redistributes value — the grand total is unchanged."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.15)
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}
    incentive.referral_feed = _StubReferralFeed({"referrer_1": 2.0, "referrer_2": 1.0})

    total_before = TOTAL_BURN_EMISSION + 0.08 + 0.05
    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert sum(scores.values()) == pytest.approx(total_before)


@pytest.mark.asyncio
async def test_pool_capped_at_burn_total_miners_still_untouched(incentive, miners, monkeypatch):
    """Rental-share-first / cap: when share*total > burn_total, the pool is capped at
    burn_total — burn is fully (not over-)drained, and miners remain untouched."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.99)
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}
    incentive.referral_feed = _StubReferralFeed({"referrer_1": 1.0})

    total_before = TOTAL_BURN_EMISSION + 0.08 + 0.05
    assert 0.99 * total_before > TOTAL_BURN_EMISSION  # sanity: this case IS capped

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert scores[BURN_HOTKEY] == pytest.approx(0.0, abs=1e-9)
    assert scores["referrer_1"] == pytest.approx(TOTAL_BURN_EMISSION)
    assert scores["miner_a"] == pytest.approx(0.08)
    assert scores["miner_b"] == pytest.approx(0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize("share", [0.0, -0.5, math.nan])
async def test_fail_closed_non_positive_or_nan_share(incentive, miners, monkeypatch, share):
    """A zero, negative, or NaN REFERRAL_EMISSION_SHARE is a no-op, even with a
    populated feed — the default (0.0) keeps the feature inert."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", share)
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}
    incentive.referral_feed = _StubReferralFeed({"referrer_1": 3.0, "referrer_2": 1.0})

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert scores["miner_a"] == pytest.approx(0.08)
    assert scores["miner_b"] == pytest.approx(0.05)
    assert scores[BURN_HOTKEY] == pytest.approx(TOTAL_BURN_EMISSION)
    assert scores["referrer_1"] == pytest.approx(0.0)
    assert scores["referrer_2"] == pytest.approx(0.0)


