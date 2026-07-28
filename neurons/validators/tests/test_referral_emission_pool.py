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

import logging
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
async def test_pool_equals_min_share_total_and_gains_losses_match(incentive, miners, monkeypatch):
    """Pool = min(share*total, burn_total); referrers' summed gain and burn's summed
    loss both equal the pool (within 1e-9)."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.1)
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}
    incentive.referral_feed = _StubReferralFeed({"referrer_1": 3.0, "referrer_2": 1.0})

    total_before = TOTAL_BURN_EMISSION + 0.08 + 0.05
    burn_total_before = TOTAL_BURN_EMISSION
    expected_pool = min(0.1 * total_before, burn_total_before)
    assert expected_pool < burn_total_before  # sanity: this case is NOT capped

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    burn_loss = burn_total_before - scores[BURN_HOTKEY]
    referrer_gain = scores["referrer_1"] + scores["referrer_2"]

    assert burn_loss == pytest.approx(expected_pool, abs=1e-9)
    assert referrer_gain == pytest.approx(expected_pool, abs=1e-9)
    # EMA-proportional split: referrer_1 (EMA 3.0) gets 3x referrer_2 (EMA 1.0).
    assert scores["referrer_1"] == pytest.approx(3 * scores["referrer_2"])


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
async def test_fail_closed_empty_feed(incentive, miners, monkeypatch):
    """An empty referral feed leaves cycle_scores equal to the no-referral baseline."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.2)
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}
    incentive.referral_feed = _StubReferralFeed({})

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert scores["miner_a"] == pytest.approx(0.08)
    assert scores["miner_b"] == pytest.approx(0.05)
    assert scores[BURN_HOTKEY] == pytest.approx(TOTAL_BURN_EMISSION)
    # referrer_1/2 are registered miners with no mining/rental incentive of their own,
    # so they're present (via the base fill-in loop) but exactly zero — not enriched.
    assert scores["referrer_1"] == pytest.approx(0.0)
    assert scores["referrer_2"] == pytest.approx(0.0)


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


@pytest.mark.asyncio
async def test_eligibility_filters_drop_ineligible_referrers(incentive, miners, monkeypatch, caplog):
    """A feed hotkey not in this cycle's miners, with non-positive EMA, or that is a
    burn hotkey contributes nothing to the pool — verified directly on
    ``_apply_referral_pool`` with a hand-built feed exercising all three cases."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.5)

    burn_scores = {BURN_HOTKEY: TOTAL_BURN_EMISSION}
    cycle_scores = dict(burn_scores)
    cycle_scores["miner_a"] = 0.08
    cycle_scores["miner_b"] = 0.05

    incentive.referral_feed = _StubReferralFeed(
        {
            "not_a_miner_this_cycle": 5.0,  # not in `miners` -> ineligible
            "referrer_1": 0.0,  # non-positive EMA -> ineligible
            "referrer_2": -1.0,  # negative EMA -> ineligible
            BURN_HOTKEY: 10.0,  # a burn hotkey -> ineligible even with EMA
        }
    )

    before = dict(cycle_scores)
    with caplog.at_level(logging.WARNING):
        await incentive._apply_referral_pool(cycle_scores, burn_scores, miners, current_epoch=None)

    # No eligible referrer -> total_ema <= 0 -> the whole step is a no-op.
    assert cycle_scores == before
    # ...but not a SILENT one: the feed named 4 referrers and every one was dropped, which
    # is a backend/validator disagreement an operator needs to see.
    assert "no eligible referrer" in caplog.text


@pytest.mark.asyncio
async def test_eligibility_filters_only_eligible_referrer_gets_pool(incentive, miners, monkeypatch):
    """When the feed mixes eligible and ineligible hotkeys, only the eligible one
    (registered this cycle, positive EMA, not a burn hotkey) receives the pool."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.1)

    burn_scores = {BURN_HOTKEY: TOTAL_BURN_EMISSION}
    cycle_scores = dict(burn_scores)
    cycle_scores["miner_a"] = 0.08
    cycle_scores["miner_b"] = 0.05
    total_before = sum(cycle_scores.values())

    incentive.referral_feed = _StubReferralFeed(
        {
            "referrer_1": 2.0,  # eligible
            "not_a_miner_this_cycle": 100.0,  # ineligible: absent from `miners`
            "referrer_2": 0.0,  # ineligible: zero EMA
            BURN_HOTKEY: 50.0,  # ineligible: burn hotkey
        }
    )

    await incentive._apply_referral_pool(cycle_scores, burn_scores, miners, current_epoch=None)

    expected_pool = min(0.1 * total_before, TOTAL_BURN_EMISSION)
    assert cycle_scores["referrer_1"] == pytest.approx(expected_pool, abs=1e-9)
    assert "referrer_2" not in cycle_scores
    assert "not_a_miner_this_cycle" not in cycle_scores
    assert cycle_scores["miner_a"] == pytest.approx(0.08)
    assert cycle_scores["miner_b"] == pytest.approx(0.05)
    assert cycle_scores[BURN_HOTKEY] == pytest.approx(TOTAL_BURN_EMISSION - expected_pool, abs=1e-9)


@pytest.mark.asyncio
async def test_no_op_when_burn_total_is_zero(incentive, miners, monkeypatch):
    """No residual burn to draw from -> the referral step is a no-op even with a
    positive share and eligible referrers (miners are never touched, and there is
    nothing to redirect since the pool is capped at burn_total == 0)."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.5)

    burn_scores: dict[str, float] = {}
    cycle_scores = {"miner_a": 0.08, "miner_b": 0.05}
    before = dict(cycle_scores)

    incentive.referral_feed = _StubReferralFeed({"referrer_1": 2.0})
    await incentive._apply_referral_pool(cycle_scores, burn_scores, miners, current_epoch=None)

    assert cycle_scores == before
    assert "referrer_1" not in cycle_scores
