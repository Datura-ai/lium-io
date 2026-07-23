"""DAH-2481 — referral emission carve-out.

A fixed, configurable fraction of the miner (non-burn) emission is redirected to a
DEDICATED referral funding wallet — a separate hotkey registered on SN51 under its
own coldkey, distinct from the burn UID 47 — so the referral emission is
distinguishable on-chain from burn. These tests exercise the single merge point,
``DefaultIncentive.calculate_final_weights`` (inherited by RentalPriceIncentive),
where burn scores and per-miner incentives are combined and the carve-out is applied.
"""

from unittest.mock import AsyncMock

import pytest
from constants import TOTAL_BURN_EMISSION
from incentive.config import IncentiveConfig
from incentive.default import DefaultIncentive
from incentive.rental_price import RentalPriceIncentive

from core.config import settings

pytest_plugins = ["fixtures.incentive_fixtures"]

# Burn wallet — UID 47, guarded by BURNER_COLDKEYS (unchanged from production).
BURN_UID = 47
BURN_HOTKEY = "hk_47"
BURN_COLDKEY = "5G694c15wAu1LKi9rpSQqJjpBfg4K1oiBxEm5QSVdVZAfp9f"

# Referral funding wallet — a DISTINCT UID on SN51 with its own coldkey.
REFERRAL_UID = 5
REFERRAL_HOTKEY = "hk_referral"
REFERRAL_COLDKEY = "5RefFund1ngWa11etCo1dkeyDedicatedForReferra1Payouts00"


@pytest.fixture
def incentive():
    """A DefaultIncentive whose burn_share is pinned to the production value.

    conftest serves TOTAL_BURN_EMISSION into the hermetic shared config, so
    __init__ resolves ``self.burn_share`` to it.
    """
    return DefaultIncentive(IncentiveConfig(), AsyncMock(), {}, {})


@pytest.fixture
def referral_env(monkeypatch):
    """Burn wallet at UID 47; referral funding wallet at a distinct UID 5 with its own
    coldkey. Referral carve-out is enabled by configuring UID + coldkey (share set
    per-test)."""
    monkeypatch.setattr(settings, "NEW_BURNERS", [BURN_UID] * 10)
    monkeypatch.setattr(settings, "ENABLE_NEW_BURN_LOGIC", True)
    monkeypatch.setattr(settings, "BURNER_COLDKEYS", {BURN_UID: BURN_COLDKEY})
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_UID", REFERRAL_UID)
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_COLDKEY", REFERRAL_COLDKEY)


@pytest.fixture
def miners(referral_env, create_neuron_info):
    """Burn wallet, referral funding wallet, and two regular miners."""
    return [
        create_neuron_info(uid=BURN_UID, hotkey=BURN_HOTKEY, coldkey=BURN_COLDKEY),
        create_neuron_info(uid=REFERRAL_UID, hotkey=REFERRAL_HOTKEY, coldkey=REFERRAL_COLDKEY),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]


@pytest.mark.asyncio
async def test_carveout_redirects_fixed_share_to_dedicated_wallet(incentive, miners, monkeypatch):
    """A 10% share moves 10% of every miner's emission onto the referral wallet, which is
    separate from (not added to) the burn wallet."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.1)
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    # Each miner keeps 90% of its emission.
    assert scores["miner_a"] == pytest.approx(0.08 * 0.9)
    assert scores["miner_b"] == pytest.approx(0.05 * 0.9)
    # The referral wallet receives exactly the carved 10% — distinguishable from burn.
    assert scores[REFERRAL_HOTKEY] == pytest.approx(0.1 * (0.08 + 0.05))
    # The burn wallet is untouched (still its full burn share).
    assert scores[BURN_HOTKEY] == pytest.approx(TOTAL_BURN_EMISSION)


@pytest.mark.asyncio
async def test_carveout_conserves_total_emission(incentive, miners, monkeypatch):
    """The carve-out only redistributes: the grand total is unchanged."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.25)
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}
    total_before = TOTAL_BURN_EMISSION + 0.08 + 0.05

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert sum(scores.values()) == pytest.approx(total_before)


@pytest.mark.asyncio
async def test_default_share_is_noop(incentive, miners, monkeypatch):
    """The default 0.0 share redirects nothing — miners, burn, and referral untouched."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.0)
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert scores["miner_a"] == pytest.approx(0.08)
    assert scores["miner_b"] == pytest.approx(0.05)
    assert scores[BURN_HOTKEY] == pytest.approx(TOTAL_BURN_EMISSION)
    assert scores[REFERRAL_HOTKEY] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_share_is_clamped_and_never_drives_miners_negative(incentive, miners, monkeypatch):
    """A misconfigured share > 1 is clamped to 1.0: miners drop to 0, never negative,
    and the whole miner emission moves to the referral wallet."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 1.5)
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert scores["miner_a"] == pytest.approx(0.0)
    assert scores["miner_b"] == pytest.approx(0.0)
    assert scores["miner_a"] >= 0.0 and scores["miner_b"] >= 0.0
    assert scores[REFERRAL_HOTKEY] == pytest.approx(0.08 + 0.05)
    assert scores[BURN_HOTKEY] == pytest.approx(TOTAL_BURN_EMISSION)


@pytest.mark.asyncio
async def test_carveout_withheld_on_coldkey_mismatch(incentive, referral_env, monkeypatch, create_neuron_info):
    """A referral wallet whose on-chain coldkey does not match the configured one gets no
    redirect — miner emission is left fully intact (fail closed)."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.1)
    miners = [
        create_neuron_info(uid=BURN_UID, hotkey=BURN_HOTKEY, coldkey=BURN_COLDKEY),
        create_neuron_info(uid=REFERRAL_UID, hotkey=REFERRAL_HOTKEY, coldkey="wrong_coldkey"),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert scores["miner_a"] == pytest.approx(0.08)
    assert scores["miner_b"] == pytest.approx(0.05)
    assert scores[REFERRAL_HOTKEY] == pytest.approx(0.0)
    assert scores[BURN_HOTKEY] == pytest.approx(TOTAL_BURN_EMISSION)


@pytest.mark.asyncio
async def test_carveout_withheld_when_coldkey_unset(incentive, miners, monkeypatch):
    """An unset (empty) REFERRAL_EMISSION_COLDKEY fails closed — nothing is redirected
    even though the UID is present, so the mechanism is inert until ops configures the
    wallet's coldkey."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.1)
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_COLDKEY", "")
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert scores["miner_a"] == pytest.approx(0.08)
    assert scores["miner_b"] == pytest.approx(0.05)
    assert scores[REFERRAL_HOTKEY] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_carveout_skipped_when_funding_uid_absent(incentive, referral_env, monkeypatch, create_neuron_info):
    """If the referral UID is not in this cycle's metagraph, miners keep everything."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.1)
    # No referral neuron (UID 5) in the set.
    miners = [
        create_neuron_info(uid=BURN_UID, hotkey=BURN_HOTKEY, coldkey=BURN_COLDKEY),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey="miner_b"),
    ]
    incentive.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}

    scores = await incentive.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert scores["miner_a"] == pytest.approx(0.08)
    assert scores["miner_b"] == pytest.approx(0.05)
    assert REFERRAL_HOTKEY not in scores


@pytest.mark.asyncio
async def test_carveout_applies_on_rental_price_incentive(miners, monkeypatch):
    """The production RentalPriceIncentive inherits the carve-out unchanged: its merged
    miner incentives (mining + rental) are carved just like DefaultIncentive's."""
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.1)
    rental = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})
    # A real cycle sets these in _on_finish_pre_process / _post_process; seed them directly.
    rental.burn_share = TOTAL_BURN_EMISSION
    rental.miner_incentives = {"miner_a": 0.08, "miner_b": 0.05}

    scores = await rental.calculate_final_weights(miners, last_mechanism_step_block=None)

    assert scores["miner_a"] == pytest.approx(0.08 * 0.9)
    assert scores["miner_b"] == pytest.approx(0.05 * 0.9)
    assert scores[REFERRAL_HOTKEY] == pytest.approx(0.1 * (0.08 + 0.05))
    assert scores[BURN_HOTKEY] == pytest.approx(TOTAL_BURN_EMISSION)
