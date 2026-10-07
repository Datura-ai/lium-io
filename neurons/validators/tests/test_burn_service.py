import pytest

from constants import TOTAL_BURN_EMISSION
from incentive.burn_service import BurnService

pytest_plugins = ["fixtures.incentive_fixtures"]

UID_47_COLDKEY = "5G694c15wAu1LKi9rpSQqJjpBfg4K1oiBxEm5QSVdVZAfp9f"


@pytest.fixture
def burn_service():
    return BurnService()


def _make_miners(create_neuron_info, uids):
    return [create_neuron_info(uid=uid, hotkey=f"hk_{uid}") for uid in uids]


@pytest.mark.asyncio
async def test_new_logic_distributes_burn_share_equally_across_unique_burners(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """Each unique burner UID should receive an equal slice of burn_share."""
    # Arrange
    monkeypatch.setattr("core.config.settings.NEW_BURNERS", [10, 20, 30, 40])
    miners = _make_miners(create_neuron_info, [10, 20, 30, 40])

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=TOTAL_BURN_EMISSION,
        last_mechanism_step_block=None,
    )

    # Assert — 4 unique burners share burn_share equally, so each gets burn_share / 4
    expected_per_burner = TOTAL_BURN_EMISSION / 4
    assert scores == {
        "hk_10": pytest.approx(expected_per_burner),
        "hk_20": pytest.approx(expected_per_burner),
        "hk_30": pytest.approx(expected_per_burner),
        "hk_40": pytest.approx(expected_per_burner),
    }
    # The full burn_share must be fully distributed (no emission lost)
    assert sum(scores.values()) == pytest.approx(TOTAL_BURN_EMISSION)


@pytest.mark.asyncio
async def test_new_logic_withholds_burn_weight_on_coldkey_mismatch(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """A configured burner UID with the wrong coldkey must not receive burn weight."""
    # Arrange
    monkeypatch.setattr("core.config.settings.NEW_BURNERS", [47, 47, 47])
    monkeypatch.setattr(
        "core.config.settings.BURNER_COLDKEYS",
        {47: UID_47_COLDKEY},
    )
    miners = [create_neuron_info(uid=47, hotkey="hk_47", coldkey="wrong_coldkey")]

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=TOTAL_BURN_EMISSION,
        last_mechanism_step_block=None,
    )

    # Assert — coldkey mismatch withholds all burn weight for UID 47
    assert scores == {}


@pytest.mark.asyncio
async def test_burn_and_mining_shares_partition_total_emission(burn_service):
    """Burn share + mining share must sum to 1.0 (full emission partition)."""
    # Act
    burn = burn_service.get_burn_share()
    mining = burn_service.get_mining_share()

    # Assert — burn comes from the constant, mining is the remainder, and they cover 100%
    assert burn == pytest.approx(TOTAL_BURN_EMISSION)
    assert mining == pytest.approx(1 - TOTAL_BURN_EMISSION)
    assert burn + mining == pytest.approx(1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "slot_count, total_slots, expected_share_ratio",
    [
        (1, 10, 0.10),  # single-slot burner gets 1/10
        (3, 10, 0.30),  # UID 47 owns 3/10 slots → 30%
        (5, 10, 0.50),  # UID 47 owns 5/10 slots → 50%
        (7, 10, 0.70),  # UID 47 owns 7/10 slots → 70%
        (4, 4, 1.00),  # entire pool collapsed to one UID
    ],
)
async def test_new_logic_aggregates_share_proportional_to_slot_count(
    slot_count,
    total_slots,
    expected_share_ratio,
    burn_service,
    monkeypatch,
    mock_settings,
    create_neuron_info,
):
    """A UID listed N times in NEW_BURNERS should receive N/total_slots of burn_share."""
    # Arrange — fill remaining slots with distinct dummy burner UIDs to reach total_slots
    target_uid = 47
    filler_uids = list(range(500, 500 + (total_slots - slot_count)))
    burner_slots = filler_uids + [target_uid] * slot_count
    monkeypatch.setattr("core.config.settings.NEW_BURNERS", burner_slots)
    monkeypatch.setattr("core.config.settings.BURNER_COLDKEYS", {})
    miners = _make_miners(create_neuron_info, [target_uid])

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=TOTAL_BURN_EMISSION,
        last_mechanism_step_block=None,
    )

    # Assert — target UID's score equals its slot fraction of the burn_share
    assert scores[f"hk_{target_uid}"] == pytest.approx(TOTAL_BURN_EMISSION * expected_share_ratio)
