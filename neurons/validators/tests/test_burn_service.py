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
async def test_new_logic_excludes_miners_that_are_not_burners(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """Non-burner miners must not appear in the resulting score map."""
    # Arrange — UIDs 200 and 300 are regular miners, not burners
    monkeypatch.setattr("core.config.settings.NEW_BURNERS", [100, 101])
    miners = _make_miners(create_neuron_info, [100, 101, 200, 300])

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=TOTAL_BURN_EMISSION,
        last_mechanism_step_block=None,
    )

    # Assert — only the two burner hotkeys should be scored
    assert set(scores.keys()) == {"hk_100", "hk_101"}


@pytest.mark.asyncio
async def test_new_logic_with_zero_burn_share_yields_zero_scores(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """A burn_share of 0 should still include burners, but with score 0."""
    # Arrange
    monkeypatch.setattr("core.config.settings.NEW_BURNERS", [100, 101])
    miners = _make_miners(create_neuron_info, [100, 101])

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=0.0,
        last_mechanism_step_block=None,
    )

    # Assert — there is nothing to distribute, so every burner gets exactly 0
    assert scores == {"hk_100": pytest.approx(0.0), "hk_101": pytest.approx(0.0)}


@pytest.mark.asyncio
async def test_new_logic_distributes_partial_burn_share(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """A burn_share smaller than TOTAL_BURN_EMISSION must still be fully distributed."""
    # Arrange — rental incentive flow may reduce burn_share below the constant
    monkeypatch.setattr("core.config.settings.NEW_BURNERS", [100, 101])
    miners = _make_miners(create_neuron_info, [100, 101])

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=0.5,
        last_mechanism_step_block=None,
    )

    # Assert — entire 0.5 burn_share split equally between the two burners
    assert sum(scores.values()) == pytest.approx(0.5)
    assert scores["hk_100"] == pytest.approx(0.25)
    assert scores["hk_101"] == pytest.approx(0.25)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "slot_count, total_slots, expected_share_ratio",
    [
        (1, 10, 0.10),  # single-slot burner gets 1/10
        (3, 10, 0.30),  # DAH-2089: UID 47 owns 3/10 slots → 30%
        (5, 10, 0.50),  # DAH-2125: UID 47 owns 5/10 slots → 50%
        (7, 10, 0.70),  # DAH-2362: UID 47 owns 7/10 slots → 70%
        (4, 4, 1.00),   # entire pool collapsed to one UID
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
    assert scores[f"hk_{target_uid}"] == pytest.approx(
        TOTAL_BURN_EMISSION * expected_share_ratio
    )


@pytest.mark.asyncio
async def test_new_logic_dah_2089_uid_47_absorbs_retired_burner_shares(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """DAH-2089 invariant: UID 47 holds 3 slots and receives exactly 30% of burn_share,
    while the 7 remaining burners keep 10% each."""
    # Arrange — DAH-2089 historical layout
    monkeypatch.setattr(
        "core.config.settings.NEW_BURNERS",
        [187, 188, 189, 190, 191, 192, 193, 47, 47, 47],
    )
    monkeypatch.setattr("core.config.settings.BURNER_COLDKEYS", {})
    miners = _make_miners(
        create_neuron_info, [187, 188, 189, 190, 191, 192, 193, 47]
    )

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=TOTAL_BURN_EMISSION,
        last_mechanism_step_block=None,
    )

    # Assert — UID 47 (3 slots) gets 30%, other 7 burners get 10% each, total = burn_share
    per_slot = TOTAL_BURN_EMISSION / 10
    assert scores["hk_47"] == pytest.approx(per_slot * 3)
    for uid in (187, 188, 189, 190, 191, 192, 193):
        assert scores[f"hk_{uid}"] == pytest.approx(per_slot)
    assert sum(scores.values()) == pytest.approx(TOTAL_BURN_EMISSION)


@pytest.mark.asyncio
async def test_new_logic_all_burn_slots_route_to_uid_47(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """Production layout: all 10 burn slots route 100% of burn_share to UID 47."""
    # Arrange
    monkeypatch.setattr(
        "core.config.settings.NEW_BURNERS",
        [47, 47, 47, 47, 47, 47, 47, 47, 47, 47],
    )
    monkeypatch.setattr(
        "core.config.settings.BURNER_COLDKEYS",
        {47: UID_47_COLDKEY},
    )
    miners = [
        create_neuron_info(uid=47, hotkey="hk_47", coldkey=UID_47_COLDKEY),
        create_neuron_info(uid=188, hotkey="hk_188", coldkey="wrong_coldkey"),
    ]

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=TOTAL_BURN_EMISSION,
        last_mechanism_step_block=None,
    )

    # Assert — only verified UID 47 receives the full burn_share
    assert scores == {"hk_47": pytest.approx(TOTAL_BURN_EMISSION)}
    assert "hk_188" not in scores


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
async def test_new_logic_dah_2362_production_layout_gives_uid_47_seventy_percent(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """Historical DAH-2362 layout: UID 47 holds 7 slots and receives 70% of burn_share."""
    # Arrange — superseded layout kept for regression coverage
    monkeypatch.setattr(
        "core.config.settings.NEW_BURNERS",
        [187, 188, 189, 47, 47, 47, 47, 47, 47, 47],
    )
    monkeypatch.setattr("core.config.settings.BURNER_COLDKEYS", {})
    miners = _make_miners(create_neuron_info, [187, 188, 189, 47])

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=TOTAL_BURN_EMISSION,
        last_mechanism_step_block=None,
    )

    # Assert — UID 47 (7 slots) gets 70%, other 3 burners get 10% each, total = burn_share
    per_slot = TOTAL_BURN_EMISSION / 10
    assert scores["hk_47"] == pytest.approx(per_slot * 7)
    for uid in (187, 188, 189):
        assert scores[f"hk_{uid}"] == pytest.approx(per_slot)
    assert sum(scores.values()) == pytest.approx(TOTAL_BURN_EMISSION)


@pytest.mark.asyncio
async def test_new_logic_skips_burner_uids_that_are_not_in_miner_list(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """A burner UID absent from the metagraph snapshot should not appear in the score map."""
    # Arrange — only UIDs 187 and 47 are actually present as miners
    monkeypatch.setattr(
        "core.config.settings.NEW_BURNERS", [187, 188, 189, 47, 47, 47]
    )
    monkeypatch.setattr("core.config.settings.BURNER_COLDKEYS", {})
    miners = _make_miners(create_neuron_info, [187, 47])

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=TOTAL_BURN_EMISSION,
        last_mechanism_step_block=None,
    )

    # Assert — missing burners 188, 189 are dropped; UID 47 still claims its 3 slots
    per_slot = TOTAL_BURN_EMISSION / 6
    assert scores == {
        "hk_187": pytest.approx(per_slot),
        "hk_47": pytest.approx(per_slot * 3),
    }


@pytest.mark.asyncio
async def test_old_logic_distributes_full_burn_share_across_burners(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """Old (main-burner) logic must still fully distribute burn_share when enabled."""
    # Arrange — disable new logic so the legacy main-burner path runs
    monkeypatch.setattr("core.config.settings.ENABLE_NEW_BURN_LOGIC", False)
    monkeypatch.setattr("core.config.settings.BURNERS", [4, 206, 207, 208])
    miners = _make_miners(create_neuron_info, [4, 206, 207, 208])

    # Act
    scores = burn_service.calculate_burn_scores(
        miners=miners,
        burn_share=TOTAL_BURN_EMISSION,
        last_mechanism_step_block=42,
    )

    # Assert — every legacy burner is scored and the totals add up to burn_share
    assert set(scores.keys()) == {"hk_4", "hk_206", "hk_207", "hk_208"}
    assert sum(scores.values()) == pytest.approx(TOTAL_BURN_EMISSION)


@pytest.mark.asyncio
async def test_old_logic_main_burner_selection_is_deterministic_per_block(
    burn_service, monkeypatch, mock_settings, create_neuron_info
):
    """Same block seed → identical score map (random.Random is seeded by block)."""
    # Arrange
    monkeypatch.setattr("core.config.settings.ENABLE_NEW_BURN_LOGIC", False)
    monkeypatch.setattr("core.config.settings.BURNERS", [4, 206, 207, 208])
    miners = _make_miners(create_neuron_info, [4, 206, 207, 208])

    # Act — call twice with the same block seed
    first = burn_service.calculate_burn_scores(
        miners=miners, burn_share=TOTAL_BURN_EMISSION, last_mechanism_step_block=42
    )
    second = burn_service.calculate_burn_scores(
        miners=miners, burn_share=TOTAL_BURN_EMISSION, last_mechanism_step_block=42
    )

    # Assert — deterministic seeding must produce the exact same distribution
    assert first == second


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "uid, expected",
    [
        (47, True),    # in NEW_BURNERS (3 slots)
        (187, True),   # in NEW_BURNERS (1 slot)
        (999, False),  # regular miner, not a burner
    ],
)
async def test_is_burner_under_new_logic(
    uid, expected, burn_service, monkeypatch, mock_settings
):
    # Arrange — DAH-2089 historical layout
    monkeypatch.setattr(
        "core.config.settings.NEW_BURNERS",
        [187, 188, 189, 190, 191, 192, 193, 47, 47, 47],
    )
    monkeypatch.setattr("core.config.settings.BURNER_COLDKEYS", {})

    # Act / Assert — membership matches the configured NEW_BURNERS list
    assert burn_service.is_burner(uid) is expected


@pytest.mark.asyncio
async def test_is_burner_under_old_logic_uses_legacy_burner_list(
    burn_service, monkeypatch, mock_settings
):
    """When new logic is disabled, is_burner must consult BURNERS, not NEW_BURNERS."""
    # Arrange
    monkeypatch.setattr("core.config.settings.ENABLE_NEW_BURN_LOGIC", False)
    monkeypatch.setattr("core.config.settings.BURNERS", [4, 206, 207, 208])
    monkeypatch.setattr("core.config.settings.NEW_BURNERS", [47])

    # Assert — 4 is a legacy burner; 47 belongs to the (disabled) new list and must not match
    assert burn_service.is_burner(4) is True
    assert burn_service.is_burner(47) is False


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
