import pytest

from incentive.burn_service import BurnService
from services.const import TOTAL_BURN_EMISSION

pytest_plugins = ["fixtures.incentive_fixtures"]


@pytest.fixture
def burn_service():
    return BurnService()


def _make_miners(create_neuron_info, uids):
    return [create_neuron_info(uid=uid, hotkey=f"hk_{uid}") for uid in uids]


class TestNewBurnLogicEqualDistribution:
    def test_equal_share_across_unique_burners(
        self, burn_service, monkeypatch, mock_settings, create_neuron_info
    ):
        monkeypatch.setattr("core.config.settings.NEW_BURNERS", [10, 20, 30, 40])
        miners = _make_miners(create_neuron_info, [10, 20, 30, 40])

        scores = burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=TOTAL_BURN_EMISSION,
            last_mechanism_step_block=None,
        )

        expected = TOTAL_BURN_EMISSION / 4
        assert scores == {
            "hk_10": pytest.approx(expected),
            "hk_20": pytest.approx(expected),
            "hk_30": pytest.approx(expected),
            "hk_40": pytest.approx(expected),
        }
        assert sum(scores.values()) == pytest.approx(TOTAL_BURN_EMISSION)

    def test_non_burner_miners_excluded(
        self, burn_service, monkeypatch, mock_settings, create_neuron_info
    ):
        monkeypatch.setattr("core.config.settings.NEW_BURNERS", [100, 101])
        miners = _make_miners(create_neuron_info, [100, 101, 200, 300])

        scores = burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=TOTAL_BURN_EMISSION,
            last_mechanism_step_block=None,
        )

        assert set(scores.keys()) == {"hk_100", "hk_101"}

    def test_zero_burn_share_yields_zero_scores(
        self, burn_service, monkeypatch, mock_settings, create_neuron_info
    ):
        monkeypatch.setattr("core.config.settings.NEW_BURNERS", [100, 101])
        miners = _make_miners(create_neuron_info, [100, 101])

        scores = burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=0.0,
            last_mechanism_step_block=None,
        )

        assert scores == {"hk_100": pytest.approx(0.0), "hk_101": pytest.approx(0.0)}

    def test_custom_burn_share_partial_distribution(
        self, burn_service, monkeypatch, mock_settings, create_neuron_info
    ):
        monkeypatch.setattr("core.config.settings.NEW_BURNERS", [100, 101])
        miners = _make_miners(create_neuron_info, [100, 101])

        scores = burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=0.5,
            last_mechanism_step_block=None,
        )

        assert sum(scores.values()) == pytest.approx(0.5)
        assert scores["hk_100"] == pytest.approx(0.25)
        assert scores["hk_101"] == pytest.approx(0.25)


class TestNewBurnLogicSlotWeighted:
    def test_duplicate_uid_accumulates_proportional_share(
        self, burn_service, monkeypatch, mock_settings, create_neuron_info
    ):
        """UID 47 occupies 3 of 10 slots → 30% of burn_share."""
        monkeypatch.setattr(
            "core.config.settings.NEW_BURNERS",
            [187, 188, 189, 190, 191, 192, 193, 47, 47, 47],
        )
        miners = _make_miners(
            create_neuron_info, [187, 188, 189, 190, 191, 192, 193, 47]
        )

        scores = burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=TOTAL_BURN_EMISSION,
            last_mechanism_step_block=None,
        )

        per_slot = TOTAL_BURN_EMISSION / 10
        assert scores["hk_47"] == pytest.approx(per_slot * 3)
        for uid in (187, 188, 189, 190, 191, 192, 193):
            assert scores[f"hk_{uid}"] == pytest.approx(per_slot)
        assert sum(scores.values()) == pytest.approx(TOTAL_BURN_EMISSION)

    def test_uid_47_receives_30_percent_of_burn_share(
        self, burn_service, monkeypatch, mock_settings, create_neuron_info
    ):
        monkeypatch.setattr(
            "core.config.settings.NEW_BURNERS",
            [187, 188, 189, 190, 191, 192, 193, 47, 47, 47],
        )
        miners = _make_miners(create_neuron_info, [47])

        scores = burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=TOTAL_BURN_EMISSION,
            last_mechanism_step_block=None,
        )

        assert scores["hk_47"] == pytest.approx(TOTAL_BURN_EMISSION * 0.3)

    def test_all_slots_owned_by_single_uid(
        self, burn_service, monkeypatch, mock_settings, create_neuron_info
    ):
        monkeypatch.setattr("core.config.settings.NEW_BURNERS", [47, 47, 47, 47])
        miners = _make_miners(create_neuron_info, [47])

        scores = burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=TOTAL_BURN_EMISSION,
            last_mechanism_step_block=None,
        )

        assert scores == {"hk_47": pytest.approx(TOTAL_BURN_EMISSION)}

    def test_missing_burner_miner_not_in_scores(
        self, burn_service, monkeypatch, mock_settings, create_neuron_info
    ):
        """A burner UID absent from the miner list contributes no score entry."""
        monkeypatch.setattr(
            "core.config.settings.NEW_BURNERS", [187, 188, 189, 47, 47, 47]
        )
        miners = _make_miners(create_neuron_info, [187, 47])

        scores = burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=TOTAL_BURN_EMISSION,
            last_mechanism_step_block=None,
        )

        per_slot = TOTAL_BURN_EMISSION / 6
        assert scores == {
            "hk_187": pytest.approx(per_slot),
            "hk_47": pytest.approx(per_slot * 3),
        }


class TestOldBurnLogic:
    def test_old_logic_distributes_main_and_other_burners(
        self, burn_service, monkeypatch, mock_settings, create_neuron_info
    ):
        monkeypatch.setattr("core.config.settings.ENABLE_NEW_BURN_LOGIC", False)
        monkeypatch.setattr("core.config.settings.BURNERS", [4, 206, 207, 208])
        miners = _make_miners(create_neuron_info, [4, 206, 207, 208])

        scores = burn_service.calculate_burn_scores(
            miners=miners,
            burn_share=TOTAL_BURN_EMISSION,
            last_mechanism_step_block=42,
        )

        assert set(scores.keys()) == {"hk_4", "hk_206", "hk_207", "hk_208"}
        assert sum(scores.values()) == pytest.approx(TOTAL_BURN_EMISSION)

    def test_old_logic_main_burner_is_deterministic_per_block(
        self, burn_service, monkeypatch, mock_settings, create_neuron_info
    ):
        monkeypatch.setattr("core.config.settings.ENABLE_NEW_BURN_LOGIC", False)
        monkeypatch.setattr("core.config.settings.BURNERS", [4, 206, 207, 208])
        miners = _make_miners(create_neuron_info, [4, 206, 207, 208])

        first = burn_service.calculate_burn_scores(
            miners=miners, burn_share=TOTAL_BURN_EMISSION, last_mechanism_step_block=42
        )
        second = burn_service.calculate_burn_scores(
            miners=miners, burn_share=TOTAL_BURN_EMISSION, last_mechanism_step_block=42
        )

        assert first == second


class TestBurnServiceHelpers:
    def test_is_burner_new_logic(self, burn_service, monkeypatch, mock_settings):
        monkeypatch.setattr(
            "core.config.settings.NEW_BURNERS",
            [187, 188, 189, 190, 191, 192, 193, 47, 47, 47],
        )

        assert burn_service.is_burner(47) is True
        assert burn_service.is_burner(187) is True
        assert burn_service.is_burner(999) is False

    def test_is_burner_old_logic(self, burn_service, monkeypatch, mock_settings):
        monkeypatch.setattr("core.config.settings.ENABLE_NEW_BURN_LOGIC", False)
        monkeypatch.setattr("core.config.settings.BURNERS", [4, 206, 207, 208])
        monkeypatch.setattr("core.config.settings.NEW_BURNERS", [47])

        assert burn_service.is_burner(4) is True
        assert burn_service.is_burner(47) is False

    def test_get_burn_share(self, burn_service):
        assert burn_service.get_burn_share() == pytest.approx(TOTAL_BURN_EMISSION)

    def test_get_mining_share(self, burn_service):
        assert burn_service.get_mining_share() == pytest.approx(1 - TOTAL_BURN_EMISSION)

    def test_burn_and_mining_shares_sum_to_one(self, burn_service):
        assert (
            burn_service.get_burn_share() + burn_service.get_mining_share()
            == pytest.approx(1.0)
        )
