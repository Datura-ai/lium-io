"""SCORING_EXCLUDED_HOTKEYS: listed hotkeys earn no subnet incentive; rentals are untouched.

The setting is a comma-separated list. Every executor of a listed hotkey is still validated and
published as before, but scores 0 in every pool (mining, unrented, spot pay) with a recorded
reason, is never a referral recipient, and any score left on it at final weights moves to burn.
"""

from unittest.mock import AsyncMock

import pytest
from constants import TOTAL_BURN_EMISSION
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.default import DefaultIncentive
from incentive.miner_incentive_log import ZeroIncentiveReason
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult
from tests.test_referral_emission_pool import BURN_HOTKEY, BURN_UID, _StubReferralFeed
from tests.test_rental_price_incentive_flow import _make_pcc_config, _make_pcc_job, _run_pcc_incentive

from core.config import settings

pytest_plugins = ["fixtures.incentive_fixtures"]

EXCLUDED = "hk_excluded"
B200 = "NVIDIA B200"


def _codes(result: JobResult) -> list[str]:
    return [reason.reason for reason in result.zero_incentive_reasons]


@pytest.fixture
def exclude(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_EXCLUDED_HOTKEYS", f" {EXCLUDED} , ,other_hk")


def test_setting_parses_a_comma_separated_list(exclude):
    assert settings.get_scoring_excluded_hotkeys() == {EXCLUDED, "other_hk"}


def test_setting_is_empty_by_default(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_EXCLUDED_HOTKEYS", "")
    assert settings.get_scoring_excluded_hotkeys() == set()


@pytest.mark.asyncio
async def test_idle_and_rented_executors_of_an_excluded_hotkey_earn_nothing(exclude, monkeypatch):
    jobs = {
        EXCLUDED: [
            _make_pcc_job("exec-idle", B200, 1),
            _make_pcc_job("exec-rented", B200, 1, is_rented=True),
        ],
        "miner_b": [
            _make_pcc_job("exec-b-idle", B200, 1),
            _make_pcc_job("exec-b-rented", B200, 1, is_rented=True),
        ],
    }

    incentive = await _run_pcc_incentive(_make_pcc_config(), jobs, monkeypatch)

    for result in jobs[EXCLUDED]:
        assert result.is_hotkey_excluded_from_scoring
        assert result.incentive == 0
        assert result.mining_score == 0
        assert not result.eligible_for_rental_share
        assert ZeroIncentiveReason.HOTKEY_EXCLUDED_FROM_SCORING in _codes(result)
    assert incentive.miner_incentives.get(EXCLUDED, 0.0) == 0
    # the excluded idle GPU does not take a slot in the unrented bucket
    assert incentive.unrented_count_by_bucket[("B200", 1)] == 1

    for result in jobs["miner_b"]:
        assert not result.is_hotkey_excluded_from_scoring
        assert result.incentive > 0
        assert ZeroIncentiveReason.HOTKEY_EXCLUDED_FROM_SCORING not in _codes(result)


@pytest.mark.asyncio
async def test_unlisted_hotkeys_score_as_before(monkeypatch):
    monkeypatch.setattr(settings, "SCORING_EXCLUDED_HOTKEYS", "")
    jobs = {EXCLUDED: [_make_pcc_job("exec-idle", B200, 1)]}

    await _run_pcc_incentive(_make_pcc_config(), jobs, monkeypatch)

    result = jobs[EXCLUDED][0]
    assert not result.is_hotkey_excluded_from_scoring
    assert result.incentive > 0


def _job(**overrides) -> JobResult:
    fields: dict = dict(
        executor_info=ExecutorSSHInfo(
            uuid="exec-1",
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
        ),
        score=1.0,
        job_score=1.0,
        job_batch_id="batch",
        log_status="success",
        log_text="ok",
        gpu_model="NVIDIA H200",
        gpu_count=8,
        collateral_deposited=True,
        sysbox_runtime=True,
    )
    fields.update(overrides)
    return JobResult(**fields)


@pytest.mark.asyncio
async def test_an_excluded_spot_node_gets_no_spot_pay(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_SPOT_NODE_PAY", True)
    incentive = RentalPriceIncentive(IncentiveConfig(), AsyncMock(), {}, {})
    spot = _job(
        is_spot=True,
        is_on_spot_pay_list=True,
        has_lium_filler=True,
        filler_revenue_per_gpu_hour=1.0,
        is_hotkey_excluded_from_scoring=True,
    )

    result = await incentive.calculate_executor_score(spot)

    assert not result.spot_pay_candidate
    assert result.mining_score == 0
    assert _codes(result) == [ZeroIncentiveReason.HOTKEY_EXCLUDED_FROM_SCORING]


@pytest.mark.asyncio
async def test_default_algorithm_zeroes_an_excluded_executor(incentive_redis_service):
    incentive = DefaultIncentive(IncentiveConfig(), incentive_redis_service, {}, {"NVIDIA H200": 8})

    result = await incentive.calculate_executor_score(_job(is_rented=True, is_hotkey_excluded_from_scoring=True))

    assert result.mining_score == 0
    assert _codes(result) == [ZeroIncentiveReason.HOTKEY_EXCLUDED_FROM_SCORING]


@pytest.fixture
def burn_env(monkeypatch):
    monkeypatch.setattr(settings, "NEW_BURNERS", [BURN_UID])
    monkeypatch.setattr(settings, "BURNER_COLDKEYS", {})


@pytest.fixture
def miners(create_neuron_info):
    return [
        create_neuron_info(uid=BURN_UID, hotkey=BURN_HOTKEY),
        create_neuron_info(uid=2, hotkey="miner_a"),
        create_neuron_info(uid=3, hotkey=EXCLUDED),
        create_neuron_info(uid=4, hotkey="referrer_1"),
    ]


@pytest.mark.asyncio
async def test_a_score_left_on_an_excluded_hotkey_moves_to_burn(burn_env, exclude, miners, monkeypatch):
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.0)
    incentive = DefaultIncentive(IncentiveConfig(), AsyncMock(), {}, {})
    incentive.miner_incentives = {"miner_a": 0.08, EXCLUDED: 0.05}

    scores = await incentive.calculate_final_weights(miners)

    assert scores[EXCLUDED] == 0
    assert scores["miner_a"] == pytest.approx(0.08)
    assert scores[BURN_HOTKEY] == pytest.approx(TOTAL_BURN_EMISSION + 0.05)
    assert sum(scores.values()) == pytest.approx(TOTAL_BURN_EMISSION + 0.08 + 0.05)


@pytest.mark.asyncio
async def test_an_excluded_hotkey_is_never_a_referral_recipient(burn_env, exclude, miners, monkeypatch):
    monkeypatch.setattr(settings, "REFERRAL_EMISSION_SHARE", 0.1)
    incentive = DefaultIncentive(IncentiveConfig(), AsyncMock(), {}, {})
    incentive.miner_incentives = {"miner_a": 0.08}
    incentive.referral_feed = _StubReferralFeed({EXCLUDED: 5.0, "referrer_1": 1.0})

    scores = await incentive.calculate_final_weights(miners)

    assert scores[EXCLUDED] == 0
    assert scores["referrer_1"] > 0
