"""DAH-4001: what the validator reports every cycle and submits at tempo under SETTLEMENT_MODE."""

import hashlib
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.backend_client import BackendClient, BackendRejected, SettledWeights
from clients.subtensor_client import fold_unregistered_into_burner, scored_registered_neurons
from core.settlement import accumulate, cycle_node_shares, fallback_vector, share_moved, tempo_index
from core.validator import UNACKED_CYCLE_REPORTS_KEY, Validator
from incentive.burn_service import verified_burner_hotkey

WINDOW = SettledWeights(
    tempo_index=342,
    window_from_block=115560,
    window_to_block=115920,
    cycle_ids=["c1", "c2"],
    hotkey_scores={"hk": 0.7, "burn": 0.3},
    withheld_count=1,
    withheld_total=0.1,
    mass_inactive_skipped=False,
)


def _validator(
    *, window: SettledWeights | None, accepted: bool, fallback: dict | None = None
) -> Validator:
    validator = Validator.__new__(Validator)
    validator.default_extra = {}
    validator.active_hotkeys = set()
    validator.miner_scores = {"hk": 0.5, "burn": 0.5}
    validator.fallback_scores = dict(fallback or {})
    validator.backend_client = MagicMock(
        get_settled_weights=AsyncMock(return_value=window),
        report_settled_weights_result=AsyncMock(return_value=None),
        report_cycle_scores=AsyncMock(return_value=None),
    )
    validator.backend_client.keypair = MagicMock(ss58_address="validator-hotkey")
    validator.redis_service = MagicMock(
        get=AsyncMock(return_value=None),
        set=AsyncMock(),
        lrange=AsyncMock(return_value=[]),
        lpush=AsyncMock(),
        ltrim=AsyncMock(),
        lrem=AsyncMock(),
    )
    validator.subtensor_client = MagicMock(
        set_weights=AsyncMock(return_value=accepted),
        get_current_block=MagicMock(return_value=123456),
        get_tempo=MagicMock(return_value=360),
        get_last_update=MagicMock(return_value=50),
        get_weights_rate_limit=MagicMock(return_value=100),
        netuid=51,
    )
    validator.subtensor_client.subtensor.get_subnet_hyperparameters.return_value = MagicMock(
        activity_cutoff=12000
    )
    return validator


@pytest.mark.asyncio
async def test_the_settled_window_is_submitted_and_its_inclusion_reported():
    validator = _validator(window=WINDOW, accepted=True)

    await validator.submit_settled_window()

    validator.backend_client.get_settled_weights.assert_awaited_once_with(342, 360)
    validator.subtensor_client.set_weights.assert_awaited_once_with(
        miner_scores=WINDOW.hotkey_scores,
        active_hotkeys=set(),
        wait_for_inclusion=True,
        include_registered_scored=True,
    )
    validator.backend_client.report_settled_weights_result.assert_awaited_once_with(342, 123456)


@pytest.mark.asyncio
async def test_a_rejected_submission_reports_no_inclusion():
    validator = _validator(window=WINDOW, accepted=False)

    await validator.submit_settled_window()

    validator.backend_client.report_settled_weights_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_backend_unreachable_submits_the_fallback_not_an_old_vector():
    validator = _validator(window=None, accepted=True, fallback={"hk": 0.2, "burn": 0.8})

    await validator.submit_settled_window()

    validator.subtensor_client.set_weights.assert_awaited_once()
    assert validator.subtensor_client.set_weights.await_args.kwargs["miner_scores"] == {
        "hk": 0.2,
        "burn": 0.8,
    }
    validator.backend_client.report_settled_weights_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_empty_window_submits_the_fallback():
    empty = WINDOW.model_copy(update={"cycle_ids": [], "hotkey_scores": {}})
    validator = _validator(window=empty, accepted=True, fallback={"burn": 1.0})

    await validator.submit_settled_window()

    assert validator.subtensor_client.set_weights.await_args.kwargs["miner_scores"] == {"burn": 1.0}


@pytest.mark.asyncio
async def test_no_window_and_no_fallback_skips_the_tempo():
    validator = _validator(window=None, accepted=True)

    await validator.submit_settled_window()

    validator.subtensor_client.set_weights.assert_not_awaited()


@pytest.mark.asyncio
async def test_set_weights_raising_is_a_failure_with_no_inclusion_report():
    validator = _validator(window=WINDOW, accepted=True)
    validator.subtensor_client.set_weights = AsyncMock(side_effect=RuntimeError("rpc down"))

    await validator.submit_settled_window()

    validator.backend_client.report_settled_weights_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_shadow_reads_the_window_and_submits_nothing_itself():
    validator = _validator(window=WINDOW, accepted=True)

    await validator.shadow_settled_window()

    validator.backend_client.get_settled_weights.assert_awaited_once_with(342, 360)
    validator.subtensor_client.set_weights.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unacknowledged_cycle_report_is_kept_and_replayed_next_cycle():
    validator = _validator(window=None, accepted=True)
    scored_at = datetime(2026, 10, 6, 10, 0, tzinfo=UTC)
    rows = [{"executor_id": "e1", "hotkey": "hk", "rented": 0.0, "idle": 0.8, "spot": False}]

    await validator.report_cycle_scores(
        {"hk": 0.8, "burn": 0.2}, rows, "burn", "2026-10-06 10:00:00", 500, scored_at
    )

    validator.redis_service.lpush.assert_awaited_once()
    kept = validator.redis_service.lpush.await_args.args[1]
    assert (
        json.loads(kept)["cycle_id"] == "2026-10-06 10:00:00"
        and json.loads(kept)["node_shares"] == rows
    )

    validator.redis_service.lrange = AsyncMock(return_value=[kept])
    validator.backend_client.report_cycle_scores = AsyncMock(
        return_value=MagicMock(cycle_id="2026-10-06 10:00:00", created=True)
    )
    await validator.report_cycle_scores(
        {"hk": 1.0}, [], "burn", "2026-10-06 10:15:00", 575, scored_at
    )

    validator.redis_service.lrem.assert_awaited_once_with(UNACKED_CYCLE_REPORTS_KEY, kept)
    assert validator.backend_client.report_cycle_scores.await_count == 2


@pytest.mark.asyncio
async def test_replay_stops_at_the_first_report_that_still_fails():
    validator = _validator(window=None, accepted=True)
    older, newer = (
        json.dumps({"cycle_id": "older", "hotkey_scores": {}}).encode(),
        json.dumps({"cycle_id": "newer", "hotkey_scores": {}}).encode(),
    )
    validator.redis_service.lrange = AsyncMock(
        return_value=[newer, older]
    )  # lpush order: newest first

    await validator._replay_unacked_cycle_reports()

    validator.backend_client.report_cycle_scores.assert_awaited_once()  # the oldest, which failed; the newer one waits
    assert validator.backend_client.report_cycle_scores.await_args.args[0]["cycle_id"] == "older"


@pytest.mark.asyncio
async def test_no_verified_burner_on_mainnet_skips_the_report_on_a_test_network_our_hotkey_stands_in(
    monkeypatch,
):
    validator = _validator(window=None, accepted=True)
    scored_at = datetime(2026, 10, 6, 10, 0, tzinfo=UTC)

    monkeypatch.setattr("core.config.settings.BITTENSOR_NETWORK", "finney")
    await validator.report_cycle_scores(
        {"hk": 1.0}, [], None, "2026-10-06 10:00:00", 500, scored_at
    )
    validator.backend_client.report_cycle_scores.assert_not_awaited()

    monkeypatch.setattr("core.config.settings.BITTENSOR_NETWORK", "test")
    await validator.report_cycle_scores(
        {"hk": 1.0}, [], None, "2026-10-06 10:00:00", 500, scored_at
    )
    assert (
        validator.backend_client.report_cycle_scores.await_args.args[0]["burn_hotkey"]
        == "validator-hotkey"
    )


def test_node_shares_describe_every_priced_result():
    def result(uuid, rented, idle, spot=False):
        return MagicMock(
            executor_info=MagicMock(uuid=uuid),
            incentive_rented=rented,
            incentive_idle=idle,
            is_spot=spot,
        )

    rows = cycle_node_shares(
        {
            "hk": [result("a", 0.0, 0.6), result("b", 0.3, 0.0), result("z", 0.0, 0.0)],
            "sp": [result("c", 0.0, 0.1, True)],
        }
    )

    assert rows == [
        {"executor_id": "a", "hotkey": "hk", "rented": 0.0, "idle": 0.6, "spot": False},
        {"executor_id": "b", "hotkey": "hk", "rented": 0.3, "idle": 0.0, "spot": False},
        {"executor_id": "c", "hotkey": "sp", "rented": 0.0, "idle": 0.1, "spot": True},
    ]


def test_the_fallback_moves_every_idle_share_to_the_burner_and_keeps_the_total():
    scores = {"hk": 0.9, "sp": 0.1, "burn": 1.0, "referrer": 0.2}
    rows = [
        {"executor_id": "a", "hotkey": "hk", "rented": 0.3, "idle": 0.6, "spot": False},
        {"executor_id": "c", "hotkey": "sp", "rented": 0.0, "idle": 0.1, "spot": True},
    ]

    vector = fallback_vector(scores, rows, "burn")

    assert vector == pytest.approx({"hk": 0.3, "sp": 0.0, "burn": 1.7, "referrer": 0.2})
    assert sum(vector.values()) == pytest.approx(sum(scores.values()))
    assert fallback_vector(scores, rows, None) == scores


def test_accumulate_and_tempo_index_and_share_moved():
    into = {"hk": 0.1}
    accumulate(into, {"hk": 0.2, "burn": 0.3})

    assert into == pytest.approx({"hk": 0.3, "burn": 0.3})
    assert tempo_index(123456, 360) == 342
    assert share_moved({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 0.0}) == pytest.approx(0.5)


def test_a_registered_hotkey_with_a_settled_score_is_paid_even_when_it_serves_no_node():
    serving = [MagicMock(uid=1, hotkey="still-here")]
    registered = [
        MagicMock(uid=1, hotkey="still-here"),
        MagicMock(uid=2, hotkey="left-yesterday"),
        MagicMock(uid=3, hotkey="never-scored"),
    ]

    extra = scored_registered_neurons(
        serving, registered, {"still-here": 0.5, "left-yesterday": 0.3}
    )

    assert [neuron.uid for neuron in extra] == [2]


def test_a_deregistered_hotkeys_share_goes_to_the_verified_burner_not_to_everyone_else():
    scores = {"here": 0.5, "gone": 0.3, "burn": 0.2}

    assert fold_unregistered_into_burner(scores, {"here", "burn"}, "burn") == {
        "here": 0.5,
        "burn": 0.5,
    }
    assert fold_unregistered_into_burner(scores, {"here", "burn"}, None) == {
        "here": 0.5,
        "burn": 0.2,
    }


def test_the_verified_burner_is_the_first_slot_only_with_its_own_coldkey(monkeypatch):
    monkeypatch.setattr("core.config.settings.ENABLE_NEW_BURN_LOGIC", True)
    monkeypatch.setattr("core.config.settings.NEW_BURNERS", [47, 47])
    monkeypatch.setattr("core.config.settings.BURNER_COLDKEYS", {47: "cold-47"})
    burner = MagicMock(uid=47, hotkey="burn-hk", coldkey="cold-47")
    impostor = MagicMock(uid=47, hotkey="burn-hk", coldkey="someone-else")

    assert verified_burner_hotkey([MagicMock(uid=1, hotkey="m"), burner]) == "burn-hk"
    assert verified_burner_hotkey([impostor]) is None
    assert verified_burner_hotkey([MagicMock(uid=1, hotkey="m")]) is None


def test_the_signed_request_message_matches_the_backend_format():
    body = b'{"a":1}'

    message = BackendClient.signed_request_message(
        "post", "/validator/hk/cycles?x=1", body, "1700000000"
    )

    assert message == "\n".join(
        [
            "lium-validator-v1",
            "POST",
            "/validator/hk/cycles?x=1",
            hashlib.sha256(body).hexdigest(),
            "1700000000",
        ]
    )


@pytest.mark.asyncio
async def test_a_signed_request_signs_the_exact_bytes_it_sends():
    client = BackendClient.__new__(BackendClient)
    client.keypair = MagicMock(ss58_address="hk", sign=MagicMock(return_value=b"\x01\x02"))
    client._request = AsyncMock(return_value=None)

    await client._signed_request(
        "POST", "validator/hk/cycles", SettledWeights, json_data={"b": 2, "a": [1, 2]}
    )

    kwargs = client._request.await_args.kwargs
    assert kwargs["raw_body"] == b'{"a":[1,2],"b":2}'
    assert (
        kwargs["extra_headers"]["signature"] == "0x0102"
        and kwargs["extra_headers"]["hotkey"] == "hk"
    )
    signed = client.keypair.sign.call_args.args[0]
    assert (
        hashlib.sha256(b'{"a":[1,2],"b":2}').hexdigest() in signed
        and "/validator/hk/cycles" in signed
    )


@pytest.mark.asyncio
async def test_a_tempo_is_submitted_once_even_while_should_set_weights_stays_true():
    validator = _validator(window=WINDOW, accepted=True)

    await validator.submit_settled_window()
    await validator.submit_settled_window()

    validator.subtensor_client.set_weights.assert_awaited_once()
    validator.subtensor_client.get_current_block.return_value = 123456 + 360
    await validator.submit_settled_window()
    assert validator.subtensor_client.set_weights.await_count == 2


@pytest.mark.asyncio
async def test_a_rejected_submission_is_retried_on_the_next_tick():
    validator = _validator(window=WINDOW, accepted=False)

    await validator.submit_settled_window()
    await validator.submit_settled_window()

    assert validator.subtensor_client.set_weights.await_count == 2


@pytest.mark.asyncio
async def test_a_cycle_report_the_backend_rejects_is_dropped_not_replayed():
    validator = _validator(window=None, accepted=True)
    validator.backend_client.report_cycle_scores = AsyncMock(side_effect=BackendRejected(422))
    scored_at = datetime(2026, 10, 6, 10, 0, tzinfo=UTC)

    await validator.report_cycle_scores(
        {"hk": 1.0}, [], "burn", "2026-10-06 10:00:00", 500, scored_at
    )

    validator.redis_service.lpush.assert_not_awaited()
