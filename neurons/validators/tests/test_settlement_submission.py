"""DAH-4001: what the validator reports every cycle and submits at tempo under SETTLEMENT_MODE."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.backend_client import (
    DEFINITIVE_REJECTIONS,
    BackendClient,
    BackendRejected,
    SettledWeights,
)
from clients.subtensor_client import fold_unregistered_into_burner, scored_registered_neurons
from core.settlement import cycle_node_shares, share_moved, tempo_index
from core.validator import PENDING_INCLUSION_KEY, UNACKED_CYCLE_REPORTS_MAX, Validator
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


def _validator(*, window: SettledWeights | None, accepted: bool) -> Validator:
    validator = Validator.__new__(Validator)
    validator.default_extra = {}
    validator.active_hotkeys = set()
    validator.miner_scores = {"hk": 0.5, "burn": 0.5}
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
        llen=AsyncMock(return_value=0),
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
async def test_backend_unreachable_submits_nothing_and_asks_again_on_the_next_tick():
    validator = _validator(window=None, accepted=True)

    first = await validator.submit_settled_window()
    validator.backend_client.get_settled_weights = AsyncMock(return_value=WINDOW)
    second = await validator.submit_settled_window()

    assert (first, second) == (False, True)
    validator.subtensor_client.set_weights.assert_awaited_once()
    assert validator.subtensor_client.set_weights.await_args.kwargs["miner_scores"] == WINDOW.hotkey_scores


@pytest.mark.asyncio
async def test_an_empty_window_submits_nothing_and_is_not_asked_for_again_in_its_tempo():
    empty = WINDOW.model_copy(update={"cycle_ids": [], "hotkey_scores": {}})
    validator = _validator(window=empty, accepted=True)

    await validator.submit_settled_window()
    await validator.submit_settled_window()

    validator.subtensor_client.set_weights.assert_not_awaited()
    validator.backend_client.get_settled_weights.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_window_whose_scores_are_all_zero_submits_nothing():
    zeros = WINDOW.model_copy(update={"hotkey_scores": {"hk": 0.0, "burn": 0.0}})
    validator = _validator(window=zeros, accepted=True)

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

    await validator.shadow_settled_window({"hk": 0.5, "burn": 0.5})

    validator.backend_client.get_settled_weights.assert_awaited_once_with(342, 360)
    validator.subtensor_client.set_weights.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failing_shadow_comparison_never_skips_the_live_weights(monkeypatch):
    monkeypatch.setattr("core.config.settings.SETTLEMENT_MODE", "shadow")
    monkeypatch.setattr("core.config.settings.DRY_RUN", False)
    validator = _validator(window=WINDOW, accepted=True)
    validator.completed_cycles_since_start = 1
    validator.subtensor_client.get_miners = AsyncMock(return_value=[])
    validator.subtensor_client.should_set_weights = AsyncMock(return_value=True)
    validator.subtensor_client.get_tempo = MagicMock(side_effect=RuntimeError("tempo read failed"))
    # ends the tick right after the weights step; sync() logs it as an unknown error
    validator.subtensor_client.get_current_block = MagicMock(side_effect=RuntimeError("stop the tick"))

    await validator.sync()

    validator.subtensor_client.set_weights.assert_awaited_once_with(
        miner_scores={"hk": 0.5, "burn": 0.5}, active_hotkeys=set()
    )
    validator.backend_client.get_settled_weights.assert_not_awaited()
    assert validator.miner_scores == {}


class RedisList:
    """The three list calls the cycle report queue uses, with Redis semantics (lpush puts the newest at the head)."""

    def __init__(self):
        self.items: list[bytes] = []

    async def lpush(self, key, value):
        self.items.insert(0, value)

    async def ltrim(self, key, max_length):
        del self.items[max_length:]

    async def lrange(self, key):
        return list(self.items)

    async def llen(self, key):
        return len(self.items)

    async def lrem(self, key, value):
        self.items.remove(value)


def _with_report_queue(validator: Validator) -> RedisList:
    queue = RedisList()
    validator.redis_service.lpush = queue.lpush
    validator.redis_service.ltrim = queue.ltrim
    validator.redis_service.lrange = queue.lrange
    validator.redis_service.lrem = queue.lrem
    validator.redis_service.llen = queue.llen
    return queue


def _report(cycle_id: str) -> bytes:
    return json.dumps({"cycle_id": cycle_id, "hotkey_scores": {}}).encode()


@pytest.mark.asyncio
async def test_an_unacknowledged_cycle_report_is_kept_and_replayed_next_cycle():
    validator = _validator(window=None, accepted=True)
    queue = _with_report_queue(validator)
    scored_at = datetime(2026, 10, 6, 10, 0, tzinfo=UTC)
    rows = [{"executor_id": "e1", "hotkey": "hk", "rented": 0.0, "idle": 0.8, "spot": False}]

    await validator.report_cycle_scores(
        {"hk": 0.8, "burn": 0.2}, rows, "burn", "2026-10-06 10:00:00", 500, scored_at
    )

    assert [json.loads(raw)["cycle_id"] for raw in queue.items] == ["2026-10-06 10:00:00"]
    assert json.loads(queue.items[0])["node_shares"] == rows

    validator.backend_client.report_cycle_scores = AsyncMock(
        return_value=MagicMock(cycle_id="any", created=True)
    )
    await validator.report_cycle_scores(
        {"hk": 1.0}, [], "burn", "2026-10-06 10:15:00", 575, scored_at
    )

    delivered = [call.args[0]["cycle_id"] for call in validator.backend_client.report_cycle_scores.await_args_list]
    assert delivered == ["2026-10-06 10:00:00", "2026-10-06 10:15:00"]  # oldest first
    assert queue.items == []


@pytest.mark.asyncio
async def test_a_hanging_backend_is_cut_off_at_the_reporting_budget_and_the_report_stays_kept(monkeypatch):
    monkeypatch.setattr("core.validator.SETTLEMENT_REPORTING_BUDGET_SECONDS", 0.05)
    validator = _validator(window=None, accepted=True)
    queue = _with_report_queue(validator)

    async def hang(payload):
        await asyncio.sleep(10)

    validator.backend_client.report_cycle_scores = hang

    await asyncio.wait_for(
        validator.report_cycle_scores(
            {"hk": 1.0}, [], "burn", "2026-10-06 10:00:00", 500, datetime(2026, 10, 6, 10, 0, tzinfo=UTC)
        ),
        timeout=1,
    )

    assert [json.loads(raw)["cycle_id"] for raw in queue.items] == ["2026-10-06 10:00:00"]


@pytest.mark.asyncio
async def test_a_full_queue_is_delivered_whole_once_the_backend_is_back():
    validator = _validator(window=None, accepted=True)
    queue = _with_report_queue(validator)
    queue.items = [_report(str(i)) for i in reversed(range(UNACKED_CYCLE_REPORTS_MAX))]  # newest at the head
    validator.backend_client.report_cycle_scores = AsyncMock(return_value=MagicMock(cycle_id="any", created=True))

    await validator.report_cycle_scores(
        {"hk": 1.0}, [], "burn", "2026-10-06 10:00:00", 500, datetime(2026, 10, 6, 10, 0, tzinfo=UTC)
    )

    delivered = [call.args[0]["cycle_id"] for call in validator.backend_client.report_cycle_scores.await_args_list]
    assert delivered == [str(i) for i in range(UNACKED_CYCLE_REPORTS_MAX)] + ["2026-10-06 10:00:00"]
    assert queue.items == []


@pytest.mark.asyncio
async def test_past_the_cap_during_an_outage_the_oldest_report_is_dropped_after_the_delivery_pass():
    validator = _validator(window=None, accepted=True)
    queue = _with_report_queue(validator)
    queue.items = [_report(str(i)) for i in reversed(range(UNACKED_CYCLE_REPORTS_MAX))]

    await validator.report_cycle_scores(
        {"hk": 1.0}, [], "burn", "2026-10-06 10:00:00", 500, datetime(2026, 10, 6, 10, 0, tzinfo=UTC)
    )

    validator.backend_client.report_cycle_scores.assert_awaited_once()  # the oldest, still failing
    assert len(queue.items) == UNACKED_CYCLE_REPORTS_MAX
    assert json.loads(queue.items[0])["cycle_id"] == "2026-10-06 10:00:00"
    assert json.loads(queue.items[-1])["cycle_id"] == "1"


@pytest.mark.asyncio
async def test_a_slowly_failing_redis_cannot_spend_the_budget_before_the_unkept_report_is_sent(monkeypatch):
    monkeypatch.setattr("core.validator.SETTLEMENT_REPORTING_BUDGET_SECONDS", 0.06)
    validator = _validator(window=None, accepted=True)

    async def slow_failure(*args):
        await asyncio.sleep(0.04)
        raise ConnectionError("redis down")

    validator.redis_service.lpush = slow_failure
    validator.redis_service.get = slow_failure
    validator.redis_service.lrange = slow_failure
    validator.redis_service.llen = slow_failure
    validator.backend_client.report_cycle_scores = AsyncMock(
        return_value=MagicMock(cycle_id="2026-10-06 10:00:00", created=True)
    )

    await validator.report_cycle_scores(
        {"hk": 1.0}, [], "burn", "2026-10-06 10:00:00", 500, datetime(2026, 10, 6, 10, 0, tzinfo=UTC)
    )

    validator.backend_client.report_cycle_scores.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_report_redis_cannot_keep_is_sent_once_directly():
    validator = _validator(window=None, accepted=True)
    validator.redis_service.lpush = AsyncMock(side_effect=ConnectionError("redis down"))
    validator.backend_client.report_cycle_scores = AsyncMock(
        return_value=MagicMock(cycle_id="2026-10-06 10:00:00", created=True)
    )

    await validator.report_cycle_scores(
        {"hk": 1.0}, [], "burn", "2026-10-06 10:00:00", 500, datetime(2026, 10, 6, 10, 0, tzinfo=UTC)
    )

    validator.backend_client.report_cycle_scores.assert_awaited_once()


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
async def test_a_hanging_shadow_read_is_cut_off_and_never_raises(monkeypatch):
    monkeypatch.setattr("core.validator.SHADOW_COMPARISON_BUDGET_SECONDS", 0.05)
    validator = _validator(window=WINDOW, accepted=True)

    async def hang(index, tempo):
        await asyncio.sleep(10)

    validator.backend_client.get_settled_weights = hang

    await asyncio.wait_for(validator.shadow_settled_window({"hk": 1.0}), timeout=1)

    assert getattr(validator, "_settled_tempo_done", None) is None


@pytest.mark.asyncio
async def test_a_cycle_with_no_burner_is_not_reported():
    validator = _validator(window=None, accepted=True)

    await validator.report_cycle_scores(
        {"hk": 1.0}, [], None, "2026-10-06 10:00:00", 500, datetime(2026, 10, 6, 10, 0, tzinfo=UTC)
    )

    validator.backend_client.report_cycle_scores.assert_not_awaited()


def test_our_hotkey_stands_in_for_the_burner_on_a_test_network_never_on_mainnet(monkeypatch):
    validator = _validator(window=None, accepted=True)
    monkeypatch.setattr("core.validator.verified_burner_hotkey", lambda miners: None)

    monkeypatch.setattr("core.config.settings.BITTENSOR_NETWORK", "finney")
    on_mainnet = validator._settlement_burner([])
    monkeypatch.setattr("core.config.settings.BITTENSOR_NETWORK", "test")
    on_testnet = validator._settlement_burner([])

    assert (on_mainnet, on_testnet) == (None, "validator-hotkey")


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


def test_tempo_index_and_share_moved():
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
    queue = _with_report_queue(validator)
    validator.backend_client.report_cycle_scores = AsyncMock(side_effect=BackendRejected(422))
    scored_at = datetime(2026, 10, 6, 10, 0, tzinfo=UTC)

    await validator.report_cycle_scores(
        {"hk": 1.0}, [], "burn", "2026-10-06 10:00:00", 500, scored_at
    )

    validator.backend_client.report_cycle_scores.assert_awaited_once()
    assert queue.items == []


@pytest.mark.asyncio
async def test_an_unacknowledged_inclusion_is_kept_and_retried_next_cycle():
    validator = _validator(window=WINDOW, accepted=True)
    validator.backend_client.report_settled_weights_result = AsyncMock(return_value=None)

    await validator.submit_settled_window()

    validator.redis_service.set.assert_any_await(PENDING_INCLUSION_KEY, json.dumps({"342": 123456}))

    validator.redis_service.get = AsyncMock(return_value=json.dumps({"342": 123456}))
    validator.backend_client.report_settled_weights_result = AsyncMock(
        return_value=MagicMock(inclusion_block=123456)
    )
    await validator.report_cycle_scores(
        {"hk": 1.0},
        [],
        "burn",
        "2026-10-06 10:00:00",
        500,
        datetime(2026, 10, 6, 10, 0, tzinfo=UTC),
    )

    validator.backend_client.report_settled_weights_result.assert_awaited_once_with(342, 123456)
    validator.redis_service.set.assert_any_await(PENDING_INCLUSION_KEY, json.dumps({}))


@pytest.mark.asyncio
async def test_a_later_confirmation_does_not_clear_an_earlier_pending_one():
    validator = _validator(window=WINDOW, accepted=True)
    validator.redis_service.get = AsyncMock(return_value=json.dumps({"342": 123456}))

    async def answer(index, block):
        return MagicMock(inclusion_block=block) if index == 343 else None

    validator.backend_client.report_settled_weights_result = AsyncMock(side_effect=answer)

    await validator._confirm_inclusion(343, 123800)

    validator.redis_service.set.assert_awaited_with(
        PENDING_INCLUSION_KEY, json.dumps({"342": 123456})
    )


@pytest.mark.asyncio
async def test_the_inclusion_is_kept_before_it_is_reported():
    validator = _validator(window=WINDOW, accepted=True)
    kept_when_reported = []

    async def report(index, block):
        kept_when_reported.extend(call.args for call in validator.redis_service.set.await_args_list)

    validator.backend_client.report_settled_weights_result = AsyncMock(side_effect=report)

    await validator.submit_settled_window()

    assert (PENDING_INCLUSION_KEY, json.dumps({"342": 123456})) in kept_when_reported


@pytest.mark.asyncio
async def test_a_pending_inclusion_is_reported_before_the_next_window_is_read():
    validator = _validator(window=WINDOW, accepted=True)
    validator.redis_service.get = AsyncMock(return_value=json.dumps({"341": 123000}))
    calls = []
    validator.backend_client.report_settled_weights_result = AsyncMock(
        side_effect=lambda index, block: calls.append(("report", index))
    )
    validator.backend_client.get_settled_weights = AsyncMock(
        side_effect=lambda index, tempo: calls.append(("read", index)) or WINDOW
    )

    await validator.submit_settled_window()

    assert calls[:2] == [("report", 341), ("read", 342)]


@pytest.mark.asyncio
async def test_the_inclusion_is_reported_with_the_block_before_submission_when_the_read_after_fails():
    validator = _validator(window=WINDOW, accepted=True)
    validator.subtensor_client.get_current_block = MagicMock(side_effect=[123456, RuntimeError("rpc down")])

    await validator.submit_settled_window()

    validator.backend_client.report_settled_weights_result.assert_awaited_once_with(342, 123456)


@pytest.mark.asyncio
async def test_a_redis_read_failure_never_overwrites_the_pending_inclusions():
    validator = _validator(window=WINDOW, accepted=True)
    validator.redis_service.get = AsyncMock(side_effect=ConnectionError("redis down"))

    await validator._confirm_inclusion(343, 123800)

    validator.redis_service.set.assert_not_awaited()


def test_only_a_wrong_request_is_dropped_a_bad_moment_is_retried():
    assert DEFINITIVE_REJECTIONS == (400, 404, 422)
    assert 403 not in DEFINITIVE_REJECTIONS and 429 not in DEFINITIVE_REJECTIONS
