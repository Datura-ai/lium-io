"""DAH-7118: a node whose only change during the cycle was a Lium filler starting keeps its last pass instead of
losing the cycle to the rental-change rule (DAH-6564)."""

import pytest

from core.config import settings
from core.validator import settle_cycle_results, specs_to_publish
from services.executor_rollout import (
    RENTAL_CHANGE_STATE_KEY,
    RolloutWindow,
    carry_rental_change_passes,
    remember_rented_scored_results,
    withhold_rental_change_verdicts,
)
from services.redis_service import last_scored_result_key, last_scored_result_vouch_key
from test_executor_rollout_grace import _scored
from test_rental_change_withheld import GPU_USAGE, MINER, NODE, _fail, _listed
from test_rented_host_gpu_fault_reset import _redis_service

BLOCK = 1000
NO_ROLLOUT = RolloutWindow(
    digest=None, previous_digest=None, started_at=None, opened_job_block=None, cycles_seen=0, grace_cycles=0
)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(settings, "RENTAL_CHANGE_CARRY_ENABLED", True)


def _owned(listed, owner):
    listed.default_job_owner_by_executor = {NODE: owner} if owner else {}
    return listed


def _idle_pass(batch="b0"):
    passed = _scored(NODE)
    passed.job_batch_id = batch
    passed.rental_pod_ids = []
    return passed


def _rented_pass(batch="b0"):
    passed = _idle_pass(batch)
    passed.is_rented, passed.rental_pod_ids, passed.rented_gpu_count = True, ["p1"], 8
    return passed


async def _previous_cycle(service, *results, block=BLOCK - 75):
    await remember_rented_scored_results({MINER: list(results)}, service, job_block=block)


async def _carry(results, start, end, service, block=BLOCK):
    cycle = {MINER: list(results)}
    await carry_rental_change_passes(cycle, service, start, end, {MINER: None}, set(), NO_ROLLOUT, job_block=block)
    return cycle[MINER]


def _filler_started(owner="lium", pods=None):
    return _listed(pods=pods), _owned(_listed(pods=pods, fillers={NODE: ["filler_a"]}), owner)


@pytest.mark.asyncio
async def test_a_lium_filler_started_on_an_idle_node_credits_its_last_pass():
    service = _redis_service()
    await _previous_cycle(service, _idle_pass())
    start, end = _filler_started()

    [credited] = await _carry([_fail(GPU_USAGE)], start, end, service)

    assert credited.carried_for_rental_change and credited.is_successful
    assert (credited.job_batch_id, credited.score, credited.has_lium_filler) == ("batch-1", 1.0, True)
    assert credited.default_job_owner == "lium"
    standing, withheld = await withhold_rental_change_verdicts({MINER: [credited]}, start, end, service, set())
    assert (standing, withheld) == ({MINER: [credited]}, [])
    # the credit counts toward the cap
    assert b"batch-1" in (await service.hgetall(RENTAL_CHANGE_STATE_KEY))[f"{MINER}:{NODE}".encode()]


@pytest.mark.asyncio
async def test_a_lium_filler_started_on_a_rented_node_credits_its_vouched_pass():
    service = _redis_service()
    await _previous_cycle(service, _rented_pass())
    start = _listed(pods={NODE: ["p1"]}, pod_fields={"gpu_count": 8})
    end = _owned(_listed(pods={NODE: ["p1"]}, fillers={NODE: ["filler_a"]}, pod_fields={"gpu_count": 8}), "lium")

    [credited] = await _carry([_fail(GPU_USAGE)], start, end, service)

    assert credited.carried_for_rental_change and credited.is_rented and credited.rented_gpu_count == 8


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(*_filler_started(owner="miner"), id="default-job"),
        pytest.param(*_filler_started(owner=None), id="unknown-owner"),
        pytest.param(_owned(_listed(fillers={NODE: ["filler_a"]}), "lium"), _listed(), id="removal"),
        pytest.param(*_filler_started(pods={NODE: ["p9"]}), id="pod-change"),
        pytest.param(
            _owned(_listed(fillers={NODE: ["filler_a"]}), "lium"),
            _owned(_listed(fillers={NODE: ["filler_a", "filler_b"]}), "lium"),
            id="mixed",
        ),
        pytest.param(_listed(), _owned(_listed(pods={NODE: ["p1"]}, fillers={NODE: ["filler_a"]}), "lium"), id="pod-and-filler"),
        # the node-wide owner names the newest run only: two added fillers may be one provider job and one Lium job
        pytest.param(_listed(), _owned(_listed(fillers={NODE: ["filler_a", "filler_b"]}), "lium"), id="two-fillers"),
    ],
)
@pytest.mark.asyncio
async def test_anything_but_an_added_lium_filler_is_withheld_as_before(start, end):
    service = _redis_service()
    await _previous_cycle(service, _idle_pass())
    failed = _fail(GPU_USAGE)

    assert await _carry([failed], start, end, service) == [failed]
    _, withheld = await withhold_rental_change_verdicts({MINER: [failed]}, start, end, service, set())
    assert [w.result for w in withheld] == [failed]


@pytest.mark.asyncio
async def test_no_pass_before_the_cycle_is_withheld():
    service = _redis_service()
    await _previous_cycle(service, _fail(GPU_USAGE, batch="b0"))
    failed = _fail(GPU_USAGE)

    assert await _carry([failed], *_filler_started(), service) == [failed]


@pytest.mark.asyncio
async def test_a_pass_vouched_for_only_by_a_reset_carry_is_withheld():
    service = _redis_service()
    await _previous_cycle(service, _rented_pass(), block=BLOCK - 150)
    carried = _rented_pass()
    carried.carried_for_renter_gpu_reset = True
    await _previous_cycle(service, carried)
    start = _listed(pods={NODE: ["p1"]}, pod_fields={"gpu_count": 8})
    end = _owned(_listed(pods={NODE: ["p1"]}, fillers={NODE: ["filler_a"]}, pod_fields={"gpu_count": 8}), "lium")
    failed = _fail(GPU_USAGE)

    assert await _carry([failed], start, end, service) == [failed]


@pytest.mark.asyncio
async def test_a_credited_pass_is_used_once():
    service = _redis_service()
    await _previous_cycle(service, _idle_pass())
    [credited] = await _carry([_fail(GPU_USAGE)], *_filler_started(), service)
    await _previous_cycle(service, credited, block=BLOCK)
    failed = _fail(GPU_USAGE, batch="batch-2")

    assert await _carry([failed], *_filler_started(), service, block=BLOCK + 75) == [failed]


@pytest.mark.asyncio
async def test_a_credited_rented_pass_is_dropped_from_redis():
    service = _redis_service()
    await _previous_cycle(service, _rented_pass())
    start = _listed(pods={NODE: ["p1"]}, pod_fields={"gpu_count": 8})
    end = _owned(_listed(pods={NODE: ["p1"]}, fillers={NODE: ["filler_a"]}, pod_fields={"gpu_count": 8}), "lium")
    [credited] = await _carry([_fail(GPU_USAGE)], start, end, service)

    await _previous_cycle(service, credited, block=BLOCK)

    assert await service.get(last_scored_result_key(MINER, NODE)) is None
    assert await service.get(last_scored_result_vouch_key(MINER, NODE)) is None


@pytest.mark.asyncio
async def test_a_credited_copy_is_not_published_and_its_gpus_count_once():
    service = _redis_service()
    await _previous_cycle(service, _idle_pass())
    [credited] = await _carry([_fail(GPU_USAGE)], *_filler_started(), service)

    assert specs_to_publish([credited]) == []
    assert settle_cycle_results({MINER: [credited], "5Other": [_scored(NODE)]}, {}) == {"H100": 8}


@pytest.mark.asyncio
async def test_the_flag_off_changes_nothing_and_consumes_nothing(monkeypatch, caplog):
    monkeypatch.setattr(settings, "RENTAL_CHANGE_CARRY_ENABLED", False)
    service = _redis_service()
    await _previous_cycle(service, _rented_pass())
    start = _listed(pods={NODE: ["p1"]}, pod_fields={"gpu_count": 8})
    end = _owned(_listed(pods={NODE: ["p1"]}, fillers={NODE: ["filler_a"]}, pod_fields={"gpu_count": 8}), "lium")
    failed = _fail(GPU_USAGE)

    with caplog.at_level("INFO"):
        assert await _carry([failed], start, end, service) == [failed]

    assert "would credit" in caplog.text
    assert await service.get(last_scored_result_vouch_key(MINER, NODE)) is not None
