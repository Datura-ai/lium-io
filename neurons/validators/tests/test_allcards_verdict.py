"""DAH-6978: the all-cards work-proof always decides, judged card by card.

- a card that gave no answer is not proven this cycle only when the backend has a Lium workload assigned to that
  card (a filler started on cards 0-3 during the run says nothing about cards 4-7);
- a node rented at the cycle snapshot runs no all-cards proof that cycle and is not offered until its cards prove
  themselves, even when the rental ended mid-cycle;
- a deferral is bounded: one retry in the next idle cycle, and the node is offered meanwhile only on a live last pass;
- only a lost connection to the whole node defers; a run that ended without an exit status is the card's verdict.
"""

from types import SimpleNamespace

import pytest
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.task.checks import capability
from services.task.checks.capability import CapabilityCheck
from services.task.messages import CapabilityMessages as Msg

from tests.helpers import FakeRedis, build_services, build_state
from tests.test_gpu_memory_health_check import _live_rental
from tests.test_matmul_allcards import (  # noqa: F401 (svc is a fixture)
    ANSWER,
    _card_index,
    _spec,
    svc,
)

PASS_KEY = "allcards_pass:miner-hotkey:executor-123"
DEFERRED_KEY = "allcards_deferred:miner-hotkey:executor-123"
OOM = SimpleNamespace(exit_status=0, stdout="UUID:  None", stderr="Failed to allocate d_A: out of memory")
NO_DEVICE = SimpleNamespace(exit_status=0, stdout="UUID:  None", stderr="no CUDA-capable device is detected")


def _node(answers: dict[int, object]):
    """Each pinned card in `answers` gives that answer (an exception is raised); every other run answers right."""
    pinned: list[int] = []

    async def run(cmd, *args, **kwargs):
        if cmd.startswith("pkill"):
            return SimpleNamespace(exit_status=0, stdout="", stderr="")
        if "CUDA_VISIBLE_DEVICES=" in cmd:
            card = _card_index(cmd)
            pinned.append(card)
            answer = answers.get(card)
            if isinstance(answer, Exception):
                raise answer
            if answer is not None:
                return answer
        return SimpleNamespace(exit_status=0, stdout=ANSWER, stderr="")

    return SimpleNamespace(run=run), pinned


def _filler_on(cards) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        all_filler_containers_by_executor={"executor-123": ["filler_e741"]},
        filler_gpu_uuids_by_executor={"executor-123": [f"GPU-{i}" for i in cards]},
    )


def _pod_on(cards) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="pod-1", container_name="pod_p1", gpu_uuids=[f"GPU-{i}" for i in cards])],
            )
        },
    )


@pytest.fixture(autouse=True)
def _no_gpu_model_numbers(monkeypatch):
    # the GPU model verdict is not under test: a pass with no speed numbers reads as measured
    monkeypatch.setattr(capability, "unmeasured_cards", lambda *_a, **_kw: set())


async def _cycle(context_factory, svc, ssh, *, redis, fresh=None, rented_data=None):  # noqa: F811
    services = build_services(validation=svc, redis=redis)
    services.backend.get_rented_executors_now.return_value = fresh
    ctx = context_factory(services=services, state=build_state(specs=_spec(8), rented_data=rented_data), ssh=ssh)
    result = await CapabilityCheck().run(ctx)
    return result, result.updates.get("state", ctx.state)


def _proven(redis: FakeRedis) -> FakeRedis:
    redis.store[PASS_KEY] = "2026-11-23T11:00:00"
    return redis


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [_filler_on(range(4)), _pod_on(range(4))], ids=["filler", "pod"])
async def test_our_own_workload_starting_on_cards_0_3_during_the_run_is_not_a_failure(context_factory, svc, fresh):  # noqa: F811
    """The H200 false positive: our filler started on cards 0-3 during the run; they ran out of memory, 4-7 answered."""
    ssh, _ = _node({i: OOM for i in range(4)})
    redis = _proven(FakeRedis())

    result, state = await _cycle(context_factory, svc, ssh, redis=redis, fresh=fresh)

    assert result.passed is True
    assert result.event.reason_code == Msg.RENTED_SKIPPED.reason
    assert result.event.what_we_saw["allcards"] == {"outcome": "deferred", "reason": "lium_workload", "listed": True}
    assert state.allcards_not_proven is False
    assert state.rented_data is fresh
    assert DEFERRED_KEY in redis.store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers",
    [
        # out of memory on card 5 too, which no workload of ours holds
        {**{i: OOM for i in range(4)}, 5: OOM},
        # timed out on cards 0-3 and on card 6 (the run is retried; the retry times out again)
        {**{i: TimeoutError() for i in range(4)}, 6: TimeoutError()},
        # a run that ended without an exit status on an unassigned card is that card's verdict
        {**{i: OOM for i in range(4)}, 7: SimpleNamespace(exit_status=None, stdout="", stderr="")},
    ],
    ids=["oom", "timeout", "no_exit_status"],
)
async def test_a_card_with_no_workload_of_ours_assigned_fails(context_factory, svc, answers):  # noqa: F811
    ssh, _ = _node(answers)

    result, _ = await _cycle(context_factory, svc, ssh, redis=_proven(FakeRedis()), fresh=_filler_on(range(4)))

    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED.reason
    assert result.event.what_we_saw["allcards"]["outcome"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [None, _filler_on([0])], ids=["no_workload", "filler_on_the_real_card"])
async def test_fake_cards_that_give_no_answer_fail(context_factory, svc, fresh):  # noqa: F811
    """The spoofer: one real card answers, the claimed others give no answer and the backend assigned none of them."""
    ssh, _ = _node({i: NO_DEVICE for i in range(1, 8)})

    result, _ = await _cycle(context_factory, svc, ssh, redis=_proven(FakeRedis()), fresh=fresh)

    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED.reason


@pytest.mark.asyncio
async def test_a_wrong_answer_is_never_waived(context_factory, svc):  # noqa: F811
    ssh, _ = _node({0: SimpleNamespace(exit_status=0, stdout="UUID: someone-elses", stderr="")})

    result, _ = await _cycle(context_factory, svc, ssh, redis=_proven(FakeRedis()), fresh=_filler_on(range(8)))

    assert result.passed is False


@pytest.mark.asyncio
async def test_a_node_rented_at_the_snapshot_runs_no_all_cards_proof_and_is_not_offered(context_factory, svc):  # noqa: F811
    """The A6000 false positive: the rental ended mid-cycle and its workload still loaded the cards. That cycle runs
    no all-cards proof; the node is offered again only after its cards prove themselves in an idle cycle."""
    ssh, pinned = _node({i: TimeoutError() for i in range(2)})
    redis = _proven(FakeRedis())

    result, state = await _cycle(context_factory, svc, ssh, redis=redis, rented_data=_live_rental())

    assert pinned == []
    assert result.passed is True
    assert result.event.what_we_saw["allcards"] == {
        "outcome": "deferred",
        "reason": "rented_at_snapshot",
        "listed": False,
    }
    assert state.allcards_not_proven is True


@pytest.mark.asyncio
async def test_a_deferral_without_a_live_pass_keeps_the_node_off_new_offers(context_factory, svc):  # noqa: F811
    """A new node, or one whose last pass expired: a deferral never lists it."""
    ssh, _ = _node({i: OOM for i in range(4)})

    result, state = await _cycle(context_factory, svc, ssh, redis=FakeRedis(), fresh=_filler_on(range(4)))

    assert result.passed is True
    assert state.allcards_not_proven is True


@pytest.mark.asyncio
async def test_a_second_deferral_in_a_row_fails_and_a_pass_resets_it(context_factory, svc):  # noqa: F811
    """A node that fakes a busy card every cycle keeps nothing: the next idle cycle is the one retry."""
    redis = _proven(FakeRedis())
    busy, _ = _node({i: OOM for i in range(4)})

    first, _ = await _cycle(context_factory, svc, busy, redis=redis, fresh=_filler_on(range(4)))
    second, _ = await _cycle(context_factory, svc, busy, redis=redis, fresh=_filler_on(range(4)))

    assert first.passed is True
    assert second.passed is False
    assert second.event.what_we_saw["allcards"]["reason"].endswith("deferred last time too")

    healthy, _ = _node({})
    passed, state = await _cycle(context_factory, svc, healthy, redis=redis)
    assert passed.passed is True and state.allcards_not_proven is False
    assert DEFERRED_KEY not in redis.store
    assert redis.ttl[PASS_KEY] == capability.ALLCARDS_PASS_VALID_SECONDS
    again, _ = await _cycle(context_factory, svc, busy, redis=redis, fresh=_filler_on(range(4)))
    assert again.passed is True


@pytest.mark.asyncio
async def test_a_deferral_never_extends_the_last_pass(context_factory, svc):  # noqa: F811
    redis = _proven(FakeRedis())
    redis.ttl[PASS_KEY] = 60
    busy, _ = _node({i: OOM for i in range(4)})

    await _cycle(context_factory, svc, busy, redis=redis, fresh=_filler_on(range(4)))

    assert redis.store[PASS_KEY] == "2026-11-23T11:00:00" and redis.ttl[PASS_KEY] == 60


@pytest.mark.asyncio
async def test_a_lost_connection_to_the_whole_node_defers(context_factory, svc):  # noqa: F811
    ssh, _ = _node({i: ConnectionResetError("connection lost") for i in range(8)})

    result, state = await _cycle(context_factory, svc, ssh, redis=_proven(FakeRedis()))

    assert result.passed is True
    assert result.event.reason_code == Msg.ALLCARDS_DEFERRED.reason
    assert state.allcards_not_proven is False


@pytest.mark.asyncio
async def test_an_honest_pass_records_the_pass(context_factory, svc):  # noqa: F811
    ssh, pinned = _node({})
    redis = FakeRedis()

    result, state = await _cycle(context_factory, svc, ssh, redis=redis)

    assert sorted(pinned) == list(range(8))
    assert result.passed is True and result.event.reason_code == Msg.VERIFY_OK.reason
    assert state.allcards_not_proven is False
    assert PASS_KEY in redis.store
