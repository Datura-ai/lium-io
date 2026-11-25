"""DAH-6564: no verdict for a ports, rental-check or GPU-usage failure when the node's rentals or fillers changed
between the cycle-start rented list and the cycle-end one."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.executor_rollout import (
    RENTAL_CHANGE_STATE_KEY,
    ExecutorRolloutTracker,
    session_kept_hotkeys,
    withhold_rental_change_verdicts,
)
from services.task.models import build_msg
from test_executor_rollout_grace import _failed, _scored, _validator_process
from test_rented_host_gpu_fault_reset import _redis_service

MINER = "5Miner"
NODE = "node-1"
PORTS, RENTAL, GPU_USAGE = "INSUFFICIENT_PORTS", "RENTAL_NOT_VERIFIED", "GPU_USAGE_HIGH"
THE_THREE = [PORTS, RENTAL, GPU_USAGE]


def _listed(
    pods: dict[str, list[str]] | None = None,
    fillers: dict[str, list[str]] | None = None,
    hotkey: str = MINER,
    ip: str = "203.0.113.5",
    port: str = "8080",
    owners: dict[str, str] | None = None,
    pod_fields: dict | None = None,
) -> RentedExecutorsResponse:
    executors = {
        uuid: RentedExecutor(
            miner_hotkey=hotkey,
            executor_ip_address=ip,
            executor_ip_port=port,
            pods=[RentedPod(pod_id=pod_id, container_name=f"c-{pod_id}", **(pod_fields or {})) for pod_id in ids],
        )
        for uuid, ids in (pods or {}).items()
    }
    return RentedExecutorsResponse(
        executors=executors, all_filler_containers_by_executor=fillers or {}, executor_owner_hotkeys=owners or {}
    )


def _fail(reason: str, uuid: str = NODE, batch: str = "batch-1", **fields):
    result = _failed(uuid, reason)
    result.job_batch_id = batch
    for name, value in fields.items():
        setattr(result, name, value)
    return result


async def _rule(results, start, end, service=None, published_early=frozenset()):
    service = service or _redis_service()
    return await withhold_rental_change_verdicts({MINER: list(results)}, start, end, service, set(published_early))


def _withheld_uuids(withheld) -> list[str]:
    return [w.result.executor_info.uuid for w in withheld]


@pytest.mark.asyncio
async def test_a_filler_started_between_the_lists_withholds_a_gpu_usage_failure():
    failed = _fail(GPU_USAGE)
    standing, withheld = await _rule([failed], _listed(), _listed(fillers={NODE: ["filler_a"]}))
    assert standing == {MINER: []}
    assert _withheld_uuids(withheld) == [NODE]


@pytest.mark.parametrize("reason", [PORTS, RENTAL])
@pytest.mark.asyncio
async def test_a_rental_that_ended_between_the_lists_withholds_the_failure(reason):
    _, withheld = await _rule([_fail(reason)], _listed(pods={NODE: ["p1"]}), _listed())
    assert _withheld_uuids(withheld) == [NODE]


@pytest.mark.asyncio
async def test_a_rental_swapped_for_another_with_the_same_count_withholds():
    _, withheld = await _rule([_fail(PORTS)], _listed(pods={NODE: ["p1"]}), _listed(pods={NODE: ["p2"]}))
    assert _withheld_uuids(withheld) == [NODE]


@pytest.mark.asyncio
async def test_a_filler_only_node_whose_filler_changed_withholds():
    _, withheld = await _rule(
        [_fail(GPU_USAGE)], _listed(fillers={NODE: ["filler_a"]}), _listed(fillers={NODE: ["filler_b"]})
    )
    assert _withheld_uuids(withheld) == [NODE]


@pytest.mark.parametrize("reason", THE_THREE)
@pytest.mark.asyncio
async def test_the_same_pods_and_fillers_in_both_lists_let_the_failure_stand(reason):
    lists = _listed(pods={NODE: ["p1"]}, fillers={NODE: ["filler_a"]})
    failed = _fail(reason)
    standing, withheld = await _rule([failed], lists, lists)
    assert (standing, withheld) == ({MINER: [failed]}, [])


@pytest.mark.asyncio
async def test_a_pod_whose_status_ports_or_gpus_changed_under_the_same_id_lets_it_stand():
    start = _listed(pods={NODE: ["p1"]}, pod_fields={"status": "RUNNING", "rented_ports": [1], "gpu_count": 1})
    end = _listed(pods={NODE: ["p1"]}, pod_fields={"status": "REBOOT_PENDING", "rented_ports": [2], "gpu_count": 2})
    _, withheld = await _rule([_fail(RENTAL)], start, end)
    assert withheld == []


@pytest.mark.parametrize("reason", ["EXECUTOR_TRANSPORT_UNREACHABLE", "VERIFYX_FAILED", "GPU_LOST"])
@pytest.mark.asyncio
async def test_another_reason_stands_when_the_set_changed(reason):
    _, withheld = await _rule([_fail(reason)], _listed(pods={NODE: ["p1"]}), _listed())
    assert withheld == []


@pytest.mark.asyncio
async def test_a_pass_is_untouched_and_clears_the_withheld_state():
    service = _redis_service()
    await _rule([_fail(PORTS, batch="b1")], _listed(pods={NODE: ["p1"]}), _listed(), service)
    passed = _scored(NODE)

    standing, withheld = await _rule([passed], _listed(pods={NODE: ["p1"]}), _listed(), service)

    assert (standing, withheld) == ({MINER: [passed]}, [])
    assert await service.hgetall(RENTAL_CHANGE_STATE_KEY) == {}


@pytest.mark.asyncio
async def test_a_change_on_another_node_of_the_miner_does_not_withhold_this_one():
    _, withheld = await _rule([_fail(PORTS)], _listed(pods={"node-2": ["p1"]}), _listed())
    assert withheld == []


@pytest.mark.parametrize(
    ("ip", "withheld_uuids"), [("198.51.100.9", []), ("203.0.113.5", [NODE])], ids=["other-endpoint", "same-endpoint"]
)
@pytest.mark.asyncio
async def test_a_rental_listed_under_another_miner_counts_only_on_this_nodes_endpoint(ip, withheld_uuids):
    start = _listed(pods={NODE: ["p1"]}, hotkey="5Other", ip=ip)
    _, withheld = await _rule([_fail(PORTS)], start, _listed())
    assert _withheld_uuids(withheld) == withheld_uuids


@pytest.mark.asyncio
async def test_a_filler_under_a_uuid_owned_by_another_hotkey_does_not_count():
    owners = {NODE: "5Other"}
    start, end = _listed(owners=owners), _listed(fillers={NODE: ["filler_a"]}, owners=owners)
    _, withheld = await _rule([_fail(GPU_USAGE)], start, end)
    assert withheld == []


@pytest.mark.asyncio
async def test_a_node_declaring_too_few_ports_stands():
    event = build_msg(
        event="x", reason=PORTS, severity="error", impact="", what={"declared_port_count": 2, "required": 4}
    )
    _, withheld = await _rule([_fail(PORTS, validation_event=event)], _listed(pods={NODE: ["p1"]}), _listed())
    assert withheld == []


@pytest.mark.asyncio
async def test_an_earlier_gpu_finding_stands_with_the_failure():
    failed = _fail(RENTAL, availability_errors=[{"reason_code": "GPU_LOST"}])
    standing, withheld = await _rule([failed], _listed(pods={NODE: ["p1"]}), _listed())
    assert (standing, withheld) == ({MINER: [failed]}, [])


@pytest.mark.asyncio
async def test_a_failure_published_early_stands():
    _, withheld = await _rule([_fail(PORTS)], _listed(), _listed(fillers={NODE: ["f"]}), published_early={NODE})
    assert withheld == []


@pytest.mark.asyncio
async def test_no_cycle_end_list_lets_every_verdict_stand():
    _, withheld = await _rule([_fail(PORTS)], _listed(pods={NODE: ["p1"]}), None)
    assert withheld == []


@pytest.mark.asyncio
async def test_the_cap_lets_the_third_withheld_cycle_since_the_last_pass_stand():
    service = _redis_service()
    start, end = _listed(pods={NODE: ["p1"]}), _listed()
    outcomes = []
    # withheld; a published failure (no change); withheld; the third since the last pass stands
    for batch, cycle_end in [("b1", end), ("b2", start), ("b3", end), ("b4", end)]:
        _, withheld = await _rule([_fail(PORTS, batch=batch)], start, cycle_end, service)
        outcomes.append(bool(withheld))
    assert outcomes == [True, False, True, False]


@pytest.mark.asyncio
async def test_a_pass_resets_the_cap():
    service = _redis_service()
    start, end = _listed(pods={NODE: ["p1"]}), _listed()
    await _rule([_fail(PORTS, batch="b1")], start, end, service)
    await _rule([_fail(PORTS, batch="b2")], start, end, service)
    await _rule([_scored(NODE)], start, end, service)
    _, withheld = await _rule([_fail(PORTS, batch="b4")], start, end, service)
    assert _withheld_uuids(withheld) == [NODE]


@pytest.mark.asyncio
async def test_a_carried_result_does_not_clear_the_state():
    service = _redis_service()
    start, end = _listed(pods={NODE: ["p1"]}), _listed()
    await _rule([_fail(PORTS, batch="b1")], start, end, service)
    await _rule([_fail(PORTS, batch="b2")], start, end, service)
    carried = _scored(NODE)
    carried.carried_for_renter_gpu_reset = True
    await _rule([carried], start, end, service)
    _, withheld = await _rule([_fail(PORTS, batch="b4")], start, end, service)
    assert withheld == []


@pytest.mark.asyncio
async def test_the_same_job_batch_id_twice_counts_once():
    service = _redis_service()
    start, end = _listed(pods={NODE: ["p1"]}), _listed()
    for _ in range(3):
        _, withheld = await _rule([_fail(PORTS, batch="b1")], start, end, service)
    assert _withheld_uuids(withheld) == [NODE]
    _, withheld = await _rule([_fail(PORTS, batch="b2")], start, end, service)
    assert _withheld_uuids(withheld) == [NODE]


@pytest.mark.parametrize("broken", ["hgetall", "write_atomically"])
@pytest.mark.asyncio
async def test_redis_failing_lets_the_verdict_stand(broken):
    service = _redis_service()
    setattr(service, broken, AsyncMock(side_effect=ConnectionError("down")))
    failed = _fail(PORTS)
    standing, withheld = await _rule([failed], _listed(pods={NODE: ["p1"]}), _listed(), service)
    assert (standing, withheld) == ({MINER: [failed]}, [])


@pytest.mark.parametrize("answer", [None, ConnectionError("down")])
@pytest.mark.asyncio
async def test_the_cycle_end_read_with_no_answer_gives_none(answer):
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    read = AsyncMock(side_effect=answer) if isinstance(answer, Exception) else AsyncMock(return_value=answer)
    validator_process.backend_client = MagicMock(get_rented_executors_now=read)
    assert await validator_process.rented_at_cycle_end({MINER: [_fail(PORTS)]}) is None
    read.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_rule_failure_and_no_reset_hold_reads_no_second_list():
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.backend_client = MagicMock(get_rented_executors_now=AsyncMock())
    await validator_process.rented_at_cycle_end({MINER: [_scored(NODE), _fail("GPU_LOST")]})
    validator_process.backend_client.get_rented_executors_now.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_cycle_end_takes_the_withheld_result_out_and_it_keeps_no_miner_active():
    validator_process = _validator_process(ExecutorRolloutTracker(redis_service=MagicMock(), grace_cycles=0))
    validator_process.fetch_executor_digest_or_none = AsyncMock(return_value=None)
    validator_process.observe_executor_rollout = AsyncMock(return_value=MagicMock(covers=lambda _: False))
    validator_process.redis_service = _redis_service()
    validator_process.miner_service = MagicMock(outage_tally=None)
    failed, healthy = _fail(PORTS), _scored("node-2")

    kept, withheld = await validator_process.withhold_verdicts_for_rollout(
        {MINER: [failed, healthy]}, 7500, "batch-1", MagicMock(), _listed(pods={NODE: ["p1"]}), _listed()
    )

    assert kept == {MINER: [healthy]}
    assert [w.result for w in withheld] == [failed]
    assert session_kept_hotkeys(withheld) == set()


OUTDATED_IMAGE = "EXECUTOR_IMAGE_OUTDATED"


@pytest.mark.asyncio
async def test_a_rental_started_after_a_free_nodes_outdated_image_read_withholds_its_zero():
    # review on lium-io#2389: a node free when its OUTDATED image was read, rented before the cycle published
    standing, withheld = await _rule([_fail(OUTDATED_IMAGE)], _listed(), _listed(pods={NODE: ["p1"]}))

    assert standing == {MINER: []}
    assert _withheld_uuids(withheld) == [NODE]


@pytest.mark.asyncio
async def test_a_free_node_on_an_outdated_image_still_free_at_the_cycle_end_stands():
    standing, withheld = await _rule([_fail(OUTDATED_IMAGE)], _listed(), _listed())

    assert [r.executor_info.uuid for r in standing[MINER]] == [NODE]
    assert withheld == []


@pytest.mark.asyncio
async def test_an_unofficial_image_stands_when_a_rental_started():
    standing, withheld = await _rule([_fail("EXECUTOR_IMAGE_UNOFFICIAL")], _listed(), _listed(pods={NODE: ["p1"]}))

    assert [r.executor_info.uuid for r in standing[MINER]] == [NODE]
    assert withheld == []
