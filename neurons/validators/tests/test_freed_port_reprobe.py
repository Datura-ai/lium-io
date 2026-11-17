"""DAH-6548: ports a rental or filler freed during the cycle are re-probed before an idle node is judged on them."""

from __future__ import annotations

import time
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.messages import PortCountMessages
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.executor_connectivity.models import PortPair, PortVerificationResult

from tests.helpers import build_context_config, build_services, build_state, make_context

UUID = "executor-123"
FREED = list(range(20000, 20010))  # the only ports the node forwards; the batch-start rental held them
WITHHELD = PortCountMessages.VERDICT_WITHHELD_FREED_PORT_REPROBE.reason


def _result(ok: list[int], status: str | None = None) -> PortVerificationResult:
    pairs = tuple(PortPair(p, p) for p in ok)
    return PortVerificationResult(
        selected_ports=pairs,
        successful_ports=pairs,
        failed_ports=(),
        dind_port=None,
        dind_ok=bool(ok),
        sysbox_runtime=True,
        status=status or ("ok" if ok else "no_working_ports"),
        declared_port_count=len(FREED),
    )


def _rented(*, pod_ports: list[int] | None = None, fillers: list[int] | None = None) -> RentedExecutorsResponse:
    executors = {}
    if pod_ports:
        executors[UUID] = RentedExecutor(
            miner_hotkey="miner-hotkey",
            executor_ip_address="127.0.0.1",
            executor_ip_port="8001",
            pods=[RentedPod(pod_id="pod-new", container_name="container_pod-new", rented_ports=pod_ports)],
        )
    return RentedExecutorsResponse(
        executors=executors, filler_ports_by_executor={UUID: fillers} if fillers else {}
    )


def _ctx(*, probe=None, now=None, first_withhold=True, started_ago: float = 0.0):
    """The cycle probed nothing it could reach (0 ports): the batch-start rental (or filler, the same shape) held
    FREED, and the port check's re-read of the rented list (state.rented_data) shows it gone."""
    redis = AsyncMock()
    redis.record_freed_port_reprobe_withheld.return_value = first_withhold
    redis.record_external_probe_miss.return_value = True
    connectivity = AsyncMock()
    connectivity.verify_ports.side_effect = probe or (lambda *a, **k: _result(FREED))
    services = build_services(redis=redis, connectivity=connectivity)
    services.backend.get_rented_executors_now.return_value = _rented() if now is None else now
    state = build_state(
        verified_port_count=0,
        specs={"verified_ports": []},
        rented_data=_rented(),
        probe_excluded_ports=FREED,
    )
    ctx = make_context(state=state, services=services, config=build_context_config(job_batch_id="b"))
    return ctx.model_copy(
        update={
            "executor": ctx.executor.model_copy(update={"uuid": UUID}),
            "started_at_monotonic": time.monotonic() - started_ago,
        }
    )


@pytest.mark.asyncio
async def test_a_rental_that_ended_during_the_cycle_frees_ports_that_pass_on_the_reprobe():
    ctx = _ctx()
    result = await PortCountCheck().run(ctx)

    assert result.passed is True
    assert result.updates["port_count"] == len(FREED)
    assert result.updates["state"].specs["verified_ports"] == FREED
    assert result.updates["state"].verified_port_pairs == [(p, p) for p in FREED]
    kwargs = ctx.services.connectivity.verify_ports.call_args.kwargs
    assert kwargs["rented_ports"] == [] and kwargs["filler_ports"] == []
    ctx.services.redis.clear_freed_port_reprobe_withheld.assert_awaited()


@pytest.mark.asyncio
async def test_the_reprobe_measures_sysbox_again_after_a_filler_stop_failed_the_scrapes_test():
    # the scrape's Sysbox test (its first DinD) failed while the filler held the node: state says no Sysbox
    measured = replace(_result(FREED), sysbox_measured=True)
    ctx = _ctx(probe=lambda *a, **k: measured)
    ctx = ctx.model_copy(update={"state": replace(ctx.state, sysbox_runtime=False)})

    result = await PortCountCheck().run(ctx)

    assert ctx.services.connectivity.verify_ports.call_args.args[3] is True  # sysbox-runc requested
    assert result.updates["state"].sysbox_runtime is True
    assert result.updates["state"].specs["sysbox_runtime"] is True  # the backend rents from specs


@pytest.mark.asyncio
async def test_freed_ports_that_do_not_answer_still_fail_insufficient_ports():
    result = await PortCountCheck().run(_ctx(probe=lambda *a, **k: _result([])))

    assert result.passed is False
    assert result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason


@pytest.mark.asyncio
async def test_a_filler_still_running_is_not_reprobed_and_its_ports_count():
    ctx = _ctx()
    ctx = ctx.model_copy(update={"state": replace(ctx.state, rented_data=_rented(fillers=FREED))})
    result = await PortCountCheck().run(ctx)

    ctx.services.connectivity.verify_ports.assert_not_called()
    assert result.updates["port_count"] == len(FREED)


@pytest.mark.asyncio
async def test_a_new_rental_on_the_freed_ports_is_judged_rented_and_its_ports_are_not_probed():
    taken = FREED[:4]
    ctx = _ctx(now=_rented(pod_ports=taken), probe=lambda *a, **k: _result(FREED[4:]))
    result = await PortCountCheck().run(ctx)

    assert ctx.services.connectivity.verify_ports.call_args.kwargs["rented_ports"] == taken
    assert result.passed is True  # rented: exempt from the floor, as today
    assert result.updates["state"].rented_data.executors[UUID].pods[0].pod_id == "pod-new"
    assert not set(taken) & set(result.updates["state"].specs["verified_ports"])


@pytest.mark.asyncio
async def test_a_rental_that_started_during_the_reprobe_is_judged_rented():
    ctx = _ctx()
    ctx.services.backend.get_rented_executors_now.side_effect = [_rented(), _rented(pod_ports=[30000])]
    result = await PortCountCheck().run(ctx)

    assert result.passed is True
    assert result.updates["state"].rented_data.executors[UUID].pods


def _raise(*a, **k):
    raise RuntimeError("ssh reset")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["read_fails", "no_budget", "probe_raises", "probe_error", "connection_lost"],
)
async def test_an_unfinished_reprobe_withholds_the_verdict(case):
    probe = {
        "probe_raises": _raise,
        "probe_error": lambda *a, **k: _result([], status="error"),
        "connection_lost": lambda *a, **k: PortVerificationResult(
            selected_ports=(), successful_ports=(), failed_ports=(), dind_port=None, dind_ok=False,
            sysbox_runtime=False, status="error", connection_lost=True,
        ),
    }.get(case)
    ctx = _ctx(probe=probe, started_ago=700 if case == "no_budget" else 0)
    if case == "read_fails":
        ctx.services.backend.get_rented_executors_now.return_value = None

    result = await PortCountCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == WITHHELD
    if case == "no_budget":
        ctx.services.connectivity.verify_ports.assert_not_called()


@pytest.mark.asyncio
async def test_the_second_affected_cycle_in_a_row_gets_its_verdict():
    ctx = _ctx(now=None, first_withhold=False)
    ctx.services.backend.get_rented_executors_now.return_value = None
    result = await PortCountCheck().run(ctx)

    assert result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason
    ctx.services.redis.clear_freed_port_reprobe_withheld.assert_awaited()


@pytest.mark.asyncio
async def test_a_redis_error_gives_the_verdict_as_before():
    ctx = _ctx()
    ctx.services.backend.get_rented_executors_now.return_value = None
    ctx.services.redis.record_freed_port_reprobe_withheld.side_effect = ConnectionError("down")

    result = await PortCountCheck().run(ctx)

    assert result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason


@pytest.mark.asyncio
async def test_nothing_skipped_means_no_reprobe():
    ctx = _ctx()
    ctx = ctx.model_copy(update={"state": replace(ctx.state, probe_excluded_ports=[])})
    result = await PortCountCheck().run(ctx)

    ctx.services.connectivity.verify_ports.assert_not_called()
    assert result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason


@pytest.mark.asyncio
async def test_a_confirmed_outside_block_credits_no_reprobed_port(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EXTERNAL_PORT_PROBE_ENFORCE", True)
    blocked = PortVerificationResult(
        selected_ports=tuple(PortPair(p, p) for p in FREED),
        successful_ports=tuple(PortPair(p, p) for p in FREED),
        failed_ports=(), dind_port=None, dind_ok=True, sysbox_runtime=True, status="ok", external_probe="blocked",
    )
    ctx = _ctx(probe=lambda *a, **k: blocked)
    ctx.services.redis.record_external_probe_miss.return_value = False  # a miss already on record

    result = await PortCountCheck().run(ctx)

    assert result.passed is False and result.updates["port_count"] == 0
    assert result.updates["state"].externally_unreachable is True


def test_the_cycle_withholds_a_run_whose_reprobe_did_not_finish():
    from services.executor_rollout import withhold_renter_gpu_reset_verdicts
    from test_executor_rollout_grace import _failed, _scored

    reprobe = _failed("node-1", WITHHELD)
    node_fault = _failed("node-2", "INSUFFICIENT_PORTS")
    scored = _scored("node-3")

    standing, withheld = withhold_renter_gpu_reset_verdicts({"5Miner": [reprobe, node_fault, scored]})

    assert standing == {"5Miner": [node_fault, scored]}
    assert [w.result for w in withheld] == [reprobe]


def test_a_run_whose_reprobe_did_not_finish_still_publishes_an_earlier_checks_recorded_failure():
    from services.executor_rollout import withhold_renter_gpu_reset_verdicts
    from test_executor_rollout_grace import _failed

    reprobe = _failed("node-1", WITHHELD)
    reprobe.availability_errors = [{"code": "GPU_COUNT_MISMATCH"}]

    standing, withheld = withhold_renter_gpu_reset_verdicts({"5Miner": [reprobe]})

    assert (standing, withheld) == ({"5Miner": [reprobe]}, [])


@pytest.mark.asyncio
async def test_a_reprobe_with_no_outside_reading_keeps_the_cycles_outside_verdict(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "EXTERNAL_PORT_PROBE_ENFORCE", True)
    ctx = _ctx()  # _result's re-probe has no outside reading (external_probe None)
    ctx = ctx.model_copy(update={"state": replace(ctx.state, externally_unreachable=True)})

    result = await PortCountCheck().run(ctx)

    assert result.passed is False and result.updates["port_count"] == 0
    assert result.updates["state"].externally_unreachable is True
    assert result.updates["state"].verified_port_pairs == []


@pytest.mark.asyncio
async def test_a_stale_pod_restarted_during_the_reprobe_still_holds_the_node():
    from types import SimpleNamespace

    from neurons.validators.src.services.task.checks.freed_port_reprobe import reprobe_freed_ports

    def _stale_pod_listed() -> RentedExecutorsResponse:
        listed = _rented(pod_ports=FREED)
        pod = listed.executors[UUID].pods[0].model_copy(update={"pod_id": "pod-stale"})
        return listed.model_copy(
            update={"executors": {UUID: listed.executors[UUID].model_copy(update={"pods": [pod]})}}
        )

    ctx = _ctx(now=_stale_pod_listed())
    ctx = ctx.model_copy(update={"state": replace(ctx.state, rented_data=_stale_pod_listed())})
    ssh = AsyncMock()
    ssh.run.return_value = SimpleNamespace(stdout="container_pod-new\n")  # docker ps: running again
    ctx = ctx.model_copy(update={"ssh": ssh})

    reprobe = await reprobe_freed_ports(ctx, min_ports=len(FREED), ended_pod_ids={"pod-stale"})

    pods = reprobe.state.rented_data.executors[UUID].pods
    assert [pod.pod_id for pod in pods] == ["pod-stale"]
