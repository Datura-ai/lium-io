"""DAH-6564: a ports, rental-check or GPU-usage failure on an idle node is judged against the node's rentals and
fillers as the backend lists them when the check runs, not as the cycle-start snapshot listed them. A failure on a
node whose ownership did not change stands."""

from unittest.mock import patch

import pytest

from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    ExecutorHealthCheckResponse,
    FillerRunActiveResponse,
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from neurons.validators.src.services.container_cleanup import ContainerCleanup
from neurons.validators.src.services.task.checks.gpu_usage import GpuUsageCheck
from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.checks.rental_verification import RentalVerificationCheck
from neurons.validators.src.services.task.messages import GpuUsageMessages, PortCountMessages
from neurons.validators.src.services.task.messages import RentalVerificationMessages
from services.const import MIN_PORT_COUNT
from tests.helpers import build_context_config, build_services, build_state, make_context
from tests.test_gpu_usage_check import FILLER, LATE_FILLER, _idle_state
from tests.test_rental_verification_check import DummyBackendClient, FillerSSHClient

LATE_FILLER_RUN = LATE_FILLER.removeprefix("filler_")


def _listed(*, pods: bool = False, fillers: list[str] | None = None, filler_ports: list[int] | None = None):
    pod = RentedPod(pod_id="pod-1", container_name="container_pod-1", rented_ports=[65000])
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8000",
                pods=[pod] if pods else [],
            )
        }
        if pods
        else {},
        all_filler_containers_by_executor={"executor-123": fillers} if fillers else {},
        filler_ports_by_executor={"executor-123": filler_ports} if filler_ports else {},
    )


# PortCountCheck: the plain floor, not only a split shortfall, re-reads the list before failing


def _port_ctx(ports: int, now: RentedExecutorsResponse | None):
    ctx = make_context(state=build_state(verified_port_count=ports))
    ctx.services.backend.get_rented_executors_now.return_value = now
    return ctx


@pytest.mark.asyncio
async def test_a_rent_started_after_the_snapshot_keeps_the_rented_exemption_below_the_floor():
    result = await PortCountCheck().run(_port_ctx(MIN_PORT_COUNT - 2, _listed(pods=True)))
    assert result.passed is True
    assert result.updates["state"].rented_data.executors["executor-123"].pods


@pytest.mark.asyncio
async def test_a_filler_started_after_the_snapshot_counts_its_ports_toward_the_floor(caplog):
    caplog.set_level("INFO")
    result = await PortCountCheck().run(_port_ctx(MIN_PORT_COUNT - 2, _listed(filler_ports=[61000, 61001])))
    assert result.passed is True
    assert result.updates["port_count"] == MIN_PORT_COUNT
    assert "Port failure excused at check time" in caplog.text


@pytest.mark.parametrize("now", [None, _listed()], ids=["no-answer", "still-idle"])
@pytest.mark.asyncio
async def test_a_floor_failure_on_a_node_still_idle_stands(now, caplog):
    caplog.set_level("INFO")
    result = await PortCountCheck().run(_port_ctx(MIN_PORT_COUNT - 2, now))
    assert result.passed is False
    assert "excused" not in caplog.text
    assert result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason


# RentalVerificationCheck: the idle-node probe failure re-reads the list


class _Backend(DummyBackendClient):
    def __init__(self, now: RentedExecutorsResponse | None, **kwargs):
        super().__init__(
            response=ExecutorHealthCheckResponse(success=False, error="Container not responding"), **kwargs
        )
        self.now = now

    async def get_rented_executors_now(self):
        return self.now


async def _verify(backend: _Backend, ssh=None):
    services = build_services(backend=backend, container_cleanup=ContainerCleanup())
    ctx = make_context(services=services, state=build_state(specs={"verified_ports": [8080]}), ssh=ssh)
    with patch("neurons.validators.src.services.task.checks.rental_verification.settings"):
        return await RentalVerificationCheck().run(ctx)


@pytest.mark.asyncio
async def test_a_rent_started_during_the_rental_check_is_not_a_failure():
    result = await _verify(_Backend(_listed(pods=True)))
    assert result.passed is True
    assert result.event.reason_code == RentalVerificationMessages.RENTED_AT_CHECK_TIME.reason
    assert result.updates["state"].rented_data.executors["executor-123"].pods


@pytest.mark.asyncio
async def test_a_filler_started_during_the_rental_check_is_judged_as_the_filler():
    result = await _verify(_Backend(_listed(fillers=[LATE_FILLER])), ssh=FillerSSHClient(running=True))
    assert result.passed is True
    assert result.event.reason_code == RentalVerificationMessages.FILLER_VERIFIED.reason


@pytest.mark.asyncio
async def test_a_filler_started_during_the_rental_check_that_is_gone_is_still_judged():
    backend = _Backend(
        _listed(fillers=[LATE_FILLER]), filler_run_active=FillerRunActiveResponse(active=True, status="RUNNING")
    )
    result = await _verify(backend, ssh=FillerSSHClient(running=False))
    assert result.event.reason_code != RentalVerificationMessages.FILLER_VERIFIED.reason


@pytest.mark.parametrize("now", [None, _listed()], ids=["no-answer", "still-idle"])
@pytest.mark.asyncio
async def test_a_rental_check_failure_on_a_node_still_idle_stands(now):
    result = await _verify(_Backend(now))
    assert result.passed is False
    assert result.event.reason_code == RentalVerificationMessages.FAILED.reason


# GpuUsageCheck: load from a filler the snapshot missed is judged by the backend's run record now


async def _usage(context_factory, filler_run: FillerRunActiveResponse | None):
    state = _idle_state(
        [{"gpu_utilization": 60, "memory_utilization": 40, "memory_used_mb": 9000}],
        [{"pid": 111, "info": "0::/../docker-a.scope", "container_name": LATE_FILLER}],
        fillers=[FILLER],
    )
    services = build_services()
    services.backend.get_filler_run_active.return_value = filler_run
    ctx = context_factory(services=services, config=build_context_config(), state=state)
    return await GpuUsageCheck().run(ctx), services


@pytest.mark.asyncio
async def test_load_from_a_filler_started_after_the_snapshot_is_the_filler(context_factory, caplog):
    caplog.set_level("INFO")
    result, services = await _usage(context_factory, FillerRunActiveResponse(active=True, status="RUNNING"))
    assert result.passed is True
    services.backend.get_filler_run_active.assert_awaited_with(LATE_FILLER_RUN)
    assert f"excused for fillers started after the snapshot: ['{LATE_FILLER}']" in caplog.text


@pytest.mark.parametrize(
    "filler_run",
    [
        None,  # no answer excuses no load
        FillerRunActiveResponse(active=False),
        FillerRunActiveResponse(active=True, status="RUNNING", executor_id="another-executor"),
    ],
    ids=["no-answer", "not-live", "another-node"],
)
@pytest.mark.asyncio
async def test_load_under_a_filler_name_the_backend_does_not_place_here_stands(context_factory, filler_run):
    result, _ = await _usage(context_factory, filler_run)
    assert result.passed is False
    assert result.event.reason_code == GpuUsageMessages.USAGE_HIGH.reason
