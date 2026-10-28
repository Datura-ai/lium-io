"""G15: a rental is the node's only when the backend's rental names the miner being validated.

The backend keys rentals by executor UUID, which the miner reports. A box listed under another miner's
rented UUID would halt as rented and keep its score without the matmul ever running.
"""

import logging

import pytest

from neurons.validators.src.services.task.checks import rented_machine
from neurons.validators.src.services.task.checks.rented_machine import TenantEnforcementCheck
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from test_rented_machine_check import DummyScoreCalculator, DummySSHClient, MockContainerCleanup

from helpers import build_context_config, build_services, build_state

EXECUTOR_UUID = "executor-123"


def _rented_by(miner_hotkey: str, port: str = "22") -> RentedExecutorsResponse:
    # The context's executor is 127.0.0.1:22 (helpers.default_executor).
    return RentedExecutorsResponse(
        executors={
            EXECUTOR_UUID: RentedExecutor(
                miner_hotkey=miner_hotkey,
                executor_ip_address="127.0.0.1",
                executor_ip_port=port,
                pods=[RentedPod(pod_id="pod-1", container_name="tenant-1", rented_ports=[])],
            )
        },
        banned_guids=[],
    )


def _ctx(context_factory, rental_miner_hotkey: str, rental_port: str = "22"):
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
    )
    state = build_state(
        gpu_processes=[],
        gpu_details=[],
        gpu_model="NVIDIA RTX 4090",
        rented_data=_rented_by(rental_miner_hotkey, rental_port),
    )
    return context_factory(
        services=services,
        config=build_context_config(),
        state=state,
        ssh=DummySSHClient(pod_running=True),
        miner_hotkey="miner-hotkey",
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )


def _mismatch_logs(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if "RENTED_OWNER_MISMATCH" in record.getMessage()]


@pytest.mark.asyncio
async def test_own_rental_logs_no_mismatch(context_factory, caplog):
    ctx = _ctx(context_factory, rental_miner_hotkey="miner-hotkey")

    with caplog.at_level(logging.WARNING, logger=rented_machine.logger.name):
        result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code != Msg.NOT_RENTED.reason
    assert _mismatch_logs(caplog) == []


@pytest.mark.asyncio
async def test_another_miners_rental_is_logged_and_validated_as_not_rented(context_factory, caplog):
    ctx = _ctx(context_factory, rental_miner_hotkey="other-miner-hotkey", rental_port="2222")

    with caplog.at_level(logging.WARNING, logger=rented_machine.logger.name):
        result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.NOT_RENTED.reason
    assert result.updates["rented"] is False
    assert len(_mismatch_logs(caplog)) == 1


@pytest.mark.asyncio
async def test_another_miners_rental_on_same_endpoint_stays_rented(context_factory, caplog):
    """Same IP and port: the renter's pod is on this box, so the probes must not run beside it."""
    ctx = _ctx(context_factory, rental_miner_hotkey="other-miner-hotkey")

    with caplog.at_level(logging.WARNING, logger=rented_machine.logger.name):
        result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code != Msg.NOT_RENTED.reason
    assert len(_mismatch_logs(caplog)) == 1
