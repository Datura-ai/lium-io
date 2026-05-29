"""Tests for StaleContainerCleanupCheck (DAH-2164 follow-up).

The check reaps orphaned non-rented rental containers BEFORE the port checks so an
orphan that outlived its rental (e.g. a BROKEN_BY_PROVIDER pod whose container the
platform does not tear down) cannot keep binding the rental port range and deadlock
port verification.
"""

import pytest
from helpers import build_services, build_state, default_executor, make_context
from neurons.validators.src.services.task.checks import StaleContainerCleanupCheck
from neurons.validators.src.services.task.checks.stale_container_cleanup import (
    StaleContainerCleanupCheck as DirectImport,
)
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory

from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)


class RecordingContainerCleanup:
    """Records cleanup() calls and returns a configurable (count, names) result."""

    def __init__(self, result=(0, [])):
        self._result = result
        self.calls = []

    async def cleanup(self, ssh_client, rented_data, executor_uuid):
        self.calls.append(
            {
                "ssh_client": ssh_client,
                "rented_data": rented_data,
                "executor_uuid": executor_uuid,
            }
        )
        return self._result


def _make_ctx(cleanup, rented_data=None):
    services = build_services(container_cleanup=cleanup)
    state = build_state(rented_data=rented_data)
    return make_context(services=services, state=state, ssh="ssh-conn-sentinel")


@pytest.mark.asyncio
async def test_runs_cleanup_and_passes():
    cleanup = RecordingContainerCleanup(result=(2, ["pod_orphan", "filler_old"]))
    ctx = _make_ctx(cleanup)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.passed is True
    assert result.event.check_id == "executor.cleanup.stale_containers"
    assert result.event.reason_code == "STALE_CLEANUP_DONE"
    assert result.event.what_we_saw["removed_count"] == 2
    assert result.event.what_we_saw["removed_containers"] == ["pod_orphan", "filler_old"]


@pytest.mark.asyncio
async def test_passes_cleanup_args_through():
    cleanup = RecordingContainerCleanup()
    executor = default_executor()
    rented_data = RentedExecutorsResponse(
        executors={
            executor.uuid: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8080",
                pods=[RentedPod(pod_id="p1", container_name="pod_p1")],
                owner_flag=False,
            )
        },
        banned_guids=[],
    )
    services = build_services(container_cleanup=cleanup)
    state = build_state(rented_data=rented_data)
    ctx = make_context(executor=executor, services=services, state=state, ssh="ssh-conn-sentinel")

    await StaleContainerCleanupCheck().run(ctx)

    assert len(cleanup.calls) == 1
    call = cleanup.calls[0]
    assert call["ssh_client"] == "ssh-conn-sentinel"
    assert call["executor_uuid"] == executor.uuid
    assert call["rented_data"] is rented_data


@pytest.mark.asyncio
async def test_noop_cleanup_still_passes():
    cleanup = RecordingContainerCleanup(result=(0, []))
    ctx = _make_ctx(cleanup)

    result = await StaleContainerCleanupCheck().run(ctx)

    assert result.passed is True
    assert result.event.what_we_saw["removed_count"] == 0


def test_check_is_not_fatal():
    # Cleanup must never change the executor's verdict on its own.
    assert StaleContainerCleanupCheck.fatal is False
    assert StaleContainerCleanupCheck is DirectImport


def test_cleanup_runs_before_port_checks_in_production_pipeline():
    """Regression guard for the deadlock: cleanup must precede the port checks so a
    fatal PortCountCheck can never halt the pipeline before the orphan is reaped."""
    check_ids = [type(c).__name__ for c in PipelineFactory.build_checks()]

    assert "StaleContainerCleanupCheck" in check_ids
    idx_cleanup = check_ids.index("StaleContainerCleanupCheck")
    idx_conn = check_ids.index("PortConnectivityCheck")
    idx_count = check_ids.index("PortCountCheck")
    idx_tenant = check_ids.index("TenantEnforcementCheck")

    assert idx_cleanup < idx_conn < idx_count < idx_tenant
