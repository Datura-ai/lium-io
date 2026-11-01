"""DAH-4709: a node whose cycle started inside its provider maintenance window fails PROVIDER_MAINTENANCE.

Half-open window [starting_at, expiring_at) against the cycle start (job_batch_id); verification is
kept; a rented node is judged after its pod states are read and a downed pod is not recovered.
"""

from datetime import UTC, datetime, timedelta

import pytest
from helpers import build_context_config, build_services, build_state
from neurons.validators.src.services.task.checks.provider_maintenance import (
    ProviderMaintenanceCheck,
    maintenance_window_at_cycle_start,
)
from neurons.validators.src.services.task.checks.rented_machine import TenantEnforcementCheck
from neurons.validators.src.services.task.messages import TenantEnforcementMessages
from neurons.validators.src.services.task.pipeline import ParallelStage
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from test_rented_machine_check import (
    DummyBackendClient,
    DummyScoreCalculator,
    DummySSHClient,
    build_rented_data,
)

from protocol.vc_protocol.compute_requests import MaintenanceWindow, RentedExecutorsResponse

EXECUTOR = "executor-123"
START = datetime(2026, 10, 23, 10, 0, tzinfo=UTC)
END = START + timedelta(hours=6)
WINDOW = MaintenanceWindow(starting_at=START, expiring_at=END)
POD = {"containers": [{"name": "tenant-123", "pod_id": "pod-1"}]}


def _batch(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _ctx(
    context_factory,
    cycle_start: datetime,
    *,
    rented_machine=None,
    ssh=None,
    backend=None,
    services=None,
):
    rented_data = build_rented_data(EXECUTOR, rented_machine) or RentedExecutorsResponse(
        executors={}, banned_guids=[]
    )
    rented_data = rented_data.model_copy(
        update={"maintenance_windows_by_executor": {EXECUTOR: WINDOW}}
    )
    services = services or build_services(
        score_calculator=DummyScoreCalculator(), backend=backend or DummyBackendClient(active=True)
    )
    return context_factory(
        services=services,
        config=build_context_config(job_batch_id=_batch(cycle_start)),
        state=build_state(gpu_processes=[], gpu_details=[], rented_data=rented_data),
        ssh=ssh,
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )


@pytest.mark.parametrize(
    ("cycle_start", "inside"),
    [
        (START - timedelta(seconds=1), False),
        (START, True),
        (END - timedelta(seconds=1), True),
        (END, False),
        (END + timedelta(seconds=1), False),
    ],
)
def test_the_window_is_half_open_on_the_cycle_start(context_factory, cycle_start, inside):
    assert (
        maintenance_window_at_cycle_start(_ctx(context_factory, cycle_start)) is not None
    ) is inside


@pytest.mark.asyncio
async def test_an_idle_node_in_its_window_fails_without_clearing_verification(context_factory):
    result = await ProviderMaintenanceCheck().run(_ctx(context_factory, START))

    assert result.passed is False
    assert ProviderMaintenanceCheck.fatal is True
    assert result.event.reason_code == "PROVIDER_MAINTENANCE"
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_an_idle_node_outside_its_window_passes(context_factory):
    result = await ProviderMaintenanceCheck().run(_ctx(context_factory, END))

    assert result.passed is True


@pytest.mark.asyncio
async def test_a_rented_node_passes_on_to_tenant_enforcement(context_factory):
    result = await ProviderMaintenanceCheck().run(_ctx(context_factory, START, rented_machine=POD))

    assert result.passed is True


@pytest.mark.parametrize("fast_path", [False, True])
def test_the_idle_check_runs_before_every_probe(fast_path):
    names: list[str] = []
    for chk in PipelineFactory.build_checks(fast_path=fast_path):
        names.extend(
            type(c).__name__ for c in (chk.checks if isinstance(chk, ParallelStage) else [chk])
        )

    at = names.index("ProviderMaintenanceCheck")
    for probe in (
        "PortConnectivityCheck",
        "RegistryPullCheck",
        "ContainerDownloadSpeedCheck",
        "VerifyXCheck",
        "RentalProbeCheck",
        "TenantEnforcementCheck",
    ):
        assert at < names.index(probe), probe


@pytest.mark.asyncio
async def test_a_rented_node_in_its_window_with_a_downed_pod_is_maintenance_not_recovered(
    context_factory,
):
    backend = DummyBackendClient(active=True)
    services = build_services(score_calculator=DummyScoreCalculator(), backend=backend)
    ctx = _ctx(
        context_factory,
        START,
        rented_machine=POD,
        ssh=DummySSHClient(pod_running=False),
        services=services,
    )

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == "PROVIDER_MAINTENANCE"
    assert "clear_verified_job_info" not in result.updates
    assert result.updates["rented"] is True
    assert {s.pod_id: s.container_state.value for s in result.updates["state"].pod_states} == {
        "pod-1": "exited"
    }
    assert backend.called_with == []
    services.pod_recovery.recover_pod_after_stale_vloopback_mount.assert_not_called()


@pytest.mark.asyncio
async def test_a_rented_node_in_its_window_with_a_running_pod_is_maintenance(context_factory):
    ctx = _ctx(context_factory, START, rented_machine=POD, ssh=DummySSHClient(pod_running=True))

    result = await TenantEnforcementCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == "PROVIDER_MAINTENANCE"
    assert {s.pod_id: s.container_state.value for s in result.updates["state"].pod_states} == {
        "pod-1": "running"
    }


@pytest.mark.asyncio
async def test_after_the_window_a_downed_pod_is_pod_not_running_again(context_factory):
    ctx = _ctx(context_factory, END, rented_machine=POD, ssh=DummySSHClient(pod_running=False))

    result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == TenantEnforcementMessages.POD_NOT_RUNNING.reason
    assert result.updates["clear_verified_job_info"] is True
