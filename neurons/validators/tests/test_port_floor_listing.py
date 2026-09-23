"""A node that declares 40000-65535 and publishes available_port_count=2 with VALIDATION_COMPLETED.

The backend lists a node only at available_port_count >= MIN_PORT_COUNT, so that run leaves the node
hidden from renters while the validator reports it as passing. The tests below walk the real selector,
probe cascade and checks to show where the 2 comes from and what each run now reports.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo

from neurons.validators.src.services.task.checks.finalize import FinalizeCheck
from neurons.validators.src.services.task.checks.port_connectivity import PortConnectivityCheck
from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.checks.rented_machine import TenantEnforcementCheck
from neurons.validators.src.services.task.messages import (
    FinalizeMessages,
    PortCountMessages,
    TenantEnforcementMessages,
)
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from services.const import BATCH_PORT_VERIFICATION_SIZE, MIN_PORT_COUNT
from services.executor_connectivity.models import DindProbeResult, PortPair
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector
from services.executor_connectivity.service import ExecutorConnectivityService

from tests.helpers import build_context_config, build_services, build_state
from tests.test_rented_machine_check import DummyBackendClient, DummySSHClient

DECLARED_RANGE = "40000-65535"
EXECUTOR_UUID = "executor-123"


def executor() -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=EXECUTOR_UUID,
        address="127.0.0.1",
        port=8001,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        port_range=DECLARED_RANGE,
    )


class HostNetworkBatch:
    """--network=host batch on a host whose firewall lets only `reachable` listeners through."""

    def __init__(self, reachable: int):
        self.reachable = reachable
        self.calls: list[list[PortPair]] = []

    async def verify(self, ports, *, ssh_client, host, log_ctx=None):
        self.calls.append(list(ports))
        return list(ports[: self.reachable]), list(ports[self.reachable :])


class PublishedPorts:
    """A -p tier: Docker's DNAT path, the one a renter's pod uses, answers on every port it publishes."""

    def __init__(self):
        self.calls: list[list[PortPair]] = []

    async def verify(self, ports, *, ssh_client, host, max_ports, log_ctx=None):
        tested = list(ports[:max_ports])
        self.calls.append(tested)
        return tested, []


class DindOk:
    async def verify(self, port, *, ssh_client, host, container_name_prefix, sysbox_runtime, log_ctx=None):
        return DindProbeResult(success=True, sysbox_runtime=True, port=port)


def connectivity(batch, semi, fallback) -> ExecutorConnectivityService:
    return ExecutorConnectivityService(
        orchestrator=ConnectivityOrchestrator(PortSelector(), PortProbe(batch, semi, fallback), DindOk())
    )


def rented_with_one_pod() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            EXECUTOR_UUID: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8001",
                pods=[RentedPod(pod_id="pod-1", container_name="container_pod-1", rented_ports=[65000, 65001])],
            )
        }
    )


def run_context(context_factory, connectivity_service, *, rented_data=None, ssh=None, backend=None):
    redis = AsyncMock()
    redis.renting_in_progress.return_value = False
    redis.record_dind_probe_miss.return_value = True
    services = build_services(
        connectivity=connectivity_service,
        redis=redis,
        **({"backend": backend} if backend is not None else {}),
    )
    return context_factory(
        executor=executor(),
        services=services,
        config=build_context_config(job_batch_id="batch-1"),
        state=build_state(rented_data=rented_data, gpu_processes=[], gpu_details=[]),
        ssh=ssh,
        score=1.0,
        job_score=1.0,
        score_warning=None,
        contract_version="1.0.3",
        collateral_deposited=True,
    )


async def apply(ctx, check):
    result = await check.run(ctx)
    return result, ctx.model_copy(update=result.updates or {})


@pytest.mark.asyncio
async def test_declared_40000_65535_publishes_two_when_the_host_network_batch_reaches_two(context_factory):
    batch, semi, fallback = HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()
    ctx = run_context(context_factory, connectivity(batch, semi, fallback))

    connectivity_result, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, ctx = await apply(ctx, PortCountCheck())

    assert connectivity_result.passed is True
    # one sample of the lowest 300 ports out of 25,536 declared; a partial batch is final
    assert len(batch.calls[0]) == BATCH_PORT_VERIFICATION_SIZE
    assert batch.calls[0][0] == PortPair(40000, 40000)
    assert semi.calls == [] and fallback.calls == []
    assert ctx.state.verified_port_count == 2
    assert ctx.state.probed_port_count == BATCH_PORT_VERIFICATION_SIZE
    assert ctx.state.declared_port_count == 65535 - 40000 + 1
    assert ctx.default_extra["probe_tier"] == "batch"
    # the count the backend's listing gate reads
    assert ctx.state.specs["available_port_count"] == 2
    assert count_result.passed is False
    assert count_result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason
    assert count_result.event.what_we_saw["probed_port_count"] == BATCH_PORT_VERIFICATION_SIZE
    assert count_result.event.what_we_saw["declared_port_count"] == 25536


@pytest.mark.asyncio
async def test_a_stale_listed_pod_carries_two_ports_through_to_validation_completed(context_factory):
    """The d9888aff shape: 2 ports, a pod in the batch-start rented list that has since ended."""
    batch = HostNetworkBatch(reachable=2)
    ctx = run_context(
        context_factory,
        connectivity(batch, PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_one_pod(),
        ssh=DummySSHClient(pod_running=False),
        backend=DummyBackendClient(active=False),
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, ctx = await apply(ctx, PortCountCheck())
    tenant_result, ctx = await apply(ctx, TenantEnforcementCheck())
    final_result, ctx = await apply(ctx, FinalizeCheck())

    assert ctx.state.specs["available_port_count"] == 2
    # exempt at the port check because the list still named the pod
    assert count_result.passed is True
    assert count_result.event.severity == "warning"
    assert count_result.event.what_we_saw["listing_hidden"] is True
    assert "Hidden from renters: only 2 verified ports, need 3" in count_result.event.impact
    # the pod is gone, the run goes on as unrented and completes
    assert tenant_result.passed is True
    assert tenant_result.event.reason_code == TenantEnforcementMessages.STALE_POD_NOT_RUNNING.reason
    assert tenant_result.event.what_we_saw["port_floor"]["available_port_count"] == 2
    assert final_result.event.reason_code == FinalizeMessages.COMPLETED.reason
    assert final_result.event.impact.startswith("Hidden from renters: only 2 verified ports, need 3.")
    assert final_result.event.what_we_saw["port_floor"] == {
        "available_port_count": 2,
        "required": MIN_PORT_COUNT,
        "listing_hidden": True,
        "probed_port_count": BATCH_PORT_VERIFICATION_SIZE,
        "declared_port_count": 25536,
    }


@pytest.mark.asyncio
async def test_stale_pod_fails_insufficient_ports_with_the_enforcement_flag(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "ENFORCE_PORT_FLOOR_ON_STALE_POD", True)
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_one_pod(),
        ssh=DummySSHClient(pod_running=False),
        backend=DummyBackendClient(active=False),
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    _, ctx = await apply(ctx, PortCountCheck())
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert TenantEnforcementCheck.fatal is True
    assert tenant_result.passed is False
    assert tenant_result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason
    assert tenant_result.event.what_we_saw["available_port_count"] == 2
    assert tenant_result.event.what_we_saw["stale_pod"]["pod_id"] == "pod-1"


@pytest.mark.asyncio
async def test_stale_pod_with_enough_ports_is_unchanged(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "ENFORCE_PORT_FLOOR_ON_STALE_POD", True)
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=50), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_one_pod(),
        ssh=DummySSHClient(pod_running=False),
        backend=DummyBackendClient(active=False),
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, ctx = await apply(ctx, PortCountCheck())
    tenant_result, ctx = await apply(ctx, TenantEnforcementCheck())
    final_result, _ = await apply(ctx, FinalizeCheck())

    assert count_result.event.severity == "info"
    assert tenant_result.passed is True
    assert tenant_result.event.severity == "info"
    assert "port_floor" not in tenant_result.event.what_we_saw
    assert "port_floor" not in final_result.event.what_we_saw
    assert final_result.event.impact == "Job score=1.0, actual score=1.0"


@pytest.mark.asyncio
async def test_topup_flag_reprobes_the_failed_ports_through_published_ports(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "PORT_PROBE_TOPUP_BELOW_FLOOR", True)
    batch, semi, fallback = HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()
    ctx = run_context(context_factory, connectivity(batch, semi, fallback))

    _, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, ctx = await apply(ctx, PortCountCheck())

    # semi-batch gets the ports the batch failed, never the two it reached
    assert semi.calls[0][0] == PortPair(40002, 40002)
    assert len(semi.calls[0]) == 50
    assert fallback.calls == []
    assert ctx.state.verified_port_count == 52
    assert ctx.default_extra["probe_tier"] == "batch+semi_batch"
    assert count_result.passed is True
    assert ctx.state.specs["available_port_count"] == 52


@pytest.mark.asyncio
async def test_topup_leaves_a_batch_at_the_floor_alone(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "PORT_PROBE_TOPUP_BELOW_FLOOR", True)
    batch, semi = HostNetworkBatch(reachable=MIN_PORT_COUNT), PublishedPorts()
    ctx = run_context(context_factory, connectivity(batch, semi, PublishedPorts()))

    _, ctx = await apply(ctx, PortConnectivityCheck())

    assert semi.calls == []
    assert ctx.state.verified_port_count == MIN_PORT_COUNT
    assert ctx.default_extra["probe_tier"] == "batch"


@pytest.mark.asyncio
async def test_topup_falls_through_to_the_sequential_tier_when_semi_batch_reaches_nothing(
    context_factory, monkeypatch
):
    monkeypatch.setattr(settings, "PORT_PROBE_TOPUP_BELOW_FLOOR", True)

    class NothingPublished(PublishedPorts):
        async def verify(self, ports, *, ssh_client, host, max_ports, log_ctx=None):
            self.calls.append(list(ports[:max_ports]))
            return [], list(ports[:max_ports])

    batch, semi, fallback = HostNetworkBatch(reachable=1), NothingPublished(), PublishedPorts()
    ctx = run_context(context_factory, connectivity(batch, semi, fallback))

    _, ctx = await apply(ctx, PortConnectivityCheck())

    assert len(fallback.calls[0]) == 10
    assert ctx.state.verified_port_count == 11
    assert ctx.default_extra["probe_tier"] == "batch+fallback"
