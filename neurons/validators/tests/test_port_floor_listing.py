"""A node that declares 40000-65535 and publishes available_port_count=2 with VALIDATION_COMPLETED.

The backend lists a node only at available_port_count >= MIN_PORT_COUNT, so that run leaves the node
hidden from renters while the validator reports it as passing. The tests below walk the real selector,
probe cascade and checks to show where the 2 comes from and what each run now reports.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo

from neurons.validators.src.services.task.checks.finalize import FinalizeCheck
from neurons.validators.src.services.task.checks.port_connectivity import PortConnectivityCheck
from neurons.validators.src.services.task.checks.port_count import (
    LISTING_PORT_CHECK_CODE,
    PortCountCheck,
    port_floor_what,
)
from neurons.validators.src.services.task.checks.rented_machine import TenantEnforcementCheck
from neurons.validators.src.services.task.messages import (
    FinalizeMessages,
    PortConnectivityMessages,
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


def executor(port_range: str = DECLARED_RANGE) -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid=EXECUTOR_UUID,
        address="127.0.0.1",
        port=8001,
        ssh_username="root",
        ssh_port=2200,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        port_range=port_range,
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


class DindFails:
    async def verify(self, port, *, ssh_client, host, container_name_prefix, sysbox_runtime, log_ctx=None):
        return DindProbeResult(success=False, sysbox_runtime=False, port=port)


def connectivity(batch, semi, fallback, dind=None) -> ExecutorConnectivityService:
    return ExecutorConnectivityService(
        orchestrator=ConnectivityOrchestrator(PortSelector(), PortProbe(batch, semi, fallback), dind or DindOk())
    )


class PerPodSSHClient(DummySSHClient):
    """`docker ps` answers per container: running only for the names in `running`."""

    def __init__(self, running: set[str]):
        super().__init__(pod_running=False)
        self.running = running

    async def run(self, command: str):
        result = await super().run(command)
        if "docker ps" in command:
            result.stdout = "container_id_123" if any(name in command for name in self.running) else ""
        return result


class PerPodBackendClient(DummyBackendClient):
    """get_pod_rental_active answers per pod: active only for the ids in `active`."""

    def __init__(self, active: set[str]):
        super().__init__(active=False)
        self.active_pods = active

    async def get_pod_rental_active(self, pod_id: str):
        record = await super().get_pod_rental_active(pod_id)
        record.active = pod_id in self.active_pods
        return record


class RentalsBackendClient(DummyBackendClient):
    """Answers the fresh rented-executors read PortConnectivityCheck makes after a failed verification."""

    def __init__(self, rented_data: RentedExecutorsResponse | None, **kwargs):
        super().__init__(**kwargs)
        self.rented_data = rented_data

    async def get_all_rented_executors(self):
        return self.rented_data


def rented_with_pods(*pod_ids: str) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            EXECUTOR_UUID: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8001",
                pods=[
                    RentedPod(pod_id=pod_id, container_name=f"container_{pod_id}", rented_ports=[65000 + i])
                    for i, pod_id in enumerate(pod_ids)
                ],
            )
        }
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


def run_context(
    context_factory,
    connectivity_service,
    *,
    rented_data=None,
    ssh=None,
    backend=None,
    port_range=DECLARED_RANGE,
    score_warning=None,
):
    redis = AsyncMock()
    redis.renting_in_progress.return_value = False
    redis.record_dind_probe_miss.return_value = True
    services = build_services(
        connectivity=connectivity_service,
        redis=redis,
        **({"backend": backend} if backend is not None else {}),
    )
    return context_factory(
        executor=executor(port_range),
        services=services,
        config=build_context_config(job_batch_id="batch-1"),
        state=build_state(rented_data=rented_data, gpu_processes=[], gpu_details=[]),
        ssh=ssh,
        score=1.0,
        job_score=1.0,
        score_warning=score_warning,
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
    """A wide-range host shape: 2 ports, a pod in the batch-start rented list that has since ended."""
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
        "listing_check": "INSUFFICIENT_VERIFIED_PORTS",
        "port_range": "40000-65535",
        "port_mappings_declared": False,
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pod_order",
    [("pod-live", "pod-stale"), ("pod-stale", "pod-live")],
    ids=["live-then-stale", "stale-then-live"],
)
async def test_a_live_rental_keeps_the_exemption_beside_a_stale_pod_with_the_enforcement_flag(
    context_factory, monkeypatch, pod_order
):
    monkeypatch.setattr(settings, "ENFORCE_PORT_FLOOR_ON_STALE_POD", True)
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_pods(*pod_order),
        ssh=PerPodSSHClient(running={"container_pod-live"}),
        backend=PerPodBackendClient(active={"pod-live"}),
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, ctx = await apply(ctx, PortCountCheck())
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert ctx.state.specs["available_port_count"] == 2
    assert count_result.passed is True
    # main's verdict for this node, flag or no flag: the stale pod is reported and the run goes on
    assert tenant_result.passed is True
    assert tenant_result.event.reason_code == TenantEnforcementMessages.STALE_POD_NOT_RUNNING.reason
    assert tenant_result.event.what_we_saw["pod_id"] == "pod-stale"


@pytest.mark.asyncio
async def test_every_listed_pod_stale_fails_insufficient_ports_with_the_enforcement_flag(
    context_factory, monkeypatch
):
    monkeypatch.setattr(settings, "ENFORCE_PORT_FLOOR_ON_STALE_POD", True)
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_pods("pod-a", "pod-b"),
        ssh=PerPodSSHClient(running=set()),
        backend=PerPodBackendClient(active=set()),
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    _, ctx = await apply(ctx, PortCountCheck())
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert tenant_result.passed is False
    assert tenant_result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason
    assert tenant_result.event.what_we_saw["stale_pod"]["pod_id"] == "pod-a"


@pytest.mark.asyncio
async def test_a_stale_pod_beside_a_down_pod_with_an_open_rental_is_not_enforced(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "ENFORCE_PORT_FLOOR_ON_STALE_POD", True)
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_pods("pod-stale", "pod-down"),
        ssh=PerPodSSHClient(running=set()),
        backend=PerPodBackendClient(active={"pod-down"}),
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    _, ctx = await apply(ctx, PortCountCheck())
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert tenant_result.passed is True
    assert tenant_result.event.reason_code == TenantEnforcementMessages.STALE_POD_NOT_RUNNING.reason


@pytest.mark.asyncio
async def test_a_stale_pod_beside_a_running_pod_with_a_closed_rental_is_not_enforced(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "ENFORCE_PORT_FLOOR_ON_STALE_POD", True)
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_pods("pod-stale", "pod-running"),
        ssh=PerPodSSHClient(running={"container_pod-running"}),
        backend=PerPodBackendClient(active=set()),
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    _, ctx = await apply(ctx, PortCountCheck())
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert tenant_result.passed is True
    assert tenant_result.event.reason_code == TenantEnforcementMessages.STALE_POD_NOT_RUNNING.reason


@pytest.mark.asyncio
async def test_a_stale_pod_beside_a_down_pod_without_a_rental_record_is_not_enforced(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "ENFORCE_PORT_FLOOR_ON_STALE_POD", True)

    class NoRecordForSecondPod(PerPodBackendClient):
        async def get_pod_rental_active(self, pod_id: str):
            return None if pod_id == "pod-unknown" else await super().get_pod_rental_active(pod_id)

    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_pods("pod-stale", "pod-unknown"),
        ssh=PerPodSSHClient(running=set()),
        backend=NoRecordForSecondPod(active=set()),
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    _, ctx = await apply(ctx, PortCountCheck())
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert tenant_result.passed is True
    assert tenant_result.event.reason_code == TenantEnforcementMessages.STALE_POD_NOT_RUNNING.reason


@pytest.mark.asyncio
async def test_a_malformed_port_range_is_a_non_fatal_verify_failure_on_an_unrented_node(context_factory):
    """Flags off, `40000:65535` (a colon, not a dash): main's verdicts, never an exception out of the check."""
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()),
        backend=RentalsBackendClient(None),
        port_range="40000:65535",
    )

    connectivity_result, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, ctx = await apply(ctx, PortCountCheck())

    assert PortConnectivityCheck.fatal is False
    assert connectivity_result.passed is False
    assert connectivity_result.event.reason_code == PortConnectivityMessages.VERIFY_FAILED.reason
    assert connectivity_result.event.what_we_saw["verification_status"] == "error"
    assert ctx.state.verified_port_count == 0
    assert ctx.state.declared_port_count is None
    assert count_result.passed is False
    assert count_result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason


@pytest.mark.asyncio
async def test_a_malformed_port_range_keeps_a_rented_node_on_its_rented_score(context_factory):
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_one_pod(),
        ssh=DummySSHClient(pod_running=True),
        backend=RentalsBackendClient(rented_with_one_pod(), active=True),
        port_range="40000:65535",
    )

    connectivity_result, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, ctx = await apply(ctx, PortCountCheck())
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert connectivity_result.event.reason_code == PortConnectivityMessages.VERIFY_FAILED.reason
    assert count_result.passed is True
    assert tenant_result.passed is True
    assert tenant_result.halt is True
    assert tenant_result.event.reason_code == TenantEnforcementMessages.ALREADY_RENTED.reason


@pytest.mark.asyncio
async def test_topup_runs_when_the_dind_miss_takes_a_batch_of_three_below_the_floor(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "PORT_PROBE_TOPUP_BELOW_FLOOR", True)
    batch, semi = HostNetworkBatch(reachable=MIN_PORT_COUNT), PublishedPorts()
    ctx = run_context(context_factory, connectivity(batch, semi, PublishedPorts(), dind=DindFails()))

    _, ctx = await apply(ctx, PortConnectivityCheck())

    # the DinD port (the batch's first) failed, so 2 were left and the -p tiers got the failed ports
    assert len(semi.calls) == 1
    assert semi.calls[0][0] == PortPair(40003, 40003)
    assert ctx.state.verified_port_count == 2 + 50
    assert ctx.default_extra["probe_tier"] == "batch+semi_batch"
    assert ctx.default_extra["dind_ok"] is False


@pytest.mark.asyncio
async def test_a_dind_miss_on_a_batch_of_three_publishes_two_with_the_topup_flag_off(context_factory):
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=MIN_PORT_COUNT), PublishedPorts(), PublishedPorts(), dind=DindFails()),
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    _, ctx = await apply(ctx, PortCountCheck())

    assert ctx.state.specs["available_port_count"] == MIN_PORT_COUNT - 1
    assert ctx.default_extra["probe_tier"] == "batch"
    assert ctx.default_extra["dind_ok"] is False


@pytest.mark.asyncio
async def test_finalize_keeps_the_score_warning_and_adds_the_port_floor_fix(context_factory):
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_one_pod(),
        ssh=DummySSHClient(pod_running=False),
        backend=DummyBackendClient(active=False),
        score_warning="GPU runtime NVML driver/library mismatch",
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    _, ctx = await apply(ctx, PortCountCheck())
    _, ctx = await apply(ctx, TenantEnforcementCheck())
    final_result, _ = await apply(ctx, FinalizeCheck())

    remediation = final_result.event.remediation
    assert remediation.startswith("No action needed.GPU runtime NVML driver/library mismatch ")
    assert f"lowest {BATCH_PORT_VERIFICATION_SIZE} free ports of the declared range" in remediation
    assert f"allow at least {MIN_PORT_COUNT} of them through the host firewall" in remediation


@pytest.mark.asyncio
async def test_the_scored_zero_failure_names_the_listing_check_and_the_declared_range(context_factory):
    ctx = run_context(context_factory, connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()))

    _, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, _ = await apply(ctx, PortCountCheck())

    # the run's own verdict is unchanged: INSUFFICIENT_PORTS, score 0
    assert count_result.passed is False
    assert count_result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason
    what = count_result.event.what_we_saw
    assert what["listing_check"] == LISTING_PORT_CHECK_CODE == "INSUFFICIENT_VERIFIED_PORTS"
    assert (what["available_port_count"], what["required"]) == (2, MIN_PORT_COUNT)
    assert what["port_range"] == DECLARED_RANGE and what["port_mappings_declared"] is False
    assert what["held_by_orphaned_containers"] == []


@pytest.mark.asyncio
async def test_the_rented_exemption_still_passes_and_names_the_listing_check(context_factory):
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=2), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_one_pod(),
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, _ = await apply(ctx, PortCountCheck())

    assert count_result.passed is True
    assert count_result.event.reason_code == PortCountMessages.PORT_COUNT_RECORDED.reason
    assert count_result.event.severity == "warning"
    what = count_result.event.what_we_saw
    assert what["listing_check"] == "INSUFFICIENT_VERIFIED_PORTS"
    assert what["port_range"] == DECLARED_RANGE
    assert what["exempt_because_rented"] is True


def test_declared_port_mappings_are_named_without_the_mappings_themselves():
    state = SimpleNamespace(
        specs={"port_range": None, "port_mappings": "[[40000, 50000], [40001, 50001]]"},
        probed_port_count=2,
        declared_port_count=2,
    )

    what = port_floor_what(state, 1)

    assert what["port_mappings_declared"] is True
    assert what["port_range"] is None
    assert "port_mappings" not in what


def test_declared_port_mappings_drop_a_declared_range_too():
    """The mappings are what the node forwards; a range left in the specs next to them is not reported."""
    mappings = "[[40000, 50000], [40001, 50001]]"
    both = SimpleNamespace(
        specs={"port_range": DECLARED_RANGE, "port_mappings": mappings},
        probed_port_count=2,
        declared_port_count=2,
    )
    range_only = SimpleNamespace(specs={"port_range": DECLARED_RANGE}, probed_port_count=2, declared_port_count=2)

    assert port_floor_what(both, 1)["port_range"] is None
    assert port_floor_what(both, 1)["port_mappings_declared"] is True
    assert port_floor_what(range_only, 1)["port_range"] == DECLARED_RANGE
    assert port_floor_what(range_only, 1)["port_mappings_declared"] is False


@pytest.mark.parametrize("declared_range", [None, ""])
def test_no_declared_range_or_mappings_reports_the_default_probed_range(declared_range):
    state = SimpleNamespace(
        specs={"port_range": declared_range, "port_mappings": None}, probed_port_count=2, declared_port_count=2
    )

    what = port_floor_what(state, 1)

    assert what["port_range"] == "20000-65535"
    assert what["port_mappings_declared"] is False
