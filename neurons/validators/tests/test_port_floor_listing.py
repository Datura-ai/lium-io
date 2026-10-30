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
from services.executor_connectivity.models import BatchResult, DindProbeResult, PortPair
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

    async def verify(self, ports, *, ssh_client, host, log_ctx=None, min_ports=3):
        self.calls.append(list(ports))
        return BatchResult(list(ports[: self.reachable]), list(ports[self.reachable :]))


class PublishedPorts:
    """A -p tier: Docker's DNAT path, the one a renter's pod uses, answers on every port it publishes
    (only the first `reachable` when set)."""

    def __init__(self, reachable: int | None = None):
        self.reachable = reachable
        self.calls: list[list[PortPair]] = []

    async def verify(self, ports, *, ssh_client, host, max_ports, log_ctx=None):
        tested = list(ports[:max_ports])
        self.calls.append(tested)
        return tested[: self.reachable], tested[self.reachable :] if self.reachable is not None else []


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
    what = count_result.event.what_we_saw
    assert (what["probed_port_count"], what["declared_port_count"]) == (
        BATCH_PORT_VERIFICATION_SIZE,
        25536,
    )
    assert (what["listing_check"], what["port_range"]) == ("INSUFFICIENT_PORTS", DECLARED_RANGE)


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
    assert count_result.event.reason_code == PortCountMessages.PORT_COUNT_RECORDED.reason
    what = count_result.event.what_we_saw
    assert what["listing_hidden"] is True and what["exempt_because_rented"] is True
    assert what["port_range"] == DECLARED_RANGE
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
        "listing_check": "INSUFFICIENT_PORTS",
        "port_range": "40000-65535",
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
async def test_a_dind_miss_on_a_batch_of_three_publishes_two(context_factory):
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


MAPPINGS = "[[40000, 50000], [40001, 50001]]"
UNPARSED = ([], "[]", "{}", "not json", "[[40000]]")


@pytest.mark.parametrize(
    ("port_range", "port_mappings", "named_range"),
    [
        (None, MAPPINGS, None),
        (DECLARED_RANGE, MAPPINGS, None),  # mappings are what the node forwards
        ("", None, "20000-65535"),
        # same parse as the platform check: no [internal, external] pair means no mappings
        *[(DECLARED_RANGE, m, DECLARED_RANGE) for m in UNPARSED],
    ],
)
def test_port_floor_what_names_the_range_or_the_mappings(port_range, port_mappings, named_range):
    specs = {"port_range": port_range, "port_mappings": port_mappings}
    state = SimpleNamespace(specs=specs, probed_port_count=0, declared_port_count=2)

    what = port_floor_what(state, 1)

    assert what["port_range"] == named_range
    assert "port_mappings" not in what


@pytest.mark.asyncio
async def test_a_range_next_to_empty_mappings_is_named_while_the_event_says_no_ports_were_probed(
    context_factory,
):
    batch = HostNetworkBatch(reachable=2)
    ctx = run_context(
        context_factory,
        connectivity(batch, PublishedPorts(), PublishedPorts()),
        backend=RentalsBackendClient(None),
    )
    executor = ctx.executor.model_copy(update={"port_mappings": "[]"})
    ctx = ctx.model_copy(update={"executor": executor})

    _, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, _ = await apply(ctx, PortCountCheck())

    assert batch.calls == []  # the selector reads the empty mappings, not the range
    assert count_result.passed is False
    what = count_result.event.what_we_saw
    assert what["port_range"] == DECLARED_RANGE
    assert (what["probed_port_count"], what["declared_port_count"]) == (0, 0)


@pytest.mark.asyncio
async def test_ports_held_by_our_filler_count_toward_the_floor_and_can_be_rented(context_factory):
    """A 6-port node whose idle filler holds 2 ports and answers 2 of the other 4 passes with 4,
    and a rent can allocate them: a rental removes the filler and frees its ports (prod 22 Oct,
    declared ranges of 6-11). The platform's rent guard (verify 2) and prepare_ports_data take
    `verified_ports` minus the pods' busy ports, so the filler ports must be in that list too."""
    rented_data = RentedExecutorsResponse(
        executors={}, filler_ports_by_executor={"executor-123": [2004, 2001]}
    )
    state = build_state(
        verified_port_count=2, specs={"verified_ports": [2003, 2005]}, rented_data=rented_data
    )
    ctx = context_factory(state=state)

    result = await PortCountCheck().run(ctx)

    assert result.passed is True
    specs = result.updates["state"].specs
    assert result.updates["port_count"] == specs["available_port_count"] == 4
    # probed ports first (rental verification uses [0]), then the filler's
    assert specs["verified_ports"] == [2003, 2005, 2001, 2004]
    # rent path: no pod holds a port (a filler creates none), so all 4 are free to allocate
    busy_pod_ports: set[int] = set()
    assert len(set(specs["verified_ports"]) - busy_pod_ports) >= MIN_PORT_COUNT


@pytest.mark.asyncio
@pytest.mark.parametrize("enforce_floor", [True, False])
@pytest.mark.parametrize(
    "rental, ports, expected",
    [
        ("stale", 2, "INSUFFICIENT_PORTS"),
        ("stale", 3, "INSUFFICIENT_PORTS"),
        ("stale", 23, "INSUFFICIENT_PORTS"),
        ("stale", 24, None),
        ("live", 2, None),
        ("live", 3, None),
        ("live", 23, None),
        ("live", 24, None),
        ("unresolved", 2, None),
        ("unresolved", 3, None),
        ("unresolved", 23, None),
        ("unresolved", 24, None),
    ],
)
async def test_an_8_gpu_split_node_needs_its_ports_once_every_listed_pod_is_stale(
    context_factory, monkeypatch, rental, ports, expected, enforce_floor
):
    from dataclasses import replace

    if ports < MIN_PORT_COUNT and not enforce_floor:
        expected = None  # the plain floor on a stale pod stays behind its flag
    monkeypatch.setattr(settings, "ENFORCE_PORT_FLOOR_ON_STALE_POD", enforce_floor)

    class NoRecordForSecondPod(PerPodBackendClient):
        async def get_pod_rental_active(self, pod_id: str):
            return None if pod_id == "pod-unknown" else await super().get_pod_rental_active(pod_id)

    pods = {"stale": ("pod-stale",), "live": ("pod-stale", "pod-live"), "unresolved": ("pod-stale", "pod-unknown")}
    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=ports), PublishedPorts(ports), PublishedPorts(ports)),
        rented_data=rented_with_pods(*pods[rental]),
        ssh=PerPodSSHClient(running={"container_pod-live"}),
        backend=NoRecordForSecondPod(active={"pod-live"}),
    )
    ctx = ctx.model_copy(
        update={"state": replace(ctx.state, gpu_count=8, supports_gpu_splitting=True, gpu_splitting_min_count=1)}
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, ctx = await apply(ctx, PortCountCheck())
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert count_result.passed is True  # rented at the snapshot: exempt here
    assert ctx.state.specs["available_port_count"] == ports
    assert tenant_result.passed is (expected is None)
    if expected:
        assert tenant_result.event.reason_code == expected
        assert tenant_result.event.what_we_saw["stale_pod"]["pod_id"] == "pod-stale"


@pytest.mark.asyncio
async def test_a_split_node_with_every_port_open_passes_when_its_only_listed_pod_is_stale(context_factory):
    """Rented at the snapshot, the probe still goes on to the tier that reaches every port, so the stale-pod
    branch judges the node's real count and does not hide a node that has all its ports open."""
    from dataclasses import replace

    ctx = run_context(
        context_factory,
        connectivity(HostNetworkBatch(reachable=3), PublishedPorts(), PublishedPorts()),
        rented_data=rented_with_pods("pod-stale"),
        ssh=PerPodSSHClient(running=set()),
        backend=PerPodBackendClient(active=set()),
    )
    ctx = ctx.model_copy(
        update={"state": replace(ctx.state, gpu_count=8, supports_gpu_splitting=True, gpu_splitting_min_count=1)}
    )

    _, ctx = await apply(ctx, PortConnectivityCheck())
    count_result, ctx = await apply(ctx, PortCountCheck())
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert count_result.passed is True
    assert ctx.state.specs["available_port_count"] >= 24
    assert tenant_result.passed is True


class Firewalled:
    """Every probe tier answers on each port except the `blocked` ones."""

    def __init__(self, blocked: set[int]):
        self.blocked = blocked
        self.calls: list[list[PortPair]] = []

    async def verify(self, ports, *, max_ports=None, **_):
        tested = list(ports if max_ports is None else ports[:max_ports])
        self.calls.append(tested)
        ok = [p for p in tested if p.external not in self.blocked]
        bad = [p for p in tested if p.external in self.blocked]
        return BatchResult(ok, bad) if max_ports is None else (ok, bad)


def stale_split_ctx(context_factory, probe, last_port, filler_ports=()):
    """8 GPUs, one per split (24 ports), declaring 40000-<last_port>; its only rental, on 40000-40002, closed."""
    from dataclasses import replace

    rented = RentedExecutorsResponse(
        filler_ports_by_executor={EXECUTOR_UUID: list(filler_ports)},
        executors={
            EXECUTOR_UUID: RentedExecutor(
                miner_hotkey="miner-hotkey",
                executor_ip_address="127.0.0.1",
                executor_ip_port="8001",
                pods=[
                    RentedPod(
                        pod_id="pod-stale",
                        container_name="container_pod-stale",
                        rented_ports=[40000, 40001, 40002],
                    )
                ],
            )
        }
    )
    ctx = run_context(
        context_factory,
        connectivity(probe, probe, probe),
        rented_data=rented,
        ssh=PerPodSSHClient(running=set()),
        backend=PerPodBackendClient(active=set()),
        port_range=f"40000-{last_port}",
    )
    return ctx.model_copy(
        update={"state": replace(ctx.state, gpu_count=8, supports_gpu_splitting=True, gpu_splitting_min_count=1)}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("last_port, passed", [(40023, True), (40022, False)])
async def test_a_split_node_declaring_exactly_its_split_target_counts_the_stale_pods_freed_ports(
    context_factory, last_port, passed
):
    """The probe skips the stale rental's 40000-40002; with every pod stale the node is re-probed, those ports
    answer and count. One declared port fewer stays short."""
    probe = Firewalled(blocked=set())
    ctx = stale_split_ctx(context_factory, probe, last_port)

    _, ctx = await apply(ctx, PortConnectivityCheck())
    _, ctx = await apply(ctx, PortCountCheck())
    probes_before = len(probe.calls)
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert ctx.state.specs["available_port_count"] == last_port - 40000 + 1 - 3  # probe skipped the rented three
    assert len(probe.calls) > probes_before  # re-probed
    assert {40000, 40001, 40002} <= {p.external for call in probe.calls[probes_before:] for p in call}
    assert tenant_result.passed is passed
    if not passed:
        assert tenant_result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason


@pytest.mark.asyncio
async def test_a_stale_split_nodes_freed_ports_that_no_longer_answer_are_not_credited(context_factory):
    """Reviewer's case: 40000-40023 declared, the closed rental's 40000-40002 now firewalled. The other 21
    answer; the re-probe finds the three dead, so the node is still short of its 24."""
    ctx = stale_split_ctx(context_factory, Firewalled(blocked={40000, 40001, 40002}), 40023)

    _, ctx = await apply(ctx, PortConnectivityCheck())
    _, ctx = await apply(ctx, PortCountCheck())
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert ctx.state.specs["available_port_count"] == 21
    assert tenant_result.passed is False
    assert tenant_result.event.reason_code == PortCountMessages.INSUFFICIENT_PORTS.reason


@pytest.mark.asyncio
async def test_a_stale_split_nodes_reprobe_keeps_crediting_its_idle_fillers_ports(context_factory):
    """Reviewer's case: 40000-40023 declared, the closed rental held 40000-40002 and our idle filler holds
    40020-40023. The re-probe skips the filler's four like the first probe; they still count, so 20 + 4 = 24."""
    probe = Firewalled(blocked=set())
    ctx = stale_split_ctx(context_factory, probe, 40023, filler_ports=[40020, 40021, 40022, 40023])

    _, ctx = await apply(ctx, PortConnectivityCheck())
    _, ctx = await apply(ctx, PortCountCheck())
    probes_before = len(probe.calls)
    tenant_result, _ = await apply(ctx, TenantEnforcementCheck())

    assert ctx.state.specs["available_port_count"] == 21  # 17 probed + 4 filler
    assert not {40020, 40021, 40022, 40023} & {p.external for call in probe.calls[probes_before:] for p in call}
    assert tenant_result.passed is True
