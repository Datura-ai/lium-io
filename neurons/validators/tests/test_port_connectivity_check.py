import asyncio
import dataclasses
import time

import pytest
from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)
from neurons.validators.src.services.executor_connectivity.models import (
    DindLogCause,
    PortPair,
    PortRangeResult,
    PortVerificationResult,
)
from neurons.validators.src.services.redis_service import VerifiedPortsMemory
from neurons.validators.src.services.task.checks import (
    port_connectivity as port_connectivity_module,
)
from neurons.validators.src.services.task.checks.port_connectivity import (
    VERIFIED_PORTS_MEMORY_TTL_SECONDS,
    PortConnectivityCheck,
)
from neurons.validators.src.services.task.checks.sysbox_required import SysboxRequiredCheck
from neurons.validators.src.services.task.messages import PortConnectivityMessages as Msg

from core.config import settings
from tests.helpers import build_context_config, build_services, build_state
from tests.test_container_download_speed_check import FakeRunner
from tests.test_host_disk import output as pressure_output


# Mock result class matching DockerConnectionCheckResult
# Mock Redis service
class DummyRedis:
    def __init__(self, *, renting_in_progress: bool = False, dind_miss_on_record: bool = False):
        self.renting_in_progress_value = renting_in_progress
        self.verified_ports = {}
        self.deferrals = {}
        # DAH-3597: whether a DinD probe miss is already on record for the executor.
        self.dind_miss_on_record = dind_miss_on_record
        self.recorded_misses: list[tuple[str, str, int]] = []
        self.cleared_misses: list[tuple[str, str]] = []
        # ticket-0494: executors whose last DinD probe that reached the container found sysbox
        self.sysbox_verified: dict[tuple[str, str], int] = {}

    async def record_sysbox_verified(self, miner_hotkey: str, executor_id: str, ttl_seconds: int) -> None:
        self.sysbox_verified[(miner_hotkey, executor_id)] = ttl_seconds

    async def clear_sysbox_verified(self, miner_hotkey: str, executor_id: str) -> None:
        self.sysbox_verified.pop((miner_hotkey, executor_id), None)

    async def is_sysbox_verified(self, miner_hotkey: str, executor_id: str) -> bool:
        return (miner_hotkey, executor_id) in self.sysbox_verified

    # DAH-5244: the last completed check's ports and the deferral counters (real and shadow)
    verified_ports: dict[tuple[str, str], dict] = {}
    deferrals: dict[tuple[str, str, bool], int] = {}

    async def record_verified_ports(self, miner_hotkey, executor_id, identity, pairs, ttl_seconds):
        self.verified_ports[(miner_hotkey, executor_id)] = {"identity": identity, "pairs": pairs, "ttl": ttl_seconds}

    async def last_verified_ports(self, miner_hotkey, executor_id, identity):
        data = self.verified_ports.get((miner_hotkey, executor_id))
        if data is None or data["identity"] != identity:
            return None
        return VerifiedPortsMemory(pairs=list(data["pairs"]), measured_at=1_790_000_000.0)

    async def disk_pressure_deferrals(self, miner_hotkey, executor_id, *, shadow):
        return self.deferrals.get((miner_hotkey, executor_id, shadow), 0)

    async def record_disk_pressure_deferral(self, miner_hotkey, executor_id, *, shadow, ttl_seconds):
        key = (miner_hotkey, executor_id, shadow)
        self.deferrals[key] = self.deferrals.get(key, 0) + 1
        return self.deferrals[key]

    async def clear_disk_pressure_deferrals(self, miner_hotkey, executor_id, *, shadow):
        self.deferrals.pop((miner_hotkey, executor_id, shadow), None)

    async def renting_in_progress(self, miner_hotkey: str, executor_uuid: str) -> bool:
        return self.renting_in_progress_value

    async def record_dind_probe_miss(self, miner_hotkey: str, executor_id: str, ttl_seconds: int) -> bool:
        self.recorded_misses.append((miner_hotkey, executor_id, ttl_seconds))
        first = not self.dind_miss_on_record
        self.dind_miss_on_record = True
        return first

    async def clear_dind_probe_miss(self, miner_hotkey: str, executor_id: str) -> None:
        self.cleared_misses.append((miner_hotkey, executor_id))
        self.dind_miss_on_record = False

    # DAH-3698: first failed port verification inside the window
    port_miss_on_record: bool = False

    async def record_port_verify_miss(self, miner_hotkey: str, executor_id: str, ttl_seconds: int) -> bool:
        first = not self.port_miss_on_record
        self.port_miss_on_record = True
        return first


# Mock backend service
class DummyBackendService:
    async def get_all_rented_executors(self):
        """Mock method that returns rented executors data."""
        # Return None or empty rented data structure as needed
        return None


# Mock connectivity service
class DummyConnectivityService:
    def __init__(
        self,
        *,
        success: bool,
        log_text: str = "",
        sysbox_runtime: bool = False,
        verified_port_count: int = 0,
        status: str | None = None,
        dind_ok: bool | None = None,
        dind_error: DindLogCause | None = None,
        port_ranges: tuple[PortRangeResult, ...] = (),
        sysbox_measured: bool | None = None,
    ):
        """
        Args:
            success: Whether verify_ports returns success
            log_text: The log message from verification
            sysbox_runtime: The sysbox runtime state to return
            verified_port_count: Number of verified working ports
            status: Override the status field (e.g., "skipped_rental_active")
            dind_ok: Override whether the DinD probe reached its container (default: success)
        """
        self.dind_error = dind_error
        self.sysbox_measured = sysbox_measured
        self.port_ranges = port_ranges
        self.success = success
        self.log_text = log_text
        self.sysbox_runtime = sysbox_runtime
        self.verified_port_count = verified_port_count
        self.status = status
        self.dind_ok = success if dind_ok is None else dind_ok
        self.called_with: dict | None = None

    async def verify_ports(
        self,
        ssh_client,
        miner_hotkey: str,
        executor_info,
        sysbox_runtime: bool,
        rented_ports: list[int] | None = None,
        rented_pod_names: list[str] | None = None,
        filler_ports: list[int] | None = None,
        log_ctx: dict | None = None,
    ) -> PortVerificationResult:
        """Mock method that mimics the real connectivity service."""
        # Track what parameters we were called with
        self.called_with = {
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "sysbox_runtime": sysbox_runtime,
            "rented_ports": rented_ports,
            "filler_ports": filler_ports,
        }

        if self.verified_port_count:
            successful_ports = tuple(
                PortPair(8000 + i, 8000 + i) for i in range(self.verified_port_count)
            )
        else:
            successful_ports = tuple()
        failed_ports = tuple()

        # Use custom status if provided, otherwise default logic
        if self.status:
            status = self.status
            error = "Rental container active, skipping port check" if status == "skipped_rental_active" else None
        else:
            status = "ok" if self.success else "no_working_ports"
            error = None if self.success else "No working ports found"

        return PortVerificationResult(
            selected_ports=successful_ports,
            successful_ports=successful_ports,
            failed_ports=failed_ports,
            dind_port=successful_ports[0] if successful_ports else None,
            dind_ok=self.dind_ok,
            sysbox_runtime=self.sysbox_runtime,
            status=status,
            error=error,
            elapsed_sec=1.0,
            dind_error=self.dind_error,
            port_ranges=self.port_ranges,
            sysbox_measured=self.sysbox_measured,
        )


@pytest.mark.parametrize(
    "rented,renting_in_progress,has_config,verify_success,sysbox_runtime,status_override,expected_pass,expected_reason",
    [
        (False, True, True, True, False, None, True, Msg.VERIFY_OK.reason),
        # Missing config - fail
        (False, False, False, True, False, None, False, Msg.CONFIG_MISSING.reason),
        # Verification succeeds
        (False, False, True, True, False, None, True, Msg.VERIFY_OK.reason),
        # Verification succeeds with sysbox runtime
        (False, False, True, True, True, None, True, Msg.VERIFY_OK.reason),
        # Verification fails
        (False, False, True, False, False, None, False, Msg.VERIFY_FAILED.reason),
        # Rental container active - should now fail with proper error message
        (False, False, True, False, False, "skipped_rental_active", False, Msg.VERIFY_FAILED.reason),
    ],
)
@pytest.mark.asyncio
async def test_port_connectivity_check(
    rented,
    renting_in_progress,
    has_config,
    verify_success,
    sysbox_runtime,
    status_override,
    expected_pass,
    expected_reason,
    context_factory,
):
    # Setup mocks
    redis_service = DummyRedis(renting_in_progress=renting_in_progress)
    backend_service = DummyBackendService()
    connectivity_service = DummyConnectivityService(
        success=verify_success,
        log_text="Port verification completed" if verify_success else "Port verification failed",
        sysbox_runtime=sysbox_runtime,
        verified_port_count=100 if verify_success else 0,
        status=status_override,
    )

    services = build_services(
        redis=redis_service,
        backend=backend_service,
        connectivity=connectivity_service,
    )

    # Setup config with or without required keys
    if has_config:
        config = build_context_config(
            job_batch_id="batch-123",
        )
    else:
        config = build_context_config(
            job_batch_id=None,
        )

    state = build_state(sysbox_runtime=False)

    # Create context
    ctx = context_factory(
        services=services,
        config=config,
        state=state,
        rented=rented,
    )

    # Run the check
    result = await PortConnectivityCheck().run(ctx)

    # Verify result
    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    # Verify service interactions based on scenario
    if not has_config:
        # Should not call connectivity service
        assert connectivity_service.called_with is None
    else:
        # Should call connectivity service
        assert connectivity_service.called_with is not None
        # Verify state update with sysbox_runtime and verified_port_count
        if "state" in result.updates:
            assert result.updates["state"].sysbox_runtime == sysbox_runtime
            if verify_success:
                assert result.updates["state"].verified_port_count == 100
            else:
                assert result.updates["state"].verified_port_count == 0


@pytest.mark.asyncio
async def test_port_connectivity_passes_filler_ports_as_exclusions(context_factory):
    """DAH-2527: an idle filler holds ports without creating a pod, so this executor has no entry
    in rented_data.executors at all. Its ports must still reach verification as exclusions, and
    must not arrive as rented_ports — a non-empty rented_ports means "has a rental" downstream."""
    connectivity_service = DummyConnectivityService(success=True, verified_port_count=3)
    services = build_services(
        redis=DummyRedis(),
        backend=DummyBackendService(),
        connectivity=connectivity_service,
    )
    state = build_state(
        rented_data=RentedExecutorsResponse(
            executors={},
            filler_ports_by_executor={"executor-123": [40001, 40003]},
        )
    )
    ctx = context_factory(
        services=services,
        config=build_context_config(job_batch_id="batch-123"),
        state=state,
    )

    await PortConnectivityCheck().run(ctx)

    assert connectivity_service.called_with["filler_ports"] == [40001, 40003]
    assert connectivity_service.called_with["rented_ports"] == []


@pytest.mark.asyncio
async def test_port_connectivity_tolerates_sysbox_downgrade_during_renting(context_factory):
    """DAH-2272 (tolerate): a rental force-removes the DinD probe mid-flight,
    flipping the probe's sysbox result to False. When renting_in_progress, the
    check must NOT record the downgrade — it preserves the last known sysbox
    value and flags the tolerated downgrade so the miner isn't penalised."""
    redis_service = DummyRedis(renting_in_progress=True)
    backend_service = DummyBackendService()
    # Ports verify OK (status "ok"), but the probe reports sysbox False because
    # its DinD container was force-removed by the concurrent rental.
    connectivity_service = DummyConnectivityService(
        success=True,
        sysbox_runtime=False,
        verified_port_count=100,
    )
    services = build_services(
        redis=redis_service,
        backend=backend_service,
        connectivity=connectivity_service,
    )
    config = build_context_config(job_batch_id="batch-123")
    # Prior known-good sysbox value.
    state = build_state(sysbox_runtime=True)
    ctx = context_factory(services=services, config=config, state=state, rented=False)

    result = await PortConnectivityCheck().run(ctx)

    assert result.passed is True
    # Prior sysbox value preserved despite the probe reporting False.
    assert result.updates["state"].sysbox_runtime is True
    assert result.updates["state"].specs["sysbox_runtime"] is True
    assert result.updates["default_extra"].get("sysbox_downgrade_tolerated") is True


@pytest.mark.asyncio
async def test_port_connectivity_records_sysbox_downgrade_when_not_renting(context_factory):
    """Control for the tolerate path: the same sysbox downgrade IS recorded
    (not tolerated) when no rental is in progress."""
    redis_service = DummyRedis(renting_in_progress=False)
    backend_service = DummyBackendService()
    connectivity_service = DummyConnectivityService(
        success=True,
        sysbox_runtime=False,
        verified_port_count=100,
    )
    services = build_services(
        redis=redis_service,
        backend=backend_service,
        connectivity=connectivity_service,
    )
    config = build_context_config(job_batch_id="batch-123")
    state = build_state(sysbox_runtime=True)
    ctx = context_factory(services=services, config=config, state=state, rented=False)

    result = await PortConnectivityCheck().run(ctx)

    assert result.passed is True
    # Downgrade recorded — nothing to tolerate without a rental in progress.
    assert result.updates["state"].sysbox_runtime is False
    assert result.updates["default_extra"].get("sysbox_downgrade_tolerated") is None


# DAH-3597: a DinD probe that never reached its container is a miss, not a sysbox verdict.


def _grace_ctx(context_factory, *, redis_service, dind_ok: bool, probe_sysbox: bool = False):
    connectivity_service = DummyConnectivityService(
        success=True,
        sysbox_runtime=probe_sysbox,
        verified_port_count=100,
        dind_ok=dind_ok,
    )
    services = build_services(
        redis=redis_service,
        backend=DummyBackendService(),
        connectivity=connectivity_service,
    )
    config = build_context_config(job_batch_id="batch-123")
    state = build_state(sysbox_runtime=True)
    return context_factory(services=services, config=config, state=state, rented=False)


@pytest.mark.asyncio
async def test_first_dind_probe_miss_keeps_the_known_sysbox_value(context_factory, monkeypatch):
    """Flag on, no miss on record, the probe never reached its container (dind_ok False): the last
    known sysbox value stays, the miss is recorded with the configured TTL, and the extra names the
    reason. Before DAH-3597 the downgrade was recorded and SysboxRequiredCheck scored the cycle 0."""
    monkeypatch.setattr(settings, "DIND_PROBE_FIRST_MISS_GRACE", True)
    monkeypatch.setattr(settings, "DIND_PROBE_FIRST_MISS_GRACE_TTL_SECONDS", 1234)
    redis_service = DummyRedis()
    ctx = _grace_ctx(context_factory, redis_service=redis_service, dind_ok=False)

    result = await PortConnectivityCheck().run(ctx)

    assert result.passed is True
    assert result.updates["state"].sysbox_runtime is True
    assert result.updates["state"].specs["sysbox_runtime"] is True
    assert result.updates["default_extra"]["sysbox_downgrade_tolerated"] is True
    assert result.updates["default_extra"]["sysbox_downgrade_tolerated_reason"] == "first_dind_probe_miss"
    assert redis_service.recorded_misses == [(ctx.miner_hotkey, ctx.executor.uuid, 1234)]


@pytest.mark.asyncio
async def test_second_dind_probe_miss_inside_the_window_records_the_downgrade(context_factory, monkeypatch):
    """Flag on, a miss already on record: the downgrade is recorded as before the flag, and the
    extra says the miss repeated. Before DAH-3597 the extra did not exist, so a repeated miss and a
    first miss read the same in the log."""
    monkeypatch.setattr(settings, "DIND_PROBE_FIRST_MISS_GRACE", True)
    redis_service = DummyRedis(dind_miss_on_record=True)
    ctx = _grace_ctx(context_factory, redis_service=redis_service, dind_ok=False)

    result = await PortConnectivityCheck().run(ctx)

    assert result.passed is True
    assert result.updates["state"].sysbox_runtime is False
    assert result.updates["state"].specs["sysbox_runtime"] is False
    assert result.updates["default_extra"].get("sysbox_downgrade_tolerated") is None
    assert result.updates["default_extra"]["dind_probe_miss_repeated"] is True


@pytest.mark.asyncio
async def test_measured_no_sysbox_verdict_is_never_tolerated_and_clears_the_miss(context_factory, monkeypatch):
    """Flag on, the probe reached its container (dind_ok True) and the hello-world run said no
    sysbox: that is a verdict, so the downgrade is recorded, and the miss record is cleared because
    the probe worked. Before DAH-3597 nothing cleared a record, so a miss an hour apart would have
    read as the second one."""
    monkeypatch.setattr(settings, "DIND_PROBE_FIRST_MISS_GRACE", True)
    redis_service = DummyRedis(dind_miss_on_record=True)
    ctx = _grace_ctx(context_factory, redis_service=redis_service, dind_ok=True, probe_sysbox=False)

    result = await PortConnectivityCheck().run(ctx)

    assert result.passed is True
    assert result.updates["state"].sysbox_runtime is False
    assert result.updates["default_extra"].get("sysbox_downgrade_tolerated") is None
    assert redis_service.recorded_misses == []
    assert redis_service.cleared_misses == [(ctx.miner_hotkey, ctx.executor.uuid)]


@pytest.mark.asyncio
async def test_dind_probe_miss_grace_is_off_by_default(context_factory, monkeypatch):
    """Flag off (the default): a probe that never reached its container records the downgrade at
    once and the miss record is never read or written. Regression: the grace applied with the flag
    off, which would change scoring before the flag's owner turned it on."""
    from core.config import Settings

    monkeypatch.delenv("DIND_PROBE_FIRST_MISS_GRACE", raising=False)
    assert Settings(_env_file=None).DIND_PROBE_FIRST_MISS_GRACE is False  # no developer .env
    monkeypatch.setattr(settings, "DIND_PROBE_FIRST_MISS_GRACE", False)
    redis_service = DummyRedis()
    ctx = _grace_ctx(context_factory, redis_service=redis_service, dind_ok=False)

    result = await PortConnectivityCheck().run(ctx)

    assert result.updates["state"].sysbox_runtime is False
    assert result.updates["default_extra"].get("sysbox_downgrade_tolerated") is None
    assert redis_service.recorded_misses == []
    assert redis_service.cleared_misses == []


@pytest.mark.parametrize("bad", [0, -1])
def test_dind_probe_grace_ttl_rejects_zero_and_negative(bad):
    """Redis rejects `SET ... EX 0` and a negative EX with the same error, so every miss would raise
    inside the check; the setting refuses both at load time."""
    from pydantic import ValidationError

    from core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, DIND_PROBE_FIRST_MISS_GRACE_TTL_SECONDS=bad)


@pytest.mark.asyncio
async def test_port_connectivity_carries_the_dind_probe_cause_into_state(context_factory):
    """DAH-2856: the probe's cause reaches ContextState.dind_probe_error (SysboxRequiredCheck reads it)
    and the check's extra, and is None when the probe had nothing to say."""
    cause = DindLogCause("DIND_INNER_DOCKERD_IPTABLES", "the inner dockerd cannot use legacy iptables")
    services = build_services(
        redis=DummyRedis(renting_in_progress=False),
        backend=DummyBackendService(),
        connectivity=DummyConnectivityService(
            success=True, sysbox_runtime=False, verified_port_count=100, dind_error=cause
        ),
    )
    ctx = context_factory(
        services=services, config=build_context_config(job_batch_id="batch-123"), state=build_state(), rented=False
    )

    result = await PortConnectivityCheck().run(ctx)

    assert result.updates["state"].dind_probe_error == cause
    assert result.updates["default_extra"]["dind_error"] == cause.text

    services = build_services(
        redis=DummyRedis(renting_in_progress=False),
        backend=DummyBackendService(),
        connectivity=DummyConnectivityService(success=True, sysbox_runtime=True, verified_port_count=100),
    )
    ctx = context_factory(
        services=services, config=build_context_config(job_batch_id="batch-123"), state=build_state(), rented=False
    )
    result = await PortConnectivityCheck().run(ctx)
    assert result.updates["state"].dind_probe_error is None
    assert "dind_error" not in result.updates["default_extra"]


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.asyncio
async def test_port_connectivity_event_carries_the_range_tallies(context_factory, success):
    ranges = (
        PortRangeResult(first=40000, last=44999, declared=5000, probed=209, answered=0),
        PortRangeResult(first=60000, last=64999, declared=5000, probed=30, answered=30),
    )
    services = build_services(
        redis=DummyRedis(renting_in_progress=False),
        backend=DummyBackendService(),
        connectivity=DummyConnectivityService(
            success=success, verified_port_count=30 if success else 0, port_ranges=ranges
        ),
    )
    ctx = context_factory(
        services=services, config=build_context_config(job_batch_id="batch-123"), state=build_state(), rented=False
    )

    result = await PortConnectivityCheck().run(ctx)

    expected = [
        {"range": "40000-44999", "declared": 5000, "probed": 209, "answered": 0},
        {"range": "60000-64999", "declared": 5000, "probed": 30, "answered": 30},
    ]
    assert result.event.context["port_ranges"] == expected
    assert "port_ranges" not in result.updates["default_extra"]
    if not success:
        assert result.event.what_we_saw["port_ranges"] == expected


# ticket-0494: a rented executor whose scrape timed out (sysbox False) and whose DinD probe never
# reached its container keeps the sysbox an earlier probe verified.


def _rented_state(*, scrape_sysbox: bool = False):
    rented = RentedExecutor(
        miner_hotkey="miner-hotkey",
        executor_ip_address="127.0.0.1",
        executor_ip_port="22",
        pods=[RentedPod(pod_id="pod-1", container_name="container_pod-1", rented_ports=[40010, 40011])],
    )
    return build_state(
        sysbox_runtime=scrape_sysbox,
        rented_data=RentedExecutorsResponse(executors={"executor-123": rented}),
    )


def _verified_ctx(context_factory, *, redis_service, state, dind_ok: bool, sysbox_measured: bool | None):
    services = build_services(
        redis=redis_service,
        backend=DummyBackendService(),
        connectivity=DummyConnectivityService(
            success=True,
            sysbox_runtime=False,
            verified_port_count=100,
            dind_ok=dind_ok,
            sysbox_measured=sysbox_measured,
        ),
    )
    return context_factory(services=services, config=build_context_config(job_batch_id="batch-123"), state=state)


@pytest.mark.asyncio
async def test_rented_node_keeps_verified_sysbox_when_the_dind_probe_is_refused(context_factory):
    """Scrape says no sysbox, the probe was refused by the busy host, an earlier probe verified
    sysbox: the cycle keeps sysbox, so the rented sysbox multiplier stays 1. Before ticket-0494 both
    fallbacks keyed off the scrape's False and the node scored 0 for the cycle."""
    redis_service = DummyRedis()
    redis_service.sysbox_verified[("miner-hotkey", "executor-123")] = 86400
    ctx = _verified_ctx(
        context_factory, redis_service=redis_service, state=_rented_state(), dind_ok=False, sysbox_measured=None
    )

    result = await PortConnectivityCheck().run(ctx)

    assert result.updates["state"].sysbox_runtime is True
    assert result.updates["state"].specs["sysbox_runtime"] is True
    assert result.updates["default_extra"]["sysbox_from_last_verified"] is True
    assert ("miner-hotkey", "executor-123") in redis_service.sysbox_verified


@pytest.mark.asyncio
async def test_rented_node_probe_that_reached_its_container_without_sysbox_is_a_verdict(context_factory):
    """The probe reached its container with sysbox-runc and it failed: no sysbox for the cycle, and
    the earlier pass is forgotten so a later refused probe does not bring it back."""
    redis_service = DummyRedis()
    redis_service.sysbox_verified[("miner-hotkey", "executor-123")] = 86400
    ctx = _verified_ctx(
        context_factory, redis_service=redis_service, state=_rented_state(), dind_ok=True, sysbox_measured=False
    )

    result = await PortConnectivityCheck().run(ctx)

    assert result.updates["state"].sysbox_runtime is False
    assert "sysbox_from_last_verified" not in result.updates["default_extra"]
    assert redis_service.sysbox_verified == {}


@pytest.mark.asyncio
async def test_rented_node_without_a_verified_sysbox_keeps_the_current_behaviour(context_factory):
    """No pass on record: a refused probe on a rented node still records no sysbox."""
    redis_service = DummyRedis()
    ctx = _verified_ctx(
        context_factory, redis_service=redis_service, state=_rented_state(), dind_ok=False, sysbox_measured=None
    )

    result = await PortConnectivityCheck().run(ctx)

    assert result.updates["state"].sysbox_runtime is False
    assert "sysbox_from_last_verified" not in result.updates["default_extra"]


@pytest.mark.asyncio
async def test_unrented_node_does_not_read_the_verified_sysbox(context_factory):
    """Only a rented node uses the memory; an unrented one is re-measured as before."""
    redis_service = DummyRedis()
    redis_service.sysbox_verified[("miner-hotkey", "executor-123")] = 86400
    ctx = _verified_ctx(
        context_factory, redis_service=redis_service, state=build_state(), dind_ok=False, sysbox_measured=None
    )

    result = await PortConnectivityCheck().run(ctx)

    assert result.updates["state"].sysbox_runtime is False


@pytest.mark.asyncio
async def test_probe_that_verifies_sysbox_is_remembered_with_the_ttl(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "SYSBOX_VERIFIED_TTL_SECONDS", 4321)
    redis_service = DummyRedis()
    ctx = _verified_ctx(
        context_factory, redis_service=redis_service, state=build_state(), dind_ok=True, sysbox_measured=True
    )

    await PortConnectivityCheck().run(ctx)

    assert redis_service.sysbox_verified == {("miner-hotkey", "executor-123"): 4321}


@pytest.mark.asyncio
async def test_redis_error_falls_back_to_the_current_behaviour(context_factory):
    class BrokenRedis(DummyRedis):
        async def is_sysbox_verified(self, miner_hotkey: str, executor_id: str) -> bool:
            raise ConnectionError("redis down")

    ctx = _verified_ctx(
        context_factory, redis_service=BrokenRedis(), state=_rented_state(), dind_ok=False, sysbox_measured=None
    )

    result = await PortConnectivityCheck().run(ctx)

    assert result.passed is True
    assert result.updates["state"].sysbox_runtime is False


# DAH-5244: the host disk is read right before the port-check containers. A node rented by someone other than
# its operator, or running a Lium-owned filler, whose Docker root stalls on `sync -f` keeps the ports its last
# completed check verified instead of a check into the stall (enforced; at most 3 cycles in a row), and in
# shadow the check runs while the event says what the gate would have done.
STALLED = pressure_output(end=120.0, rc="")
QUICK = pressure_output()
MEMORY_PAIRS = [(40020, 40020), (40021, 40021), (40022, 40022)]


def _filler_state(*, owner: str | None = "lium", sysbox_runtime: bool = False):
    return build_state(
        sysbox_runtime=sysbox_runtime,
        rented_data=RentedExecutorsResponse(
            executors={},
            filler_ports_by_executor={"executor-123": [40010, 40011]},
            default_job_owner_by_executor={} if owner is None else {"executor-123": owner},
        ),
    )


def _self_rented_state(*, filler_owner: str | None = None):
    rented = RentedExecutor(
        miner_hotkey="miner-hotkey",
        executor_ip_address="127.0.0.1",
        executor_ip_port="22",
        pods=[RentedPod(pod_id="pod-1", container_name="container_pod-1", rented_ports=[40010])],
        self_rent=True,
    )
    return build_state(
        sysbox_runtime=False,
        rented_data=RentedExecutorsResponse(
            executors={"executor-123": rented},
            filler_ports_by_executor={} if filler_owner is None else {"executor-123": [40011]},
            default_job_owner_by_executor={} if filler_owner is None else {"executor-123": filler_owner},
        ),
    )


@pytest.fixture(autouse=True)
def fresh_saved_completed(monkeypatch):
    # the module-level set outlives the check instance; a test starts as a freshly started validator does
    monkeypatch.setattr(port_connectivity_module, "_COMPLETED_CHECK_SAVED_AT", {})


def _gate_ctx(
    context_factory, *, redis_service, state, reading: str, connectivity=None, executor=None, miner_hotkey="miner-hotkey"
):
    connectivity = connectivity or DummyConnectivityService(success=True, sysbox_runtime=True, verified_port_count=100)
    services = build_services(redis=redis_service, backend=DummyBackendService(), connectivity=connectivity)
    ctx = context_factory(
        services=services,
        config=build_context_config(job_batch_id="batch-123"),
        state=state,
        runner=FakeRunner(reading),
        executor=executor,
        miner_hotkey=miner_hotkey,
    )
    return ctx, connectivity


def _remember(redis_service, ctx, pairs=MEMORY_PAIRS):
    # a completed check this validator process saved: the memory in Redis and the readiness to republish it
    redis_service.verified_ports[(ctx.miner_hotkey, "executor-123")] = {
        "identity": PortConnectivityCheck._port_setup_fingerprint(ctx),
        "pairs": pairs,
        "ttl": 86400,
    }
    port_connectivity_module._COMPLETED_CHECK_SAVED_AT[(ctx.miner_hotkey, "executor-123")] = time.monotonic()


@pytest.mark.asyncio
@pytest.mark.parametrize("state_builder", [_rented_state, _filler_state], ids=["rented", "filler"])
async def test_enforced_pressure_republishes_the_last_verified_ports_without_a_check(
    context_factory, monkeypatch, state_builder
):
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    redis_service.sysbox_verified[("miner-hotkey", "executor-123")] = 86400
    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=state_builder(), reading=STALLED)
    _remember(redis_service, ctx)

    result = await PortConnectivityCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.VERIFY_DEFERRED.reason
    assert connectivity.called_with is None
    state = result.updates["state"]
    assert state.verified_port_count == 3 and state.verified_port_pairs == MEMORY_PAIRS
    assert state.specs["verified_ports"] == [40020, 40021, 40022]
    assert state.sysbox_runtime is True and state.specs["sysbox_runtime"] is True  # the remembered sysbox
    assert state.specs["disk_pressure"]["pressure"] is True and state.specs["disk_pressure"]["syncfs_timed_out"] is True


@pytest.mark.asyncio
async def test_a_deferred_cycle_drops_a_grace_marker_from_the_providers_specs(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    state = dataclasses.replace(_rented_state(), specs={"port_verify_first_miss": True})
    ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=state, reading=STALLED)
    _remember(redis_service, ctx)

    result = await PortConnectivityCheck().run(ctx)

    assert result.event.reason_code == Msg.VERIFY_DEFERRED.reason
    assert "port_verify_first_miss" not in result.updates["state"].specs
    assert redis_service.deferrals[("miner-hotkey", "executor-123", False)] == 1
    assert result.updates["default_extra"]["disk_pressure_deferrals_in_a_row"] == 1
    # the memory is what it was: a republish never refreshes it
    assert redis_service.verified_ports[("miner-hotkey", "executor-123")]["pairs"] == MEMORY_PAIRS


@pytest.mark.asyncio
@pytest.mark.parametrize("state_builder", [_rented_state, _filler_state], ids=["rented", "filler"])
async def test_a_republished_memory_drops_the_ports_a_rental_or_filler_took_since(
    context_factory, monkeypatch, state_builder
):
    # the memory predates the pod/filler holding 40010-40011; the rental probe puts ssh on a republished port
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=state_builder(), reading=STALLED)
    _remember(redis_service, ctx, pairs=[(40010, 40010), (40011, 40011), (40012, 40012)])

    result = await PortConnectivityCheck().run(ctx)

    state = result.updates["state"]
    assert result.event.reason_code == Msg.VERIFY_DEFERRED.reason
    assert state.verified_port_pairs == [(40012, 40012)] and state.specs["verified_ports"] == [40012]
    assert result.updates["default_extra"]["verified_ports_now_busy"] == 2
    assert redis_service.verified_ports[("miner-hotkey", "executor-123")]["pairs"][0] == (40010, 40010)


# DAH-3698: one failed port verification must not zero a split remainder's idle pay.


def _port_ctx(context_factory, redis_service, *, success: bool, connectivity=None, specs=None):
    services = build_services(
        redis=redis_service,
        backend=DummyBackendService(),
        connectivity=connectivity
        or DummyConnectivityService(success=success, verified_port_count=8 if success else 0),
    )
    return context_factory(
        services=services,
        config=build_context_config(job_batch_id="batch-123"),
        state=build_state(specs=specs or {}),
        rented=False,
    )


class PartialConnectivityService(DummyConnectivityService):
    """`ok` of 8 probed ports answered; the rest failed. Status is "ok" from one answering port up."""

    def __init__(self, ok: int):
        super().__init__(success=ok > 0, verified_port_count=ok)
        self.ok = ok

    async def verify_ports(self, *args, **kwargs) -> PortVerificationResult:
        result = await super().verify_ports(*args, **kwargs)
        failed = tuple(PortPair(9000 + i, 9000 + i) for i in range(8 - self.ok))
        return dataclasses.replace(result, failed_ports=failed, selected_ports=result.successful_ports + failed)


class BrokenPortMissRedis(DummyRedis):
    async def record_port_verify_miss(self, miner_hotkey: str, executor_id: str, ttl_seconds: int) -> bool:
        raise ConnectionError("redis down")


@pytest.mark.asyncio
@pytest.mark.parametrize("ok", [0, 1, 2, 7])
async def test_a_partial_port_loss_is_a_miss_first_forgiven_then_counted(context_factory, ok):
    redis_service = DummyRedis()

    first = await PortConnectivityCheck().run(
        _port_ctx(context_factory, redis_service, success=False, connectivity=PartialConnectivityService(ok))
    )
    second = await PortConnectivityCheck().run(
        _port_ctx(context_factory, redis_service, success=False, connectivity=PartialConnectivityService(ok))
    )

    assert first.updates["state"].specs.get("port_verify_first_miss") is True
    assert "port_verify_first_miss" not in second.updates["state"].specs
    assert len(second.updates["state"].specs["verified_ports"]) == ok


@pytest.mark.asyncio
async def test_every_probed_port_answering_keeps_the_miss_on_record(context_factory):
    redis_service = DummyRedis()
    redis_service.port_miss_on_record = True

    await PortConnectivityCheck().run(
        _port_ctx(context_factory, redis_service, success=True, connectivity=PartialConnectivityService(8))
    )

    assert redis_service.port_miss_on_record is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_builder, remembered, reading",
    [
        (build_state, True, STALLED),  # idle: the writer would be the miner's
        (_self_rented_state, True, STALLED),  # self-rented
        (_rented_state, False, STALLED),  # never verified under this identity
        (_rented_state, True, QUICK),  # no pressure
    ],
    ids=["idle", "owner", "no_memory", "no_pressure"],
)
async def test_enforced_gate_runs_the_check_when_not_eligible_or_nothing_to_republish(
    context_factory, monkeypatch, state_builder, remembered, reading
):
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=state_builder(), reading=reading)
    if remembered:
        _remember(redis_service, ctx)

    result = await PortConnectivityCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.VERIFY_OK.reason
    assert connectivity.called_with is not None
    assert result.updates["state"].specs["disk_pressure"]["pressure"] is (reading is STALLED)
    # the completed check is the new memory and the streak is over
    assert redis_service.verified_ports[("miner-hotkey", "executor-123")]["pairs"] == [
        (8000 + i, 8000 + i) for i in range(100)
    ]
    assert ("miner-hotkey", "executor-123", False) not in redis_service.deferrals


@pytest.mark.asyncio
async def test_the_fourth_pressured_cycle_runs_the_check_and_ends_the_streak(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    deferred = []
    for _ in range(5):
        ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
        _remember(redis_service, ctx) if not redis_service.verified_ports else None
        result = await PortConnectivityCheck().run(ctx)
        deferred.append(connectivity.called_with is None)
        assert result.passed
    assert deferred == [True, True, True, False, True]


@pytest.mark.asyncio
async def test_a_remembered_identity_that_changed_is_not_republished(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
    _remember(redis_service, ctx)
    redis_service.verified_ports[("miner-hotkey", "executor-123")]["identity"] = "10.0.0.1|40000-41000|"

    await PortConnectivityCheck().run(ctx)

    assert connectivity.called_with is not None


@pytest.mark.asyncio
async def test_shadow_runs_the_check_and_simulates_the_deferrals(context_factory):
    redis_service = DummyRedis()
    would_defer = []
    for cycle in range(5):
        ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
        if cycle == 0:
            _remember(redis_service, ctx)
        result = await PortConnectivityCheck().run(ctx)
        assert result.passed and result.event.reason_code == Msg.VERIFY_OK.reason
        assert connectivity.called_with is not None
        would_defer.append(result.updates["state"].specs["disk_pressure"]["pressure"])
    # the simulated streak: three "would defer", the forced fourth cycle clears it, then it starts again
    assert would_defer == [True] * 5
    assert redis_service.deferrals[("miner-hotkey", "executor-123", True)] == 1
    assert ("miner-hotkey", "executor-123", False) not in redis_service.deferrals


@pytest.mark.asyncio
async def test_reading_off_runs_the_check_and_still_rewrites_the_memory(context_factory, monkeypatch):
    # the memory follows every completed check, the reading on or off: a check failed with the reading off may
    # not be undone by the memory of an earlier one once the reading is back
    monkeypatch.setattr(settings, "DISK_PRESSURE_CHECK_ENABLED", False)
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
    _remember(redis_service, ctx)

    result = await PortConnectivityCheck().run(ctx)

    assert connectivity.called_with is not None and len(ctx.runner.commands) == 0
    assert "disk_pressure" not in result.updates["state"].specs
    assert redis_service.verified_ports[("miner-hotkey", "executor-123")]["pairs"] == [
        (8000 + i, 8000 + i) for i in range(100)
    ]


@pytest.mark.asyncio
async def test_a_check_failed_with_the_reading_off_is_what_a_pressured_cycle_finds(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=QUICK)
    await PortConnectivityCheck().run(ctx)  # the success on record
    monkeypatch.setattr(settings, "DISK_PRESSURE_CHECK_ENABLED", False)
    unreachable = DummyConnectivityService(success=False, status="externally_unreachable")
    ctx, _ = _gate_ctx(
        context_factory, redis_service=redis_service, state=_rented_state(), reading=QUICK, connectivity=unreachable
    )
    failed = await PortConnectivityCheck().run(ctx)
    monkeypatch.setattr(settings, "DISK_PRESSURE_CHECK_ENABLED", True)
    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)

    pressured = await PortConnectivityCheck().run(ctx)

    assert failed.event.reason_code == Msg.VERIFY_FAILED.reason
    assert pressured.event.reason_code == Msg.VERIFY_DEFERRED.reason
    assert connectivity.called_with is None and pressured.updates["state"].verified_port_count == 0


@pytest.mark.asyncio
async def test_a_reading_that_fails_is_not_pressure(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading="")
    _remember(redis_service, ctx)

    result = await PortConnectivityCheck().run(ctx)

    assert connectivity.called_with is not None
    assert result.updates["state"].specs["disk_pressure"] == {**result.updates["state"].specs["disk_pressure"], "error": "no_output", "pressure": False}


@pytest.mark.asyncio
async def test_a_deferred_cycle_takes_sysbox_from_the_verified_memory_never_from_the_scrape(
    context_factory, monkeypatch
):
    # the scrape is the host's own word; SysboxRequiredCheck fires on a filler-only node, so a deferral may not
    # pass it on a value the validator's probe never confirmed
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    ctx, connectivity = _gate_ctx(
        context_factory, redis_service=redis_service, state=_filler_state(sysbox_runtime=True), reading=STALLED
    )
    _remember(redis_service, ctx)

    result = await PortConnectivityCheck().run(ctx)

    assert result.event.reason_code == Msg.VERIFY_DEFERRED.reason and connectivity.called_with is None
    assert result.updates["state"].sysbox_runtime is False
    assert result.updates["state"].specs["sysbox_runtime"] is False
    assert result.updates["default_extra"]["sysbox_runtime"] is False


@pytest.mark.asyncio
async def test_a_deferral_redis_refuses_to_record_runs_the_check(context_factory, monkeypatch):
    # Redis reachable, reads fine, writes refused (maxmemory + noeviction): the streak could never grow
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)

    class WriteRefusingRedis(DummyRedis):
        async def record_disk_pressure_deferral(self, miner_hotkey, executor_id, *, shadow, ttl_seconds):
            raise ConnectionError("OOM command not allowed when used memory > 'maxmemory'")

    redis_service = WriteRefusingRedis()
    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
    _remember(redis_service, ctx)

    result = await PortConnectivityCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.VERIFY_OK.reason
    assert connectivity.called_with is not None
    assert result.updates["state"].port_check_deferred is False


@pytest.mark.asyncio
async def test_a_deferral_slot_past_the_bound_runs_the_check(context_factory, monkeypatch):
    # DAH-5244 review: the count is read before the write; a cycle that took the last slot in between must not
    # let this one defer a fourth time
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)

    class SlotTakenInBetween(DummyRedis):
        async def record_disk_pressure_deferral(self, miner_hotkey, executor_id, *, shadow, ttl_seconds):
            key = (miner_hotkey, executor_id, shadow)
            self.deferrals[key] = self.deferrals.get(key, 0) + 2  # the other cycle's INCR landed first
            return self.deferrals[key]

    redis_service = SlotTakenInBetween()
    redis_service.deferrals[("miner-hotkey", "executor-123", False)] = 2
    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
    _remember(redis_service, ctx)

    result = await PortConnectivityCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.VERIFY_OK.reason
    assert connectivity.called_with is not None
    assert result.updates["state"].port_check_deferred is False


@pytest.mark.asyncio
async def test_a_deferred_cycle_is_marked_in_the_state_and_a_completed_one_is_not(context_factory, monkeypatch):
    # lium-io#1937's successor stores "last good ports" from the state: a republished cycle must be tellable apart
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
    _remember(redis_service, ctx)

    deferred = await PortConnectivityCheck().run(ctx)
    completed = await PortConnectivityCheck().run(
        _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=QUICK)[0]
    )

    assert deferred.event.reason_code == Msg.VERIFY_DEFERRED.reason
    assert deferred.updates["state"].port_check_deferred is True
    assert completed.event.reason_code == Msg.VERIFY_OK.reason
    assert completed.updates["state"].port_check_deferred is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_builder, eligible",
    [
        (_rented_state, True),  # a rental not contracted as a self-rent
        (_self_rented_state, False),
        (lambda: _filler_state(owner="lium"), True),
        (lambda: _filler_state(owner="miner"), False),
        (lambda: _filler_state(owner=None), False),  # a backend that names no owner
        (lambda: _self_rented_state(filler_owner="miner"), False),
        (lambda: _self_rented_state(filler_owner="lium"), False),  # the self-rent veto outranks the filler
    ],
    ids=[
        "rental",
        "self_rent",
        "lium_filler",
        "miner_filler",
        "unowned_filler",
        "self_rent_and_miner_filler",
        "self_rent_and_lium_filler",
    ],
)
async def test_only_a_third_party_rental_or_a_lium_filler_may_defer(context_factory, monkeypatch, state_builder, eligible):
    # owner_flag is not read: the backend sends it False for every rental
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=state_builder(), reading=STALLED)
    _remember(redis_service, ctx)

    result = await PortConnectivityCheck().run(ctx)

    assert (connectivity.called_with is None) is eligible
    assert (result.event.reason_code == Msg.VERIFY_DEFERRED.reason) is eligible


class WriteRefusingRedis(DummyRedis):
    # Redis reachable, reads fine, writes refused (maxmemory + noeviction) while `refuse_writes` is on
    refuse_writes = False

    async def record_verified_ports(self, miner_hotkey, executor_id, identity, pairs, ttl_seconds):
        if self.refuse_writes:
            raise ConnectionError("OOM command not allowed when used memory > 'maxmemory'")
        await super().record_verified_ports(miner_hotkey, executor_id, identity, pairs, ttl_seconds)


async def _complete(context_factory, redis_service, *, miner_hotkey="miner-hotkey", success=True):
    connectivity = DummyConnectivityService(success=success, verified_port_count=100 if success else 0)
    ctx, _ = _gate_ctx(
        context_factory, redis_service=redis_service, state=_rented_state(), reading=QUICK, connectivity=connectivity,
        miner_hotkey=miner_hotkey,
    )
    return await PortConnectivityCheck().run(ctx)


async def _pressured(context_factory, redis_service, *, miner_hotkey="miner-hotkey", state=None):
    ctx, connectivity = _gate_ctx(
        context_factory, redis_service=redis_service, state=state or _rented_state(), reading=STALLED,
        miner_hotkey=miner_hotkey,
    )
    result = await PortConnectivityCheck().run(ctx)
    return result.event.reason_code == Msg.VERIFY_DEFERRED.reason and connectivity.called_with is None


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_first", [True, False], ids=["blocked_then_other_saves", "other_saves_then_blocked"])
async def test_two_miners_reporting_one_uuid_do_not_change_each_others_permission_to_defer(
    context_factory, monkeypatch, blocked_first
):
    # the uuid is the miner's word; Redis scopes the memory by (miner, executor) and so must the readiness
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = WriteRefusingRedis()

    async def block_miner_a():
        await _complete(context_factory, redis_service, miner_hotkey="miner-a")  # the success on record
        redis_service.refuse_writes = True
        await _complete(context_factory, redis_service, miner_hotkey="miner-a", success=False)  # not recorded
        redis_service.refuse_writes = False

    if blocked_first:
        await block_miner_a()
        await _complete(context_factory, redis_service, miner_hotkey="miner-b")
    else:
        await _complete(context_factory, redis_service, miner_hotkey="miner-b")
        await block_miner_a()

    assert await _pressured(context_factory, redis_service, miner_hotkey="miner-a") is False
    assert await _pressured(context_factory, redis_service, miner_hotkey="miner-b") is True
    assert redis_service.verified_ports[("miner-a", "executor-123")]["pairs"] == [
        (8000 + i, 8000 + i) for i in range(100)
    ]


@pytest.mark.asyncio
async def test_a_restarted_validator_runs_the_check_before_it_trusts_the_memory_in_redis(
    context_factory, monkeypatch
):
    # the refused zero-port check is forgotten with the process; the older success in Redis is not a verdict
    # this process saw saved, so the first pressured cycle is a real check
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = WriteRefusingRedis()
    await _complete(context_factory, redis_service)
    redis_service.refuse_writes = True
    await _complete(context_factory, redis_service, success=False)
    redis_service.refuse_writes = False
    monkeypatch.setattr(port_connectivity_module, "_COMPLETED_CHECK_SAVED_AT", {})  # the restart

    checked_after_restart = not await _pressured(context_factory, redis_service)
    deferred_once_saved = await _pressured(context_factory, redis_service)

    assert checked_after_restart and deferred_once_saved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ports_saved, sysbox_forgotten",
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["ok_ok", "ports_ok_sysbox_fail", "ports_fail_sysbox_ok", "fail_fail"],
)
async def test_a_filler_node_that_lost_sysbox_defers_only_when_both_memories_were_updated(
    context_factory, monkeypatch, ports_saved, sysbox_forgotten
):
    # SysboxRequired rejected the completed no-sysbox cycle; a deferral on a memory the update of which failed
    # would pass it one cycle later
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)

    class PartlyRefusingRedis(WriteRefusingRedis):
        refuse_sysbox_clear = False

        async def clear_sysbox_verified(self, miner_hotkey, executor_id):
            if self.refuse_sysbox_clear:
                raise ConnectionError("OOM command not allowed when used memory > 'maxmemory'")
            await super().clear_sysbox_verified(miner_hotkey, executor_id)

    redis_service = PartlyRefusingRedis()
    redis_service.sysbox_verified[("miner-hotkey", "executor-123")] = 86400
    ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=_filler_state(), reading=QUICK)
    _remember(redis_service, ctx)
    redis_service.refuse_writes = not ports_saved
    redis_service.refuse_sysbox_clear = not sysbox_forgotten
    def without_sysbox():
        return DummyConnectivityService(success=True, sysbox_runtime=False, verified_port_count=100, dind_ok=True)

    ctx, _ = _gate_ctx(
        context_factory, redis_service=redis_service, state=_filler_state(), reading=QUICK, connectivity=without_sysbox()
    )
    completed = await PortConnectivityCheck().run(ctx)
    redis_service.refuse_writes = redis_service.refuse_sysbox_clear = False
    ctx, connectivity = _gate_ctx(
        context_factory, redis_service=redis_service, state=_filler_state(), reading=STALLED, connectivity=without_sysbox()
    )

    pressured = await PortConnectivityCheck().run(ctx)

    completed_sysbox = await SysboxRequiredCheck().run(context_factory(state=completed.updates["state"]))
    pressured_sysbox = await SysboxRequiredCheck().run(context_factory(state=pressured.updates["state"]))
    assert completed_sysbox.passed is False and pressured_sysbox.passed is False
    assert (pressured.event.reason_code == Msg.VERIFY_DEFERRED.reason) is (ports_saved and sysbox_forgotten)
    assert (connectivity.called_with is None) is (ports_saved and sysbox_forgotten)


@pytest.mark.asyncio
async def test_a_completed_check_redis_refused_to_record_blocks_the_deferral_until_one_is_saved(
    context_factory, monkeypatch
):
    # the older success stays in Redis while the latest completed check rejected every port
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = WriteRefusingRedis()
    ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=QUICK)
    await PortConnectivityCheck().run(ctx)  # the success on record
    redis_service.refuse_writes = True
    no_ports = DummyConnectivityService(success=False)
    ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=QUICK, connectivity=no_ports)
    await PortConnectivityCheck().run(ctx)  # completed with no ports, not recorded
    redis_service.refuse_writes = False

    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
    blocked = await PortConnectivityCheck().run(ctx)
    ctx, connectivity_after = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
    deferred = await PortConnectivityCheck().run(ctx)

    assert blocked.event.reason_code == Msg.VERIFY_OK.reason and connectivity.called_with is not None
    assert redis_service.verified_ports[("miner-hotkey", "executor-123")]["pairs"] == [
        (8000 + i, 8000 + i) for i in range(100)
    ]
    assert deferred.event.reason_code == Msg.VERIFY_DEFERRED.reason and connectivity_after.called_with is None


@pytest.mark.asyncio
@pytest.mark.parametrize("ready_before", [True, False], ids=["ready_before", "not_ready_before"])
@pytest.mark.parametrize(
    "interrupted_method, interruption",
    [
        ("renting_in_progress", ConnectionError("redis down")),
        ("record_verified_ports", asyncio.CancelledError()),
        ("clear_sysbox_verified", asyncio.CancelledError()),
    ],
    ids=["pending_rental_read_fails", "cancelled_during_port_write", "cancelled_during_sysbox_delete"],
)
async def test_a_check_interrupted_before_all_of_it_is_saved_leaves_no_readiness_behind(
    context_factory, monkeypatch, interrupted_method, interruption, ready_before
):
    # the scrape said sysbox, the probe said none: the Redis sysbox memory must go before a deferral may
    # republish it. An error raised out of run() or a cancellation thrown into an await is not `except Exception`
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)

    class InterruptingRedis(DummyRedis):
        interrupt = False

        async def _interrupted(self, *args, **kwargs):
            if self.interrupt:
                raise interruption
            return await getattr(super(), interrupted_method)(*args, **kwargs)

    setattr(InterruptingRedis, interrupted_method, InterruptingRedis._interrupted)
    redis_service = InterruptingRedis()
    redis_service.sysbox_verified[("miner-hotkey", "executor-123")] = 86400
    ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=_filler_state(), reading=QUICK)
    _remember(redis_service, ctx)
    if not ready_before:
        port_connectivity_module._COMPLETED_CHECK_SAVED_AT.clear()

    def without_sysbox():
        return DummyConnectivityService(success=True, sysbox_runtime=False, verified_port_count=100, dind_ok=True)

    redis_service.interrupt = True
    ctx, _ = _gate_ctx(
        context_factory,
        redis_service=redis_service,
        state=_filler_state(sysbox_runtime=True),
        reading=QUICK,
        connectivity=without_sysbox(),
    )
    with pytest.raises(type(interruption)):
        await PortConnectivityCheck().run(ctx)
    redis_service.interrupt = False
    ctx, connectivity = _gate_ctx(
        context_factory, redis_service=redis_service, state=_filler_state(), reading=STALLED, connectivity=without_sysbox()
    )

    pressured = await PortConnectivityCheck().run(ctx)

    assert pressured.event.reason_code == Msg.VERIFY_OK.reason and connectivity.called_with is not None
    pressured_sysbox = await SysboxRequiredCheck().run(context_factory(state=pressured.updates["state"]))
    assert pressured_sysbox.passed is False


@pytest.mark.asyncio
async def test_readiness_outlives_the_port_memory_by_nothing_and_a_deferral_does_not_refresh_it(
    context_factory, monkeypatch
):
    # a departed executor's key goes with the next completed check anywhere; an expired key is no readiness
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    saved_at = port_connectivity_module._COMPLETED_CHECK_SAVED_AT
    expired = time.monotonic() - VERIFIED_PORTS_MEMORY_TTL_SECONDS - 1
    ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
    _remember(redis_service, ctx)
    saved_at[("departed-miner", "executor-gone")] = expired
    ctx_expired, _ = _gate_ctx(
        context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED, miner_hotkey="miner-late"
    )
    _remember(redis_service, ctx_expired)
    saved_at[("miner-late", "executor-123")] = expired
    recent_saved_at = saved_at[("miner-hotkey", "executor-123")]

    expired_deferred = await _pressured(context_factory, redis_service, miner_hotkey="miner-late")
    recent_deferred = await _pressured(context_factory, redis_service)

    assert expired_deferred is False and recent_deferred is True
    assert ("departed-miner", "executor-gone") not in saved_at  # pruned by the check miner-late completed
    assert saved_at[("miner-late", "executor-123")] > expired  # its real check is a fresh completion
    assert saved_at[("miner-hotkey", "executor-123")] == recent_saved_at  # a deferral refreshes nothing


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_builder, republished_sysbox",
    [(_rented_state, True), (_filler_state, False)],
    ids=["rented", "filler"],
)
async def test_a_cycle_without_sysbox_on_an_unrented_node_forgets_the_verified_sysbox(
    context_factory, monkeypatch, state_builder, republished_sysbox
):
    # SysboxRequired fails the completed cycle of a filler-only node that lost sysbox; a deferral may not pass
    # it on the older memory. A rented node keeps its fallback: it needs no sysbox to pass.
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    redis_service.sysbox_verified[("miner-hotkey", "executor-123")] = 86400
    without_sysbox = DummyConnectivityService(success=True, sysbox_runtime=False, verified_port_count=100, dind_ok=True)
    ctx, _ = _gate_ctx(
        context_factory, redis_service=redis_service, state=state_builder(), reading=QUICK, connectivity=without_sysbox
    )
    completed = await PortConnectivityCheck().run(ctx)
    ctx, connectivity = _gate_ctx(context_factory, redis_service=redis_service, state=state_builder(), reading=STALLED)

    deferred = await PortConnectivityCheck().run(ctx)

    assert completed.event.reason_code == Msg.VERIFY_OK.reason
    assert deferred.event.reason_code == Msg.VERIFY_DEFERRED.reason and connectivity.called_with is None
    assert deferred.updates["state"].sysbox_runtime is republished_sysbox
    assert (("miner-hotkey", "executor-123") in redis_service.sysbox_verified) is republished_sysbox


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"address": "10.0.0.2"},
        {"port_range": "41000-42000"},
        {"port_mappings": "[[40020, 50020]]"},
        {"ssh_port": 40020},
    ],
    ids=["address", "range", "mappings", "ssh_port"],
)
async def test_a_memory_from_another_port_setup_is_not_republished(context_factory, monkeypatch, changed):
    # the selector excludes the ssh port too: a memory predating an sshd moved onto a verified port misses
    monkeypatch.setattr(settings, "DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED", True)
    redis_service = DummyRedis()
    ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
    _remember(redis_service, ctx)
    ctx, connectivity = _gate_ctx(
        context_factory,
        redis_service=redis_service,
        state=_rented_state(),
        reading=STALLED,
        executor=ctx.executor.model_copy(update=changed),
    )

    result = await PortConnectivityCheck().run(ctx)

    assert result.event.reason_code == Msg.VERIFY_OK.reason and connectivity.called_with is not None


@pytest.mark.asyncio
async def test_every_reading_is_logged_at_info_on_its_own_line(context_factory, caplog):
    # the port event repeats cycle after cycle and LoggerSink drops a repeat to DEBUG; the shadow readings do not
    redis_service = DummyRedis()
    with caplog.at_level("INFO", logger=port_connectivity_module.__name__):
        for _ in range(2):
            ctx, _ = _gate_ctx(context_factory, redis_service=redis_service, state=_rented_state(), reading=STALLED)
            await PortConnectivityCheck().run(ctx)

    readings = [r for r in caplog.records if "DISK_PRESSURE_READING" in r.getMessage()]
    assert [r.levelname for r in readings] == ["INFO", "INFO"]
    assert all(r.msg.extra["disk_pressure"]["pressure"] is True for r in readings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("redis_service", "success"),
    [
        pytest.param(DummyRedis(), False, id="repeated_failure"),
        pytest.param(DummyRedis(), True, id="ok"),
        pytest.param(BrokenPortMissRedis(), False, id="redis_error_on_failure"),
        pytest.param(BrokenPortMissRedis(), True, id="redis_error_on_ok"),
    ],
)
async def test_a_grace_marker_in_the_providers_specs_never_survives(context_factory, redis_service, success):
    redis_service.port_miss_on_record = True  # a repeat, so this cycle earns no grace of its own

    result = await PortConnectivityCheck().run(
        _port_ctx(context_factory, redis_service, success=success, specs={"port_verify_first_miss": True})
    )

    assert "port_verify_first_miss" not in result.updates["state"].specs


@pytest.mark.asyncio
async def test_only_the_first_failed_port_verification_is_marked_for_grace(context_factory):
    redis_service = DummyRedis()

    first = await PortConnectivityCheck().run(_port_ctx(context_factory, redis_service, success=False))
    second = await PortConnectivityCheck().run(_port_ctx(context_factory, redis_service, success=False))

    assert first.updates["state"].specs.get("port_verify_first_miss") is True
    assert "port_verify_first_miss" not in second.updates["state"].specs


@pytest.mark.asyncio
async def test_a_passing_port_verification_keeps_the_miss_on_record(context_factory):
    redis_service = DummyRedis()
    redis_service.port_miss_on_record = True

    ok = await PortConnectivityCheck().run(_port_ctx(context_factory, redis_service, success=True))

    assert "port_verify_first_miss" not in ok.updates["state"].specs
    assert redis_service.port_miss_on_record is True


@pytest.mark.asyncio
async def test_alternating_pass_and_failure_earns_grace_only_once(context_factory):
    # success -> failure -> success -> failure inside the window: only the first failure is forgiven
    redis_service = DummyRedis()
    marks = []
    for success in (True, False, True, False):
        result = await PortConnectivityCheck().run(_port_ctx(context_factory, redis_service, success=success))
        marks.append(result.updates["state"].specs.get("port_verify_first_miss") is True)

    assert marks == [False, True, False, False]


@pytest.mark.asyncio
async def test_a_persisting_port_miss_restarts_the_window_instead_of_earning_fresh_grace():
    import asyncio

    from fakeredis.aioredis import FakeRedis
    from services.redis_service import RedisService

    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis()
    service.lock = asyncio.Lock()
    key = "port_verify_miss:hk:ex"

    assert await service.record_port_verify_miss("hk", "ex", 3600) is True
    await service.redis.expire(key, 5)  # the window nearly over
    assert await service.record_port_verify_miss("hk", "ex", 3600) is False
    assert await service.redis.ttl(key) > 3000


@pytest.mark.asyncio
async def test_concurrent_deferrals_take_distinct_slots_of_the_streak():
    # DAH-5244 review: a read-then-set counter let two cycles take the same slot of the bounded streak
    import asyncio

    from fakeredis.aioredis import FakeRedis

    from services.redis_service import RedisService

    class AnswersOnALaterTurn(FakeRedis):
        async def get(self, key):
            await asyncio.sleep(0)  # a socket's answer arrives on a later loop turn; fakeredis answers at once
            return await super().get(key)

    service = RedisService.__new__(RedisService)
    service.redis = AnswersOnALaterTurn()
    service.lock = asyncio.Lock()

    counts = await asyncio.gather(
        *(service.record_disk_pressure_deferral("hk", "ex", shadow=False, ttl_seconds=3600) for _ in range(2))
    )

    assert sorted(counts) == [1, 2]
    assert await service.disk_pressure_deferrals("hk", "ex", shadow=False) == 2
    assert await service.redis.ttl("disk_pressure_deferrals:hk:ex") > 3000
    await service.clear_disk_pressure_deferrals("hk", "ex", shadow=False)
    assert await service.disk_pressure_deferrals("hk", "ex", shadow=False) == 0
