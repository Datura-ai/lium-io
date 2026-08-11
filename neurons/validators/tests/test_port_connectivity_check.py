import pytest
from dataclasses import dataclass, field

from neurons.validators.src.protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from neurons.validators.src.services.executor_connectivity.models import PortPair, PortVerificationResult
from neurons.validators.src.services.task.checks.port_connectivity import PortConnectivityCheck
from neurons.validators.src.services.task.messages import PortConnectivityMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


# Mock result class matching DockerConnectionCheckResult
# Mock Redis service
class DummyRedis:
    def __init__(self, *, renting_in_progress: bool = False):
        self.renting_in_progress_value = renting_in_progress

    async def renting_in_progress(self, miner_hotkey: str, executor_uuid: str) -> bool:
        return self.renting_in_progress_value


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
    ):
        """
        Args:
            success: Whether verify_ports returns success
            log_text: The log message from verification
            sysbox_runtime: The sysbox runtime state to return
            verified_port_count: Number of verified working ports
            status: Override the status field (e.g., "skipped_rental_active")
        """
        self.success = success
        self.log_text = log_text
        self.sysbox_runtime = sysbox_runtime
        self.verified_port_count = verified_port_count
        self.status = status
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
            dind_ok=self.success,
            sysbox_runtime=self.sysbox_runtime,
            status=status,
            error=error,
            elapsed_sec=1.0,
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
            elif status_override == "skipped_rental_active":
                # DAH-2647: the check was skipped, so it reports no count rather than zero.
                assert result.updates["state"].verified_port_count is None
            else:
                assert result.updates["state"].verified_port_count == 0


@pytest.mark.parametrize(
    "status,probed_port_count,expected_count",
    [
        ("ok", 3, 3),
        ("no_working_ports", 0, 0),
        ("no_ports", 0, None),
        ("error", 0, None),
    ],
)
@pytest.mark.asyncio
async def test_port_connectivity_reports_no_count_when_it_never_probed(
    status, probed_port_count, expected_count, context_factory
):
    """DAH-2647: no_ports and error carry no evidence about the executor's ports, so they must
    reach PortCountCheck as an absent count rather than as a measured zero."""
    ctx = context_factory(
        services=build_services(
            redis=DummyRedis(),
            backend=DummyBackendService(),
            connectivity=DummyConnectivityService(
                success=status == "ok", verified_port_count=probed_port_count, status=status
            ),
        ),
        config=build_context_config(job_batch_id="batch-123"),
        state=build_state(),
    )

    result = await PortConnectivityCheck().run(ctx)

    assert result.updates["state"].verified_port_count == expected_count


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
