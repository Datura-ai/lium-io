import pytest
from dataclasses import dataclass

from neurons.validators.src.services.task.checks.port_connectivity import PortConnectivityCheck
from neurons.validators.src.services.task.messages import PortConnectivityMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


# Mock result class matching DockerConnectionCheckResult
@dataclass
class MockConnectivityResult:
    success: bool
    log_text: str | None = None
    sysbox_runtime: bool = False


# Mock Redis service
class DummyRedis:
    def __init__(self, *, renting_in_progress: bool = False):
        self.renting_in_progress_value = renting_in_progress

    async def renting_in_progress(self, miner_hotkey: str, executor_uuid: str) -> bool:
        return self.renting_in_progress_value


# Mock connectivity service
class DummyConnectivityService:
    def __init__(self, *, success: bool, log_text: str = "", sysbox_runtime: bool = False):
        """
        Args:
            success: Whether verify_ports returns success
            log_text: The log message from verification
            sysbox_runtime: The sysbox runtime state to return
        """
        self.success = success
        self.log_text = log_text
        self.sysbox_runtime = sysbox_runtime
        self.called_with: dict | None = None

    async def verify_ports(
        self,
        ssh_client,
        job_batch_id: str,
        miner_hotkey: str,
        executor_info,
        private_key: str,
        public_key: str,
        sysbox_runtime: bool,
    ) -> MockConnectivityResult:
        """Mock method that mimics the real connectivity service."""
        # Track what parameters we were called with
        self.called_with = {
            "job_batch_id": job_batch_id,
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_info.uuid,
            "private_key": private_key,
            "public_key": public_key,
            "sysbox_runtime": sysbox_runtime,
        }

        return MockConnectivityResult(
            success=self.success,
            log_text=self.log_text,
            sysbox_runtime=self.sysbox_runtime,
        )


@pytest.mark.parametrize(
    "rented,renting_in_progress,has_config,verify_success,sysbox_runtime,expected_pass,expected_reason",
    [
        # Already rented - skip check
        (True, False, True, True, False, True, Msg.SKIPPED_RENTED.reason),
        # Renting in progress - skip verification
        (False, True, True, True, False, True, Msg.RENTING_IN_PROGRESS.reason),
        # Missing config - fail
        (False, False, False, True, False, False, Msg.CONFIG_MISSING.reason),
        # Verification succeeds
        (False, False, True, True, False, True, Msg.VERIFY_OK.reason),
        # Verification succeeds with sysbox runtime
        (False, False, True, True, True, True, Msg.VERIFY_OK.reason),
        # Verification fails
        (False, False, True, False, False, False, Msg.VERIFY_FAILED.reason),
    ],
)
@pytest.mark.asyncio
async def test_port_connectivity_check(
    rented,
    renting_in_progress,
    has_config,
    verify_success,
    sysbox_runtime,
    expected_pass,
    expected_reason,
    context_factory,
):
    # Setup mocks
    redis_service = DummyRedis(renting_in_progress=renting_in_progress)
    connectivity_service = DummyConnectivityService(
        success=verify_success,
        log_text="Port verification completed" if verify_success else "Port verification failed",
        sysbox_runtime=sysbox_runtime,
    )

    services = build_services(
        redis=redis_service,
        connectivity=connectivity_service,
    )

    # Setup config with or without required keys
    if has_config:
        config = build_context_config(
            job_batch_id="batch-123",
            port_private_key="private-key-data",
            port_public_key="public-key-data",
        )
    else:
        config = build_context_config(
            job_batch_id=None,
            port_private_key=None,
            port_public_key=None,
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
    if rented:
        # Should not call Redis or connectivity service
        assert connectivity_service.called_with is None
    elif renting_in_progress:
        # Should call Redis but not connectivity service
        assert connectivity_service.called_with is None
        # Verify updates
        assert result.updates.get("renting_in_progress") is True
    elif not has_config:
        # Should not call connectivity service
        assert connectivity_service.called_with is None
    else:
        # Should call connectivity service
        assert connectivity_service.called_with is not None
        assert connectivity_service.called_with["job_batch_id"] == "batch-123"
        assert connectivity_service.called_with["private_key"] == "private-key-data"
        assert connectivity_service.called_with["public_key"] == "public-key-data"

        # Verify state update with sysbox_runtime
        if "state" in result.updates:
            assert result.updates["state"].sysbox_runtime == sysbox_runtime
