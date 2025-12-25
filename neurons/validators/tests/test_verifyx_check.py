import pytest

from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from neurons.validators.src.services.task.messages import VerifyXMessages as Msg

from tests.helpers import build_context_config, build_services, build_state


# Mock VerifyX response matching the real VerifyXResponse
class MockVerifyXResponse:
    def __init__(self, data: dict | None = None, error: str | None = None):
        self.data = data
        self.error = error


# Mock VerifyX service
class DummyVerifyXService:
    def __init__(self, *, success: bool, error_msg: str = "", updated_specs: dict | None = None):
        """
        Args:
            success: Whether verification succeeds
            error_msg: Error message if verification fails
            updated_specs: Additional specs to return (ram, hard_disk, network)
        """
        self.success = success
        self.error_msg = error_msg
        self.updated_specs = updated_specs or {}
        self.called_with: dict | None = None

    async def validate_verifyx_and_process_job(
        self,
        *,
        shell,
        executor_info,
        default_extra: dict,
        machine_spec: dict,
    ) -> MockVerifyXResponse:
        """Mock method that mimics the real verifyx service."""
        # Track what parameters we were called with
        self.called_with = {
            "shell": shell,
            "executor_info": executor_info,
            "default_extra": default_extra,
            "machine_spec": machine_spec,
        }

        if self.success:
            # Return successful response with updated specs
            data = {
                "success": True,
                **self.updated_specs,
            }
            return MockVerifyXResponse(data=data)
        else:
            # Return failure with error
            if self.error_msg:
                return MockVerifyXResponse(error=self.error_msg)
            else:
                data = {"success": False, "errors": "Verification failed"}
                return MockVerifyXResponse(data=data)


@pytest.mark.parametrize(
    "verifyx_enabled,has_specs,verify_success,error_msg,updated_specs,expected_pass,expected_reason",
    [
        # VerifyX disabled - should pass without calling service
        (False, True, True, "", {}, True, Msg.DISABLED.reason),
        # VerifyX enabled but no specs - should fail
        (True, False, True, "", {}, False, Msg.NO_SPECS.reason),
        # VerifyX succeeds with updated specs
        (
            True,
            True,
            True,
            "",
            {"ram": {"total": "64GB"}, "hard_disk": {"total": "1TB"}, "network": {"download_speed": 1000}},
            True,
            Msg.VERIFY_SUCCESS.reason,
        ),
        # VerifyX succeeds without additional specs
        (True, True, True, "", {}, True, Msg.VERIFY_SUCCESS.reason),
        # VerifyX fails with error message
        (True, True, False, "Checksum mismatch", {}, False, Msg.VERIFY_FAILED.reason),
        # VerifyX fails with data errors
        (True, True, False, "", {}, False, Msg.VERIFY_FAILED.reason),
    ],
)
@pytest.mark.asyncio
async def test_verifyx_check(
    verifyx_enabled,
    has_specs,
    verify_success,
    error_msg,
    updated_specs,
    expected_pass,
    expected_reason,
    context_factory,
):
    # Setup mock service
    verifyx_service = DummyVerifyXService(
        success=verify_success,
        error_msg=error_msg,
        updated_specs=updated_specs,
    )

    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=verifyx_enabled)

    # Setup specs
    base_specs = {"gpu": {"count": 2}, "cpu": {"cores": 8}} if has_specs else {}
    state = build_state(specs=base_specs)

    # Create context
    ctx = context_factory(services=services, config=config, state=state)

    # Run the check
    result = await VerifyXCheck().run(ctx)

    # Verify result
    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    # Verify service interactions based on scenario
    if not verifyx_enabled:
        # Service should not be called when disabled
        assert verifyx_service.called_with is None
    elif not has_specs:
        # Service should not be called when no specs
        assert verifyx_service.called_with is None
    else:
        # Service should be called
        assert verifyx_service.called_with is not None
        assert verifyx_service.called_with["machine_spec"] == base_specs

        # Verify specs update on success
        if verify_success and "state" in result.updates:
            updated_state = result.updates["state"]
            # Base specs should still be there
            assert updated_state.specs.get("gpu") == base_specs["gpu"]
            assert updated_state.specs.get("cpu") == base_specs["cpu"]
            # Additional specs should be merged if provided
            if "ram" in updated_specs:
                assert updated_state.specs.get("ram") == updated_specs["ram"]
            if "hard_disk" in updated_specs:
                assert updated_state.specs.get("hard_disk") == updated_specs["hard_disk"]
            if "network" in updated_specs:
                assert updated_state.specs.get("network") == updated_specs["network"]
