import pytest

from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.messages import CapabilityMessages as Msg
from neurons.validators.src.services.matrix_validation_service import ValidationResult

from tests.helpers import build_context_config, build_services, build_state


# Mock validation service - mimics the real ValidationService interface
class DummyValidationService:
    def __init__(self, *, success: bool, should_raise: bool = False, error_message: str = ""):
        """
        Args:
            success: Whether validate_gpu_model_and_process_job should return success/failure
            should_raise: Whether to raise an exception instead of returning
            error_message: The exception message if should_raise=True or error in ValidationResult
        """
        self.success = success
        self.should_raise = should_raise
        self.error_message = error_message
        # Track what parameters the check called us with
        self.called_with: dict | None = None

    async def validate_gpu_model_and_process_job(
        self,
        *,
        ssh_client,
        executor_info,
        default_extra,
        machine_spec,
    ):
        """Mock method that mimics the real validation service."""
        # Store the call parameters so we can verify them in tests
        self.called_with = {
            "ssh_client": ssh_client,
            "executor_info": executor_info,
            "default_extra": default_extra,
            "machine_spec": machine_spec,
        }

        # Simulate an exception if requested
        if self.should_raise:
            raise RuntimeError(self.error_message)

        # Return ValidationResult instead of bool
        return ValidationResult(
            success=self.success,
            expected_uuid="test-uuid-123",
            returned_uuid="test-uuid-123" if self.success else "wrong-uuid",
            stdout="UUID: test-uuid-123" if self.success else "UUID: wrong-uuid",
            stderr="",
            error_message="" if self.success else self.error_message or "Validation failed"
        )


@pytest.mark.parametrize(
    "specs,success,should_raise,error_msg,expected_pass,expected_reason",
    [
        # No specs - should fail immediately without calling service
        ({}, False, False, "", False, Msg.NO_SPECS.reason),
        # Validation succeeds
        ({"gpu": {"count": 2}}, True, False, "", True, Msg.VERIFY_OK.reason),
        # Validation fails (returns False)
        ({"gpu": {"count": 2}}, False, False, "", False, Msg.VERIFY_FAILED.reason),
        # Validation raises exception
        ({"gpu": {"count": 2}}, False, True, "GPU not accessible in container", False, Msg.VERIFY_FAILED.reason),
    ],
)
@pytest.mark.asyncio
async def test_capability_check(
    specs,
    success,
    should_raise,
    error_msg,
    expected_pass,
    expected_reason,
    context_factory,
):
    # Create our mock validation service with the test parameters
    validation_service = DummyValidationService(
        success=success,
        should_raise=should_raise,
        error_message=error_msg,
    )

    # Inject the mock service into the context services
    services = build_services(validation=validation_service)
    config = build_context_config()
    state = build_state(specs=specs)

    # Create context with our mocked services
    ctx = context_factory(services=services, config=config, state=state)

    # Run the check
    result = await CapabilityCheck().run(ctx)

    # Verify the result
    assert result.passed is expected_pass
    assert result.event.reason_code == expected_reason

    # If specs were empty, the service shouldn't have been called
    if not specs:
        assert validation_service.called_with is None
    else:
        # Otherwise, verify the service was called with correct parameters
        assert validation_service.called_with is not None
        assert validation_service.called_with["machine_spec"] == specs
        assert validation_service.called_with["executor_info"] == ctx.executor
