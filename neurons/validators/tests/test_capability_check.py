import pytest

from neurons.validators.src.services.task.checks.capability import CapabilityCheck
from neurons.validators.src.services.task.messages import CapabilityMessages as Msg
from neurons.validators.src.services.matrix_validation_service import ValidationResult
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod

from tests.helpers import build_context_config, build_services, build_state


# Mock validation service - mimics the real ValidationService interface
class DummyValidationService:
    def __init__(
        self,
        *,
        success: bool,
        should_raise: bool = False,
        error_message: str = "",
        metrics: dict | None = None,
    ):
        """
        Args:
            success: Whether validate_gpu_model_and_process_job should return success/failure
            should_raise: Whether to raise an exception instead of returning
            error_message: The exception message if should_raise=True or error in ValidationResult
            metrics: Optional FP32 TFLOPS metrics dict to surface on the ValidationResult
        """
        self.success = success
        self.should_raise = should_raise
        self.error_message = error_message
        self.metrics = metrics
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
            error_message="" if self.success else self.error_message or "Validation failed",
            metrics=self.metrics,
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


@pytest.mark.asyncio
async def test_capability_check_skips_active_filler_only_executor(context_factory):
    validation_service = DummyValidationService(success=True)
    services = build_services(validation=validation_service)
    state = build_state(
        specs={"gpu": {"count": 1}},
        rented_data=RentedExecutorsResponse(
            executors={},
            banned_guids=[],
            filler_containers_by_executor={"executor-123": "filler_active"},
        ),
    )
    ctx = context_factory(services=services, state=state)

    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_SKIPPED.reason
    assert result.event.what_we_saw == {"filler_container": "filler_active"}
    assert validation_service.called_with is None


@pytest.mark.asyncio
async def test_capability_check_runs_for_customer_rental_even_with_filler_mapping(context_factory):
    validation_service = DummyValidationService(success=True)
    services = build_services(validation=validation_service)
    state = build_state(
        specs={"gpu": {"count": 1}},
        rented_data=RentedExecutorsResponse(
            executors={
                "executor-123": RentedExecutor(
                    miner_hotkey="test-miner",
                    executor_ip_address="127.0.0.1",
                    executor_ip_port="22",
                    pods=[
                        RentedPod(
                            pod_id="pod-1",
                            container_name="container_test",
                            rented_ports=[8080],
                        )
                    ],
                )
            },
            banned_guids=[],
            filler_containers_by_executor={"executor-123": "filler_active"},
        ),
    )
    ctx = context_factory(services=services, state=state)

    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFY_OK.reason
    assert validation_service.called_with is not None


@pytest.mark.asyncio
async def test_capability_check_captures_metrics_into_state(context_factory):
    """On success with metrics present, the check writes gpu_metrics into pipeline state."""
    metrics = {"device": "NVIDIA A100", "cc": "8.0", "fp32_tflops": 51.2}
    validation_service = DummyValidationService(success=True, metrics=metrics)
    services = build_services(validation=validation_service)
    state = build_state(specs={"gpu": {"count": 1}})
    ctx = context_factory(services=services, state=state)

    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFY_OK.reason
    assert "state" in result.updates
    assert result.updates["state"].gpu_metrics == metrics


@pytest.mark.asyncio
async def test_capability_check_no_metrics_leaves_state_none(context_factory):
    """Fail-safe: success with no metrics leaves state untouched (no gpu_metrics)."""
    validation_service = DummyValidationService(success=True, metrics=None)
    services = build_services(validation=validation_service)
    state = build_state(specs={"gpu": {"count": 1}})
    ctx = context_factory(services=services, state=state)

    result = await CapabilityCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFY_OK.reason
    assert "state" not in result.updates
