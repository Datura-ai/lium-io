import logging

import pytest

from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from neurons.validators.src.services.task.messages import (
    VERIFYX_DEBUG_DOC_URL,
    VerifyXMessages as Msg,
)

from tests.helpers import build_context_config, build_services, build_state


# Mock VerifyX response matching the real VerifyXResponse
class MockVerifyXResponse:
    def __init__(
        self,
        data: dict | None = None,
        error: str | None = None,
        diagnostics: dict | None = None,
    ):
        self.data = data
        self.error = error
        self.diagnostics = diagnostics


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
            {"ram": {"total": "64GB"}, "hard_disk": {"total": "1TB"}, "network": {"download_speed": 1000, "upload_speed": 60.0}},
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
                # verifyx speeds are stored under their own additive keys (do not overwrite speedtest values)
                assert updated_state.specs["network"].get("verifyx_download_speed") == updated_specs["network"].get("download_speed")
                if updated_specs["network"].get("upload_speed") is not None:
                    assert updated_state.specs["network"].get("verifyx_upload_speed") == updated_specs["network"].get("upload_speed")
                    assert isinstance(updated_state.specs["network"].get("ema_verifyx_upload_speed"), float)


class _VerifyXServiceReturning:
    """Test double that returns a caller-supplied VerifyXResponse."""

    def __init__(self, response: MockVerifyXResponse):
        self._response = response

    async def validate_verifyx_and_process_job(
        self, *, shell, executor_info, default_extra, machine_spec
    ):
        return self._response


@pytest.mark.parametrize(
    "failure_class,expected_reason",
    [
        ("SSH_TRANSPORT", Msg.VERIFY_FAILED_SSH_TRANSPORT.reason),
        ("EXECUTOR_CRASH", Msg.VERIFY_FAILED_EXECUTOR_CRASH.reason),
        ("EMPTY_RESPONSE", Msg.VERIFY_FAILED_EMPTY_RESPONSE.reason),
        ("CIPHER_REJECTED", Msg.VERIFY_FAILED_CIPHER_REJECTED.reason),
        ("UNKNOWN", Msg.VERIFY_FAILED.reason),
    ],
)
@pytest.mark.asyncio
async def test_verifyx_failure_classes_pick_per_class_template(
    failure_class, expected_reason, context_factory
):
    diagnostics = {
        "failure_class": failure_class,
        "exit_status": 1 if failure_class == "EXECUTOR_CRASH" else 0,
        "stdout_len": 12 if failure_class == "EMPTY_RESPONSE" else 0,
        "stderr_tail": "boom" if failure_class == "EXECUTOR_CRASH" else None,
        "transport_error": (
            "ConnectionLost: timeout" if failure_class == "SSH_TRANSPORT" else None
        ),
    }
    verifyx_service = _VerifyXServiceReturning(
        MockVerifyXResponse(error="failure", diagnostics=diagnostics)
    )
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(specs={"gpu": {"count": 1}})
    ctx = context_factory(services=services, config=config, state=state)

    result = await VerifyXCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == expected_reason
    # Diagnostic keys surfaced into what_we_saw
    assert result.event.what_we_saw["failure_class"] == failure_class
    assert "exit_status" in result.event.what_we_saw
    assert "stdout_len" in result.event.what_we_saw
    assert "stderr_tail" in result.event.what_we_saw
    # help_uri populated on every failure class
    assert result.event.help_uri == VERIFYX_DEBUG_DOC_URL


@pytest.mark.asyncio
async def test_verifyx_failure_without_diagnostics_falls_back_to_generic_template(
    context_factory,
):
    verifyx_service = _VerifyXServiceReturning(
        MockVerifyXResponse(error="opaque failure", diagnostics=None)
    )
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(specs={"gpu": {"count": 1}})
    ctx = context_factory(services=services, config=config, state=state)

    result = await VerifyXCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED.reason
    # No diagnostic keys when no diagnostics available
    assert "failure_class" not in result.event.what_we_saw
    assert result.event.help_uri == VERIFYX_DEBUG_DOC_URL


@pytest.mark.asyncio
async def test_verifyx_success_path_emits_no_verifyx_failed_log(
    context_factory, caplog
):
    """Regression: success path MUST NOT emit the new VerifyX failure log line."""
    updated_specs = {"ram": {"total": "64GB"}, "hard_disk": {"total": "1TB"}}
    verifyx_service = DummyVerifyXService(success=True, updated_specs=updated_specs)
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(specs={"gpu": {"count": 1}, "cpu": {"cores": 4}})
    ctx = context_factory(services=services, config=config, state=state)

    with caplog.at_level(logging.ERROR):
        result = await VerifyXCheck().run(ctx)

    assert result.passed is True
    # Success path never populates diagnostics
    response_attr = getattr(result.event, "what_we_saw", {})
    assert "failure_class" not in response_attr
    # No ERROR log lines naming the new VerifyX failure event
    assert not any(
        "VerifyX validation failed" in rec.getMessage() for rec in caplog.records
    )
