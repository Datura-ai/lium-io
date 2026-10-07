
import pytest

from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from neurons.validators.src.services.task.messages import (
    VerifyXMessages as Msg,
)
from protocol.vc_protocol.compute_requests import NetworkEMA, RentedExecutorsResponse

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
        challenge_config_overrides: dict | None = None,
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
        # VerifyX succeeds without network data → EMA=0 < threshold → fails
        (True, True, True, "", {}, False, Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason),
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


def _rented_data_with_ema(executor_uuid: str, *, download: float | None = None, upload: float | None = None) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        banned_guids=[],
        network_ema={
            executor_uuid: NetworkEMA(
                ema_verifyx_download_speed=download,
                ema_verifyx_upload_speed=upload,
            )
        },
    )


@pytest.mark.asyncio
async def test_verifyx_success_ema_below_threshold_fails_check(context_factory):
    """Verifyx passes but EMA drops below 100 Mbps threshold → check fails with specific reason."""
    # prev_ema=110, current=0 → new EMA = 0.5*0 + 0.5*110 = 55 < 100
    verifyx_service = DummyVerifyXService(success=True, updated_specs={"network": {}})
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(
        specs={"gpu": {"count": 1}},
        rented_data=_rented_data_with_ema("executor-123", download=110.0),
    )
    ctx = context_factory(services=services, config=config, state=state)

    result = await VerifyXCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(55.0)


@pytest.mark.asyncio
async def test_verifyx_success_ema_at_threshold_passes(context_factory):
    """EMA exactly at 100 Mbps passes (threshold is strictly less-than)."""
    # prev_ema=None, current=100 → EMA = 100.0 (bootstrap)
    verifyx_service = DummyVerifyXService(
        success=True,
        updated_specs={"network": {"download_speed": 100.0, "upload_speed": 50.0}},
    )
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(specs={"gpu": {"count": 1}}, rented_data=None)
    ctx = context_factory(services=services, config=config, state=state)

    result = await VerifyXCheck().run(ctx)

    assert result.passed is True


