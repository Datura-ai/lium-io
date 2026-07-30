import logging

import pytest

from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck
from neurons.validators.src.services.task.messages import (
    VERIFYX_DEBUG_DOC_URL,
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
async def test_verifyx_skips_active_filler_and_reuses_prev_ema(context_factory):
    verifyx_service = DummyVerifyXService(
        success=True,
        updated_specs={"network": {"download_speed": 500.0, "upload_speed": 100.0}},
    )
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(
        specs={"gpu": {"count": 1}},
        rented_data=RentedExecutorsResponse(
            executors={},
            banned_guids=[],
            filler_containers_by_executor={"executor-123": "filler_active"},
            network_ema={
                "executor-123": NetworkEMA(
                    ema_verifyx_download_speed=350.0,
                    ema_verifyx_upload_speed=70.0,
                )
            },
        ),
    )
    ctx = context_factory(services=services, config=config, state=state)

    result = await VerifyXCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.FILLER_SKIPPED.reason
    assert verifyx_service.called_with is None
    network = result.updates["state"].specs["network"]
    assert network["ema_verifyx_download_speed"] == pytest.approx(350.0)
    assert network["ema_verifyx_upload_speed"] == pytest.approx(70.0)
    assert "verifyx_download_speed" not in network


@pytest.mark.asyncio
async def test_verifyx_failure_without_prev_ema_leaves_specs_empty(context_factory):
    """No prior EMA anywhere — failure branch must not fabricate a state update."""
    verifyx_service = _VerifyXServiceReturning(MockVerifyXResponse(error="failure"))
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(
        specs={"gpu": {"count": 1}},
        rented_data=RentedExecutorsResponse(executors={}, banned_guids=[], network_ema={}),
    )
    ctx = context_factory(services=services, config=config, state=state)

    result = await VerifyXCheck().run(ctx)

    assert result.passed is False
    assert "state" not in result.updates


@pytest.mark.asyncio
async def test_verifyx_failure_without_rented_data_does_not_update_state(context_factory):
    """rented_data is None — failure branch must not attempt to read from it."""
    verifyx_service = _VerifyXServiceReturning(MockVerifyXResponse(error="failure"))
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(specs={"gpu": {"count": 1}}, rented_data=None)
    ctx = context_factory(services=services, config=config, state=state)

    result = await VerifyXCheck().run(ctx)

    assert result.passed is False
    assert "state" not in result.updates


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


@pytest.mark.asyncio
async def test_verifyx_success_null_network_bootstraps_ema_to_zero_and_fails(context_factory):
    """First run: verifyx passes but network failed → EMA=0.0 < threshold → check fails."""
    verifyx_service = DummyVerifyXService(success=True, updated_specs={"network": {}})
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(specs={"gpu": {"count": 1}}, rented_data=None)
    ctx = context_factory(services=services, config=config, state=state)

    result = await VerifyXCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == pytest.approx(0.0)
    assert result.updates["state"].specs["network"]["ema_verifyx_upload_speed"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_verifyx_success_null_network_decays_prev_ema(context_factory):
    """Verifyx passes but network failed → EMA decays: compute_ema(prev, 0.0) = 0.5 × prev."""
    verifyx_service = DummyVerifyXService(success=True, updated_specs={"network": {}})
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(
        specs={"gpu": {"count": 1}},
        rented_data=_rented_data_with_ema("executor-123", download=500.0, upload=100.0),
    )
    ctx = context_factory(services=services, config=config, state=state)

    result = await VerifyXCheck().run(ctx)

    assert result.passed is True
    net = result.updates["state"].specs["network"]
    # compute_ema(500.0, 0.0) = 0.5 * 500 = 250.0
    assert net["ema_verifyx_download_speed"] == pytest.approx(250.0)
    # compute_ema(100.0, 0.0) = 0.5 * 100 = 50.0
    assert net["ema_verifyx_upload_speed"] == pytest.approx(50.0)
    # raw speed keys must NOT be set when there was no measurement
    assert "verifyx_download_speed" not in net
    assert "verifyx_upload_speed" not in net


@pytest.mark.asyncio
async def test_verifyx_success_null_network_repeated_decays_below_threshold(context_factory):
    """5 consecutive verifyx-pass / network-fail cycles from 500 Mbps → EMA below 100 Mbps."""
    # 500 * 0.5^5 = 15.625 < 100 Mbps
    ema = 500.0
    for _ in range(5):
        verifyx_service = DummyVerifyXService(success=True, updated_specs={"network": {}})
        services = build_services(verifyx=verifyx_service)
        config = build_context_config(verifyx_enabled=True)
        state = build_state(
            specs={"gpu": {"count": 1}},
            rented_data=_rented_data_with_ema("executor-123", download=ema),
        )
        ctx = context_factory(services=services, config=config, state=state)
        result = await VerifyXCheck().run(ctx)
        ema = result.updates["state"].specs["network"]["ema_verifyx_download_speed"]

    assert ema < 100.0


@pytest.mark.asyncio
async def test_verifyx_keeps_the_scrapes_disk_breakdown(context_factory):
    """VerifyX measures only total/used/free, so its dict must not evict the scrape's own
    docker usage fields (DAH-2514) — they would be dropped on every node where VerifyX runs."""
    # Arrange
    verifyx_service = DummyVerifyXService(
        success=True,
        updated_specs={"hard_disk": {"total": 1000, "used": 400, "free": 600}, "network": {}},
    )
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(
        specs={
            "gpu": {"count": 1},
            "hard_disk": {"total": 999, "used": 399, "free": 600, "images": 120, "volumes": 250},
        }
    )
    ctx = context_factory(services=services, config=config, state=state)

    # Act
    result = await VerifyXCheck().run(ctx)

    # Assert — VerifyX wins on the fields it measures, the breakdown survives
    hard_disk = result.updates["state"].specs["hard_disk"]
    assert hard_disk["total"] == 1000
    assert hard_disk["images"] == 120
    assert hard_disk["volumes"] == 250


@pytest.mark.asyncio
async def test_verifyx_success_path_emits_no_verifyx_failed_log(
    context_factory, caplog
):
    """Regression: success path MUST NOT emit the new VerifyX failure log line."""
    updated_specs = {"ram": {"total": "64GB"}, "hard_disk": {"total": "1TB"}, "network": {"download_speed": 500.0, "upload_speed": 100.0}}
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
