from __future__ import annotations

import pytest
from neurons.validators.src.services.task.checks.sysbox_required import SysboxRequiredCheck
from neurons.validators.src.services.task.messages import SysboxRequiredMessages as Msg
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)

from core.config import settings
from tests.helpers import build_state


def _rented_data() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            "executor-123": RentedExecutor(
                miner_hotkey="test-miner",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[RentedPod(pod_id="pod-1", container_name="container_test", rented_ports=[8080, 8081])],
            )
        },
    )


@pytest.mark.asyncio
async def test_no_sysbox_unrented_fails(context_factory):
    """Unrented executor without sysbox is rejected, but its verification is NOT cleared."""
    ctx = context_factory(state=build_state(sysbox_runtime=False))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.SYSBOX_MISSING.reason
    assert "clear_verified_job_info" not in result.updates


@pytest.mark.asyncio
async def test_no_sysbox_rented_passes(context_factory):
    """Rented executor without sysbox is left untouched so live rentals are not disrupted."""
    ctx = context_factory(state=build_state(sysbox_runtime=False, rented_data=_rented_data()))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SYSBOX_OK.reason


@pytest.mark.asyncio
async def test_sysbox_present_passes(context_factory):
    """Executor with sysbox passes regardless of rental status."""
    ctx = context_factory(state=build_state(sysbox_runtime=True))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SYSBOX_OK.reason


@pytest.mark.asyncio
async def test_disabled_flag_skips_enforcement(context_factory, monkeypatch):
    """With the kill-switch off, no-sysbox unrented executors are not rejected."""
    monkeypatch.setattr(settings, "REQUIRE_SYSBOX_FOR_UNRENTED", False)
    ctx = context_factory(state=build_state(sysbox_runtime=False))

    result = await SysboxRequiredCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DISABLED.reason
