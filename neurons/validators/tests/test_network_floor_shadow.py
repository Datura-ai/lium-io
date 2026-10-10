"""NETWORK_FLOOR shadow: a GPU-count-scaled bandwidth floor is logged, never scored."""

import logging
from unittest.mock import patch

import pytest
from neurons.validators.src.services.task.checks import NetworkFloorShadowCheck, RentedGpuHealthShadowCheck
from neurons.validators.src.services.task.checks import network_floor as module
from neurons.validators.src.services.task.checks.network_floor import scaled_download_floor_mbps
from neurons.validators.src.services.task.checks.verifyx import MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS
from neurons.validators.src.services.task.messages import NetworkFloorMessages as Msg
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from protocol.vc_protocol.compute_requests import NetworkEMA, RentedExecutorsResponse

from tests.helpers import build_context_config, build_services, build_state, default_executor


def _ctx(context_factory, *, gpu_count: int, download: float | None, upload: float | None = None):
    executor = default_executor()
    ema = {executor.uuid: NetworkEMA(ema_verifyx_download_speed=download, ema_verifyx_upload_speed=upload)}
    state = build_state(
        specs={"gpu": {"count": gpu_count}},
        gpu_count=gpu_count,
        rented_data=RentedExecutorsResponse(executors={}, network_ema=ema),
    )
    return context_factory(
        services=build_services(), config=build_context_config(), state=state, executor=executor
    )


@pytest.fixture
def enabled():
    with patch.object(module.settings, "NETWORK_FLOOR_SCALED_SHADOW_ENABLED", True):
        yield


def test_the_floor_scales_linearly_from_one_to_the_full_gpu_count():
    assert scaled_download_floor_mbps(1) == 100.0
    assert scaled_download_floor_mbps(2) == pytest.approx(228.6)
    assert scaled_download_floor_mbps(4) == pytest.approx(485.7)
    assert scaled_download_floor_mbps(8) == 1000.0
    assert scaled_download_floor_mbps(16) == 1000.0
    assert scaled_download_floor_mbps(0) == 100.0


def test_the_existing_gate_is_unchanged():
    assert MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS == 100.0


@pytest.mark.asyncio
async def test_disabled_by_default(context_factory, monkeypatch):
    monkeypatch.delenv("NETWORK_FLOOR_SCALED_SHADOW_ENABLED", raising=False)
    with patch.object(module, "settings", type(module.settings)(_env_file=None)):
        result = await NetworkFloorShadowCheck().run(_ctx(context_factory, gpu_count=8, download=150.0))

    assert result.passed and result.event.reason_code == Msg.DISABLED.reason


@pytest.mark.asyncio
async def test_a_large_node_under_its_floor_is_logged_but_passes(context_factory, enabled, caplog):
    ctx = _ctx(context_factory, gpu_count=8, download=400.0, upload=300.0)

    with caplog.at_level(logging.WARNING, logger=module.__name__):
        result = await NetworkFloorShadowCheck().run(ctx)

    assert result.passed is True and NetworkFloorShadowCheck.fatal is False
    assert "score" not in result.updates
    assert result.event.reason_code == Msg.SHADOW_BELOW.reason
    seen = result.event.what_we_saw
    assert seen["download_floor_mbps"] == 1000.0 and seen["upload_floor_mbps"] == 500.0
    assert seen["below"] == ["download", "upload"] and seen["verdict"] == "below_floor"
    assert any("network_floor_shadow" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_a_small_node_at_the_same_speed_is_ok(context_factory, enabled):
    result = await NetworkFloorShadowCheck().run(_ctx(context_factory, gpu_count=1, download=400.0, upload=300.0))

    assert result.event.reason_code == Msg.SHADOW_OK.reason
    assert result.event.what_we_saw["below"] == []


@pytest.mark.asyncio
async def test_only_the_upload_under_its_floor(context_factory, enabled):
    result = await NetworkFloorShadowCheck().run(_ctx(context_factory, gpu_count=8, download=1200.0, upload=200.0))

    assert result.event.what_we_saw["below"] == ["upload"]


@pytest.mark.asyncio
async def test_no_ema_is_no_verdict(context_factory, enabled):
    result = await NetworkFloorShadowCheck().run(_ctx(context_factory, gpu_count=8, download=None))

    assert result.passed and result.event.reason_code == Msg.NO_READING.reason


@pytest.mark.parametrize(
    "build",
    [PipelineFactory.build_checks, PipelineFactory.build_dry_run_checks, PipelineFactory.build_fast_path_checks],
)
def test_the_check_runs_before_the_rented_halt(build):
    kinds = [type(check) for check in build()]

    assert kinds.index(NetworkFloorShadowCheck) == kinds.index(RentedGpuHealthShadowCheck) + 1
