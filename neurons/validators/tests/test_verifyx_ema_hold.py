"""The VerifyX EMA is not moved by a cycle the node did not pass for another reason.

The incident: a cycle that failed the cached-image check still seeded the EMA while the image pull
shared the link, and the next cycle failed the 100 Mbps gate at 86.7.
"""

import asyncio
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from core.config import Settings, settings
from neurons.validators.src.services.task.checks.cached_template_verification import (
    CachedTemplateVerificationCheck,
)
from neurons.validators.src.services.task.checks.verifyx import _EMA_KEYS, VerifyXCheck, hold_verifyx_ema
from neurons.validators.src.services.task.messages import CachedTemplateMessages
from neurons.validators.src.services.task.messages import VerifyXMessages as Msg
from protocol.vc_protocol.compute_requests import DefaultDockerImage
from services.redis_service import RedisService
from neurons.validators.src.services.task.pipeline import CheckResult, Pipeline
from neurons.validators.src.services.task.result_handler import ResultHandler
from protocol.vc_protocol.compute_requests import NetworkEMA, RentedExecutorsResponse
from protocol.vc_protocol.validator_requests import ValidationEvent

from tests.helpers import build_context_config, build_services, build_state
from tests.test_verifyx_check import MockVerifyXResponse

EXECUTOR = "executor-123"  # tests.helpers.default_executor().uuid
IMAGE_CHECK = "executor.validate.cached_template"


@pytest.fixture(autouse=True)
def _hold_enabled(monkeypatch):
    # The hold ships off; every test below runs it on unless it says otherwise.
    monkeypatch.setattr(settings, "VERIFYX_EMA_HOLD_ENABLED", True)


def _never_measured() -> RentedExecutorsResponse:
    return RentedExecutorsResponse(executors={}, banned_guids=[], network_ema={})


def _known(download: float, upload: float = 50.0) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        banned_guids=[],
        network_ema={
            EXECUTOR: NetworkEMA(
                ema_verifyx_download_speed=download, ema_verifyx_upload_speed=upload
            )
        },
    )


def _event(check_id: str, *, failed: bool) -> ValidationEvent:
    return ValidationEvent(
        event="x",
        reason_code="X",
        severity="error" if failed else "info",
        impact="x",
        when=datetime(2026, 9, 20, tzinfo=UTC),
        check_id=check_id,
        what_we_saw={"steps_failed": check_id} if failed else {},
    )


def _measured_specs(download: float, ema_download: float, ema_upload: float) -> dict:
    return {
        "gpu": {"count": 1},
        "network": {
            "download_speed": 900.0,
            "verifyx_download_speed": download,
            "ema_verifyx_download_speed": ema_download,
            "verifyx_upload_speed": 40.0,
            "ema_verifyx_upload_speed": ema_upload,
        },
    }


async def _publish(context_factory, *, specs, rented_data, event, success, image_cached=None):
    ctx = context_factory(
        state=build_state(
            specs=specs, rented_data=rented_data, recommended_image_cached=image_cached
        ),
        ssh_pub_keys=[],
    )
    result = await ResultHandler(redis_service=None, dry_run=True).handle_result(
        context=ctx,
        miner_info=SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1"),
        executor_info=ctx.executor,
        verified_job_info={},
        log_text="x",
        success=success,
        validation_event=event,
    )
    return result.spec["network"]


_ABSENT = None
_STORED = {"ema_verifyx_download_speed": 200.0, "ema_verifyx_upload_speed": 50.0}
_PASSED = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("specs", "rented_data", "failed_check", "image_cached", "hold", "ema", "logged"),
    [
        # the raw samples are still published in every row: only the smoothed value is held
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), _never_measured(), IMAGE_CHECK, None, True,
            (_ABSENT, _ABSENT), True, id="failed-another-check-does-not-seed",
        ),
        pytest.param(
            _measured_specs(50.0, 125.0, 45.0), _known(200.0, 50.0), IMAGE_CHECK, None, True,
            (200.0, 50.0), True, id="failed-cycle-keeps-the-previous-ema",
        ),
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), None, IMAGE_CHECK, None, True,
            (120.0, 40.0), False, id="failed-cycle-without-the-backends-answer-holds-nothing",
        ),
        # the cached-image check passed (advisory, or a fresh node held as pending) without the image
        pytest.param(
            _measured_specs(20.0, 160.0, 40.0), _known(300.0), _PASSED, False, True,
            (160.0, 40.0), False, id="passing-uncached-cycle-of-a-known-node-publishes",
        ),
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), _never_measured(), _PASSED, False, True,
            (_ABSENT, _ABSENT), True, id="passing-uncached-cycle-does-not-seed",
        ),
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), _never_measured(), _PASSED, True, True,
            (120.0, 40.0), False, id="passing-cached-cycle-seeds",
        ),
        # every node looks never-measured without rented_data; a backend blip must not hold them all
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), None, _PASSED, False, True,
            (120.0, 40.0), False, id="passing-uncached-cycle-without-the-backends-answer-publishes",
        ),
        pytest.param(
            _measured_specs(60.0, 80.0, 40.0), _known(100.0), VerifyXCheck.check_id, None, True,
            (80.0, 40.0), False, id="verifyx-failing-its-own-gate-moves-the-ema",
        ),
        pytest.param(
            _measured_specs(300.0, 250.0, 45.0), _known(200.0), _PASSED, True, True,
            (250.0, 45.0), False, id="passing-cycle-publishes-the-new-ema",
        ),
        pytest.param(
            {"gpu": {"count": 1}, "network": {"download_speed": 900.0}}, _known(200.0),
            "executor.validate.port_count", None, True, (_ABSENT, _ABSENT), False,
            id="failure-before-verifyx-ran-adds-no-ema",
        ),
        pytest.param(
            {"gpu": {"count": 1}, "network": dict(_STORED)}, _known(200.0, 50.0), IMAGE_CHECK, None,
            True, (200.0, 50.0), False, id="stored-ema-only-is-published-unchanged-and-not-logged",
        ),
        pytest.param(
            {"gpu": {"count": 1}, "network": dict(_STORED)}, _known(200.0, 50.0), IMAGE_CHECK, None,
            False, (200.0, 50.0), False, id="flag-off-stored-ema-only-is-not-logged",
        ),
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), _never_measured(), IMAGE_CHECK, None, False,
            (120.0, 40.0), True, id="flag-off-failed-cycle-seeds-as-before",
        ),
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), _never_measured(), _PASSED, False, False,
            (120.0, 40.0), True, id="flag-off-passing-uncached-cycle-seeds-as-before",
        ),
    ],
)
async def test_what_a_cycle_publishes(
    monkeypatch, context_factory, caplog, specs, rented_data, failed_check, image_cached, hold, ema, logged
):
    if not hold:
        assert Settings.model_fields["VERIFYX_EMA_HOLD_ENABLED"].default is False
    monkeypatch.setattr(settings, "VERIFYX_EMA_HOLD_ENABLED", hold)

    with caplog.at_level(logging.INFO):
        network = await _publish(
            context_factory,
            specs=specs,
            rented_data=rented_data,
            event=_event(failed_check or "pipeline.finalize", failed=failed_check is not None),
            success=failed_check is None,
            image_cached=image_cached,
        )

    raw = {key: value for key, value in specs["network"].items() if key not in _EMA_KEYS}
    expected_ema = {key: value for key, value in zip(_EMA_KEYS, ema) if value is not _ABSENT}
    assert network == {**raw, **expected_ema}
    hold_logs = [r for r in caplog.records if r.getMessage().startswith(("VerifyX EMA held", "VerifyX EMA hold"))]
    assert bool(hold_logs) is logged


# --- cycles through the real pipeline and result handler ---------------------------------------


class _ProbeSequence:
    def __init__(self, *downloads: float):
        self._downloads = list(downloads)
        self.calls = 0

    async def validate_verifyx_and_process_job(
        self, *, shell, executor_info, default_extra, machine_spec
    ):
        self.calls += 1
        download = self._downloads.pop(0)
        return MockVerifyXResponse(
            data={
                "success": True,
                "ram": {"total": 64},
                "hard_disk": {"total": 1000},
                "network": {"download_speed": download, "upload_speed": 40.0},
            }
        )


class _PassingUncachedCheck:
    """The cached-image check passing without the image: advisory, or a fresh node held as pending."""

    check_id = IMAGE_CHECK
    fatal = True

    async def run(self, ctx):
        return CheckResult(
            passed=True,
            event=_event(IMAGE_CHECK, failed=False),
            updates={"state": replace(ctx.state, recommended_image_cached=False)},
        )


class _Sink:
    async def emit(self, event):
        pass


async def _cycle(context_factory, probes, rented_data, *, after=()):
    ctx = context_factory(
        services=build_services(verifyx=probes),
        config=build_context_config(verifyx_enabled=True),
        state=build_state(specs={"gpu": {"count": 1}}, rented_data=rented_data),
        ssh_pub_keys=[],
    )
    checks = [VerifyXCheck(), *after]
    ok, events, last = await Pipeline(checks, _Sink()).run(ctx)
    result = await ResultHandler(redis_service=None, dry_run=True).handle_result(
        context=last,
        miner_info=SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1"),
        executor_info=last.executor,
        verified_job_info={},
        log_text="x",
        success=ok,
        validation_event=events[-1],
    )
    return ok, events[-1], result.spec["network"]


def _as_backend_would_store(network: dict) -> RentedExecutorsResponse:
    ema = network.get("ema_verifyx_download_speed")
    if ema is None:
        return _never_measured()
    return _known(ema, network.get("ema_verifyx_upload_speed") or 0.0)


# --- the incident's two cycles through the real cached-image check -------------------------
# With the grace of lium-io#1461 a fresh node without the image passes as PENDING instead of failing
# the check; the rows that turn it on skip on a tree without it.

_IMAGE = DefaultDockerImage(
    docker_image="daturaai/torch", docker_image_tag="2.4.0", docker_image_size=12_000_000_000
)
_IMAGE_REF = "daturaai/torch:2.4.0"
_FIRST_SWEEP_RUNNING = json.dumps(
    {
        "schema_version": 1,
        "sweep_count": 0,
        "last_outcome": "pulling",
        "outcome_counts": {},
        "first_sweep_ok_at": None,
        "images": {_IMAGE_REF: {"last_outcome": "pulling"}},
    }
)

needs_the_grace = pytest.mark.skipif(
    not hasattr(CachedTemplateMessages, "PENDING"),
    reason="needs the fresh-node grace of lium-io#1461",
)


def _redis_service():
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis(server=FakeServer())
    service.lock = asyncio.Lock()
    return service


async def _cycle_with_the_image_check(context_factory, probes, rented_data, redis, *, cached):
    backend = Mock()
    backend.get_default_docker_image = AsyncMock(return_value=[_IMAGE])
    ssh = AsyncMock()
    ssh.run = AsyncMock(
        side_effect=[Mock(exit_status=0, stdout="[]")]
        if cached
        else [Mock(exit_status=1, stdout=""), Mock(exit_status=0, stdout=_FIRST_SWEEP_RUNNING)]
    )
    ctx = context_factory(
        services=build_services(verifyx=probes, backend=backend, redis=redis),
        config=build_context_config(verifyx_enabled=True),
        state=build_state(
            gpu_model="NVIDIA H200",
            specs={"gpu": {"count": 1, "driver": "580.95.05"}},
            rented_data=rented_data,
        ),
        ssh=ssh,
        ssh_pub_keys=[],
    )
    ok, events, last = await Pipeline(
        [VerifyXCheck(), CachedTemplateVerificationCheck()], _Sink()
    ).run(ctx)
    result = await ResultHandler(redis_service=None, dry_run=True).handle_result(
        context=last,
        miner_info=SimpleNamespace(miner_hotkey="miner-hotkey", job_batch_id="batch-1"),
        executor_info=last.executor,
        verified_job_info={},
        log_text="x",
        success=ok,
        validation_event=events[-1],
    )
    return ok, events, result.spec["network"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("grace", "hold", "cycle_2_passes"),
    [
        pytest.param(False, False, False, id="main"),
        # the grace alone: cycle 1 passes as PENDING and seeds 120
        pytest.param(True, False, False, id="grace", marks=needs_the_grace),
        # the hold alone: cycle 1 fails the image check and is held
        pytest.param(False, True, True, id="hold"),
        # both: cycle 1 passes as PENDING and is held
        pytest.param(True, True, True, id="grace-and-hold", marks=needs_the_grace),
    ],
)
async def test_the_incident_with_the_fresh_node_grace(
    monkeypatch, context_factory, grace, hold, cycle_2_passes
):
    monkeypatch.setattr(settings, "CACHED_TEMPLATE_CUTOFF", datetime.utcnow() - timedelta(days=1))
    # Settings is a pydantic model: setting a field this tree lacks raises even with raising=False.
    if hasattr(settings, "CACHED_TEMPLATE_FRESH_NODE_GRACE_ENABLED"):
        monkeypatch.setattr(settings, "CACHED_TEMPLATE_FRESH_NODE_GRACE_ENABLED", grace)
    monkeypatch.setattr(settings, "VERIFYX_EMA_HOLD_ENABLED", hold)
    monkeypatch.setattr(settings, "VERIFYX_COLD_SAMPLE_RETRY_ENABLED", True)
    redis = _redis_service()

    # Cycle 1: a fresh node, its executor's first pre-pull still running; VerifyX samples 120.
    ok, events, network = await _cycle_with_the_image_check(
        context_factory, _ProbeSequence(120.0), _never_measured(), redis, cached=False
    )
    assert ok is grace
    assert (
        events[-1].reason_code
        == (CachedTemplateMessages.PENDING if grace else CachedTemplateMessages.NOT_CACHED).reason
    )
    assert ("ema_verifyx_download_speed" in network) is not hold

    # Cycle 2: the image is on disk now; the first sample reads 53.4, a re-measure 238.
    probes = _ProbeSequence(53.4, 238.0)
    ok, events, network = await _cycle_with_the_image_check(
        context_factory, probes, _as_backend_would_store(network), redis, cached=True
    )
    verifyx_event = events[0]
    assert ok is cycle_2_passes
    if cycle_2_passes:
        assert probes.calls == 2
        assert verifyx_event.what_we_saw["cold_sample_retry"]["used"] == "retry"
        assert network["ema_verifyx_download_speed"] == pytest.approx(238.0)
    else:
        assert probes.calls == 1
        assert verifyx_event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
        assert verifyx_event.what_we_saw["ema_verifyx_download_speed"] == pytest.approx(86.7)


@needs_the_grace
@pytest.mark.asyncio
async def test_a_measured_node_held_as_pending_publishes_its_sample(monkeypatch, context_factory):
    monkeypatch.setattr(settings, "CACHED_TEMPLATE_CUTOFF", datetime.utcnow() - timedelta(days=1))
    monkeypatch.setattr(settings, "CACHED_TEMPLATE_FRESH_NODE_GRACE_ENABLED", True)

    ok, events, network = await _cycle_with_the_image_check(
        context_factory, _ProbeSequence(200.0), _known(300.0, 40.0), _redis_service(), cached=False
    )

    assert ok is True
    assert events[-1].reason_code == CachedTemplateMessages.PENDING.reason
    assert network["ema_verifyx_download_speed"] == pytest.approx(250.0)


# --- composition with an upstream step that already keeps the previous EMA -----------------
# With the scrape's own speed test removed (lium-io#1419), `network` holds only what VerifyX
# adds, and VerifyX itself republishes the stored EMA on a probe fallback or a malformed
# reading; a node VerifyX does not run on carries only its stored ema_verifyx_*. The hold
# restores from the same stored value, so whichever lands first it never compounds.


def _hold(context_factory, specs, rented_data):
    ctx = context_factory(state=build_state(specs=specs, rented_data=rented_data), ssh_pub_keys=[])
    return hold_verifyx_ema(ctx, specs)


def test_holding_an_ema_already_kept_upstream_changes_nothing(context_factory):
    specs = {"network": {"ema_verifyx_download_speed": 200.0, "ema_verifyx_upload_speed": 50.0}}

    once = _hold(context_factory, specs, _known(200.0, 50.0))
    twice = _hold(context_factory, once, _known(200.0, 50.0))

    assert once["network"] == twice["network"] == specs["network"]


def test_the_hold_touches_only_the_keys_this_cycle_wrote(context_factory):
    specs = {"network": {"ema_verifyx_download_speed": 60.0}}

    held = _hold(context_factory, specs, _known(200.0, 50.0))

    assert held["network"] == {"ema_verifyx_download_speed": 200.0}


@pytest.mark.asyncio
async def test_a_held_node_moves_again_on_its_next_passing_cycle(context_factory):
    # The hold is per cycle: a failed cycle keeps the stored EMA, the next passing cycle moves it.
    held = await _publish(
        context_factory,
        specs=_measured_specs(50.0, 125.0, 45.0),
        rented_data=_known(200.0, 50.0),
        event=_event(IMAGE_CHECK, failed=True),
        success=False,
    )
    assert held["ema_verifyx_download_speed"] == 200.0

    moved = await _publish(
        context_factory,
        specs=_measured_specs(100.0, 150.0, 45.0),
        rented_data=_as_backend_would_store(held),
        event=_event("pipeline.finalize", failed=False),
        success=True,
        image_cached=True,
    )
    assert moved["ema_verifyx_download_speed"] == 150.0


@pytest.mark.asyncio
async def test_a_passing_node_without_the_image_is_judged_on_its_samples(context_factory):
    # Stored EMA 300, the node now measures 20 Mbps and keeps passing the cached-image check without
    # the image. Each passing cycle publishes its sample, so the gate fails it on cycle 2 at 90.
    rented_data, results = _known(300.0, 40.0), []
    for _ in range(2):
        ok, last_event, network = await _cycle(
            context_factory,
            _ProbeSequence(20.0),
            rented_data,
            after=[_PassingUncachedCheck()],
        )
        results.append((ok, network["ema_verifyx_download_speed"], last_event.reason_code))
        rented_data = _as_backend_would_store(network)

    assert results[0][:2] == (True, pytest.approx(160.0))
    assert results[1] == (
        False,
        pytest.approx(90.0),
        Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason,
    )
