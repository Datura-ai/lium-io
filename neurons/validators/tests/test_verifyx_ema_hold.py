"""The VerifyX download EMA is not moved by a cycle the node did not pass for another reason.

A node failed VERIFYX_FAILED_NETWORK_SPEED_TOO_SLOW (EMA 86.7 < 100) one batch after a cycle that
failed the cached-image check had still run VerifyX and seeded the EMA (alpha 0.5), most likely
while the executor's mandatory multi-GB image pull shared the link; it passed the cycle after that.
Such a cycle now publishes the EMA the backend held before it, so a never-measured node stays
never-measured and its next cycle gets the cold-sample retry. A passing cycle without the
image cached (the fresh-node grace's PENDING, lium-io#1461) does not seed a never-measured node
either; a node with a stored EMA publishes its sample on every passing cycle.
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

from core.config import settings
from neurons.validators.src.services.task.checks.cached_template_verification import (
    CachedTemplateVerificationCheck,
)
from neurons.validators.src.services.task.checks.verifyx import VerifyXCheck, hold_verifyx_ema
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


@pytest.mark.asyncio
async def test_a_cycle_that_failed_another_check_does_not_seed_the_ema(context_factory):
    network = await _publish(
        context_factory,
        specs=_measured_specs(120.0, 120.0, 40.0),
        rented_data=_never_measured(),
        event=_event(IMAGE_CHECK, failed=True),
        success=False,
    )

    assert "ema_verifyx_download_speed" not in network
    assert "ema_verifyx_upload_speed" not in network
    # The raw samples are still published: only the smoothed value is held.
    assert network["verifyx_download_speed"] == 120.0
    assert network["download_speed"] == 900.0


@pytest.mark.asyncio
async def test_a_known_node_keeps_its_previous_ema_on_a_failed_cycle(context_factory):
    network = await _publish(
        context_factory,
        specs=_measured_specs(50.0, 125.0, 45.0),
        rented_data=_known(200.0, 50.0),
        event=_event(IMAGE_CHECK, failed=True),
        success=False,
    )

    assert network["ema_verifyx_download_speed"] == 200.0
    assert network["ema_verifyx_upload_speed"] == 50.0


@pytest.mark.asyncio
async def test_a_passing_cycle_without_the_image_publishes_its_sample(context_factory):
    # The cached-image check passed (advisory, or holding a fresh node as pending) but the image is
    # not on disk: the cycle passed, so its sample moves the EMA.
    network = await _publish(
        context_factory,
        specs=_measured_specs(20.0, 160.0, 40.0),
        rented_data=_known(300.0),
        event=_event("pipeline.finalize", failed=False),
        success=True,
        image_cached=False,
    )

    assert network["ema_verifyx_download_speed"] == 160.0
    assert network["ema_verifyx_upload_speed"] == 40.0


@pytest.mark.asyncio
async def test_a_passing_cycle_without_the_image_does_not_seed_a_never_measured_node(
    context_factory,
):
    network = await _publish(
        context_factory,
        specs=_measured_specs(120.0, 120.0, 40.0),
        rented_data=_never_measured(),
        event=_event("pipeline.finalize", failed=False),
        success=True,
        image_cached=False,
    )

    assert "ema_verifyx_download_speed" not in network
    assert "ema_verifyx_upload_speed" not in network
    assert network["verifyx_download_speed"] == 120.0


@pytest.mark.asyncio
async def test_a_never_measured_node_with_the_image_is_seeded_by_a_passing_cycle(context_factory):
    network = await _publish(
        context_factory,
        specs=_measured_specs(120.0, 120.0, 40.0),
        rented_data=_never_measured(),
        event=_event("pipeline.finalize", failed=False),
        success=True,
        image_cached=True,
    )

    assert network["ema_verifyx_download_speed"] == 120.0


@pytest.mark.asyncio
async def test_without_the_backends_answer_a_passing_uncached_cycle_publishes_its_sample(
    context_factory,
):
    # Every node looks never-measured without rented_data; a backend blip must not hold them all.
    network = await _publish(
        context_factory,
        specs=_measured_specs(120.0, 120.0, 40.0),
        rented_data=None,
        event=_event("pipeline.finalize", failed=False),
        success=True,
        image_cached=False,
    )

    assert network["ema_verifyx_download_speed"] == 120.0


@pytest.mark.asyncio
async def test_verifyx_failing_its_own_gate_still_moves_the_ema(context_factory):
    network = await _publish(
        context_factory,
        specs=_measured_specs(60.0, 80.0, 40.0),
        rented_data=_known(100.0),
        event=_event(VerifyXCheck.check_id, failed=True),
        success=False,
    )

    assert network["ema_verifyx_download_speed"] == 80.0


@pytest.mark.asyncio
async def test_a_passing_cycle_publishes_the_new_ema(context_factory):
    network = await _publish(
        context_factory,
        specs=_measured_specs(300.0, 250.0, 45.0),
        rented_data=_known(200.0),
        event=_event("pipeline.finalize", failed=False),
        success=True,
        image_cached=True,
    )

    assert network["ema_verifyx_download_speed"] == 250.0
    assert network["ema_verifyx_upload_speed"] == 45.0


@pytest.mark.asyncio
async def test_a_failure_before_verifyx_ran_adds_no_ema(context_factory):
    network = await _publish(
        context_factory,
        specs={"gpu": {"count": 1}, "network": {"download_speed": 900.0}},
        rented_data=_known(200.0),
        event=_event("executor.validate.port_count", failed=True),
        success=False,
    )

    assert network == {"download_speed": 900.0}


# --- the two cycles from the evidence, through the real pipeline and result handler ------------


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


class _ImageNotCachedCheck:
    """The fatal cached-image check failing after VerifyX passed, as in the evidence."""

    check_id = IMAGE_CHECK
    fatal = True

    async def run(self, ctx):
        return CheckResult(passed=False, event=_event(IMAGE_CHECK, failed=False))


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


async def _cycle(context_factory, probes, rented_data, *, image_check: bool, after=()):
    ctx = context_factory(
        services=build_services(verifyx=probes),
        config=build_context_config(verifyx_enabled=True),
        state=build_state(specs={"gpu": {"count": 1}}, rented_data=rented_data),
        ssh_pub_keys=[],
    )
    checks = [VerifyXCheck()] + ([_ImageNotCachedCheck()] if image_check else []) + list(after)
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


@pytest.mark.asyncio
async def test_the_next_cycle_of_a_node_seeded_by_a_failed_cycle_gets_the_cold_retry(
    monkeypatch, context_factory
):
    monkeypatch.setattr(settings, "VERIFYX_COLD_SAMPLE_RETRY_ENABLED", True)

    # Cycle 1: VerifyX passes on a sample taken while the image pull shares the link, then the
    # cached-image check fails the cycle.
    ok, last_event, network = await _cycle(
        context_factory, _ProbeSequence(120.0), _never_measured(), image_check=True
    )
    assert ok is False
    assert last_event.what_we_saw["steps_failed"] == IMAGE_CHECK
    assert "ema_verifyx_download_speed" not in network

    # Cycle 2: the node is still never-measured, so a cold first sample is re-measured and the
    # better sample seeds the EMA. Seeded from cycle 1 it would have read (53.4 + 120) / 2 = 86.7.
    probes = _ProbeSequence(53.4, 238.0)
    ok, last_event, network = await _cycle(
        context_factory, probes, _as_backend_would_store(network), image_check=False
    )

    assert probes.calls == 2
    assert ok is True
    assert last_event.reason_code == Msg.VERIFY_SUCCESS.reason
    assert last_event.what_we_saw["cold_sample_retry"]["used"] == "retry"
    assert network["ema_verifyx_download_speed"] == pytest.approx(238.0)


@pytest.mark.asyncio
async def test_without_the_hold_the_same_two_cycles_fail_on_the_seeded_ema(context_factory):
    # The regression this guards: cycle 2 judged against the EMA cycle 1 seeded.
    probes = _ProbeSequence(53.4)
    ok, last_event, _ = await _cycle(context_factory, probes, _known(120.0), image_check=False)

    assert probes.calls == 1
    assert ok is False
    assert last_event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    assert last_event.what_we_saw["ema_verifyx_download_speed"] == pytest.approx(86.7)


# --- the flag ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_the_flag_off_a_failed_cycle_seeds_the_ema_as_before(
    monkeypatch, context_factory, caplog
):
    monkeypatch.setattr(settings, "VERIFYX_EMA_HOLD_ENABLED", False)

    with caplog.at_level(logging.INFO):
        network = await _publish(
            context_factory,
            specs=_measured_specs(120.0, 120.0, 40.0),
            rented_data=_never_measured(),
            event=_event(IMAGE_CHECK, failed=True),
            success=False,
        )

    assert network["ema_verifyx_download_speed"] == 120.0
    assert network["ema_verifyx_upload_speed"] == 40.0
    assert any("would not have moved it" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_with_the_flag_off_a_passing_uncached_cycle_seeds_the_ema_as_before(
    monkeypatch, context_factory, caplog
):
    monkeypatch.setattr(settings, "VERIFYX_EMA_HOLD_ENABLED", False)

    with caplog.at_level(logging.INFO):
        network = await _publish(
            context_factory,
            specs=_measured_specs(120.0, 120.0, 40.0),
            rented_data=_never_measured(),
            event=_event("pipeline.finalize", failed=False),
            success=True,
            image_cached=False,
        )

    assert network["ema_verifyx_download_speed"] == 120.0
    assert any("would not have moved it" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_with_the_flag_off_the_two_cycles_play_out_as_before(monkeypatch, context_factory):
    monkeypatch.setattr(settings, "VERIFYX_EMA_HOLD_ENABLED", False)
    monkeypatch.setattr(settings, "VERIFYX_COLD_SAMPLE_RETRY_ENABLED", True)

    ok, _, network = await _cycle(
        context_factory, _ProbeSequence(120.0), _never_measured(), image_check=True
    )
    assert ok is False
    assert network["ema_verifyx_download_speed"] == pytest.approx(120.0)

    probes = _ProbeSequence(53.4)
    ok, last_event, _ = await _cycle(
        context_factory, probes, _as_backend_would_store(network), image_check=False
    )

    assert probes.calls == 1
    assert ok is False
    assert last_event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    assert last_event.what_we_saw["ema_verifyx_download_speed"] == pytest.approx(86.7)


@pytest.mark.asyncio
async def test_a_hold_that_changes_nothing_is_not_logged(monkeypatch, context_factory, caplog):
    stored = {"ema_verifyx_download_speed": 200.0, "ema_verifyx_upload_speed": 50.0}
    for enabled in (True, False):
        monkeypatch.setattr(settings, "VERIFYX_EMA_HOLD_ENABLED", enabled)
        with caplog.at_level(logging.INFO):
            await _publish(
                context_factory,
                specs={"gpu": {"count": 1}, "network": dict(stored)},
                rented_data=_known(200.0, 50.0),
                event=_event(IMAGE_CHECK, failed=True),
                success=False,
            )

    assert not any("VerifyX EMA hold" in r.getMessage() for r in caplog.records)
    assert not any("VerifyX EMA held" in r.getMessage() for r in caplog.records)


# --- the grace of lium-io#1461 and this hold, both on ---------------------------------------
# The grace holds a fresh node without the image as a passing PENDING, so with it on the incident's
# first cycle passes instead of failing the cached-image check. These run the real check and skip
# on a tree without the grace.

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


@needs_the_grace
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("grace", "hold", "cycle_2_passes"),
    [
        (False, False, False),  # main
        (True, False, False),  # the grace alone: cycle 1 passes as PENDING and seeds 120
        (False, True, True),  # the hold alone: cycle 1 fails the image check and is held
        (True, True, True),  # both: cycle 1 passes as PENDING and is held
    ],
)
async def test_the_incident_with_the_fresh_node_grace(
    monkeypatch, context_factory, grace, hold, cycle_2_passes
):
    monkeypatch.setattr(settings, "CACHED_TEMPLATE_CUTOFF", datetime.utcnow() - timedelta(days=1))
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
async def test_a_node_carrying_only_its_stored_ema_publishes_it_unchanged(context_factory):
    stored = {"ema_verifyx_download_speed": 200.0, "ema_verifyx_upload_speed": 50.0}
    network = await _publish(
        context_factory,
        specs={"gpu": {"count": 1}, "network": dict(stored)},
        rented_data=_known(200.0, 50.0),
        event=_event(IMAGE_CHECK, failed=True),
        success=False,
    )

    assert network == stored


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
            image_check=False,
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
