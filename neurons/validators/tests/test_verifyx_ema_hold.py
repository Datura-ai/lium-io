import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.config import Settings, settings
from neurons.validators.src.services.task.checks.cached_template_verification import (
    CachedTemplateVerificationCheck,
)
from neurons.validators.src.services.task.checks.verifyx import _EMA_KEYS, VerifyXCheck
from neurons.validators.src.services.task.messages import CachedTemplateMessages
from neurons.validators.src.services.task.messages import VerifyXMessages as Msg
from neurons.validators.src.services.task.pipeline import Pipeline
from neurons.validators.src.services.task.result_handler import ResultHandler
from protocol.vc_protocol.validator_requests import ValidationEvent

from tests.helpers import build_context_config, build_services, build_state
from tests.test_cached_template_verification import (
    _IMAGE,
    _backend,
    _fake_redis_service,
    _prefetch_doc,
    _result,
    _ssh_seq,
)
from tests.test_verifyx_cold_sample_retry import _known, _never_measured, _probe, _ProbeSequence

IMAGE_CHECK = "executor.validate.cached_template"


@pytest.fixture(autouse=True)
def _hold_enabled(monkeypatch):
    # The hold ships off; every test below runs it on unless it says otherwise.
    monkeypatch.setattr(settings, "VERIFYX_EMA_HOLD_ENABLED", True)


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
    network = {"download_speed": 900.0, "verifyx_download_speed": download, "verifyx_upload_speed": 40.0}
    network |= dict(zip(_EMA_KEYS, (ema_download, ema_upload)))
    return {"gpu": {"count": 1}, "network": network}


async def _published_network(ctx, *, success, event) -> dict:
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
_HELD = "VerifyX EMA held"
_WOULD_HOLD = "would not have moved it"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("specs", "rented_data", "failed_check", "image_cached", "hold", "ema", "log"),
    [
        # the raw samples are still published in every row: only the smoothed value is held
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), _never_measured(), IMAGE_CHECK, None, True,
            (_ABSENT, _ABSENT), _HELD, id="failed-another-check-does-not-seed",
        ),
        pytest.param(
            _measured_specs(50.0, 125.0, 45.0), _known(200.0), IMAGE_CHECK, None, True,
            (200.0, 50.0), _HELD, id="failed-cycle-keeps-the-previous-ema",
        ),
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), None, IMAGE_CHECK, None, True,
            (120.0, 40.0), None, id="failed-cycle-without-the-backends-answer-holds-nothing",
        ),
        # the cached-image check passed (advisory, or a fresh node held as pending) without the image
        pytest.param(
            _measured_specs(20.0, 160.0, 40.0), _known(300.0), _PASSED, False, True,
            (160.0, 40.0), None, id="passing-uncached-cycle-of-a-known-node-publishes",
        ),
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), _never_measured(), _PASSED, False, True,
            (_ABSENT, _ABSENT), _HELD, id="passing-uncached-cycle-does-not-seed",
        ),
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), _never_measured(), _PASSED, True, True,
            (120.0, 40.0), None, id="passing-cached-cycle-seeds",
        ),
        # every node looks never-measured without rented_data; a backend blip must not hold them all
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), None, _PASSED, False, True,
            (120.0, 40.0), None, id="passing-uncached-cycle-without-the-backends-answer-publishes",
        ),
        pytest.param(
            _measured_specs(60.0, 80.0, 40.0), _known(100.0), VerifyXCheck.check_id, None, True,
            (80.0, 40.0), None, id="verifyx-failing-its-own-gate-moves-the-ema",
        ),
        pytest.param(
            _measured_specs(300.0, 250.0, 45.0), _known(200.0), _PASSED, True, True,
            (250.0, 45.0), None, id="passing-cycle-publishes-the-new-ema",
        ),
        pytest.param(
            {"gpu": {"count": 1}, "network": {"download_speed": 900.0}}, _known(200.0),
            "executor.validate.port_count", None, True, (_ABSENT, _ABSENT), None,
            id="failure-before-verifyx-ran-adds-no-ema",
        ),
        pytest.param(
            {"gpu": {"count": 1}, "network": {"ema_verifyx_download_speed": 60.0}}, _known(200.0),
            IMAGE_CHECK, None, True, (200.0, _ABSENT), _HELD,
            id="hold-touches-only-the-keys-this-cycle-wrote",
        ),
        pytest.param(
            {"gpu": {"count": 1}, "network": dict(_STORED)}, _known(200.0), IMAGE_CHECK, None,
            True, (200.0, 50.0), None, id="stored-ema-only-is-published-unchanged-and-not-logged",
        ),
        pytest.param(
            {"gpu": {"count": 1}, "network": dict(_STORED)}, _known(200.0), IMAGE_CHECK, None,
            False, (200.0, 50.0), None, id="flag-off-stored-ema-only-is-not-logged",
        ),
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), _never_measured(), IMAGE_CHECK, None, False,
            (120.0, 40.0), _WOULD_HOLD, id="flag-off-failed-cycle-seeds-as-before",
        ),
        pytest.param(
            _measured_specs(120.0, 120.0, 40.0), _never_measured(), _PASSED, False, False,
            (120.0, 40.0), _WOULD_HOLD, id="flag-off-passing-uncached-cycle-seeds-as-before",
        ),
    ],
)
async def test_what_a_cycle_publishes(
    monkeypatch, context_factory, caplog, specs, rented_data, failed_check, image_cached, hold, ema, log
):
    if not hold:
        assert Settings.model_fields["VERIFYX_EMA_HOLD_ENABLED"].default is False
    monkeypatch.setattr(settings, "VERIFYX_EMA_HOLD_ENABLED", hold)
    state = build_state(specs=specs, rented_data=rented_data, recommended_image_cached=image_cached)

    with caplog.at_level(logging.INFO):
        network = await _published_network(
            context_factory(state=state, ssh_pub_keys=[]),
            success=failed_check is None,
            event=_event(failed_check or "pipeline.finalize", failed=failed_check is not None),
        )

    raw = {key: value for key, value in specs["network"].items() if key not in _EMA_KEYS}
    expected_ema = {key: value for key, value in zip(_EMA_KEYS, ema) if value is not _ABSENT}
    assert network == {**raw, **expected_ema}
    hold_logs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("VerifyX EMA h")]
    assert len(hold_logs) == (log is not None)
    assert all(log in message for message in hold_logs)


# --- the incident's two cycles through the real VerifyX and cached-image checks -------------


def _as_backend_would_store(network: dict):
    ema = network.get("ema_verifyx_download_speed")
    return _never_measured() if ema is None else _known(ema)


async def _cycle_with_the_image_check(context_factory, probes, rented_data, redis, *, cached):
    ssh = (
        _ssh_seq(_result(stdout="[]"))
        if cached
        else _ssh_seq(_result(exit_status=1), _result(stdout=_prefetch_doc(pull_error=None, last_outcome="pulling")))
    )
    ctx = context_factory(
        services=build_services(verifyx=probes, backend=_backend(images=[_IMAGE]), redis=redis),
        config=build_context_config(verifyx_enabled=True),
        state=build_state(
            gpu_model="NVIDIA H200",
            specs={"gpu": {"count": 1, "driver": "580.95.05"}},
            rented_data=rented_data,
        ),
        ssh=ssh,
        ssh_pub_keys=[],
    )
    pipeline = Pipeline([VerifyXCheck(), CachedTemplateVerificationCheck()], AsyncMock())
    ok, events, last = await pipeline.run(ctx)
    return ok, events, await _published_network(last, success=ok, event=events[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("grace", "hold", "cold_retry", "cycle_2_passes", "cycle_2_ema"),
    [
        pytest.param(False, False, True, False, 86.7, id="main"),
        # the grace alone: cycle 1 passes as PENDING and seeds 120
        pytest.param(True, False, True, False, 86.7, id="grace"),
        # the hold alone: cycle 1 fails the image check and is held
        pytest.param(False, True, True, True, 238.0, id="hold"),
        # both: cycle 1 passes as PENDING and is held
        pytest.param(True, True, True, True, 238.0, id="grace-and-hold"),
        # the hold without the re-measure: cycle 2 is judged on its one cold sample
        pytest.param(False, True, False, False, 53.4, id="hold-without-cold-retry"),
    ],
)
async def test_the_incident_with_the_fresh_node_grace(
    monkeypatch, context_factory, grace, hold, cold_retry, cycle_2_passes, cycle_2_ema
):
    monkeypatch.setattr(settings, "CACHED_TEMPLATE_CUTOFF", datetime.utcnow() - timedelta(days=1))
    monkeypatch.setattr(settings, "CACHED_TEMPLATE_FRESH_NODE_GRACE_ENABLED", grace)
    monkeypatch.setattr(settings, "VERIFYX_EMA_HOLD_ENABLED", hold)
    monkeypatch.setattr(settings, "VERIFYX_COLD_SAMPLE_RETRY_ENABLED", cold_retry)
    redis = _fake_redis_service()

    # Cycle 1: a fresh node, its executor's first pre-pull still running; VerifyX samples 120.
    ok, events, network = await _cycle_with_the_image_check(
        context_factory, _ProbeSequence(_probe(120.0)), _never_measured(), redis, cached=False
    )
    assert ok is grace
    assert (
        events[-1].reason_code
        == (CachedTemplateMessages.PENDING if grace else CachedTemplateMessages.NOT_CACHED).reason
    )
    assert ("ema_verifyx_download_speed" in network) is not hold

    # Cycle 2: the image is on disk now; the first sample reads 53.4, a re-measure 238.
    probes = _ProbeSequence(_probe(53.4), _probe(238.0))
    ok, events, network = await _cycle_with_the_image_check(
        context_factory, probes, _as_backend_would_store(network), redis, cached=True
    )
    verifyx_event = events[0]
    assert ok is cycle_2_passes
    if cycle_2_passes:
        assert probes.calls == 2
        assert verifyx_event.what_we_saw["cold_sample_retry"]["used"] == "retry"
        assert network["ema_verifyx_download_speed"] == pytest.approx(cycle_2_ema)
    else:
        assert probes.calls == 1
        assert verifyx_event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
        assert verifyx_event.what_we_saw["ema_verifyx_download_speed"] == pytest.approx(cycle_2_ema)
