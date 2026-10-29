import logging
import time

import pytest
from helpers import FakeRedis, build_context_config, build_services, build_state
from neurons.validators.src.services.inspector_validation_service import (
    InspectorValidationResponse,
)
from neurons.validators.src.services.task.checks import inspector
from neurons.validators.src.services.task.checks.inspector import InspectorRentedCheck
from test_inspector_check import DummyInspectorService, _healthy_report, _rented_data

ERROR = InspectorValidationResponse(error="inspector did not answer")
CLEAN = InspectorValidationResponse(report=_healthy_report())
JUST_STARTED = InspectorValidationResponse(
    report=_healthy_report(health={"collector_started_unix": int(time.time())})
)


async def _run_cycles(context_factory, redis, *responses):
    results = []
    for response in responses:
        ctx = context_factory(
            config=build_context_config(),
            services=build_services(inspector=DummyInspectorService(response), redis=redis),
            state=build_state(rented_data=_rented_data("executor-123")),
        )
        results.append(await InspectorRentedCheck().run(ctx))
    return results


def _streak_logs(caplog) -> int:
    return sum("INSPECTOR_BLIND_STREAK" == record.getMessage().split(" ")[0] for record in caplog.records)


@pytest.mark.asyncio
async def test_a_blind_streak_is_logged_from_the_limit_and_still_passes(
    context_factory, caplog, monkeypatch
):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_LIMIT", 3)

    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        results = await _run_cycles(context_factory, FakeRedis(), ERROR, ERROR, ERROR, ERROR)

    assert [result.passed for result in results] == [True, True, True, True]
    assert _streak_logs(caplog) == 2


@pytest.mark.asyncio
async def test_a_seeing_cycle_ends_the_streak(context_factory, caplog, monkeypatch):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_LIMIT", 3)
    redis = FakeRedis()

    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        await _run_cycles(context_factory, redis, ERROR, ERROR, CLEAN, ERROR, ERROR)

    assert _streak_logs(caplog) == 0
    assert list(redis.store.values()) == ["2"]


@pytest.mark.asyncio
async def test_a_failed_canary_counts_as_a_blind_cycle(context_factory, caplog, monkeypatch):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_LIMIT", 2)
    no_canary = InspectorValidationResponse(report=_healthy_report(canary_ok=False))

    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        await _run_cycles(context_factory, FakeRedis(), ERROR, no_canary)

    assert _streak_logs(caplog) == 1


@pytest.mark.asyncio
async def test_a_blind_streak_fails_the_check_under_enforcement(context_factory, monkeypatch):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_LIMIT", 2)
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_ENFORCEMENT_ENABLED", True)

    results = await _run_cycles(context_factory, FakeRedis(), ERROR, ERROR, CLEAN)

    assert [result.passed for result in results] == [True, False, True]
    # the check is non-fatal: only this flag makes calculate_scores zero the score
    assert results[1].updates["inspector_blind_streak_passed"] is False
    assert "inspector_blind_streak_passed" not in results[2].updates


@pytest.mark.asyncio
async def test_a_blind_streak_in_shadow_sets_no_score_flag(context_factory, monkeypatch):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_LIMIT", 1)

    results = await _run_cycles(context_factory, FakeRedis(), ERROR, ERROR)

    assert all("inspector_blind_streak_passed" not in result.updates for result in results)


@pytest.mark.asyncio
async def test_a_just_started_collector_counts_as_a_blind_cycle(
    context_factory, caplog, monkeypatch
):
    # the collector is restarted before every read, so a host that kills it reads as just started
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_LIMIT", 3)
    redis = FakeRedis()

    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        await _run_cycles(context_factory, redis, ERROR, ERROR, JUST_STARTED, ERROR)

    assert _streak_logs(caplog) == 2
    assert list(redis.store.values()) == ["4"]


@pytest.mark.asyncio
async def test_a_host_killing_the_collector_every_cycle_reaches_the_limit(
    context_factory, monkeypatch
):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_LIMIT", 3)
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_ENFORCEMENT_ENABLED", True)

    results = await _run_cycles(
        context_factory, FakeRedis(), JUST_STARTED, JUST_STARTED, JUST_STARTED
    )

    assert [result.passed for result in results] == [True, True, False]
    assert results[2].updates["inspector_blind_streak_passed"] is False


@pytest.mark.asyncio
async def test_a_redis_outage_never_fails_the_check(context_factory, monkeypatch):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_LIMIT", 1)
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_ENFORCEMENT_ENABLED", True)

    results = await _run_cycles(context_factory, FakeRedis(failing=True), ERROR, ERROR)

    assert [result.passed for result in results] == [True, True]


def test_blind_streak_enforcement_is_off_by_default():
    fields = type(inspector.settings).model_fields
    assert fields["INSPECTOR_BLIND_STREAK_ENFORCEMENT_ENABLED"].default is False


@pytest.mark.asyncio
async def test_a_provider_finding_with_a_failed_canary_still_counts_as_blind(
    context_factory, caplog, monkeypatch
):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_LIMIT", 2)
    finding_no_canary = InspectorValidationResponse(
        report=_healthy_report(canary_ok=False, findings=[{"kind": "malicious-process"}])
    )

    with caplog.at_level(logging.WARNING, logger=inspector.logger.name):
        await _run_cycles(context_factory, FakeRedis(), ERROR, finding_no_canary)

    assert _streak_logs(caplog) == 1


@pytest.mark.asyncio
async def test_a_provider_finding_from_a_seeing_sensor_ends_the_streak(
    context_factory, monkeypatch
):
    monkeypatch.setattr(inspector.settings, "INSPECTOR_BLIND_STREAK_LIMIT", 3)
    redis = FakeRedis()
    finding = InspectorValidationResponse(
        report=_healthy_report(findings=[{"kind": "malicious-process"}])
    )

    await _run_cycles(context_factory, redis, ERROR, ERROR, finding)

    assert list(redis.store.values()) == []
