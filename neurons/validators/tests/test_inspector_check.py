from __future__ import annotations

from typing import Any

import pytest
from helpers import build_context_config, build_services, build_state
from neurons.validators.src.services.const import SECONDS_PER_BLOCK
from neurons.validators.src.services.inspector_validation_service import (
    InspectorValidationResponse,
)
from neurons.validators.src.services.task.checks.inspector import InspectorRentedCheck
from neurons.validators.src.services.task.messages import InspectorMessages as Msg
from protocol.vc_protocol.compute_requests import (
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)

from core.config import settings


class DummyInspectorService:
    def __init__(self, response: InspectorValidationResponse) -> None:
        self.response = response
        self.called = False

    async def validate_rented_executor(self, shell, ssh, executor, default_extra):
        self.called = True
        return self.response


def _rented_data(executor_uuid: str) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={
            executor_uuid: RentedExecutor(
                miner_hotkey="miner",
                executor_ip_address="127.0.0.1",
                executor_ip_port="22",
                pods=[
                    RentedPod(
                        pod_id="pod-1",
                        container_name="pod_pod-1",
                    )
                ],
            )
        },
        banned_guids=[],
    )


def _inspector_context(context_factory, **overrides):
    return context_factory(
        config=build_context_config(inspector_enabled=True),
        **overrides,
    )


def _healthy_report(**overrides: Any) -> dict:
    report = {
        "canary_ok": True,
        "health": {"collector_started_unix": 1},
        "findings": [],
        "summary": {"malicious": 0},
    }
    report.update(overrides)
    return report


@pytest.mark.asyncio
async def test_inspector_check_skips_when_disabled(context_factory):
    service = DummyInspectorService(InspectorValidationResponse(report={}))
    ctx = context_factory(
        config=build_context_config(inspector_enabled=False),
        services=build_services(inspector=service),
        state=build_state(rented_data=_rented_data("executor-123")),
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.DISABLED.reason
    assert service.called is False


@pytest.mark.asyncio
async def test_inspector_check_skips_without_customer_rented_pods(context_factory):
    service = DummyInspectorService(InspectorValidationResponse(report={}))
    rented_data = RentedExecutorsResponse(
        executors={},
        filler_containers_by_executor={"executor-123": "filler_1"},
        banned_guids=[],
    )
    ctx = _inspector_context(
        context_factory,
        services=build_services(inspector=service),
        state=build_state(rented_data=rented_data),
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.passed is True
    assert result.halt is False
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert service.called is False


@pytest.mark.asyncio
async def test_inspector_check_logs_findings_without_score_change(context_factory):
    service = DummyInspectorService(
        InspectorValidationResponse(
            report={
                **_healthy_report(),
                "findings": [{"kind": "malicious-process"}],
                "summary": {"malicious": 1},
            }
        )
    )
    ctx = _inspector_context(
        context_factory,
        services=build_services(inspector=service),
        state=build_state(rented_data=_rented_data("executor-123")),
        score=0.42,
        job_score=0.24,
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.passed is True
    assert result.halt is False
    assert result.event.reason_code == Msg.MALICIOUS_FINDINGS.reason
    assert result.event.severity == "warning"
    assert result.event.what_we_saw["findings"] == [{"kind": "malicious-process"}]
    assert "score" not in result.updates
    assert "job_score" not in result.updates


@pytest.mark.asyncio
async def test_inspector_check_malicious_payload_persists_report(context_factory):
    report = {
        **_healthy_report(),
        "health": {
            "ok": True,
            "collector_started_unix": 1,
            "events_dropped": 0,
            "bytes_buffered": 0,
        },
        "findings": [{"kind": "malicious-process"}],
        "summary": {"findings": 1, "total_events_seen": 99},
    }
    service = DummyInspectorService(InspectorValidationResponse(report=report))
    ctx = _inspector_context(
        context_factory,
        services=build_services(inspector=service),
        state=build_state(rented_data=_rented_data("executor-123")),
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.event.reason_code == Msg.MALICIOUS_FINDINGS.reason
    assert "report" not in result.event.what_we_saw
    assert result.event.what_we_saw["summary"] == {"findings": 1, "total_events_seen": 99}
    assert result.event.what_we_saw["findings"] == [{"kind": "malicious-process"}]
    event = result.updates["state"].inspector_event
    assert event["outcome"] == "MALICIOUS"
    assert event["pod_ids"] == ["pod-1"]
    assert event["report"] == report
    assert event["error"] is None


@pytest.mark.asyncio
async def test_inspector_check_attaches_inspector_event_on_clean(context_factory):
    service = DummyInspectorService(InspectorValidationResponse(report=_healthy_report()))
    ctx = _inspector_context(
        context_factory,
        services=build_services(inspector=service),
        state=build_state(rented_data=_rented_data("executor-123")),
    )

    result = await InspectorRentedCheck().run(ctx)

    event = result.updates["state"].inspector_event
    assert event["outcome"] == "CLEAN"
    assert event["reason_code"] == Msg.CLEAN.reason
    assert event["job_batch_id"] == "batch-1"
    assert event["executor_id"] == "executor-123"
    assert event["report"]["canary_ok"] is True
    assert event["error"] is None


@pytest.mark.asyncio
async def test_inspector_check_attaches_inspector_event_on_error(context_factory):
    service = DummyInspectorService(
        InspectorValidationResponse(
            error="transport failed",
            diagnostics={"reason": Msg.FAILED_SSH_TRANSPORT.reason},
            message=Msg.FAILED_SSH_TRANSPORT,
        )
    )
    ctx = _inspector_context(
        context_factory,
        services=build_services(inspector=service),
        state=build_state(rented_data=_rented_data("executor-123")),
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.passed is True
    assert result.halt is False
    assert result.event.reason_code == Msg.FAILED_SSH_TRANSPORT.reason
    assert result.event.what_we_saw["error"] == "transport failed"
    event = result.updates["state"].inspector_event
    assert event["outcome"] == "ERROR"
    assert event["report"] is None
    assert event["error"]["error"] == "transport failed"
    assert event["context"] == {}


@pytest.mark.asyncio
async def test_inspector_check_skipped_has_no_inspector_event(context_factory):
    service = DummyInspectorService(InspectorValidationResponse(report={}))
    ctx = _inspector_context(
        context_factory,
        services=build_services(inspector=service),
        state=build_state(rented_data=RentedExecutorsResponse(executors={}, banned_guids=[])),
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.event.reason_code == Msg.SKIPPED.reason
    assert "state" not in result.updates


@pytest.mark.asyncio
async def test_inspector_check_warns_when_collector_started_recently(context_factory, monkeypatch):
    now = 1_700_000_000
    monkeypatch.setattr(
        "neurons.validators.src.services.task.checks.inspector.time.time",
        lambda: now,
    )
    service = DummyInspectorService(
        InspectorValidationResponse(
            report={
                "canary_ok": True,
                "health": {"collector_started_unix": now - 60},
                "findings": [],
                "summary": {"malicious": 0},
            }
        )
    )
    ctx = _inspector_context(
        context_factory,
        services=build_services(inspector=service),
        state=build_state(rented_data=_rented_data("executor-123")),
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.event.reason_code == Msg.COLLECTOR_RECENTLY_STARTED.reason
    assert result.event.severity == "warning"
    warnings = result.event.what_we_saw["warnings"]
    assert len(warnings) == 1
    assert warnings[0]["reason"] == "INSPECTOR_COLLECTOR_RECENTLY_STARTED"
    assert warnings[0]["collector_age_seconds"] == 60
    event = result.updates["state"].inspector_event
    assert event["outcome"] == "COLLECTOR_RECENTLY_STARTED"


@pytest.mark.asyncio
async def test_inspector_check_no_warning_when_collector_old_enough(context_factory, monkeypatch):
    now = 1_700_000_000
    threshold = settings.BLOCKS_FOR_JOB * SECONDS_PER_BLOCK
    monkeypatch.setattr(
        "neurons.validators.src.services.task.checks.inspector.time.time",
        lambda: now,
    )
    service = DummyInspectorService(
        InspectorValidationResponse(
            report={
                "canary_ok": True,
                "health": {"collector_started_unix": now - threshold},
                "findings": [],
                "summary": {"malicious": 0},
            }
        )
    )
    ctx = _inspector_context(
        context_factory,
        services=build_services(inspector=service),
        state=build_state(rented_data=_rented_data("executor-123")),
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.event.reason_code == Msg.CLEAN.reason
    assert "warnings" not in result.event.what_we_saw


@pytest.mark.asyncio
async def test_inspector_check_canary_failure_is_not_clean(context_factory):
    service = DummyInspectorService(
        InspectorValidationResponse(report=_healthy_report(canary_ok=False))
    )
    ctx = _inspector_context(
        context_factory,
        services=build_services(inspector=service),
        state=build_state(rented_data=_rented_data("executor-123")),
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.event.reason_code == Msg.CANARY_FAILED.reason
    assert result.event.what_we_saw["canary_ok"] is False
    event = result.updates["state"].inspector_event
    assert event["outcome"] == "CANARY_FAILED"


@pytest.mark.asyncio
async def test_inspector_check_collector_not_running(context_factory):
    service = DummyInspectorService(
        InspectorValidationResponse(
            report={
                "canary_ok": True,
                "health": {},
                "findings": [],
                "summary": {"malicious": 0},
            }
        )
    )
    ctx = _inspector_context(
        context_factory,
        services=build_services(inspector=service),
        state=build_state(rented_data=_rented_data("executor-123")),
    )

    result = await InspectorRentedCheck().run(ctx)

    assert result.event.reason_code == Msg.COLLECTOR_NOT_RUNNING.reason
    event = result.updates["state"].inspector_event
    assert event["outcome"] == "COLLECTOR_NOT_RUNNING"
