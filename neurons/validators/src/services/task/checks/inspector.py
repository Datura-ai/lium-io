from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from services.const import SECONDS_PER_BLOCK
from services.inspector_validation_service import InspectorValidationResponse

from core.config import settings

from ..messages import InspectorMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from ..models import ValidationEvent
from protocol.vc_protocol.compute_requests import RentedPod


class InspectorRentedCheck:
    check_id = "executor.validate.inspector_rented"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not ctx.config.inspector_enabled:
            event = render_message(
                Msg.DISABLED,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=True, event=event)

        rented_executor = (
            ctx.state.rented_data.executors.get(ctx.executor.uuid)
            if ctx.state.rented_data
            else None
        )
        rented_pods = rented_executor.pods if rented_executor else []
        if not rented_pods:
            event = render_message(
                Msg.SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"executor_uuid": ctx.executor.uuid},
            )
            return CheckResult(passed=True, event=event)

        extra = {
            **ctx.default_extra,
            "rented": True,
            "rented_pods": [{"name": p.container_name, "pod_id": p.pod_id} for p in rented_pods],
        }

        result = await ctx.services.inspector.validate_rented_executor(
            ctx.services.shell,
            ctx.ssh,
            ctx.executor,
            ctx.default_extra,
        )

        if result.error:
            diagnostics = result.diagnostics or {}
            event = render_message(
                result.message or Msg.VALIDATION_ERROR,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "error": result.error,
                    **diagnostics,
                },
                extra=extra,
            )
            inspector_event = _build_inspector_event(
                ctx, event, rented_pods, result, outcome="ERROR"
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={
                    "default_extra": extra,
                    "state": replace(ctx.state, inspector_event=inspector_event),
                },
            )

        report = dict(result.report or {})
        findings = _findings(report)
        warnings = _collector_start_warnings(report)
        if findings:
            what: dict[str, Any] = {
                "findings": findings,
                "summary": report.get("summary", {}),
                "canary_ok": report.get("canary_ok"),
            }
            if warnings:
                what["warnings"] = warnings
            event = render_message(
                Msg.MALICIOUS_FINDINGS,
                ctx=ctx,
                check_id=self.check_id,
                what=what,
                extra=extra,
            )
            inspector_event = _build_inspector_event(
                ctx, event, rented_pods, result, outcome="MALICIOUS", report=report
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={
                    "default_extra": extra,
                    "state": replace(ctx.state, inspector_event=inspector_event),
                },
            )

        clean_what: dict[str, Any] = {
            "summary": report.get("summary", {}),
            "canary_ok": report.get("canary_ok"),
        }
        if _canary_failed(report):
            outcome = "CANARY_FAILED"
            event = render_message(
                Msg.CANARY_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what=clean_what,
                extra=extra,
            )
        elif _collector_not_running(report):
            outcome = "COLLECTOR_NOT_RUNNING"
            event = render_message(
                Msg.COLLECTOR_NOT_RUNNING,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    **clean_what,
                    "health": report.get("health") or {},
                },
                extra=extra,
            )
        elif warnings:
            outcome = "COLLECTOR_RECENTLY_STARTED"
            event = render_message(
                Msg.COLLECTOR_RECENTLY_STARTED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    **clean_what,
                    "warnings": warnings,
                },
                extra=extra,
            )
        else:
            outcome = "CLEAN"
            event = render_message(
                Msg.CLEAN,
                ctx=ctx,
                check_id=self.check_id,
                what=clean_what,
                extra=extra,
            )
        inspector_event = _build_inspector_event(
            ctx, event, rented_pods, result, outcome=outcome, report=report
        )
        return CheckResult(
            passed=True,
            event=event,
            updates={
                "default_extra": extra,
                "state": replace(ctx.state, inspector_event=inspector_event),
            },
        )


def _build_inspector_event(
    ctx: Context,
    event: ValidationEvent,
    rented_pods: list[RentedPod],
    result: InspectorValidationResponse,
    *,
    outcome: str,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "executor_id": ctx.executor.uuid,
        "miner_hotkey": ctx.miner_hotkey,
        "job_batch_id": ctx.config.job_batch_id or "",
        "pod_ids": [p.pod_id for p in rented_pods],
        "outcome": outcome,
        "reason_code": event.reason_code,
        "pipeline_id": event.pipeline_id,
        "trace_id": event.trace_id,
        "when": event.when.isoformat(),
    }
    if result.error:
        payload["report"] = None
        payload["error"] = {**(result.diagnostics or {}), "error": result.error}
        payload["context"] = {}
    else:
        payload["report"] = report
        payload["error"] = None
        payload["context"] = result.diagnostics or {}
    return payload

def _canary_failed(report: dict[str, Any]) -> bool:
    return report.get("canary_ok") is False


def _collector_not_running(report: dict[str, Any]) -> bool:
    health = report.get("health") or {}
    return not health.get("collector_started_unix")


def _collector_start_warnings(report: dict[str, Any]) -> list[dict[str, Any]]:
    health = report.get("health") or {}
    collector_started = health.get("collector_started_unix")
    if not collector_started:
        return []

    age = int(time.time()) - int(collector_started)
    if age >= settings.BLOCKS_FOR_JOB * SECONDS_PER_BLOCK:
        return []

    return [
        {
            "reason": "INSPECTOR_COLLECTOR_RECENTLY_STARTED",
            "collector_age_seconds": age,
            "collector_started_unix": collector_started,
            "impact": "Inspector collection window may be incomplete",
        }
    ]


def _findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings = report.get("findings") or []
    if not isinstance(findings, list):
        return [{"value": findings}]
    return [item if isinstance(item, dict) else {"value": item} for item in findings]
