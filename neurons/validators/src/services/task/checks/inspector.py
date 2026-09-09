from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from services.const import SECONDS_PER_BLOCK
from services.inspector_validation_service import InspectorValidationResponse
from services.redis_service import STREAMING_LOG_CHANNEL

from core.config import settings
from core.utils import _m, get_extra_info

from ..inspector_verdict import InspectorVerdict, build_verdict, renter_access_event
from ..messages import InspectorMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from ..models import ValidationEvent
from protocol.vc_protocol.compute_requests import RentedPod

logger = logging.getLogger(__name__)


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

        sensor_attested = _sensor_attested(ctx)
        result = await ctx.services.inspector.validate_rented_executor(
            ctx.services.shell,
            ctx.ssh,
            ctx.executor,
            ctx.default_extra,
            sensor_attested=sensor_attested,
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
        if findings is None:
            # a report whose findings are not a list of objects is a broken sensor, not a
            # provider caught in the act — it must not zero a score or request a quarantine
            event = render_message(
                Msg.VALIDATION_ERROR,
                ctx=ctx,
                check_id=self.check_id,
                what={"error": "inspector report findings are not a list of objects", "findings": report.get("findings")},
                extra=extra,
            )
            inspector_event = _build_inspector_event(
                ctx, event, rented_pods, result, outcome="ERROR", report=report
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={"default_extra": extra, "state": replace(ctx.state, inspector_event=inspector_event)},
            )
        warnings = _collector_start_warnings(report)
        enforce = settings.INSPECTOR_ENFORCE_ENABLED
        verdict = build_verdict(
            report,
            findings,
            rented_pod_ids=[p.pod_id for p in rented_pods],
            sensor_attested=sensor_attested,
            enforce=enforce,
        )
        if verdict.provider_origin:
            what: dict[str, Any] = {
                "findings": verdict.provider_findings,
                "platform_findings": len(verdict.platform_findings),
                "verdict": verdict.as_payload(),
                "summary": report.get("summary", {}),
                "canary_ok": report.get("canary_ok"),
            }
            if warnings:
                what["warnings"] = warnings
            event = render_message(
                Msg.MALICIOUS_FINDINGS,
                ctx=ctx,
                check_id=self.check_id,
                severity="error" if enforce else None,
                impact=(
                    "Provider-origin access to a rented pod: score zeroed, quarantine requested"
                    if enforce
                    else "Provider-origin access to a rented pod recorded; score unchanged (INSPECTOR_ENFORCE_ENABLED off)"
                ),
                what=what,
                extra=extra,
            )
            inspector_event = _build_inspector_event(
                ctx, event, rented_pods, result, outcome="MALICIOUS", report=report, verdict=verdict
            )
            await _tell_renters(ctx, verdict, when=event.when.isoformat())
            updates: dict[str, Any] = {
                "default_extra": extra,
                "state": replace(ctx.state, inspector_event=inspector_event),
            }
            if enforce:
                # Non-fatal check: passed=False alone changes nothing downstream, the score gate
                # in calculate_scores reads this flag (same mechanics as cpu_truth_passed).
                updates["inspector_passed"] = False
            return CheckResult(passed=not enforce, event=event, updates=updates)

        clean_what: dict[str, Any] = {
            "summary": report.get("summary", {}),
            "canary_ok": report.get("canary_ok"),
            "verdict": verdict.as_payload(),
        }
        if verdict.platform_findings:
            # every finding was one of our own execs seen from a host whose Tetragon lost the
            # sshd ancestry — recorded, not malicious (the 8 Sep false positives)
            clean_what["platform_findings"] = verdict.platform_findings
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
        elif verdict.platform_findings:
            outcome = "CLEAN"
            event = render_message(
                Msg.PLATFORM_ORIGIN_ONLY,
                ctx=ctx,
                check_id=self.check_id,
                what=clean_what,
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
            ctx, event, rented_pods, result, outcome=outcome, report=report, verdict=verdict
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
    verdict: InspectorVerdict | None = None,
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
        # `context` is the one free-form field InspectorEventRequest carries to the backend;
        # the verdict rides there (evidence hashes, sensor state, requested action + ban source)
        # so the consumer can act on it without re-deriving the classification.
        payload["context"] = {
            **(result.diagnostics or {}),
            **({"verdict": verdict.as_payload()} if verdict is not None else {}),
        }
    return payload


def _sensor_attested(ctx: Context) -> bool:
    # The sensor (libinspector.so + Tetragon) ships inside the executor image. On a dstack CVM
    # the image is part of the measured stack the validator checked against TDX_WHITELIST, so
    # the report comes from a measured binary; elsewhere its checksum was read through the
    # provider's own shell and proves nothing — say so in the verdict.
    return bool(ctx.tdx_attestation_passed)


async def _tell_renters(ctx: Context, verdict: InspectorVerdict, *, when: str) -> None:
    redis = ctx.services.redis
    if redis is None or not verdict.affected_pod_ids:
        return
    for pod_id in verdict.affected_pod_ids:
        try:
            await redis.publish(
                STREAMING_LOG_CHANNEL,
                {
                    "logs": [renter_access_event(verdict, pod_id=pod_id, when=when)],
                    "miner_hotkey": ctx.miner_hotkey,
                    "executor_uuid": ctx.executor.uuid,
                    "pod_id": pod_id,
                },
            )
        except Exception as exc:  # the verdict itself is already in the inspector event
            logger.warning(
                _m(
                    "Failed to publish provider_access_detected to the pod stream",
                    extra=get_extra_info({**ctx.default_extra, "pod_id": pod_id, "error": str(exc)}),
                )
            )


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


def _findings(report: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The report's findings as a list of objects, or None when the report is malformed."""
    findings = report.get("findings") or []
    if not isinstance(findings, list) or not all(isinstance(item, dict) for item in findings):
        return None
    return findings
