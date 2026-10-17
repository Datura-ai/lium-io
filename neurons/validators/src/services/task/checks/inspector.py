from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from core.utils import _m, get_extra_info
from services.const import SECONDS_PER_BLOCK
from services.inspector_validation_service import InspectorValidationResponse

from core.config import settings

from ..inspector_verdict import InspectorVerdict, build_verdict, host_write_findings
from ..messages import InspectorMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from ..models import ValidationEvent
from protocol.vc_protocol.compute_requests import RentedPod

logger = logging.getLogger(__name__)

# Outcomes in which the sensor saw nothing of the rental: an error, a failed canary, a stopped
# collector, or one that only just started (the collector is restarted before every read, so a host
# that kills it always reads as just started).
BLIND_OUTCOMES = frozenset(
    {"ERROR", "CANARY_FAILED", "COLLECTOR_NOT_RUNNING", "COLLECTOR_RECENTLY_STARTED"}
)
BLIND_STREAK_KEY_PREFIX = "inspector:blind_streak"
BLIND_STREAK_TTL_SECONDS = 24 * 60 * 60


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
            passed = await _record_blind_streak_and_decide_pass(ctx, "ERROR")
            return CheckResult(
                passed=passed,
                event=event,
                updates={
                    "default_extra": extra,
                    "state": replace(ctx.state, inspector_event=inspector_event),
                    **_score_gate_updates_for_failed_streak(passed),
                },
            )

        report = dict(result.report or {})
        findings = _findings(report)
        if findings is None:
            # a report whose findings are not a list of objects is a broken sensor, not a
            # provider caught in the act — it must not be recorded as a MALICIOUS finding
            event = render_message(
                Msg.VALIDATION_ERROR,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "error": "inspector report findings are not a list of objects",
                    "findings_type": type(report.get("findings")).__name__,
                    "findings_preview": repr(report.get("findings"))[:200],
                },
                extra=extra,
            )
            inspector_event = _build_inspector_event(
                ctx, event, rented_pods, result, outcome="ERROR", report=report
            )
            passed = await _record_blind_streak_and_decide_pass(ctx, "ERROR")
            return CheckResult(
                passed=passed,
                event=event,
                updates={
                    "default_extra": extra,
                    "state": replace(ctx.state, inspector_event=inspector_event),
                    **_score_gate_updates_for_failed_streak(passed),
                },
            )
        warnings = _collector_start_warnings(report)
        verdict = build_verdict(
            report,
            findings,
            rented_pod_ids=[p.pod_id for p in rented_pods],
            sensor_attested=sensor_attested,
        )
        if verdict.provider_origin:
            # a finding seen while the canary failed or the collector was down or just started
            # does not show the sensor saw everything, so that cycle still counts as blind
            await _record_blind_streak_and_decide_pass(ctx, _sensor_outcome(report, warnings))
            return self._record_provider_origin(
                ctx,
                verdict=verdict,
                report=report,
                warnings=warnings,
                rented_pods=rented_pods,
                result=result,
                extra=extra,
            )

        clean_what: dict[str, Any] = {
            "summary": report.get("summary", {}),
            "canary_ok": report.get("canary_ok"),
            "verdict": verdict.as_payload().model_dump(),
        }
        if verdict.platform_findings:
            # every finding was one of our own execs seen from a host whose Tetragon lost the
            # sshd ancestry — recorded, not malicious (the 8 Sep false positives); the findings
            # themselves are in the inspector event's report
            clean_what["platform_findings"] = len(verdict.platform_findings)
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
        passed = await _record_blind_streak_and_decide_pass(ctx, outcome)
        return CheckResult(
            passed=passed,
            event=event,
            updates={
                "default_extra": extra,
                "state": replace(ctx.state, inspector_event=inspector_event),
                **_score_gate_updates_for_failed_streak(passed),
            },
        )

    def _record_provider_origin(
        self,
        ctx: Context,
        *,
        verdict: InspectorVerdict,
        report: dict[str, Any],
        warnings: list[dict[str, Any]],
        rented_pods: list[RentedPod],
        result: InspectorValidationResponse,
        extra: dict[str, Any],
    ) -> CheckResult:
        """A provider-origin verdict becomes the MALICIOUS event and the inspector event that
        carries its evidence. It is a record only: the check passes, nothing reaches the score
        and no renter or backend is asked to act — except a host write into a rented pod under
        INSPECTOR_HOST_WRITE_ENFORCEMENT_ENABLED (DAH-4548), which fails the check and the score."""
        writes = host_write_findings(verdict.provider_findings, [p.pod_id for p in rented_pods])
        enforced = bool(writes) and settings.INSPECTOR_HOST_WRITE_ENFORCEMENT_ENABLED
        if writes:
            logger.warning(
                _m(
                    "INSPECTOR_HOST_WRITE",
                    extra=get_extra_info(
                        {
                            **ctx.default_extra,
                            "enforced": enforced,
                            "host_writes": len(writes),
                            "commands": [str(f.get("command"))[:160] for f in writes[:5]],
                        }
                    ),
                )
            )
        if enforced:
            impact = "The host wrote into a rented pod (docker cp, archive or volume write); node failed"
        elif verdict.affected_pod_ids:
            impact = "Provider-origin access to a rented pod recorded; score unchanged"
        else:
            impact = "Provider-origin finding on a container that is not a rented pod recorded; score unchanged"
        what: dict[str, Any] = {
            "findings": verdict.provider_findings,
            "platform_findings": len(verdict.platform_findings),
            "verdict": verdict.as_payload().model_dump(),
            "summary": report.get("summary", {}),
            "canary_ok": report.get("canary_ok"),
        }
        if warnings:
            what["warnings"] = warnings
        event = render_message(
            Msg.MALICIOUS_FINDINGS,
            ctx=ctx,
            check_id=self.check_id,
            impact=impact,
            what=what,
            extra=extra,
        )
        inspector_event = _build_inspector_event(
            ctx, event, rented_pods, result, outcome="MALICIOUS", report=report, verdict=verdict
        )
        return CheckResult(
            passed=not enforced,
            event=event,
            updates={
                "default_extra": extra,
                "state": replace(ctx.state, inspector_event=inspector_event),
                **({"inspector_host_write_passed": False} if enforced else {}),
            },
        )


async def _record_blind_streak_and_decide_pass(ctx: Context, outcome: str) -> bool:
    """Counts the rented cycles in a row whose inspector outcome saw nothing of the rental.

    Each of those outcomes passes on its own (a sensor can break honestly), so a host that keeps
    the sensor dead for a whole rental is never looked at. The streak is logged from
    INSPECTOR_BLIND_STREAK_LIMIT on; it fails the check only under
    INSPECTOR_BLIND_STREAK_ENFORCEMENT_ENABLED. The check is non-fatal, so the caller also sets
    `inspector_blind_streak_passed` (`_score_gate_updates_for_failed_streak`), which
    calculate_scores zeroes on. A collector that only just started counts as blind. A Redis outage
    decides nothing."""
    key = f"{BLIND_STREAK_KEY_PREFIX}:{ctx.miner_hotkey}:{ctx.executor.uuid}"
    try:
        if outcome not in BLIND_OUTCOMES:
            await ctx.services.redis.delete(key)
            return True
        streak = int(await ctx.services.redis.get(key) or 0) + 1
        await ctx.services.redis.set(key, str(streak), ex=BLIND_STREAK_TTL_SECONDS)
    except Exception:
        logger.warning(
            _m("INSPECTOR_BLIND_STREAK_UNAVAILABLE", extra=get_extra_info(ctx.default_extra)),
            exc_info=True,
        )
        return True
    if streak < settings.INSPECTOR_BLIND_STREAK_LIMIT:
        return True
    enforced = settings.INSPECTOR_BLIND_STREAK_ENFORCEMENT_ENABLED
    logger.warning(
        _m(
            "INSPECTOR_BLIND_STREAK",
            extra=get_extra_info(
                {**ctx.default_extra, "streak": streak, "outcome": outcome, "enforced": enforced}
            ),
        )
    )
    return not enforced


def _score_gate_updates_for_failed_streak(passed: bool) -> dict[str, bool]:
    """The context flag calculate_scores zeroes the score on; set only when the streak failed."""
    return {} if passed else {"inspector_blind_streak_passed": False}


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
        # the verdict rides there (evidence hashes, sensor state, affected pods) so a reader
        # does not have to re-derive the classification.
        payload["context"] = {
            **(result.diagnostics or {}),
            **({"verdict": verdict.as_payload().model_dump()} if verdict is not None else {}),
        }
    return payload


def _sensor_attested(ctx: Context) -> bool:
    # The sensor (libinspector.so + Tetragon) ships inside the executor image. On a dstack CVM
    # the image is part of the measured stack — but only when ENABLE_ATTESTATION_WHITELIST is on
    # does the validator compare that stack against TDX_WHITELIST (attestation_service._verify_tdx);
    # with the flag off (prod today) a passed attestation says the quote is genuine, not that the
    # image is ours, so the report is not from a measured binary and the shell checksum stays.
    # Elsewhere the checksum was read through the provider's own shell and proves nothing — say
    # so in the verdict.
    return bool(ctx.tdx_attestation_passed) and settings.ENABLE_ATTESTATION_WHITELIST


def _sensor_outcome(report: dict[str, Any], warnings: list[dict[str, Any]]) -> str:
    """The blind-streak outcome of a cycle with a provider-origin finding."""
    if _canary_failed(report):
        return "CANARY_FAILED"
    if _collector_not_running(report):
        return "COLLECTOR_NOT_RUNNING"
    if warnings:
        return "COLLECTOR_RECENTLY_STARTED"
    return "MALICIOUS"


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
    findings = report.get("findings")
    if findings is None:
        findings = []
    if not isinstance(findings, list) or not all(isinstance(item, dict) for item in findings):
        return None
    return findings
