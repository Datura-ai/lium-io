import json
import logging

from core.utils import _m, get_extra_info

from .inspector_verdict import canonical_sha256

logger = logging.getLogger(__name__)

# the backend closes the socket with 1009 on a bigger frame (its ws_max_size); resending that frame can never succeed
MAX_FRAME_BYTES = 4 * 1024 * 1024
# room left for the fields and delivery stamps the connector adds to an event report
EVENT_REPORT_MAX_BYTES = MAX_FRAME_BYTES - 64 * 1024


def _size(value) -> int:
    return len(json.dumps(value).encode())


def fit_event_report(event: dict) -> dict:
    """Keeps the leading findings of an oversized event report that fit one frame; counts the rest.

    Runs before the report is published to Redis: a message over the pub/sub output buffer limit closes the connector's
    subscription and drops everything queued for it. Measured as `RedisService.publish` encodes it, which is never
    smaller than the connector's compact frame. Not split into several reports: the backend stores one row per cycle.
    """
    report = event.get("report")
    findings = report.get("findings") if isinstance(report, dict) else None
    if not isinstance(findings, list) or _size(event) <= EVENT_REPORT_MAX_BYTES:
        return event
    empty = {**report, "findings": [], "findings_omitted": len(findings)}
    room = EVENT_REPORT_MAX_BYTES - _size({**event, "report": empty})
    kept = 0
    for finding in findings:
        room -= _size(finding) + 2  # the ", " between list items
        if room < 0:
            break
        kept += 1
    fitted = {**report, "findings": findings[:kept], "findings_omitted": len(findings) - kept}
    logger.warning(
        _m(
            "Trimmed an event report to fit one frame",
            extra=get_extra_info(
                {
                    "executor_id": event.get("executor_id"),
                    "miner_hotkey": event.get("miner_hotkey"),
                    "findings_kept": kept,
                    "findings_omitted": len(findings) - kept,
                }
            ),
        )
    )
    result = {**event, "report": fitted}
    context = event.get("context") or {}
    verdict = context.get("verdict")
    if isinstance(verdict, dict) and "report_sha256" in verdict:
        # the verdict hashed the full report; it must match the report actually stored
        result["context"] = {**context, "verdict": {**verdict, "report_sha256": canonical_sha256(fitted)}}
    return result
