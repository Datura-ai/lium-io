import json
import logging

from services.task.event_report_fit import EVENT_REPORT_MAX_BYTES, MAX_FRAME_BYTES, fit_event_report
from services.task.inspector_verdict import canonical_sha256


def _event(findings: list[dict]) -> dict:
    return {
        "miner_hotkey": "hk",
        "executor_id": "ex",
        "job_batch_id": "jb",
        "pod_ids": ["p1"],
        "outcome": "MALICIOUS",
        "reason_code": "R",
        "report": {"findings": findings, "health": {"ok": True}},
        "context": {},
        "when": "2026-10-28T04:07:32+00:00",
    }


def _published_size(event: dict) -> int:
    return len(json.dumps(event).encode())  # as RedisService.publish encodes it


def test_a_small_report_goes_unchanged():
    event = _event([{"kind": "a"}])
    assert fit_event_report(event) is event


def test_a_report_with_40000_findings_is_published_under_one_frame(caplog):
    findings = [{"i": i, "command": "é" * 200} for i in range(40_000)]  # ~50 MB escaped
    with caplog.at_level(logging.WARNING, logger="services.task.event_report_fit"):
        fitted = fit_event_report(_event(findings))

    kept = fitted["report"]["findings"]
    assert _published_size(fitted) <= EVENT_REPORT_MAX_BYTES < MAX_FRAME_BYTES
    assert kept and kept == findings[: len(kept)]
    assert fitted["report"]["findings_omitted"] == len(findings) - len(kept)
    assert fitted["report"]["health"] == {"ok": True}
    (trim,) = [r.msg.extra for r in caplog.records if "Trimmed" in r.getMessage()]
    assert (trim["executor_id"], trim["miner_hotkey"]) == ("ex", "hk")
    assert trim["findings_kept"] == len(kept)


def test_one_finding_bigger_than_a_frame_is_left_out_with_the_rest_counted():
    fitted = fit_event_report(_event([{"command": "x" * MAX_FRAME_BYTES}, {"kind": "b"}]))
    assert fitted["report"]["findings"] == []
    assert fitted["report"]["findings_omitted"] == 2


def test_a_trimmed_reports_verdict_hash_matches_the_report_published():
    event = _event([{"i": i, "command": "x" * 2000} for i in range(5_000)])
    event["context"] = {"verdict": {"report_sha256": canonical_sha256(event["report"]), "kind": "v"}}

    fitted = fit_event_report(event)

    assert fitted["report"]["findings_omitted"] > 0
    assert fitted["context"]["verdict"]["report_sha256"] == canonical_sha256(fitted["report"])
    assert fitted["context"]["verdict"]["kind"] == "v"
    assert event["context"]["verdict"]["report_sha256"] == canonical_sha256(event["report"])


def test_a_report_too_big_outside_its_findings_is_not_published(caplog):
    event = _event([{"kind": "a"}])
    event["context"] = {"error": "x" * MAX_FRAME_BYTES}

    with caplog.at_level(logging.ERROR, logger="services.task.event_report_fit"):
        assert fit_event_report(event) is None

    (drop,) = [r.msg.extra for r in caplog.records if "Dropped" in r.getMessage()]
    assert (drop["executor_id"], drop["miner_hotkey"]) == ("ex", "hk")
