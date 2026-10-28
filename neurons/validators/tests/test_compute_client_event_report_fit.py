from clients.compute_client import EVENT_REPORT_MAX_BYTES, MAX_FRAME_BYTES, fit_event_report
from protocol.vc_protocol.validator_requests import InspectorEventRequest


def _event(findings: list[dict]) -> InspectorEventRequest:
    return InspectorEventRequest(
        miner_hotkey="hk",
        validator_hotkey="vk",
        executor_id="ex",
        job_batch_id="jb",
        pod_ids=["p1"],
        outcome="MALICIOUS",
        reason_code="R",
        report={"findings": findings, "health": {"ok": True}},
        when="2026-10-28T04:07:32+00:00",
    )


def test_a_small_report_goes_unchanged():
    event = _event([{"kind": "a"}])
    assert fit_event_report(event) is event


def test_an_oversized_report_keeps_the_leading_findings_that_fit_one_frame():
    findings = [{"i": i, "command": "é" * 2000} for i in range(10_000)]  # ~40 MB escaped
    fitted = fit_event_report(_event(findings))

    size = len(fitted.model_dump_json().encode())
    kept = fitted.report["findings"]
    assert size <= EVENT_REPORT_MAX_BYTES < MAX_FRAME_BYTES
    assert kept and kept == findings[: len(kept)]
    assert fitted.report["findings_omitted"] == len(findings) - len(kept)
    assert fitted.report["health"] == {"ok": True}


def test_one_finding_bigger_than_a_frame_is_left_out_with_the_rest_counted():
    fitted = fit_event_report(_event([{"command": "x" * MAX_FRAME_BYTES}, {"kind": "b"}]))
    assert fitted.report["findings"] == []
    assert fitted.report["findings_omitted"] == 2
