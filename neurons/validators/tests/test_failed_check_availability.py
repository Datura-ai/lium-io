"""DAH-5395: a fatal check that means the node cannot serve a rental hides it until a cycle passes."""

from datura.requests.miner_requests import ExecutorSSHInfo
from services.task.availability import (
    AvailabilityErrorCode,
    ReachSource,
    ReachTarget,
    availability_errors,
    build_availability_event,
    failed_check_event,
    silence_availability_errors_on_our_own_outage,
)
from services.task.models import JobResult, build_msg


def _event(check_id: str, reason: str = "SOME_REASON", severity: str = "error"):
    return build_msg(event="x", reason=reason, severity=severity, impact="x", check_id=check_id)


def _errors_the_service_sends(ok: bool, events: list) -> list[dict]:
    # the same composition as TaskService.create_task
    failed = failed_check_event(ok, events)
    return [e.model_dump(mode="json") for e in availability_errors([*events, failed] if failed else events)]


def test_a_failed_check_that_makes_the_node_unusable_hides_it() -> None:
    events = [_event("gpu.scrape.machine_spec", "OK", "info"), _event("gpu.validate.verifyx", "VERIFYX_FAILED")]

    errors = _errors_the_service_sends(False, events)

    assert [e["reason_code"] for e in errors] == [AvailabilityErrorCode.VERIFICATION_FAILED]
    assert errors[0]["reach_source"] == "validator"
    assert errors[0]["reach_target"] == "node"
    assert errors[0]["what_we_saw"] == {"failed_check": "gpu.validate.verifyx", "failed_reason": "VERIFYX_FAILED"}


def test_a_policy_score_zero_keeps_the_node_listed() -> None:
    # a run that completes with score 0 (excluded hotkey, idle-only exclusion, a score-zero gate) is ok=True
    events = [_event("gpu.validate.verifyx", "OK", "info"), _event("score", "SCORING_EXCLUDED", "warning")]

    assert failed_check_event(True, events) is None
    assert _errors_the_service_sends(True, events) == []


def test_failed_checks_that_keep_the_listing_add_nothing() -> None:
    for check_id in (
        "executor.validate.duplicate",
        "executor.validate.image_version",
        "executor.validate.rented_state",
        "executor.validate.cached_template",
    ):
        assert _errors_the_service_sends(False, [_event(check_id)]) == [], check_id


def test_a_check_that_already_reported_an_availability_error_is_not_doubled() -> None:
    lost = build_availability_event(
        code=AvailabilityErrorCode.GPU_LOST, reach_source=ReachSource.VALIDATOR, reach_target=ReachTarget.GPU,
        event_text="lost", impact="x", remediation="x", what_we_saw={},
    )

    errors = _errors_the_service_sends(False, [lost])

    assert [e["reason_code"] for e in errors] == ["GPU_LOST"]


def test_the_next_pass_clears_it() -> None:
    assert _errors_the_service_sends(False, [_event("gpu.validate.fingerprint")])
    # an empty list (not None) is what makes the backend clear the stored entry
    assert _errors_the_service_sends(True, [_event("score", "OK", "info")]) == []


def _result(errors):
    return JobResult(
        executor_info=ExecutorSSHInfo(
            uuid="node", address="10.0.0.5", port=8080, ssh_username="root", ssh_port=2200,
            python_path="/usr/bin/python3", root_dir="/root/app",
        ),
        score=0, job_score=0, job_batch_id="b", log_status="info", log_text="", gpu_model=None, gpu_count=0,
        availability_errors=errors,
    )


def test_most_of_a_cycle_failing_is_treated_as_our_own_outage() -> None:
    failed = _errors_the_service_sends(False, [_event("gpu.validate.verifyx")])
    results = [_result(list(failed)) for _ in range(8)] + [_result([]) for _ in range(2)]

    assert silence_availability_errors_on_our_own_outage(results) == 10
    assert all(r.availability_errors is None for r in results)
