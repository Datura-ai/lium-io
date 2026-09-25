"""ExecutorSpecRequest since protocol 1.4.0: `pod_ssh` (the cycle's SSH probe of each rented pod) and
`validation_event` (the structured event whose `reason_code` the backend stores) ride the wire; a report from an
older validator, which sends neither, reads as None for both."""

import pydantic
import pytest

from lium_protocol import PROTOCOL_VERSION, VALIDATOR_MESSAGES
from lium_protocol.validator_to_backend import (
    POD_STATES_MAX_ITEMS,
    CouldNotLookCode,
    ExecutorSpecRequest,
    PodSshObservation,
    PodSshResult,
)

OLD_VALIDATOR_REPORT = {
    "message_type": "ExecutorSpecRequest",
    "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    "miner_coldkey": "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy",
    "validator_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    "executor_uuid": "6f1d2c3b-4a5e-4f60-9b7c-8d9e0f1a2b3c",
    "executor_ip": "203.0.113.10",
    "executor_port": 8001,
    "specs": None,
    "score": None,
    "log_text": "SSH connection failed",
    "job_batch_id": "2026-09-25T10:15:00+00:00",
}
EVENT = {
    "event": "No result from the miner for this node",
    "reason_code": "EXECUTOR_RESULT_MISSING",
    "severity": "error",
    "category": "availability",
    "impact": "The node could not be validated this cycle",
    "when": "2026-09-25T10:16:40Z",
}


def test_the_fields_arrived_with_a_minor_bump() -> None:
    major, minor, _patch = (int(part) for part in PROTOCOL_VERSION.split("."))
    assert (major, minor) >= (1, 4)
    assert {"pod_ssh", "validation_event"} <= set(ExecutorSpecRequest.model_fields)


def test_an_older_validator_report_reads_as_none() -> None:
    report = VALIDATOR_MESSAGES.parse_obj(OLD_VALIDATOR_REPORT)
    assert report.pod_ssh is None
    assert report.validation_event is None


def test_an_observation_defaults_to_no_errno_and_a_healthy_fleet() -> None:
    seen = PodSshObservation.model_validate({"pod_id": "P1", "result": "banner"})
    assert (seen.result, seen.errno, seen.fleet_ok) == (PodSshResult.banner, None, True)


def test_a_result_outside_the_four_is_refused() -> None:
    with pytest.raises(pydantic.ValidationError):
        PodSshObservation.model_validate({"pod_id": "P1", "result": "open"})


def test_a_result_missing_report_round_trips() -> None:
    message = {
        **OLD_VALIDATOR_REPORT,
        "score": 0.0,
        "validation_event": {**EVENT, "trace_id": "t-1", "extra_key": "kept"},
        "pod_ssh": [{"pod_id": "P1", "result": "timeout", "errno": 110, "fleet_ok": False}],
    }
    report = VALIDATOR_MESSAGES.parse_obj(message)
    assert report.validation_event.reason_code == CouldNotLookCode.EXECUTOR_RESULT_MISSING
    assert report.validation_event.model_dump()["extra_key"] == "kept"
    assert report.pod_ssh == [PodSshObservation(pod_id="P1", result=PodSshResult.timeout, errno=110, fleet_ok=False)]
    assert VALIDATOR_MESSAGES.parse_obj(report.model_dump(mode="json")) == report


def test_an_unknown_reason_code_still_reads() -> None:
    report = VALIDATOR_MESSAGES.parse_obj({**OLD_VALIDATOR_REPORT, "validation_event": {**EVENT, "reason_code": "NEW"}})
    assert report.validation_event.reason_code == "NEW"


def test_pod_ssh_is_bounded_like_pod_states() -> None:
    too_many = [{"pod_id": f"P{i}", "result": "banner"} for i in range(POD_STATES_MAX_ITEMS + 1)]
    with pytest.raises(pydantic.ValidationError):
        ExecutorSpecRequest.model_validate({**OLD_VALIDATOR_REPORT, "pod_ssh": too_many})
