"""FailedContainerRequest since protocol 1.1.0: `build_log_tail` (DAH-3504) and `step_detail` (DAH-3505)
ride the wire and survive a round trip; a message from an older validator, which omits them, reads as None.

Regression: on protocol 1.0.0 the mirror had neither field, so a backend reading the wire through it
dropped both silently (pydantic ignores unknown keys) and the renter never saw the build's last lines."""

import json

import pytest

from lium_protocol import PROTOCOL_VERSION, VALIDATOR_MESSAGES
from lium_protocol.validator_to_backend import FailedContainerRequest

OLD_VALIDATOR_FAILURE = {
    "message_type": "FailedRequest",
    "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    "executor_id": "6f1d2c3b-4a5e-4f60-9b7c-8d9e0f1a2b3c",
    "pod_id": "0b1c2d3e-4f50-4617-8293-a4b5c6d7e8f9",
    "error_type": "ContainerCreationFailed",
    "error_code": "UnknownError",
    "msg": "Failed create_container",
    "failure_step": "docker_build",
}
BUILD_TAIL = "#7 2.114 ERROR: No matching distribution found for torch==9.9.9\n#7 ERROR: exit code: 1"
VOLUME_DETAIL = "Docker SDK create volume failed: 'NoneType' object has no attribute 'settimeout'"


def test_the_fields_arrived_with_a_minor_bump() -> None:
    """1.0.0 had neither field; adding optional fields is a minor bump (README), so any 1.x from 1.1 carries them."""
    major, minor, _patch = (int(part) for part in PROTOCOL_VERSION.split("."))
    assert (major, minor) >= (1, 1)
    assert {"build_log_tail", "step_detail"} <= set(FailedContainerRequest.model_fields)


@pytest.mark.parametrize(
    "extra",
    # one exception per failure: a CustomBuildFailed sets build_log_tail, a volume-step error sets step_detail
    [
        {"build_log_tail": BUILD_TAIL, "step_detail": None},
        {"build_log_tail": None, "step_detail": VOLUME_DETAIL},
    ],
    ids=["build_log_tail", "step_detail"],
)
def test_round_trip_keeps_the_fields(extra: dict) -> None:
    message = VALIDATOR_MESSAGES.parse(json.dumps({**OLD_VALIDATOR_FAILURE, **extra}))
    assert isinstance(message, FailedContainerRequest)
    assert message.build_log_tail == extra["build_log_tail"]
    assert message.step_detail == extra["step_detail"]
    again = VALIDATOR_MESSAGES.parse(message.model_dump_json())
    assert again == message
    dumped = json.loads(message.model_dump_json())
    assert dumped["build_log_tail"] == extra["build_log_tail"]
    assert dumped["step_detail"] == extra["step_detail"]


def test_a_message_from_an_older_validator_reads_as_none() -> None:
    message = VALIDATOR_MESSAGES.parse(json.dumps(OLD_VALIDATOR_FAILURE))
    assert isinstance(message, FailedContainerRequest)
    assert message.build_log_tail is None
    assert message.step_detail is None
    # and the round trip does not invent a value for either
    again = VALIDATOR_MESSAGES.parse(message.model_dump_json())
    assert again.build_log_tail is None and again.step_detail is None
