"""lium_protocol ↔ the validator's own models: the recorded wire messages (lium_protocol/recorded/)
go through both, and both must read every field the same way (DAH-3247).

This is the replay test of the protocol package on the validator's side: the recordings are what
the two peers' models serialise today, `payload_models` / `vc_protocol` are what the validator
runs, and `lium_protocol` is the one copy both repos are meant to pin. A field one of them drops,
renames or types differently fails here before it fails on the socket.
"""

import json
import sys
from pathlib import Path

import pydantic
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "lium_protocol") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "lium_protocol"))

import lium_protocol  # noqa: E402
from lium_protocol.backend_to_validator import SOCKET_REPLIES  # noqa: E402
from lium_protocol.http import HTTP_MODELS  # noqa: E402
from lium_protocol.recorded import json_keys, recorded  # noqa: E402
from payload_models import payloads  # noqa: E402
from protocol.vc_protocol import compute_requests, validator_requests  # noqa: E402

# declared in both enums, sent by neither peer, and the validator has no model for it
VALIDATOR_TYPES_WITHOUT_A_VALIDATOR_MODEL = {"MachineSpecRequest"}
# the validator sends this body to the backend and builds it ad hoc: no model to compare
HTTP_BODIES_WITHOUT_A_VALIDATOR_MODEL = {"PodHostRebootRecoveredRequest"}
# fields the backend puts on the wire that the validator's model does not declare (so ignores):
# the known drift, by recording. A new entry here is a wire field the validator silently drops —
# decide whether the validator should read it before adding it.
VALIDATOR_IGNORES: dict[str, set[str]] = {
    "RentedMachineResponse": {"machines[].rented_ports", "machines[].containers[].rented_ports"},
}


def _ids(direction: str) -> list[str]:
    return [f"{entry['expect']}[{i}]" for i, entry in enumerate(recorded(direction))]


def _validator_parse_validator_message(message: dict) -> pydantic.BaseModel:
    """The validator's own model for a message it sends: the cycle messages live in vc_protocol,
    the answers to container requests in payload_models; both dispatch on `message_type` through
    datura's BaseRequest.parse."""
    raw = json.dumps(message)
    if message["message_type"] in {member.value for member in validator_requests.RequestType}:
        return validator_requests.BaseValidatorRequest.parse(raw)
    return payloads.BaseValidatorResponse.parse(raw)


def _dropped(recorded_value: object, view: object, path: str = "") -> set[str]:
    """Paths present in the recorded message that `view` (a model's JSON dump) does not carry."""
    if isinstance(recorded_value, dict):
        if not isinstance(view, dict):
            return {path or "$"}
        dropped: set[str] = set()
        for key, value in recorded_value.items():
            here = f"{path}.{key}" if path else key
            dropped |= {here} if key not in view else _dropped(value, view[key], here)
        return dropped
    if isinstance(recorded_value, list):
        if not isinstance(view, list) or len(view) != len(recorded_value):
            return {path}
        dropped = set()
        for item, seen in zip(recorded_value, view, strict=True):
            dropped |= _dropped(item, seen, f"{path}[]")
        return dropped
    return set()


def _disagreements(
    recorded_value: object, ours: object, theirs: object, path: str = ""
) -> list[str]:
    """Recorded leaf paths where the two models' views differ; a path the validator's model does not
    carry is not a disagreement (it is a drop, checked separately)."""
    if isinstance(recorded_value, dict):
        out: list[str] = []
        for key, value in recorded_value.items():
            here = f"{path}.{key}" if path else key
            if isinstance(theirs, dict) and key in theirs and isinstance(ours, dict):
                out += _disagreements(value, ours.get(key), theirs[key], here)
        return out
    if isinstance(recorded_value, list) and isinstance(ours, list) and isinstance(theirs, list):
        out = []
        for item, mine, its in zip(recorded_value, ours, theirs, strict=True):
            out += _disagreements(item, mine, its, f"{path}[]")
        return out
    return [] if ours == theirs else [f"{path}: lium_protocol {ours!r} != validator {theirs!r}"]


def _same_wire_view(
    message: dict,
    ours: pydantic.BaseModel,
    theirs: pydantic.BaseModel,
    *,
    ignores: set[str] = frozenset(),
) -> None:
    """Both models read every recorded field (but the validator's documented ignores) and agree,
    leaf by leaf, on every field the validator reads."""
    our_view = json_keys(ours.model_dump(mode="json"))
    their_view = json_keys(theirs.model_dump(mode="json"))
    assert _dropped(message, our_view) == set(), "lium_protocol drops a recorded field"
    assert _dropped(message, their_view) == ignores, "the validator's model drops a recorded field"
    assert _disagreements(message, our_view, their_view) == []


@pytest.mark.parametrize(
    "entry", recorded("validator_to_backend"), ids=_ids("validator_to_backend")
)
def test_what_the_validator_sends_parses_the_same_on_both_sides(entry: dict) -> None:
    message = entry["message"]
    ours = lium_protocol.VALIDATOR_MESSAGES.parse_obj(message)
    assert type(ours).__name__ == entry["expect"]
    if message["message_type"] in VALIDATOR_TYPES_WITHOUT_A_VALIDATOR_MODEL:
        with pytest.raises(KeyError):
            _validator_parse_validator_message(message)
        return
    theirs = _validator_parse_validator_message(message)
    assert type(theirs).__name__ == entry["expect"]
    _same_wire_view(message, ours, theirs)


@pytest.mark.parametrize(
    "entry", recorded("backend_to_validator"), ids=_ids("backend_to_validator")
)
def test_what_the_backend_sends_parses_through_the_validators_own_parser(entry: dict) -> None:
    """The exact code paths compute_client.py takes for a backend message."""
    message = entry["message"]
    raw = json.dumps(message)
    ours = lium_protocol.BACKEND_MESSAGES.parse_obj(message)
    assert type(ours).__name__ == entry["expect"]
    if entry["expect"] == "ForcedValidationCycleRequest":
        theirs = pydantic.TypeAdapter(payloads.ForcedValidationCycleRequest).validate_json(raw)
    elif entry["expect"] == "GetEstimateRequest":
        theirs = pydantic.TypeAdapter(payloads.GetEstimateRequest).validate_json(raw)
        # the validator's model has no message_type: it ignores the discriminator, reads the rest
        assert _dropped(message, json_keys(theirs.model_dump(mode="json"))) == {"message_type"}
        assert _dropped(message, json_keys(ours.model_dump(mode="json"))) == set()
        return
    elif entry["expect"] == "DuplicateExecutorsResponse":
        theirs = payloads.DuplicateExecutorsResponse.model_validate_json(raw)
    else:
        theirs = payloads.BaseServerRequest.parse(raw)
    assert type(theirs).__name__ == entry["expect"]
    _same_wire_view(message, ours, theirs)


@pytest.mark.parametrize(
    "entry",
    recorded("http") + recorded("socket_replies"),
    ids=_ids("http") + _ids("socket_replies"),
)
def test_http_bodies_and_socket_replies_parse_the_same_on_both_sides(entry: dict) -> None:
    """HTTP bodies via vc_protocol.compute_requests; the three typeless socket replies the same way,
    which is what compute_client.handle_message does with the raw text
    (`Response.model_validate_json`, `TypeAdapter(RentedMachineResponse)`)."""
    body = entry["message"]
    ours = {**HTTP_MODELS, **SOCKET_REPLIES}[entry["expect"]].model_validate(body)
    if entry["expect"] in HTTP_BODIES_WITHOUT_A_VALIDATOR_MODEL:
        assert not hasattr(compute_requests, entry["expect"])
        return
    theirs = getattr(compute_requests, entry["expect"]).model_validate(body)
    if isinstance(body, list):
        assert len(ours.root) == len(theirs.root) == len(body)
        return
    _same_wire_view(body, ours, theirs, ignores=VALIDATOR_IGNORES.get(entry["expect"], set()))


def test_every_validator_wire_type_is_in_lium_protocol() -> None:
    """The union check the other way: a type the validator can emit today is in the package."""
    validator_emits = {member.value for member in validator_requests.RequestType} | {
        member.value for member in payloads.ContainerResponseType
    }
    package = {member.value for member in lium_protocol.ValidatorMessageType}
    assert validator_emits <= package, sorted(validator_emits - package)
    backend_sends = {member.value for member in payloads.ContainerRequestType} | {
        "ForcedValidationCycleRequest",
        "GetEstimateRequest",  # compute_client parses it with a TypeAdapter, no enum member here
    }
    assert backend_sends <= {member.value for member in lium_protocol.BackendMessageType}


def test_enums_the_validator_emits_are_members_of_the_package_enums() -> None:
    from lium_protocol import validator_to_backend as v2b

    def values(enum_cls) -> set:
        return {member.value for member in enum_cls}

    assert values(payloads.FailedContainerErrorCodes) <= values(v2b.FailedContainerErrorCodes)
    assert values(payloads.FailedContainerErrorTypes) <= values(v2b.FailedContainerErrorTypes)
    assert values(payloads.VolumeEncryptionStatus) == values(v2b.VolumeEncryptionStatus)
    assert values(payloads.ContainerWarningCode) <= values(v2b.ContainerWarningCode)
    assert values(payloads.WorkloadKind) == values(lium_protocol.WorkloadKind)


def test_the_excluded_executor_id_is_the_validators_failed_miner_uuid() -> None:
    """The backend keys emission eligibility off it; the validator names the value differently."""
    from core.validator import FAILED_MINER_EXECUTOR_UUID
    from lium_protocol.validator_to_backend import EXCLUDED_PROVIDER_EMISSION_EXECUTOR_ID

    assert EXCLUDED_PROVIDER_EMISSION_EXECUTOR_ID == FAILED_MINER_EXECUTOR_UUID
