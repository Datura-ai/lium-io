"""The registries: every wire type has exactly one model, dispatch is by `message_type`, and what is
not a message of this protocol is a ProtocolError — the error type both peers' loops already catch."""

import json

import pydantic
import pytest

from lium_protocol import (
    BACKEND_MESSAGES,
    VALIDATOR_MESSAGES,
    BackendMessageType,
    Message,
    ProtocolError,
    Registry,
    ValidatorMessageType,
)
from lium_protocol.backend_to_validator import ContainerDeleteRequest
from lium_protocol.validator_to_backend import ContainerDeleted, ExecutorSpecRequest

A_DELETED = {
    "message_type": "ContainerDeleted",
    "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    "executor_id": "6f1d2c3b-4a5e-4f60-9b7c-8d9e0f1a2b3c",
    "pod_id": "0b1c2d3e-4f50-4617-8293-a4b5c6d7e8f9",
}


@pytest.mark.parametrize("registry", [VALIDATOR_MESSAGES, BACKEND_MESSAGES], ids=["validator", "backend"])
def test_every_wire_type_has_a_model_and_every_model_a_wire_type(registry: Registry) -> None:
    assert registry.missing_types() == []
    for wire_value, model in registry.models().items():
        assert model.wire_type().value == wire_value
        assert isinstance(model.wire_type(), registry.type_enum)


def test_the_two_enums_do_not_share_wire_values() -> None:
    """One socket, two directions: a value on both sides would make a message ambiguous to a reader."""
    both = {m.value for m in ValidatorMessageType} & {m.value for m in BackendMessageType}
    assert both == set()


def test_parse_dispatches_on_message_type() -> None:
    message = VALIDATOR_MESSAGES.parse(json.dumps(A_DELETED))
    assert type(message) is ContainerDeleted
    assert message.workload_kind.value == "CUSTOMER_RENTAL"  # the default
    assert (message.sent_at, message.forwarded_at, message.queue_depth) == (None, None, None)


def test_parse_accepts_bytes_and_keeps_unknown_keys_out_of_the_way() -> None:
    raw = json.dumps({**A_DELETED, "a_field_from_a_newer_validator": 1}).encode()
    message = VALIDATOR_MESSAGES.parse(raw)
    assert type(message) is ContainerDeleted
    assert "a_field_from_a_newer_validator" not in message.model_dump()


@pytest.mark.parametrize(
    "text,reason",
    [
        ("{not json", "not JSON"),
        (json.dumps({"pod_id": "x"}), "no message_type"),
        (json.dumps({"message_type": "ContainerVaporised"}), "unknown message_type"),
        (
            json.dumps({"message_type": "ContainerDeleted", "pod_id": "x"}),
            "fields the model refuses",
        ),
        (
            json.dumps({**A_DELETED, "message_type": "ContainerDeleteRequest"}),
            "the other direction's type",
        ),
    ],
)
def test_what_is_not_a_message_is_a_protocol_error(text: str, reason: str) -> None:
    with pytest.raises(ProtocolError) as excinfo:
        VALIDATOR_MESSAGES.parse(text)
    assert excinfo.value.msg  # never empty: a consumer logs it
    if reason == "unknown message_type":
        assert excinfo.value.msg == "unknown message_type 'ContainerVaporised'"


def test_backend_registry_parses_its_own_direction() -> None:
    message = BACKEND_MESSAGES.parse(
        json.dumps(
            {
                **A_DELETED,
                "message_type": "ContainerDeleteRequest",
                "container_name": "container_0b1c2d3e",
            }
        )
    )
    assert type(message) is ContainerDeleteRequest


def test_a_consumer_registry_overrides_one_model_and_inherits_the_rest() -> None:
    """The backend's use: its own ExecutorSpecRequest with typed specs, everything else from here."""

    class Specs(pydantic.BaseModel):
        gpu_count: int

    class TypedExecutorSpecRequest(ExecutorSpecRequest):
        specs: Specs | None = None  # type: ignore[assignment]

    mine: Registry = Registry(ValidatorMessageType, base=VALIDATOR_MESSAGES)
    mine.register(TypedExecutorSpecRequest)
    assert mine.model_for("ExecutorSpecRequest") is TypedExecutorSpecRequest
    assert mine.model_for("ContainerDeleted") is ContainerDeleted
    assert VALIDATOR_MESSAGES.model_for("ExecutorSpecRequest") is ExecutorSpecRequest  # the base is untouched


def test_a_refused_message_never_carries_the_input_into_the_error() -> None:
    """A BackupContainerRequest missing one field: the error names the field, not the credentials next to it."""
    message = {
        "message_type": "BackupContainerRequest",
        "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "executor_id": "6f1d2c3b-4a5e-4f60-9b7c-8d9e0f1a2b3c",
        "pod_id": "0b1c2d3e-4f50-4617-8293-a4b5c6d7e8f9",
        "auth_token": "placeholder-jwt-that-must-not-leak",
        "repository_password": "placeholder-password-that-must-not-leak",
    }
    with pytest.raises(ProtocolError) as excinfo:
        BACKEND_MESSAGES.parse_obj(message)
    errors = json.loads(excinfo.value.msg)  # the error list itself, not a string of it
    assert {e["loc"][0] for e in errors} >= {"source_volume", "backup_volume_info", "backup_log_id"}
    assert "must-not-leak" not in excinfo.value.msg
    assert all("input" not in e and "url" not in e for e in errors)


def test_register_refuses_a_second_class_for_a_bound_wire_type() -> None:
    class Twin(ContainerDeleted):
        pass

    with pytest.raises(TypeError):
        VALIDATOR_MESSAGES.register(Twin)  # ContainerDeleted is bound in this registry itself
    mine: Registry = Registry(ValidatorMessageType, base=VALIDATOR_MESSAGES)
    assert mine.register(Twin) is Twin  # inherited from base: overriding is the point
    with pytest.raises(TypeError):
        mine.register(ContainerDeleted)  # and now Twin is bound in `mine`
    assert VALIDATOR_MESSAGES.model_for("ContainerDeleted") is ContainerDeleted


def test_package_version_is_the_protocol_version() -> None:
    import importlib.metadata

    from lium_protocol import PROTOCOL_VERSION

    assert importlib.metadata.version("lium-protocol") == PROTOCOL_VERSION


def test_register_refuses_a_model_of_the_other_enum_or_without_a_default() -> None:
    with pytest.raises(TypeError):
        VALIDATOR_MESSAGES.register(ContainerDeleteRequest)

    class NoDefault(Message):
        message_type: ValidatorMessageType

    with pytest.raises(TypeError):
        Registry(ValidatorMessageType).register(NoDefault)
