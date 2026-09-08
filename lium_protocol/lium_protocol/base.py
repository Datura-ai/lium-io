"""What every message on the validator↔backend wire has in common.

A message is one JSON object with a `message_type` string that names its model. `Registry.parse`
turns the text into the right model, or raises `ProtocolError` — the same shape as the
`ValidationError` both sides define today (`datura.requests.base`, backend `protocol/base.py`), so
a consumer can swap its parser without changing its error handling.

The registry is explicit (`Registry.register`) instead of walking `__subclasses__`: a consumer that
subclasses a message to add typed fields (the backend types `ExecutorSpecRequest.specs` as its
`MachineSpecs`) registers its subclass in its own registry and the wire type still maps to exactly one
model — with subclass discovery, two classes carrying the same `message_type` default would race.
"""

from __future__ import annotations

import enum
import json
from typing import Any, Generic, TypeVar

import pydantic


class ProtocolError(Exception):
    """Text that is not a message of this protocol: not JSON, no/unknown `message_type`, or fields the
    model refuses. `.msg` is the human-readable reason; pydantic's error list travels as JSON in it."""

    def __init__(self, msg: str):
        super().__init__(msg)
        self.msg = msg

    @classmethod
    def from_json_decode_error(cls, exc: json.JSONDecodeError) -> ProtocolError:
        return cls(exc.args[0])

    @classmethod
    def from_pydantic_validation_error(cls, exc: pydantic.ValidationError) -> ProtocolError:
        return cls(json.dumps(exc.json()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.msg})"


class WorkloadKind(enum.Enum):
    """What a container is for: a renter's pod or a Lium/miner default job (filler)."""

    CUSTOMER_RENTAL = "CUSTOMER_RENTAL"
    FILLER = "FILLER"


class DeliveryStamps(pydantic.BaseModel):
    """DAH-2792: epoch seconds. `sent_at` by the producer, `forwarded_at` / `queue_depth` by the
    validator's connector before `ws.send()`; the backend turns the gaps into histograms. All None from
    a peer that predates the stamps."""

    sent_at: float | None = None
    forwarded_at: float | None = None
    queue_depth: int | None = None


class Message(pydantic.BaseModel):
    """A wire message. Subclasses set `message_type` to one enum member as the field default; that
    default is the discriminator `Registry` dispatches on. Unknown keys are kept and ignored, never an
    error: both peers add fields on their own release schedule (see README — additive changes only)."""

    model_config = pydantic.ConfigDict(extra="ignore")

    message_type: enum.Enum

    @classmethod
    def wire_type(cls) -> enum.Enum:
        default = cls.model_fields["message_type"].default
        if not isinstance(default, enum.Enum):
            raise TypeError(f"{cls.__name__} has no message_type default; only concrete messages have one")
        return default


class _Envelope(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")
    message_type: str


M = TypeVar("M", bound=Message)


class Registry(Generic[M]):
    """`message_type` value → model, for one direction of the wire.

    `parse(text)` reads the envelope, looks the type up, and validates the whole object against that
    model. A consumer builds its own registry from ours (`Registry(base=VALIDATOR_MESSAGES)`) and
    `register`s subclasses to override a model with a more precisely typed one."""

    def __init__(self, type_enum: type[enum.Enum], *, base: Registry[M] | None = None):
        self.type_enum = type_enum
        self._models: dict[str, type[M]] = dict(base._models) if base else {}

    def register(self, model: type[M]) -> type[M]:
        """Decorator and function: `@REGISTRY.register` on a message class, or `REGISTRY.register(Sub)`.
        The model's `message_type` default must be a member of this registry's enum."""
        wire_type = model.wire_type()
        if not isinstance(wire_type, self.type_enum):
            raise TypeError(f"{model.__name__}.message_type is {wire_type!r}, not a {self.type_enum.__name__}")
        self._models[wire_type.value] = model
        return model

    def model_for(self, wire_type: str | enum.Enum) -> type[M]:
        key = wire_type.value if isinstance(wire_type, enum.Enum) else wire_type
        try:
            return self._models[key]
        except KeyError:
            raise ProtocolError(f"unknown message_type {key!r}") from None

    def models(self) -> dict[str, type[M]]:
        """Every registered model by wire value, in registration order (the schema snapshot's order)."""
        return dict(self._models)

    def missing_types(self) -> list[str]:
        """Enum members with no model — a test asserts this is empty for the shipped registries."""
        return [member.value for member in self.type_enum if member.value not in self._models]

    def parse(self, text: str | bytes) -> M:
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProtocolError.from_json_decode_error(exc) from None
        return self.parse_obj(payload)

    def parse_obj(self, payload: Any) -> M:
        try:
            envelope = _Envelope.model_validate(payload)
        except pydantic.ValidationError as exc:
            raise ProtocolError.from_pydantic_validation_error(exc) from None
        model = self.model_for(envelope.message_type)
        try:
            return model.model_validate(payload)
        except pydantic.ValidationError as exc:
            raise ProtocolError.from_pydantic_validation_error(exc) from None
