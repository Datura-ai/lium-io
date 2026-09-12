"""lium_protocol — the validator↔backend wire, once.

Every message the validator sends the backend (`validator_to_backend`), every message the backend sends
the validator (`backend_to_validator`) and every HTTP body the validator reads (`http`), as pydantic v2
models with no dependency on either application. `snapshots/lium_protocol.v1.json` is the committed JSON
Schema of all of it; `python -m lium_protocol.schema --check` fails when the models drift from it.

Versioning: PROTOCOL_VERSION is semver over the wire, not over this code. Adding an optional field or an
enum member is a minor bump; making a field required, removing one, or changing a type is a major bump
(and a new snapshot file). Consumers pin a tag `lium-protocol-v<PROTOCOL_VERSION>` of lium-io.
"""

PROTOCOL_VERSION = "1.0.0"

from .backend_to_validator import BACKEND_MESSAGES, BackendMessage, BackendMessageType  # noqa: E402
from .base import DeliveryStamps, Message, ProtocolError, Registry, WorkloadKind  # noqa: E402
from .validator_to_backend import VALIDATOR_MESSAGES, ValidatorMessage, ValidatorMessageType  # noqa: E402

__all__ = [
    "PROTOCOL_VERSION",
    "BACKEND_MESSAGES",
    "BackendMessage",
    "BackendMessageType",
    "DeliveryStamps",
    "Message",
    "ProtocolError",
    "Registry",
    "VALIDATOR_MESSAGES",
    "ValidatorMessage",
    "ValidatorMessageType",
    "WorkloadKind",
]
