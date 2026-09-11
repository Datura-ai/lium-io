"""Wire types of the validator's one-call rental create (`POST /rent`, liumd deploy).

Schema id `lium.local_rent/1`. The validator signs the intent with its hotkey (the same signing
blob as `/verify`); the executor makes the rental container from the validator's own run spec
(`datura.rental_spec`, the very dataclass the SSH path hands to docker-py) with docker-py on the
host, waits for it to run, and answers with what it made. The answer is evidence, not a verdict:
the validator's SSH execs that follow (keys, sshd, environment) are what prove the container is
there and reachable — a local answer only moves the create.
"""

from typing import Any, Literal

from datura.requests.validator_requests import (
    LOCAL_RENT_CAPABILITY,
    LOCAL_RENT_SCHEMA,
    LocalVerifyWireModel,
)
from pydantic import Field

# Every document on this wire refuses a field its schema does not name (`extra="forbid"`, as the
# `/verify` wire and the Rust liumd's `deny_unknown_fields` do); `populate_by_name` for `schema`.
WireModel = LocalVerifyWireModel

SCHEMA = LOCAL_RENT_SCHEMA
CAPABILITY = LOCAL_RENT_CAPABILITY

STEP_NAMES = ("image", "container", "ready")


class ReadyStep(WireModel):
    """Wait for the container to run (the SSH path's `docker ps -q --filter name=` poll) and, when
    asked, for the published sshd port to answer an SSH banner from this host — which says nothing
    about reachability from outside; the validator's own connection does."""

    running_timeout_s: int = Field(default=10, ge=1, le=60)
    ssh_host_port: int | None = Field(default=None, ge=1, le=65535)
    ssh_timeout_s: int = Field(default=10, ge=1, le=60)


class RentSteps(WireModel):
    # `docker image inspect` of the spec's image before the create: present or not, and its digest.
    image: bool = False
    # The wire form of `ContainerRunSpec` (`datura.rental_spec.spec_to_wire`); parsed and bounded
    # by `spec_from_wire`, never by hand here.
    container: dict[str, Any] | None = None
    ready: ReadyStep | None = None


class RentIntentBody(WireModel):
    schema_id: Literal["lium.local_rent/1"] = Field(alias="schema", default=SCHEMA)
    nonce: str = Field(min_length=16, max_length=128)
    issued_at: int
    expires_at: int
    executor_uuid: str = Field(min_length=1, max_length=128)
    # sha256 (hex) of the executor's SSH host public key line — the one the miner reports and the
    # validator pins for SSH. The executor compares it with its own: a captured intent replayed to
    # another executor is refused there (401). Signed with the rest.
    ssh_host_key_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    # Seconds the executor may spend before answering with whatever finished; capped by settings.
    deadline_s: int = Field(default=60, ge=5, le=600)
    steps: RentSteps = Field(default_factory=RentSteps)



class RentIntent(RentIntentBody):
    signature: str = Field(min_length=1, max_length=1024)


class RentStepResult(WireModel):
    status: str  # ok | failed | timeout | skipped
    ms: int = 0
    data: dict[str, Any] | None = None
    error: str | None = None


class RentResult(WireModel):
    schema_id: str = Field(alias="schema", default=SCHEMA)
    nonce: str
    executor_uuid: str
    executor_version: str
    started_at: int
    elapsed_ms: int
    deadline_hit: bool = False
    # True when the call did not end with a running container and nothing the executor made remains
    # (removed, or never materialised): the validator's SSH fallback starts from the host as it was.
    # False with `container.status == "ok"`: the container is there and the validator frees the name.
    rolled_back: bool = False
    steps: dict[str, RentStepResult]

