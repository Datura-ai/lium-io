"""Wire types of the validator's one-call local verification (`POST /verify`).

Schema id `lium.local_verify/1`. The validator signs the intent with its hotkey; the executor
runs the steps it names locally — the same scripts the SSH-driven checks run — and answers with one
result document. The document is evidence, not a verdict: the validator unseals and scores the
matmul and VerifyX outputs with the code it already has, so a local run and an SSH run of the same
challenge are judged by the same function.
"""

from typing import Literal

from datura.requests.validator_requests import (
    LOCAL_VERIFY_CAPABILITY,
    LOCAL_VERIFY_SCHEMA,
    DeviceChallenge,  # noqa: F401 — re-exported for the service and the tests
    LocalVerifyWireModel,
    MatmulStep,
    VerifyXStep,
)
from pydantic import Field

# One definition for both sides (datura): the validator client imports the same names.
SCHEMA = LOCAL_VERIFY_SCHEMA
CAPABILITY = LOCAL_VERIFY_CAPABILITY
# The intent's step challenges (`DeviceChallenge`, `MatmulStep`, `VerifyXStep`) and the
# `extra="forbid"` base are datura's — the validator builds the intent from the very models the
# executor parses. Re-exported here for the service and the tests.
WireModel = LocalVerifyWireModel

STEP_NAMES = ("matmul", "verifyx", "docker", "ports", "inspector")


class VerifySteps(WireModel):
    matmul: MatmulStep | None = None
    verifyx: VerifyXStep | None = None
    docker: bool = False
    ports: bool = False
    inspector: bool = False


class VerifyIntentBody(WireModel):
    """What the validator signs. `signature` covers the canonical JSON of these fields."""

    # Only this schema is understood: another version is refused (422) rather than run as v1.
    schema_id: Literal["lium.local_verify/1"] = Field(alias="schema", default=SCHEMA)
    nonce: str = Field(min_length=16, max_length=128)
    issued_at: int  # unix seconds, validator clock
    expires_at: int  # unix seconds; refused after this
    executor_uuid: str = Field(min_length=1, max_length=128)
    # The miner this executor belongs to, as the validator knows it. Signed like every other
    # field, it binds the intent to that miner's executors: one relayed to another provider's
    # executor is refused there (401) instead of running a GPU suite the sender never earned.
    # The executor cannot check `executor_uuid` (it does not know its own); it does know its miner.
    miner_hotkey: str = Field(min_length=1, max_length=128)
    # Seconds the executor may spend before answering with whatever finished; capped by settings.
    deadline_s: int = Field(default=300, ge=5, le=3600)
    # Run the two GPU/RAM probes side by side. The validator sets this only at first-pass sizes:
    # a full-size VerifyX (75 % of RAM) beside the matmul's host matrices OOMs 64–128 GB hosts.
    parallel_gpu: bool = False
    steps: VerifySteps = Field(default_factory=VerifySteps)


class VerifyIntent(VerifyIntentBody):
    signature: str = Field(min_length=1, max_length=1024)


class ScriptRun(WireModel):
    """What one script run produced: exit status, capped stdout, the stderr tail."""

    status: Literal["ok", "failed", "timeout", "skipped"]
    ms: int = 0
    exit_status: int | None = None
    stdout: str | None = None
    stderr_tail: str | None = None
    stdout_sha256: str | None = None
    error: str | None = None


# --- typed `data` per step: the shape two repos read, stated once -------------------------------


class CardRun(ScriptRun):
    """One card's pinned matmul run (`matmul.data.per_card[]`)."""

    card_index: int


class MatmulData(WireModel):
    per_card: list[CardRun]


class VerifyXData(WireModel):
    # The validator compares the library digest before it trusts a response (core/checksums).
    lib_sha256: str | None


class ContainerFact(WireModel):
    name: str
    status: str
    image: str | None
    created: str | None


class DiskFact(WireModel):
    total_bytes: int
    free_bytes: int
    used_bytes: int


class DockerFacts(WireModel):
    server_version: str | None
    root_dir: str | None
    runtimes: list[str]
    default_runtime: str | None
    sysbox_runtime: bool
    disk: DiskFact | None
    containers: list[ContainerFact]


class PortFacts(WireModel):
    port_range: str | None
    port_mappings: str | None
    configured: int
    sampled: int
    published_by_docker: list[int]
    free_ports: int


class InspectorFacts(WireModel):
    lib_present: bool
    lib_sha256: str | None
    script_present: bool


StepData = MatmulData | VerifyXData | DockerFacts | PortFacts | InspectorFacts


class StepResult(ScriptRun):
    """One step's evidence. `status` is about the run, not the verdict: `ok` means the step produced
    output the validator can judge; the judging happens on the validator."""

    data: StepData | None = None


class VerifyResult(WireModel):
    schema_id: str = Field(alias="schema", default=SCHEMA)
    nonce: str
    executor_uuid: str
    executor_version: str
    started_at: int
    elapsed_ms: int
    deadline_hit: bool = False
    steps: dict[str, StepResult]
    # No executor signing key exists today (design §3: a key on the adversary's host proves only
    # that the adversary's host signed). The field is here so a future key can fill it.
    signer: str = "none"
    signature: str | None = None
