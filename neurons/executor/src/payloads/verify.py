"""Wire types of the validator's one-call local verification (`POST /verify`).

Schema id `lium.local_verify/1`. The validator signs the intent with its hotkey; the executor
runs the steps it names locally — the same scripts the SSH-driven checks run — and answers with one
result document. The document is evidence, not a verdict: the validator unseals and scores the
matmul and VerifyX outputs with the code it already has, so a local run and an SSH run of the same
challenge are judged by the same function.
"""

from typing import Literal

from datura.requests.validator_requests import LOCAL_VERIFY_CAPABILITY, LOCAL_VERIFY_SCHEMA
from pydantic import BaseModel, Field, model_validator

# One definition for both sides (datura): the validator client imports the same names.
SCHEMA = LOCAL_VERIFY_SCHEMA
CAPABILITY = LOCAL_VERIFY_CAPABILITY

STEP_NAMES = ("matmul", "verifyx", "docker", "ports", "inspector")
# The largest card count one host can claim; bounds the matmul fan-out an intent can ask for.
MAX_DEVICES = 64


class DeviceChallenge(BaseModel):
    """One card's own challenge for the all-cards work-proof: a run pinned to `index`
    (`CUDA_VISIBLE_DEVICES`) with a cipher text sealed for that card alone (seeds may repeat, as
    on the SSH path). The validator derives the cipher text per device (domain-separated from the
    intent's nonce), so the output of one real run unseals for one card only — a host with fewer
    cards than it claims cannot answer for all of them with a single computation."""

    index: int = Field(ge=0)
    seed: int
    cipher_text: str = Field(min_length=1, max_length=4096)


class MatmulStep(BaseModel):
    """The capability matmul challenge, exactly the arguments `decrypt_challenge.py` takes."""

    dim_n: int
    dim_k: int
    seed: int
    cipher_text: str = Field(min_length=1, max_length=4096)
    # The all-cards work-proof: one pinned run per card, each with its own challenge; None = one
    # unpinned run with the challenge above.
    devices: list[DeviceChallenge] | None = Field(default=None, max_length=MAX_DEVICES)

    @model_validator(mode="after")
    def _one_challenge_per_card(self) -> "MatmulStep":
        if not self.devices:
            return self
        indexes = [d.index for d in self.devices]
        if len(set(indexes)) != len(indexes):
            raise ValueError("devices: the same card index twice")
        ciphers = {d.cipher_text for d in self.devices} | {self.cipher_text}
        if len(ciphers) != len(self.devices) + 1:
            raise ValueError("devices: every card needs its own cipher_text")
        return self


class VerifyXStep(BaseModel):
    """The VerifyX challenge, exactly the arguments `verifyx_executor.py` takes."""

    seed: int
    cipher_text: str = Field(min_length=1, max_length=65536)


class VerifySteps(BaseModel):
    matmul: MatmulStep | None = None
    verifyx: VerifyXStep | None = None
    docker: bool = False
    ports: bool = False
    inspector: bool = False


class VerifyIntentBody(BaseModel):
    """What the validator signs. `signature` covers the canonical JSON of these fields."""

    # Only this schema is understood: another version is refused (422) rather than run as v1.
    schema_id: Literal["lium.local_verify/1"] = Field(alias="schema", default=SCHEMA)
    nonce: str = Field(min_length=16, max_length=128)
    issued_at: int  # unix seconds, validator clock
    expires_at: int  # unix seconds; refused after this
    executor_uuid: str = Field(min_length=1, max_length=128)
    # Seconds the executor may spend before answering with whatever finished; capped by settings.
    deadline_s: int = Field(default=300, ge=5, le=3600)
    # Run the two GPU/RAM probes side by side. The validator sets this only at first-pass sizes:
    # a full-size VerifyX (75 % of RAM) beside the matmul's host matrices OOMs 64–128 GB hosts.
    parallel_gpu: bool = False
    steps: VerifySteps = Field(default_factory=VerifySteps)

    model_config = {"populate_by_name": True}


class VerifyIntent(VerifyIntentBody):
    signature: str = Field(min_length=1, max_length=1024)


class ScriptRun(BaseModel):
    """What one script run produced: exit status, capped stdout, the stderr tail."""

    status: str  # ok | failed | timeout | skipped
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


class MatmulData(BaseModel):
    per_card: list[CardRun]


class VerifyXData(BaseModel):
    # The validator compares the library digest before it trusts a response (core/checksums).
    lib_sha256: str | None


class ContainerFact(BaseModel):
    name: str
    status: str
    image: str | None
    created: str | None


class DiskFact(BaseModel):
    total_bytes: int
    free_bytes: int
    used_bytes: int


class DockerFacts(BaseModel):
    server_version: str | None
    root_dir: str | None
    runtimes: list[str]
    default_runtime: str | None
    sysbox_runtime: bool
    disk: DiskFact | None
    containers: list[ContainerFact]


class PortFacts(BaseModel):
    port_range: str | None
    port_mappings: str | None
    configured: int
    sampled: int
    published_by_docker: list[int]
    free: int


class InspectorFacts(BaseModel):
    lib_present: bool
    lib_sha256: str | None
    script_present: bool


StepData = MatmulData | VerifyXData | DockerFacts | PortFacts | InspectorFacts


class StepResult(ScriptRun):
    """One step's evidence. `status` is about the run, not the verdict: `ok` means the step produced
    output the validator can judge; the judging happens on the validator."""

    data: StepData | None = None


class VerifyResult(BaseModel):
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

    model_config = {"populate_by_name": True}
