"""Wire types of the validator's one-call local verification (`POST /verify`).

Schema id `lium.local_verify/1`. The validator signs the intent with its hotkey; the executor
runs the steps it names locally — the same scripts the SSH-driven checks run — and answers with one
result document. The document is evidence, not a verdict: the validator unseals and scores the
matmul and VerifyX outputs with the code it already has, so a local run and an SSH run of the same
challenge are judged by the same function.
"""

from typing import Annotated, Any, Literal

from datura.requests.validator_requests import LOCAL_VERIFY_CAPABILITY, LOCAL_VERIFY_SCHEMA
from pydantic import BaseModel, Field

# One definition for both sides (datura): the validator client imports the same names.
SCHEMA = LOCAL_VERIFY_SCHEMA
CAPABILITY = LOCAL_VERIFY_CAPABILITY

STEP_NAMES = ("matmul", "verifyx", "docker", "ports", "inspector")
# The largest card count one host can claim; bounds the matmul fan-out an intent can ask for.
MAX_DEVICES = 64


class MatmulStep(BaseModel):
    """The capability matmul challenge, exactly the arguments `decrypt_challenge.py` takes."""

    dim_n: int
    dim_k: int
    seed: int
    cipher_text: str = Field(min_length=1, max_length=4096)
    # CUDA device indexes to pin one run to each (the all-cards work-proof); None = one unpinned run.
    devices: list[Annotated[int, Field(ge=0)]] | None = Field(default=None, max_length=MAX_DEVICES)


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


class StepResult(BaseModel):
    """One step's evidence. `status` is about the run, not the verdict: `ok` means the step produced
    output the validator can judge; the judging happens on the validator."""

    status: str  # ok | failed | timeout | skipped
    ms: int = 0
    exit_status: int | None = None
    stdout: str | None = None
    stderr_tail: str | None = None
    stdout_sha256: str | None = None
    data: dict[str, Any] | None = None
    error: str | None = None


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
