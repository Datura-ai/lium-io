"""liumd deploy (DAH-2834): the rental container made by ONE signed `POST /rent` on the executor.

The SSH path's create is four dependent round trips through the miner hop and the SSH tunnel —
the Docker SDK `create` + `start`, the `docker ps` running poll, the first exec. An executor with
the route on advertises `local_rent/1` and `local_rent_port` on `/version` (one plain GET over
the executor's API port; no port → the SDK path, `no_tunnel_port`). It accepts the validator's own
run spec (`datura.rental_spec`, the very dataclass the SDK path hands to docker-py) and makes the
container with the same docker-py calls on the host, waits for it to run and for the published
sshd port to answer, and reports what it made. What it reports is evidence, not the proof: the SSH
execs that follow (keys, sshd bootstrap, environment) are unchanged and are what prove the
container reachable.

The answer is unsigned, like `/verify`'s, so the executor serves `/rent` to loopback peers only
and the intent rides the rental's own SSH session (`LocalVerifyClient.post_signed`: a
direct-tcpip channel of the `asyncssh` connection the SDK path already holds, host key pinned).
A proxy on the miner's port-forward can neither read the spec nor rewrite the answer — the same
trust root as the SSH path (taiberium on #1339).

Only a spec that carries nothing private takes this path (`carries_only_public_fields`): the
executor's API is plain HTTP inside the miner's own process, so a renter's startup command,
entrypoint or environment never travels on it — such a rental keeps the SDK path. Any refusal,
timeout or malformed answer → the SDK path as today (`LocalRentUnavailable.reason` is the log
label).
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import asyncssh
import pydantic
from datura.rental_spec import ContainerRunSpec, carries_only_public_fields, spec_to_wire
from datura.requests.validator_requests import (
    LOCAL_RENT_CAPABILITY,
    LOCAL_RENT_SCHEMA,
    RentContainerData,
    RentImageData,
    RentReadyData,
)
from services.local_verify_client import (
    INTENT_TTL_SECONDS,
    LocalVerifyClient,
    LocalVerifyUnavailable,
    StepEvidence,
    _wire_int,
)

SCHEMA = LOCAL_RENT_SCHEMA
CAPABILITY = LOCAL_RENT_CAPABILITY
EXECUTOR_VERSION_MAX_CHARS = 64

# The executor's `ready` step: how long it waits for State.Running (the SSH path's poll is 10 tries
# of 1 s). The SSH-banner wait is the caller's choice (`ssh_wait_s`, 0 = none), capped at what the
# executor's `ReadyStep.ssh_timeout_s` accepts (`le=60`).
SSH_WAIT_MAX_S = 60
RUNNING_TIMEOUT_S = 10

LocalRentUnavailable = LocalVerifyUnavailable

# Non-answers after which the executor cannot have acted on the intent: the route is absent or off
# (404), the body was refused before any create — 401 from the route's checks, 403 from a peer
# that was not its loopback, 422 from an old image's `MinerMiddleware` (no `data_to_sign`) or from
# the route's own model validation — the executor was busy / had seen the nonce (409), the tunnel
# listener was never bound (`post_signed`'s `tunnel: …` transport error or `tunnel listener not
# bound` timeout: nothing was sent), or the local connect to that listener failed (aiohttp's
# `ClientConnector*` errors — the type name `post_signed` puts first in the detail). Every other
# non-answer — a total timeout, a tunnel that closed without a status line (sshd could not open
# the channel, OR the session broke after the send: aiohttp sees the same disconnect), a 5xx, an
# answer that could not be read — leaves it open whether a container of the spec's name exists
# over there, and the SDK fallback frees the name before its own `docker run` (`may_have_acted`).
NEVER_ACTED_REASONS = frozenset({"not_supported", "busy_or_replay"})
NEVER_CONNECTED_PREFIX = "ClientConnector"  # ClientConnectorError and its DNS/SSL/cert subclasses
NEVER_BOUND_PREFIXES = ("tunnel: ", "tunnel listener not bound")  # `post_signed` before any send
TUNNEL_CLOSED_PREFIX = "tunnel to "  # `post_signed`'s `refused` for a channel that closed unanswered
NEVER_ACTED_STATUS_PREFIX = "status 422"  # `post_signed` labels any non-200/401/403/404/409 `http_error: status <n>`


def may_have_acted(reason: str, detail: str = "") -> bool:
    if reason in NEVER_ACTED_REASONS:
        return False
    if reason == "refused":
        # 401/403 text is the executor's refusal (never acted); a closed tunnel is not an answer.
        return detail.startswith(TUNNEL_CLOSED_PREFIX)
    if reason == "http_error" and detail.startswith(NEVER_ACTED_STATUS_PREFIX):
        return False
    if reason in ("transport", "timeout") and detail.startswith(NEVER_BOUND_PREFIXES):
        return False
    return not (reason == "transport" and detail.startswith(NEVER_CONNECTED_PREFIX))


# The executor answers within `deadline_s`, then may spend its rollback bound removing what it made
# — by id 10 s (REMOVE_TIMEOUT_SECONDS), by label up to 20 s (a list and a remove) — and the answer
# rides after it. The margin covers the by-id case and the connect (5 s); a by-label rollback that
# runs past the budget is a client timeout, which frees the name here (`may_have_acted`).
EXECUTOR_ROLLBACK_MARGIN_S = 15
EXECUTOR_DEADLINE_MIN_S = 5


def executor_deadline_s(timeout_s: int) -> int:
    return max(EXECUTOR_DEADLINE_MIN_S, timeout_s - EXECUTOR_ROLLBACK_MARGIN_S)


def host_key_sha256(host_key_line: str | None) -> str | None:
    """The digest the executor computes of its own SSH host public key line (the miner-reported
    `ssh_host_key` the validator already pins for SSH): the intent is bound to it."""
    if not host_key_line or not host_key_line.strip():
        return None
    return hashlib.sha256(host_key_line.strip().encode("utf-8")).hexdigest()


def eligible(spec: ContainerRunSpec, host_key: str | None) -> str | None:
    """None when the spec may travel on the executor's HTTP API; otherwise why not (the label)."""
    if not carries_only_public_fields(spec):
        return "private_fields"
    if host_key_sha256(host_key) is None:
        return "no_host_key"  # nothing to bind the intent to (the SDK path would refuse too)
    return None


def build_intent(
    *,
    executor_uuid: str,
    host_key: str,
    spec: ContainerRunSpec,
    deadline_s: int,
    ssh_host_port: int | None,
    ssh_wait_s: int = 0,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    ready: dict[str, Any] = {"running_timeout_s": RUNNING_TIMEOUT_S}
    if ssh_host_port is not None and ssh_wait_s > 0:
        # the executor's ReadyStep bounds ssh_timeout_s at SSH_WAIT_MAX_S; a larger setting would be
        # a 422 on every rent (one wasted call each) instead of a longer wait
        ready.update({"ssh_host_port": ssh_host_port, "ssh_timeout_s": min(ssh_wait_s, SSH_WAIT_MAX_S)})
    return {
        "schema": SCHEMA,
        "nonce": secrets.token_hex(16),
        "issued_at": int(now),
        "expires_at": int(now) + INTENT_TTL_SECONDS,
        "executor_uuid": executor_uuid,
        "ssh_host_key_sha256": host_key_sha256(host_key),
        "deadline_s": deadline_s,
        "steps": {"image": True, "container": spec_to_wire(spec), "ready": ready},
    }


@dataclass
class RentStepEvidence:
    """One step as the validator reads it: how the run went. The step's evidence is typed beside
    it on the answer (`image`, `container`, `ready`), never read back out of a dict."""

    status: str
    ms: int = 0
    error: str | None = None


@dataclass
class LocalRentAnswer:
    nonce: str
    executor_uuid: str
    executor_version: str
    elapsed_ms: int
    deadline_hit: bool
    rolled_back: bool
    steps: dict[str, RentStepEvidence] = field(default_factory=dict)
    # Each step's evidence in the executor's own model (datura, one definition for both ends);
    # None when the step did not run, carried none, or carried a shape the model refuses — that
    # step is then `malformed` in `steps`, so the answer never reads as `created`.
    image: RentImageData | None = None
    container: RentContainerData | None = None
    ready: RentReadyData | None = None
    round_trip_ms: int = 0

    def step(self, name: str) -> RentStepEvidence:
        return self.steps.get(name) or RentStepEvidence(status="skipped")

    @property
    def created(self) -> bool:
        """The container is up on the executor's word: made (the container step `ok` with its
        evidence), running, sshd answering when asked. The validator's own execs are what confirm it."""
        return (
            not self.deadline_hit
            and not self.rolled_back
            and self.step("container").status == "ok"
            and self.container is not None
            and self.step("ready").status == "ok"
            and self.ready is not None
        )

    @property
    def may_hold_the_name(self) -> bool:
        """The executor could not prove that nothing of its making remains (a container it made and
        could not remove, or a create its deadline cut before the daemon answered): the SSH
        fallback's `docker run` of the same name would collide unless the name is freed first.
        A create the daemon refused, or an intent refused before any create, is `rolled_back`."""
        return not self.created and not self.rolled_back


def parse_answer(raw: Any, *, intent: dict[str, Any], round_trip_ms: int) -> LocalRentAnswer:
    if not isinstance(raw, dict):
        raise LocalRentUnavailable("malformed", "answer is not an object")
    if raw.get("schema") != SCHEMA:
        raise LocalRentUnavailable("schema_mismatch", f"got {raw.get('schema')!r}"[:100])
    if raw.get("nonce") != intent["nonce"]:
        raise LocalRentUnavailable("nonce_mismatch", "answer does not echo the intent nonce")
    if raw.get("executor_uuid") != intent["executor_uuid"]:
        raise LocalRentUnavailable("executor_mismatch", f"got {raw.get('executor_uuid')!r}"[:100])
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, dict):
        raise LocalRentUnavailable("malformed", "steps is not an object")
    steps: dict[str, RentStepEvidence] = {}
    image = _step_data(raw_steps, "image", RentImageData, steps)
    container = _step_data(raw_steps, "container", RentContainerData, steps)
    ready = _step_data(raw_steps, "ready", RentReadyData, steps)
    return LocalRentAnswer(
        nonce=raw["nonce"],
        executor_uuid=raw["executor_uuid"],
        executor_version=str(raw.get("executor_version") or "")[:EXECUTOR_VERSION_MAX_CHARS],
        elapsed_ms=_wire_int(raw.get("elapsed_ms")),
        deadline_hit=bool(raw.get("deadline_hit")),
        rolled_back=bool(raw.get("rolled_back")),
        steps=steps,
        image=image,
        container=container,
        ready=ready,
        round_trip_ms=round_trip_ms,
    )


_StepData = TypeVar("_StepData", RentImageData, RentContainerData, RentReadyData)


def _step_data(
    raw_steps: dict[str, Any], name: str, model: type[_StepData], steps: dict[str, RentStepEvidence]
) -> _StepData | None:
    """Read one step: its run status into `steps`, its `data` as the step's own model. A `data`
    the model refuses (a field it does not name, a value of another type) makes the step
    `malformed`, as an unknown status does — the validator does not act on evidence it cannot
    read, and `created` stays False."""
    if name not in raw_steps:
        return None
    evidence = StepEvidence.from_wire(raw_steps[name])
    steps[name] = RentStepEvidence(status=evidence.status, ms=evidence.ms, error=evidence.error)
    if evidence.status == "malformed":
        return None
    raw_data = raw_steps[name].get("data")
    if raw_data is None:
        return None
    try:
        return model.model_validate(raw_data)
    except pydantic.ValidationError as exc:
        steps[name] = RentStepEvidence(
            status="malformed", ms=evidence.ms, error=f"{name} data is not a {model.__name__}: {exc.error_count()} error(s)"
        )
        return None


class LocalRentClient(LocalVerifyClient):
    """The verify client's transport (`/version` read, signing, the SSH tunnel, bounds and error
    labels), one more route."""

    async def rent(
        self, ssh: asyncssh.SSHClientConnection, local_rent_port: int, intent: dict[str, Any]
    ) -> LocalRentAnswer:
        """The signed intent through `ssh` to `127.0.0.1:<local_rent_port>/rent` on the executor
        (`post_signed`); the answer must be to this intent (`parse_answer`)."""
        raw, round_trip_ms = await self.post_signed(ssh, local_rent_port, "/rent", intent)
        return parse_answer(raw, intent=intent, round_trip_ms=round_trip_ms)
