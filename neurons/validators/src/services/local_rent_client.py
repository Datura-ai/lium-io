"""liumd deploy (DAH-2834): the rental container made by ONE signed `POST /rent` on the executor.

The SSH path's create is four dependent round trips through the miner hop and the SSH tunnel —
the Docker SDK `create` + `start`, the `docker ps` running poll, the first exec. An executor with
the route on (it advertises `local_rent/1` on `/version`; the rent path does not spend a round trip
reading it — it posts, and a 404/422 is the "not here" answer) accepts the validator's own run spec
(`datura.rental_spec`, the very dataclass the SDK path hands to docker-py) and makes the container
with the same docker-py calls on the host, waits for it to run and for the published sshd port to
answer, and reports what it made. What it reports is evidence, not the proof: the SSH execs that follow (keys, sshd
bootstrap, environment) are unchanged and are what prove the container reachable.

Only a spec that carries nothing private takes this path (`carries_only_public_fields`): the
executor's API port is plain HTTP, so a renter's startup command, entrypoint or environment
never travels on it — such a rental keeps the SSH tunnel. Any refusal, timeout or malformed
answer → the SDK path as today (`LocalRentUnavailable.reason` is the log label).
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from datura.rental_spec import ContainerRunSpec, carries_only_public_fields, spec_to_wire
from datura.requests.validator_requests import LOCAL_RENT_CAPABILITY, LOCAL_RENT_SCHEMA
from services.local_verify_client import (
    INTENT_TTL_SECONDS,
    LocalVerifyClient,
    LocalVerifyUnavailable,
    StepEvidence,
    _wire_int,
)

SCHEMA = LOCAL_RENT_SCHEMA
CAPABILITY = LOCAL_RENT_CAPABILITY
STEP_NAMES = ("image", "container", "ready")
EXECUTOR_VERSION_MAX_CHARS = 64

# The executor's `ready` step: how long it waits for State.Running (the SSH path's poll is 10 tries
# of 1 s). The SSH-banner wait is the caller's choice (`ssh_wait_s`, 0 = none).
RUNNING_TIMEOUT_S = 10

LocalRentUnavailable = LocalVerifyUnavailable

# Non-answers after which the executor cannot have acted on the intent: the route is absent or off
# (404), the body was refused before any create — 401 from the route's checks, 422 from an old
# image's `MinerMiddleware` (no `data_to_sign`) or from the route's own model validation — the
# executor was busy / had seen the nonce (409), or the connection was never made (aiohttp's
# connect-phase errors: refused, DNS, connect timeout — the type name `post_signed` puts first in
# the detail). Every other non-answer — a total timeout, a connection that broke after the send, a
# 5xx, an answer that could not be read — leaves it open whether a container of the spec's name
# exists over there, and the SDK fallback frees the name before its own `docker run`
# (`may_have_acted`).
NEVER_ACTED_REASONS = frozenset({"not_supported", "refused", "busy_or_replay"})
NEVER_CONNECTED_PREFIX = "ClientConnector"  # ClientConnectorError and its DNS/SSL/cert subclasses
NEVER_ACTED_STATUS_PREFIX = "status 422"  # `post_signed` labels any non-200/401/404/409 `http_error: status <n>`


def may_have_acted(reason: str, detail: str = "") -> bool:
    if reason in NEVER_ACTED_REASONS:
        return False
    if reason == "http_error" and detail.startswith(NEVER_ACTED_STATUS_PREFIX):
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
        ready.update({"ssh_host_port": ssh_host_port, "ssh_timeout_s": ssh_wait_s})
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
class LocalRentAnswer:
    nonce: str
    executor_uuid: str
    executor_version: str
    elapsed_ms: int
    deadline_hit: bool
    rolled_back: bool
    steps: dict[str, StepEvidence] = field(default_factory=dict)
    round_trip_ms: int = 0

    def step(self, name: str) -> StepEvidence:
        return self.steps.get(name) or StepEvidence(status="skipped")

    @property
    def created(self) -> bool:
        """The container is up on the executor's word: made, running, sshd answering when asked.
        The validator's own execs are what confirm it."""
        return (
            not self.deadline_hit
            and not self.rolled_back
            and self.step("container").status == "ok"
            and self.step("ready").status == "ok"
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
    steps = raw.get("steps")
    if not isinstance(steps, dict):
        raise LocalRentUnavailable("malformed", "steps is not an object")
    return LocalRentAnswer(
        nonce=raw["nonce"],
        executor_uuid=raw["executor_uuid"],
        executor_version=str(raw.get("executor_version") or "")[:EXECUTOR_VERSION_MAX_CHARS],
        elapsed_ms=_wire_int(raw.get("elapsed_ms")),
        deadline_hit=bool(raw.get("deadline_hit")),
        rolled_back=bool(raw.get("rolled_back")),
        steps={name: StepEvidence.from_wire(steps[name]) for name in STEP_NAMES if name in steps},
        round_trip_ms=round_trip_ms,
    )


class LocalRentClient(LocalVerifyClient):
    """The verify client's transport (same base URL, signing, bounds and error labels), one more
    route."""

    async def rent(self, executor_info, intent: dict[str, Any]) -> LocalRentAnswer:
        raw, round_trip_ms = await self.post_signed(executor_info, "/rent", intent)
        return parse_answer(raw, intent=intent, round_trip_ms=round_trip_ms)
