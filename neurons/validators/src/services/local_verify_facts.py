"""liumd phase 2 (DAH-2834): the read-only host facts `POST /verify` returns, parsed with closed
sets and bounds before any check reads them.

The executor's `docker` / `ports` / `inspector` steps are host-reported — an executor can say
anything here, exactly as it can in the stdout of a `docker ps` run over SSH. They therefore may
stand in for *read-only SSH commands* whose answer the host controlled anyway, and never for a probe
that proves something. What consumes them (`speed/LIUMD_PHASE2.md` §2):

- `docker.containers` + `docker.now` → `ContainerCleanup.cleanup`: the stale-container candidate list
  and each candidate's age (replaces `docker ps -a --filter` and one `inspect .Created` + `date +%s`
  per NON-stale candidate). The removal itself stays SSH-proven: a candidate the fact calls stale is
  re-read over SSH before `docker rm`.
- `ports.published_by_docker` → `PortSelector`: host ports docker already publishes are not probed
  (fewer failed binds). A fact can only REMOVE candidates from a set the validator built itself.
- `inspector.lib_sha256` → observe-only: logged against the validator's expected digest so the
  agreement rate is known before anything consumes it (jam6099's area).

NEVER replaced by a fact, whatever the executor reports — the list the phase-2 design keeps:
- the DinD probe and the sysbox proof (`docker run --rm hello-world` under sysbox, the validator's
  own inbound connection): `docker.sysbox_runtime` seeds nothing and decides nothing;
- the port connect-back (`PortTester`): `ports.published_by_docker` says which ports NOT to try,
  never which ports work;
- the filler liveness verdict (`RentalVerificationCheck._verify_filler_alive`, a penalty path) and
  the removal verdict of the stale cleanup (`docker rm` after an SSH re-read);
- the cached-template verdict (`CachedTemplateVerificationCheck`, fatal after the cutoff);
- the inspector run itself (`inspector_executor.py` over SSH).

Bounds: ≤ MAX_CONTAINERS containers, names in docker's own grammar and ≤ 128 chars, status from
docker's closed set, `created` an RFC 3339 string the validator parses (else the container carries
no age and is left to the SSH path), ≤ MAX_PORTS published ports each 1..65535, the digest 64 hex.
Anything outside a bound makes that fact absent (`None`), never an error and never a verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

MAX_CONTAINERS = 512
# The host's clock (`now`, epoch seconds): a positive int below 2**40 (year ≈ 36 800) — outside
# that the fact cannot age containers; an unbounded int would overflow the float division downstream.
HOST_NOW_MAX = 2**40
MAX_PORTS = 4096
NAME_MAX_CHARS = 128
IMAGE_MAX_CHARS = 256
# docker/api/types/container: the states `State.Status` can carry.
DOCKER_STATUSES = frozenset(
    {"created", "restarting", "running", "removing", "paused", "exited", "dead"}
)
# daemon/names: [a-zA-Z0-9][a-zA-Z0-9_.-]*
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")  # fullmatch: `$` would allow a trailing newline
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RFC3339_FRACTION_RE = re.compile(r"(\.\d{1,9})(?=Z|[+-]\d{2}:\d{2}$)")


@dataclass(frozen=True)
class HostContainer:
    name: str
    status: str  # one of DOCKER_STATUSES
    created_at: int | None  # epoch seconds, None when the executor's string did not parse
    image: str | None = None


@dataclass(frozen=True)
class LocalFacts:
    """What the early facts call learnt; every field is optional evidence, never a verdict."""

    containers: tuple[HostContainer, ...] | None = None  # None = the docker fact was unusable
    host_now: int | None = None  # the host clock at collection, epoch seconds
    published_ports: frozenset[int] | None = None  # None = the ports fact was unusable
    inspector_lib_sha256: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    round_trip_ms: int = 0
    executor_elapsed_ms: int = 0
    steps: dict[str, str] = field(default_factory=dict)  # step -> status as answered

    def can_age_containers(self) -> bool:
        return self.containers is not None and self.host_now is not None


def parse_created(value: Any) -> int | None:
    """Docker's `Created` (RFC 3339 with up to nine fractional digits, `Z` or an offset) → epoch
    seconds; None for anything else."""
    if not isinstance(value, str) or not (20 <= len(value) <= 40):
        return None
    text = _RFC3339_FRACTION_RE.sub(lambda m: m.group(1)[:7], value)  # Python parses ≤ 6 digits
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _host_now(value: Any) -> int | None:
    now = _int(value)
    return now if now is not None and 0 < now < HOST_NOW_MAX else None


def parse_containers(data: Any) -> tuple[HostContainer, ...] | None:
    raw = data.get("containers") if isinstance(data, dict) else None
    if not isinstance(raw, list) or len(raw) > MAX_CONTAINERS:
        return None
    out: list[HostContainer] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        name, status = item.get("name"), item.get("status")
        if not isinstance(name, str) or len(name) > NAME_MAX_CHARS or not _NAME_RE.fullmatch(name):
            return None
        if not isinstance(status, str) or status not in DOCKER_STATUSES:
            return None
        image = item.get("image")
        out.append(
            HostContainer(
                name=name,
                status=status,
                created_at=parse_created(item.get("created")),
                image=image[:IMAGE_MAX_CHARS] if isinstance(image, str) else None,
            )
        )
    return tuple(out)


def parse_published_ports(data: Any) -> frozenset[int] | None:
    raw = data.get("published_by_docker") if isinstance(data, dict) else None
    if not isinstance(raw, list) or len(raw) > MAX_PORTS:
        return None
    ports: set[int] = set()
    for port in raw:
        value = _int(port)
        if value is None or not (1 <= value <= 65535):
            return None
        ports.add(value)
    return frozenset(ports)


def parse_inspector_digest(data: Any) -> str | None:
    digest = data.get("lib_sha256") if isinstance(data, dict) else None
    return digest if isinstance(digest, str) and _SHA256_RE.fullmatch(digest) else None


def parse_facts(
    steps: dict[str, Any],
    *,
    capabilities: set[str] | frozenset[str],
    round_trip_ms: int,
    executor_elapsed_ms: int,
) -> LocalFacts:
    """`steps` is `{name: StepEvidence}` from `local_verify_client.parse_answer` (statuses already
    from the closed set); a step that is not `ok` contributes nothing."""

    def data_of(name: str) -> Any:
        step = steps.get(name)
        return step.data if step is not None and step.status == "ok" else None

    docker = data_of("docker")
    return LocalFacts(
        containers=parse_containers(docker),
        host_now=_host_now(docker.get("now")) if isinstance(docker, dict) else None,
        published_ports=parse_published_ports(data_of("ports")),
        inspector_lib_sha256=parse_inspector_digest(data_of("inspector")),
        capabilities=frozenset(capabilities),
        round_trip_ms=round_trip_ms,
        executor_elapsed_ms=executor_elapsed_ms,
        steps={name: step.status for name, step in steps.items()},
    )
