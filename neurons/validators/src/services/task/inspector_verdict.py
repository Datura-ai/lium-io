"""Turn an Inspector report into a verdict the validator can act on (DAH-3275).

The sensor reports every `docker exec` / `nsenter` / memory read against a rented pod. Many of
those are the platform's own: the liveness exec (`checks/rented_machine.py`, `cat
/root/.ssh/authorized_keys`), the executor's disk metric (`executor/services/hardware_service.py`,
`df -k <mount>`), the validator's miner jobs (`miner_jobs/restore_storage.py` `mkdir`/`chown`/`tar
… --strip-components=1`, `miner_jobs/backup_storage.py` `du`/`tar -czf`), key injection, the
sshd bootstrap and the gocryptfs setup scripts (`docker_service.py`, `sh -c '<script>'`). They
all run from inside the executor container — the validator's SSH session lands there — so the
sensor tags them `nested_from:executor-…` with `host=false`. On hosts where Tetragon lost the
ancestry to sshd they surfaced as findings: 607 of the 632 MALICIOUS rounds on 8 Sep.

Classification is by that ancestry, as `design/RENTER_DATA_PRIVACY.md` row 16 intends: a Docker
control-plane action (`DockerExec`, and the `docker cp` / `docker rm` / … the executor runs on a
pod's behalf) from inside the executor container is *platform-origin* (the sensor itself already
drops docker-policy findings whose ancestry reaches sshd or pid 1, so these are the ones it could
not trust); anything from the host, a `NamespaceEnter`, a memory read, an exec the sensor could not attribute is *provider-origin* and is
what the check, the score gate and the renter event act on. The payloads are too many and too
script-shaped for an exact allow-list to be honest, so the platform execs' payloads are recorded
in the verdict (`platform_payloads`) for the daily digest instead of gating anything. What keeps
the tag trustworthy — a provider `docker exec`-ing into the executor container to borrow its
ancestry — is the verifier's job: DAH-3278 (stack-binary ancestry gating, sysbox-fs exemption).
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

NESTED_FROM_TAG_PREFIX = "nested_from:"
POD_CONTAINER_PREFIX = "pod_"
# the pod's volume is `volume_<pod_id>` (docker_service.py create flow; the sensor's
# RENTAL_VOLUME_PREFIXES) — an OverlayFsRead on it is a finding about that pod
VOLUME_CONTAINER_PREFIX = "volume_"

_EXEC_OPTIONS_WITH_VALUE = {"-u", "--user", "-e", "--env", "-w", "--workdir"}
_EXEC_FLAGS = {"-i", "-t", "-it", "-ti", "-d", "--detach", "--privileged", "--interactive", "--tty"}
_SHELL_WRAPPERS = {"sh", "/bin/sh", "bash", "/bin/bash", "/usr/bin/sh", "/usr/bin/bash"}
_SHELL_COMMAND_FLAGS = {"-c", "-lc", "-ec", "-lec"}
# The sensor's `RuntimeInterferenceKind` (celium-gpu-verifier inspector/src/collector/analysis/
# types.rs, serde PascalCase) — the only strings a renter is shown as a finding kind. Anything else
# — the report is produced on the provider's root when the sensor is unattested — is shown as
# `unknown` and kept raw only inside the evidence. Keep in step with types.rs and
# inspector_summary/sql.py.
KNOWN_FINDING_KINDS = frozenset(
    {
        "DockerExec", "DockerAttach", "DockerCp", "DockerRun", "DockerCreate", "DockerStart", "DockerStop",
        "DockerKill", "DockerPause", "DockerRm", "DockerRestart", "DockerRename", "DockerUpdate", "DockerCommit",
        "DockerExport", "DockerPrune", "DockerInspect", "DockerPull", "DockerPush", "DockerBuild", "DockerLoad",
        "DockerImport", "DockerRmi", "DockerNetworkConnect", "DockerNetworkDisconnect", "DockerNetworkRm",
        "DockerVolumeRm", "DockerSave", "NamespaceEnter", "DockerSocketWrite", "OverlayFsRead", "OverlayFsWrite",
        "ProcFsRead", "ProcFsWrite", "ContainerChroot", "ProcessAttach", "ProcessMemoryRead", "ProcessMemoryWrite",
        "DockerVolumeMount",
    }
)
# fs.rs / mount.rs kinds: no container, the path is in `command`
_PATH_SHAPED_KINDS = frozenset({"OverlayFsRead", "OverlayFsWrite", "DockerVolumeMount"})
# The kinds the sensor's two docker policies emit (`tamper-docker-cli`, `tamper-docker-sock-write`;
# docker_cli.rs `finding_is_rental_interference`): every `Docker*` kind except the path-shaped
# `DockerVolumeMount`. The executor drives a pod's whole life through the Docker API from inside
# its own container — create, start, `docker cp` for a restore, `docker rm` when the rental ends —
# so any of these with the executor's ancestry is the platform's, not only the exec.
_DOCKER_CONTROL_PLANE_KINDS = (
    frozenset(kind for kind in KNOWN_FINDING_KINDS if kind.startswith("Docker")) - _PATH_SHAPED_KINDS
)
RENTAL_VOLUME_TAG = "rental_volume"
# the renter's pod-log entry carries at most this many evidence hashes; the finding count is the
# sensor's to choose (21,694 OverlayFsRead in one day, 8 Sep), the full list stays in the
# inspector event's `context.verdict`
_RENTER_EVIDENCE_MAX = 20
UNKNOWN_KIND = "unknown"
_PAYLOAD_PREVIEW_CHARS = 160
_PAYLOAD_PREVIEW_MAX = 20
_UNMATCHED_MAX = 20
_UNMATCHED_NAME_CHARS = 128

SENSOR_ATTESTED = "attested"
SENSOR_UNATTESTED = "unattested"
ACTION_QUARANTINE = "quarantine"
ACTION_NONE = "none"
BAN_SOURCE = "inspector_auto"
RENTER_EVENT = "provider_access_detected"


class VerdictPayload(BaseModel):
    """The verdict as the inspector event carries it to the backend (`context.verdict`).

    Wire keys: `sensor` is the attestation state (`attested` / `unattested`), `finding_kinds` the
    sensor's kinds (`KNOWN_FINDING_KINDS`, else `unknown`). `model_dump()` at the emit boundary.
    """

    provider_findings: int
    platform_findings: int
    finding_kinds: list[str]
    affected_pod_ids: list[str]
    evidence_sha256: list[str]
    report_sha256: str
    sensor: str
    enforce: bool
    action: str
    ban_source: str | None
    # the platform execs' payloads, deduplicated, for the daily digest (never gate on them)
    platform_payloads: list[str]
    # provider findings on a container that is not in the rented list: recorded, no renter told
    unmatched_containers: list[str]
    unmatched_containers_count: int


class RenterAccessEvent(BaseModel):
    """One pod-log entry the renter sees in their pod's event stream."""

    log_text: str
    log_status: str
    log_tag: str
    event: str
    pod_id: str
    when: str
    finding_kinds: list[str]
    provider_findings: int
    report_sha256: str
    evidence_sha256: list[str]
    evidence_sha256_truncated: bool
    sensor: str
    action: str


@dataclass(frozen=True)
class InspectorVerdict:
    provider_findings: list[dict[str, Any]]
    platform_findings: list[dict[str, Any]]
    evidence_sha256: list[str]
    report_sha256: str
    finding_kinds: list[str]
    affected_pod_ids: list[str]
    sensor_attestation: str
    enforce: bool
    action: str
    # the platform execs' payloads, deduplicated, for the daily digest (never gate on them)
    platform_payloads: list[str] = field(default_factory=list)
    # provider findings on a container that is not in the rented list: recorded, no renter told
    unmatched_containers: list[str] = field(default_factory=list)
    unmatched_containers_count: int = 0

    @property
    def provider_origin(self) -> bool:
        return bool(self.provider_findings)

    def as_payload(self) -> VerdictPayload:
        return VerdictPayload(
            provider_findings=len(self.provider_findings),
            platform_findings=len(self.platform_findings),
            finding_kinds=self.finding_kinds,
            affected_pod_ids=self.affected_pod_ids,
            evidence_sha256=self.evidence_sha256,
            report_sha256=self.report_sha256,
            sensor=self.sensor_attestation,
            enforce=self.enforce,
            action=self.action,
            ban_source=BAN_SOURCE if self.action == ACTION_QUARANTINE else None,
            platform_payloads=self.platform_payloads,
            unmatched_containers=self.unmatched_containers,
            unmatched_containers_count=self.unmatched_containers_count,
        )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def exec_payload(command: str) -> str | None:
    """The command a `docker exec` ran inside the target container, or None if not an exec.

    `docker exec -u 0 -i pod_x sh -c 'cat /root/.ssh/authorized_keys'` → the cat; a bare
    `docker exec pod_x df -k /` → the df. Options and a `sh -c` wrapper are stripped; anything
    that does not parse is not a platform exec.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if "exec" not in tokens:
        return None
    index = tokens.index("exec") + 1
    while index < len(tokens) and tokens[index].startswith("-"):
        if tokens[index] in _EXEC_OPTIONS_WITH_VALUE:
            index += 2
        elif tokens[index] in _EXEC_FLAGS or "=" in tokens[index]:
            index += 1
        else:
            return None
    if index >= len(tokens) or not tokens[index].startswith(POD_CONTAINER_PREFIX):
        return None
    payload = tokens[index + 1 :]
    if len(payload) >= 3 and payload[0] in _SHELL_WRAPPERS and payload[1] in _SHELL_COMMAND_FLAGS:
        payload = payload[2:]
        if len(payload) == 1:
            # the shell's -c argument is one string; normalise its whitespace
            return " ".join(payload[0].split())
    return " ".join(payload) if payload else None


def is_executor_stack_container(name: str) -> bool:
    """The sensor's own rule for the executor stack (docker_cli.rs): the compose project may be
    `executor`, `executor-…` or `lium-executor-executor-1`."""
    return name == "executor" or name.startswith("executor-") or "-executor-" in name


def _kind(finding: dict[str, Any]) -> str | None:
    """`kind` as a string, or None — the report is peer-produced, a field may be any JSON type."""
    kind = finding.get("kind")
    return kind if isinstance(kind, str) else None


def _tags(finding: dict[str, Any]) -> list[str]:
    tags = finding.get("tags")
    return [tag for tag in tags if isinstance(tag, str)] if isinstance(tags, list) else []


def nested_from_executor(finding: dict[str, Any]) -> bool:
    for tag in _tags(finding):
        if tag.startswith(NESTED_FROM_TAG_PREFIX):
            if is_executor_stack_container(tag[len(NESTED_FROM_TAG_PREFIX) :]):
                return True
    return False


def is_platform_origin(finding: dict[str, Any]) -> bool:
    """A Docker control-plane action the sensor traced to inside the executor container
    (`host=false`, a `nested_from:<executor-stack container>` tag). The sensor already drops
    docker-policy findings whose ancestry reaches sshd or the container's pid 1, so every such
    finding is one whose ancestry it could not trust; until DAH-3278 hardens that on the verifier,
    all of them count as the platform's. Everything else — an nsenter, a memory or overlayfs read,
    anything from the host — is the provider's."""
    if _kind(finding) not in _DOCKER_CONTROL_PLANE_KINDS:
        return False
    if finding.get("host") is True:
        return False
    return nested_from_executor(finding)


def finding_kind(finding: dict[str, Any]) -> str:
    """The sensor's kind, or `unknown` for anything outside its vocabulary."""
    kind = _kind(finding)
    return kind if kind in KNOWN_FINDING_KINDS else UNKNOWN_KIND


def _named_container(finding: dict[str, Any]) -> str | None:
    container = finding.get("container")
    docker = finding.get("docker") or {}
    for name in (container, docker.get("resolved_name") if isinstance(docker, dict) else None):
        if isinstance(name, str) and name:
            return name
    return None


def pod_id_of(name: str) -> str | None:
    """`pod_<id>` and `volume_<id>` both belong to pod `<id>`."""
    for prefix in (POD_CONTAINER_PREFIX, VOLUME_CONTAINER_PREFIX):
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix) :]
    return None


def volume_name_from_path(path: str) -> str | None:
    """The sensor's own rule (rental.rs `volume_name_from_path`): the segment after `volumes`
    (`/var/lib/docker/volumes/<name>/_data/…`) or after `propagated-mount` (vloopback)."""
    segments = [segment for segment in path.split("/") if segment]
    for first, second in zip(segments, segments[1:]):
        if first in ("volumes", "propagated-mount"):
            return second
    return None


def _named_resource(finding: dict[str, Any]) -> str | None:
    """What the finding is about: the container it names, or — for the path-shaped kinds the
    sensor emits with `container: None` (`OverlayFsRead`/`OverlayFsWrite` from fs.rs,
    `DockerVolumeMount` from mount.rs; the path is in `command`, tag `rental_volume`) — the
    volume named in that path."""
    name = _named_container(finding)
    if name is not None:
        return name
    if _kind(finding) in _PATH_SHAPED_KINDS or RENTAL_VOLUME_TAG in _tags(finding):
        command = finding.get("command")
        if isinstance(command, str):
            return volume_name_from_path(command)
    return None


def _platform_payloads(platform: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for finding in platform:
        payload = exec_payload(str(finding.get("command") or "")) or "(not a docker exec argv)"
        payload = payload[:_PAYLOAD_PREVIEW_CHARS]
        if payload not in seen:
            seen.append(payload)
        if len(seen) >= _PAYLOAD_PREVIEW_MAX:
            break
    return seen


def build_verdict(
    report: dict[str, Any],
    findings: list[dict[str, Any]],
    *,
    rented_pod_ids: list[str],
    sensor_attested: bool,
    enforce: bool,
) -> InspectorVerdict:
    provider: list[dict[str, Any]] = []
    platform: list[dict[str, Any]] = []
    for finding in findings:
        (platform if is_platform_origin(finding) else provider).append(finding)

    rented = set(rented_pod_ids)
    affected: set[str] = set()
    unmatched: set[str] = set()
    has_unnamed_finding = False
    for finding in provider:
        name = _named_resource(finding)
        pod_id = pod_id_of(name) if name else None
        if name is None:
            has_unnamed_finding = True
        elif pod_id in rented:
            affected.add(pod_id)
        else:
            # a pod that left or joined since the rented list was fetched, or not a pod at all:
            # recorded, but no renter is told about a container that was not theirs
            unmatched.add(name)
    if has_unnamed_finding:
        # the sensor named no container at all: every renter on this host is told
        affected |= rented
    action = ACTION_QUARANTINE if (provider and enforce) else ACTION_NONE
    names = sorted(unmatched)
    return InspectorVerdict(
        provider_findings=provider,
        platform_findings=platform,
        evidence_sha256=[canonical_sha256(f) for f in provider],
        report_sha256=canonical_sha256(report),
        finding_kinds=sorted({finding_kind(f) for f in provider}),
        affected_pod_ids=sorted(affected),
        sensor_attestation=SENSOR_ATTESTED if sensor_attested else SENSOR_UNATTESTED,
        enforce=enforce,
        action=action,
        platform_payloads=_platform_payloads(platform),
        unmatched_containers=[name[:_UNMATCHED_NAME_CHARS] for name in names[:_UNMATCHED_MAX]],
        unmatched_containers_count=len(names),
    )


def renter_access_event(
    verdict: InspectorVerdict,
    *,
    pod_id: str,
    when: str,
) -> RenterAccessEvent:
    """One pod-log entry the renter sees in their pod's event stream."""
    kinds = ", ".join(verdict.finding_kinds) or "access"
    sensor = "attested sensor" if verdict.sensor_attestation == SENSOR_ATTESTED else "unattested sensor"
    # `quarantine` only asks the backend to act; the ban itself is the backend's
    outcome = (
        " and asked to take the host off the marketplace." if verdict.action == ACTION_QUARANTINE else "."
    )
    return RenterAccessEvent(
        log_text=(
            f"Provider-side access to this pod detected ({kinds}; {sensor}). "
            f"Lium has recorded the evidence{outcome}"
        ),
        log_status="error",
        log_tag=RENTER_EVENT,
        event=RENTER_EVENT,
        pod_id=pod_id,
        when=when,
        finding_kinds=verdict.finding_kinds,
        provider_findings=len(verdict.provider_findings),
        report_sha256=verdict.report_sha256,
        evidence_sha256=verdict.evidence_sha256[:_RENTER_EVIDENCE_MAX],
        evidence_sha256_truncated=len(verdict.evidence_sha256) > _RENTER_EVIDENCE_MAX,
        sensor=verdict.sensor_attestation,
        action=verdict.action,
    )
