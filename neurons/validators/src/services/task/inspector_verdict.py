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

Classification is by that ancestry, as `design/RENTER_DATA_PRIVACY.md` row 16 intends: a
`DockerExec` from inside the executor container is *platform-origin* (the sensor itself already
drops execs whose ancestry reaches sshd or pid 1, so these are the ones it could not trust); anything from the host, a
`NamespaceEnter`, a memory read, an exec the sensor could not attribute is *provider-origin* and is
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

NESTED_FROM_TAG_PREFIX = "nested_from:"
POD_CONTAINER_PREFIX = "pod_"
# the pod's volume is `volume_<pod_id>` (docker_service.py create flow; the sensor's
# RENTAL_VOLUME_PREFIXES) — an OverlayFsRead on it is a finding about that pod
VOLUME_CONTAINER_PREFIX = "volume_"

_EXEC_OPTIONS_WITH_VALUE = {"-u", "--user", "-e", "--env", "-w", "--workdir"}
_EXEC_FLAGS = {"-i", "-t", "-it", "-ti", "-d", "--detach", "--privileged", "--interactive", "--tty"}
_SHELL_WRAPPERS = {"sh", "/bin/sh", "bash", "/bin/bash", "/usr/bin/sh", "/usr/bin/bash"}
_SHELL_COMMAND_FLAGS = {"-c", "-lc", "-ec", "-lec"}
_DOCKER_EXEC_KINDS = {"DockerExec"}
_PATH_SHAPED_KINDS = {"OverlayFsRead", "OverlayFsWrite", "DockerVolumeMount"}
RENTAL_VOLUME_TAG = "rental_volume"
# the renter's pod-log entry carries at most this many evidence hashes; the finding count is the
# sensor's to choose (21,694 OverlayFsRead in one day, 8 Sep), the full list stays in the
# inspector event's `context.verdict`
_RENTER_EVIDENCE_MAX = 20
# The sensor's `RuntimeInterferenceKind` (celium-gpu-verifier inspector/src/collector/analysis/
# types.rs, serde PascalCase) — the only strings a renter is shown as a class. Anything else — the
# report is produced on the provider's root when the sensor is unattested — is shown as `unknown`
# and kept raw only inside the evidence. Keep in step with types.rs and inspector_summary/sql.py.
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


@dataclass(frozen=True)
class InspectorVerdict:
    provider_findings: list[dict[str, Any]]
    platform_findings: list[dict[str, Any]]
    evidence: list[str]
    report_sha256: str
    classes: list[str]
    affected_pod_ids: list[str]
    sensor: str
    enforce: bool
    action: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def provider_origin(self) -> bool:
        return bool(self.provider_findings)

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider_findings": len(self.provider_findings),
            "platform_findings": len(self.platform_findings),
            "classes": self.classes,
            "affected_pod_ids": self.affected_pod_ids,
            "evidence_sha256": self.evidence,
            "report_sha256": self.report_sha256,
            "sensor": self.sensor,
            "enforce": self.enforce,
            "action": self.action,
            "ban_source": BAN_SOURCE if self.action == ACTION_QUARANTINE else None,
            **self.extra,
        }


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
    """A `docker exec` the sensor traced to inside the executor container (`host=false`, a
    `nested_from:<executor-stack container>` tag). The sensor already drops execs whose ancestry
    reaches sshd or the container's pid 1, so every such finding is one whose ancestry it could
    not trust; until DAH-3278 hardens that on the verifier, all of them count as the platform's.
    Everything else is the provider's."""
    if _kind(finding) not in _DOCKER_EXEC_KINDS:
        return False
    if finding.get("host") is True:
        return False
    return nested_from_executor(finding)


def finding_class(finding: dict[str, Any]) -> str:
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
    unnamed = False
    for finding in provider:
        name = _named_resource(finding)
        pod_id = pod_id_of(name) if name else None
        if name is None:
            unnamed = True
        elif pod_id in rented:
            affected.add(pod_id)
        else:
            # a pod that left or joined since the rented list was fetched, or not a pod at all:
            # recorded, but no renter is told about a container that was not theirs
            unmatched.add(name)
    if unnamed:
        # the sensor named no container at all: every renter on this host is told
        affected |= rented
    classes = sorted({finding_class(f) for f in provider})
    action = ACTION_QUARANTINE if (provider and enforce) else ACTION_NONE
    extra: dict[str, Any] = {"platform_payloads": _platform_payloads(platform)}
    if unmatched:
        names = sorted(unmatched)
        extra["unmatched_containers"] = [name[:_UNMATCHED_NAME_CHARS] for name in names[:_UNMATCHED_MAX]]
        extra["unmatched_containers_count"] = len(names)
    return InspectorVerdict(
        provider_findings=provider,
        platform_findings=platform,
        evidence=[canonical_sha256(f) for f in provider],
        report_sha256=canonical_sha256(report),
        classes=classes,
        affected_pod_ids=sorted(affected),
        sensor=SENSOR_ATTESTED if sensor_attested else SENSOR_UNATTESTED,
        enforce=enforce,
        action=action,
        extra=extra,
    )


def renter_access_event(
    verdict: InspectorVerdict,
    *,
    pod_id: str,
    when: str,
) -> dict[str, Any]:
    """One pod-log entry the renter sees in their pod's event stream."""
    classes = ", ".join(verdict.classes) or "access"
    sensor = "attested sensor" if verdict.sensor == SENSOR_ATTESTED else "unattested sensor"
    return {
        "log_text": (
            f"Provider-side access to this pod detected ({classes}; {sensor}). "
            "Lium has recorded the evidence"
            + (" and removed the host from the marketplace." if verdict.action == ACTION_QUARANTINE else ".")
        ),
        "log_status": "error",
        "log_tag": RENTER_EVENT,
        "event": RENTER_EVENT,
        "pod_id": pod_id,
        "when": when,
        "classes": verdict.classes,
        "provider_findings": len(verdict.provider_findings),
        "report_sha256": verdict.report_sha256,
        "evidence_sha256": verdict.evidence[:_RENTER_EVIDENCE_MAX],
        "evidence_sha256_truncated": len(verdict.evidence) > _RENTER_EVIDENCE_MAX,
        "sensor": verdict.sensor,
        "action": verdict.action,
    }
