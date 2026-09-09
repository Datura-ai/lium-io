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
`DockerExec` from inside the executor container is *platform-origin*; anything from the host, a
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

EXECUTOR_CONTAINER_TAG_PREFIX = "nested_from:executor"
POD_CONTAINER_PREFIX = "pod_"

_EXEC_OPTIONS_WITH_VALUE = {"-u", "--user", "-e", "--env", "-w", "--workdir"}
_EXEC_FLAGS = {"-i", "-t", "-it", "-ti", "-d", "--detach", "--privileged", "--interactive", "--tty"}
_SHELL_WRAPPERS = {"sh", "/bin/sh", "bash", "/bin/bash", "/usr/bin/sh", "/usr/bin/bash"}
_SHELL_COMMAND_FLAGS = {"-c", "-lc", "-ec", "-lec"}
_DOCKER_EXEC_KINDS = {"DockerExec"}
# The sensor's finding classes the renter may be told about, verbatim. Anything else — the report
# is produced on the provider's root when the sensor is unattested — is shown as `unknown` and
# kept raw only inside the evidence.
KNOWN_FINDING_KINDS = frozenset(
    {"DockerExec", "NamespaceEnter", "ProcFsRead", "ProcessMemoryRead", "PtraceAttach", "FileRead", "MountAccess"}
)
UNKNOWN_KIND = "unknown"
_PAYLOAD_PREVIEW_CHARS = 160
_PAYLOAD_PREVIEW_MAX = 20

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


def is_platform_origin(finding: dict[str, Any]) -> bool:
    """A `docker exec` that the sensor traced to inside the executor container (`host=false`,
    `nested_from:executor-…`). Everything else is the provider's."""
    if finding.get("kind") not in _DOCKER_EXEC_KINDS:
        return False
    if finding.get("host") is True:
        return False
    tags = finding.get("tags") or []
    return any(isinstance(tag, str) and tag.startswith(EXECUTOR_CONTAINER_TAG_PREFIX) for tag in tags)


def finding_class(finding: dict[str, Any]) -> str:
    kind = finding.get("kind")
    return kind if isinstance(kind, str) and kind in KNOWN_FINDING_KINDS else UNKNOWN_KIND


def _named_container(finding: dict[str, Any]) -> str | None:
    container = finding.get("container")
    docker = finding.get("docker") or {}
    for name in (container, docker.get("resolved_name") if isinstance(docker, dict) else None):
        if isinstance(name, str) and name:
            return name
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
        name = _named_container(finding)
        if name is None:
            unnamed = True
        elif name.startswith(POD_CONTAINER_PREFIX) and name[len(POD_CONTAINER_PREFIX) :] in rented:
            affected.add(name[len(POD_CONTAINER_PREFIX) :])
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
        extra["unmatched_containers"] = sorted(unmatched)
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
        "evidence_sha256": verdict.evidence,
        "sensor": verdict.sensor,
        "action": verdict.action,
    }
