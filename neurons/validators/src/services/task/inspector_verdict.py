"""Turn an Inspector report into a verdict the validator can act on (DAH-3275).

The sensor reports every `docker exec` / `nsenter` / memory read against a rented pod. Some of
those are ours: the validator's own liveness exec (`cat /root/.ssh/authorized_keys`), the
executor's disk metric (`df -k`), the executor's legacy volume restore (`tar`). They run from
inside the executor container (the SSH session lands there), so the sensor tags them
`nested_from:executor-…` and `host=false`; on hosts where Tetragon loses the ancestry to sshd
they still surface as findings — 607 of the 632 MALICIOUS rounds on 8 Sep were exactly that.

A finding is *platform-origin* only when BOTH hold: it came from inside the executor container
and its exec payload is one of the exact commands the platform runs. Everything else is
*provider-origin* and is what the check, the score gate and the renter event act on.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any

EXECUTOR_CONTAINER_TAG_PREFIX = "nested_from:executor"
POD_CONTAINER_PREFIX = "pod_"

# Exact exec payloads the platform runs against a rented pod. Keep this list in step with the
# code that runs them: rented_machine.py (_check_pod_running), the executor's
# hardware_service.py (disk metric) — nothing else may be added without a matching caller.
PLATFORM_EXEC_PAYLOADS = frozenset(
    {
        "cat /root/.ssh/authorized_keys",
        "df -k /",
        "df -k /lium-cipher",
    }
)
# The executor's legacy tar restore streams an archive into one directory of the pod.
PLATFORM_EXEC_PAYLOAD_PATTERNS = (
    re.compile(r"^tar --xattrs --acls -xzpf - -C /[A-Za-z0-9._/-]+$"),
)
_EXEC_OPTIONS_WITH_VALUE = {"-u", "--user", "-e", "--env", "-w", "--workdir"}
_EXEC_FLAGS = {"-i", "-t", "-it", "-ti", "-d", "--detach", "--privileged", "--interactive", "--tty"}
_SHELL_WRAPPERS = {"sh", "/bin/sh", "bash", "/bin/bash", "/usr/bin/sh", "/usr/bin/bash"}
_DOCKER_EXEC_KINDS = {"DockerExec"}

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
    if len(payload) >= 3 and payload[0] in _SHELL_WRAPPERS and payload[1] == "-c":
        payload = payload[2:]
        if len(payload) == 1:
            # the shell's -c argument is one string; normalise its whitespace
            return " ".join(payload[0].split())
    return " ".join(payload) if payload else None


def is_platform_origin(finding: dict[str, Any]) -> bool:
    if finding.get("kind") not in _DOCKER_EXEC_KINDS:
        return False
    if finding.get("host") is True:
        return False
    tags = finding.get("tags") or []
    if not any(isinstance(tag, str) and tag.startswith(EXECUTOR_CONTAINER_TAG_PREFIX) for tag in tags):
        return False
    payload = exec_payload(str(finding.get("command") or ""))
    if payload is None:
        return False
    if payload in PLATFORM_EXEC_PAYLOADS:
        return True
    return any(pattern.match(payload) for pattern in PLATFORM_EXEC_PAYLOAD_PATTERNS)


def _finding_pod_id(finding: dict[str, Any], rented_pod_ids: set[str]) -> str | None:
    container = finding.get("container")
    docker = finding.get("docker") or {}
    candidates = [container, docker.get("resolved_name") if isinstance(docker, dict) else None]
    for name in candidates:
        if isinstance(name, str) and name.startswith(POD_CONTAINER_PREFIX):
            pod_id = name[len(POD_CONTAINER_PREFIX) :]
            if pod_id in rented_pod_ids:
                return pod_id
    return None


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
    affected = sorted({pod for pod in (_finding_pod_id(f, rented) for f in provider) if pod})
    if provider and not affected:
        # the sensor did not name the pod: every renter on this host is told
        affected = sorted(rented)
    classes = sorted({str(f.get("kind") or "unknown") for f in provider})
    action = ACTION_QUARANTINE if (provider and enforce) else ACTION_NONE
    return InspectorVerdict(
        provider_findings=provider,
        platform_findings=platform,
        evidence=[canonical_sha256(f) for f in provider],
        report_sha256=canonical_sha256(report),
        classes=classes,
        affected_pod_ids=affected,
        sensor=SENSOR_ATTESTED if sensor_attested else SENSOR_UNATTESTED,
        enforce=enforce,
        action=action,
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
