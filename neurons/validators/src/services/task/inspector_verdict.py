"""Turn an Inspector report into a verdict the validator records (DAH-3275).

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
what the MALICIOUS event records, with the rented pods it names (or every rented pod when it names no
container at all) in `affected_pod_ids`; a finding on any other container is recorded under
`unmatched_containers`. The verdict is evidence only: no score, renter message or ban reads it.
The executor's own backup and restore (`executor/src/storage/restic.py`) never
reach this module: the sensor classifies the encrypted flow's `nsenter -t <pid> -U` as benign
(nsenter.rs: a `CLONE_NEWUSER`-only `setns`, judged by nstype, never by a container name) and the
unencrypted flow's `-v volume_<pod_id>:/workspace` bind as a runtime mount (mount.rs: runc /
container-stack binaries), so neither is a finding — a name-based exemption here would only let a
provider's `lium-storage-…`-named container borrow the platform's origin. The payloads are too many and too
script-shaped for an exact allow-list to be honest, so the platform execs' payloads are recorded
in the verdict (`platform_exec_commands`) for the daily digest instead of gating anything. What keeps
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
# types.rs, serde PascalCase) — the only strings the verdict publishes as a finding kind. Anything else
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
UNKNOWN_KIND = "unknown"
_PAYLOAD_PREVIEW_CHARS = 160
_PAYLOAD_PREVIEW_MAX = 20
_UNMATCHED_MAX = 20
_UNMATCHED_NAME_CHARS = 128

SENSOR_ATTESTED = "attested"
SENSOR_UNATTESTED = "unattested"


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
    # the platform execs' payloads, deduplicated, for the daily digest (never gate on them)
    platform_exec_commands: list[str]
    # provider findings on a container that is not in the rented list
    unmatched_containers: list[str]
    unmatched_containers_count: int
    # provider findings that named no container and no rental volume: each one marks every rented
    # pod on the host
    unnamed_findings: int
    # platform findings by kind, for the daily digest
    platform_kind_counts: dict[str, int]


@dataclass(frozen=True)
class InspectorVerdict:
    provider_findings: list[dict[str, Any]]
    platform_findings: list[dict[str, Any]]
    evidence_sha256: list[str]
    report_sha256: str
    finding_kinds: list[str]
    affected_pod_ids: list[str]
    sensor_attestation: str
    # the platform execs' payloads, deduplicated, for the daily digest (never gate on them)
    platform_exec_commands: list[str] = field(default_factory=list)
    # provider findings on a container that is not in the rented list
    unmatched_containers: list[str] = field(default_factory=list)
    unmatched_containers_count: int = 0
    # provider findings that named no container and no rental volume (every rented pod is marked)
    unnamed_findings: int = 0

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
            platform_exec_commands=self.platform_exec_commands,
            unmatched_containers=self.unmatched_containers,
            unmatched_containers_count=self.unmatched_containers_count,
            unnamed_findings=self.unnamed_findings,
            platform_kind_counts=_kind_counts(self.platform_findings),
        )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def docker_exec_command(command: str) -> str | None:
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
    anything from the host — is the provider's. The executor's own backup/restore nsenter and volume
    bind (restic.py) are benign at the sensor (module docstring) and never arrive here; a
    `NamespaceEnter` that does arrive is a real pid/mnt/net entry, whatever container it came from."""
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


def _platform_exec_commands(platform: list[dict[str, Any]]) -> list[str]:
    commands: list[str] = []
    for finding in platform:
        payload = docker_exec_command(str(finding.get("command") or "")) or "(not a docker exec argv)"
        payload = payload[:_PAYLOAD_PREVIEW_CHARS]
        if payload not in commands:
            commands.append(payload)
        if len(commands) >= _PAYLOAD_PREVIEW_MAX:
            break
    return commands


def _kind_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        kind = finding_kind(finding)
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def build_verdict(
    report: dict[str, Any],
    findings: list[dict[str, Any]],
    *,
    rented_pod_ids: list[str],
    sensor_attested: bool,
) -> InspectorVerdict:
    provider: list[dict[str, Any]] = []
    platform: list[dict[str, Any]] = []
    for finding in findings:
        (platform if is_platform_origin(finding) else provider).append(finding)

    rented = set(rented_pod_ids)
    affected: set[str] = set()
    unmatched: set[str] = set()
    unnamed = 0
    for finding in provider:
        name = _named_resource(finding)
        pod_id = pod_id_of(name) if name else None
        if name is None:
            unnamed += 1
        elif pod_id in rented:
            affected.add(pod_id)
        else:
            # a pod that left or joined since the rented list was fetched, or not a pod at all
            unmatched.add(name)
    if unnamed:
        # the sensor named no container at all: every rented pod on this host is affected
        affected |= rented
    names = sorted(unmatched)
    return InspectorVerdict(
        provider_findings=provider,
        platform_findings=platform,
        evidence_sha256=[canonical_sha256(f) for f in provider],
        report_sha256=canonical_sha256(report),
        finding_kinds=sorted({finding_kind(f) for f in provider}),
        affected_pod_ids=sorted(affected),
        sensor_attestation=SENSOR_ATTESTED if sensor_attested else SENSOR_UNATTESTED,
        platform_exec_commands=_platform_exec_commands(platform),
        unmatched_containers=[name[:_UNMATCHED_NAME_CHARS] for name in names[:_UNMATCHED_MAX]],
        unmatched_containers_count=len(names),
        unnamed_findings=unnamed,
    )


# Docker's copy-up of the image's home files into a freshly created, empty volume (the container
# create flow mounts `volume_<pod>` on the home dir): a write of these at the volume's top level is
# Docker populating it, not a provider writing into the rental. The sensor drops dockerd itself, so
# on 16 Oct the copy-ups arrived attributed to the host's init (pid 1, parent `<kernel>`) or to an
# unresolved process (empty binary, no parent); a host shell editing `.bashrc` has neither shape
# and counts.
_VOLUME_COPY_UP_FILES = frozenset({".bashrc", ".profile"})


def _is_volume_copy_up(finding: dict[str, Any]) -> bool:
    parent = finding.get("parent")
    writer_is_init = (
        finding.get("pid") == 1 and isinstance(parent, dict) and parent.get("binary") == "<kernel>"
    )
    writer_is_unresolved = not finding.get("binary") and not parent
    if not (writer_is_init or writer_is_unresolved):
        return False
    path = str(finding.get("command") or "")
    segments = [segment for segment in path.split("/") if segment]
    for index, segment in enumerate(segments[:-1]):
        if segment in ("volumes", "propagated-mount"):
            rest = segments[index + 2 :]
            if segment == "volumes" and rest[:1] == ["_data"]:
                rest = rest[1:]
            # shortcut: matches by name at the volume root, not by time; a later overwrite of
            # just these two files is missed — add the container's start time to the report if that matters
            return len(rest) == 1 and rest[0] in _VOLUME_COPY_UP_FILES
    return False


def _is_copy_into_container(finding: dict[str, Any]) -> bool:
    """`docker cp <src> <ctr>:<dst>` (the CLI) or `PUT …/archive` (the socket) — not a copy out."""
    command = finding.get("command")
    if not isinstance(command, str):
        return False
    if command.startswith(("PUT ", "GET ", "HEAD ")):
        return command.startswith("PUT ") and "/archive" in command
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if "cp" not in tokens:
        return False
    positional = [token for token in tokens[tokens.index("cp") + 1 :] if not token.startswith("-")]
    return len(positional) == 2 and ":" in positional[1] and ":" not in positional[0]


def host_write_findings(
    provider_findings: list[dict[str, Any]], rented_pod_ids: list[str]
) -> list[dict[str, Any]]:
    """The provider findings in which the host wrote into a rented pod (DAH-4548): a `docker cp`
    into it, an archive PUT, a write on its volume other than Docker's copy-up, a write into its
    processes' memory, a `docker update` of it. `provider_findings` already excludes the platform's own (the executor's
    `docker cp` for restores and key injection carry its ancestry; the scheduled Lium execs are
    suppressed at the sensor); anything with the executor's ancestry is dropped here too."""
    rented = set(rented_pod_ids)
    writes = []
    for finding in provider_findings:
        if nested_from_executor(finding):
            continue
        name = _named_resource(finding)
        if not name or pod_id_of(name) not in rented:
            continue
        kind = _kind(finding)
        if kind in ("DockerCp", "DockerSocketWrite"):
            hit = _is_copy_into_container(finding)
        elif kind == "OverlayFsWrite":
            hit = not _is_volume_copy_up(finding)
        else:
            # ProcFsWrite: a `/proc/<pid>/mem` write (`procfs mem`);
            # DockerUpdate: the host changing a rented pod's limits (CLI or POST …/update)
            hit = kind in ("ProcessMemoryWrite", "ProcFsWrite", "DockerUpdate")
        if hit:
            writes.append(finding)
    return writes
