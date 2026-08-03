from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from storage.models import StorageAction, StorageOperationSpec, WorkspaceMode


class WorkspaceResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalWorkspace:
    path: Path


@dataclass(frozen=True)
class DockerVolumeWorkspace:
    image: str
    volume_name: str
    path: PurePosixPath
    read_only: bool


@dataclass(frozen=True)
class DockerUserNamespaceWorkspace:
    image: str
    container_name: str
    container_id: str
    pid: int
    path: PurePosixPath
    read_only: bool


@dataclass(frozen=True)
class DockerEncryptedVolumeWorkspace:
    image: str
    volume_name: str
    path: PurePosixPath
    volume_passphrase: str


ResolvedWorkspace = (
    LocalWorkspace
    | DockerVolumeWorkspace
    | DockerUserNamespaceWorkspace
    | DockerEncryptedVolumeWorkspace
)


@dataclass(frozen=True)
class ContainerIdentity:
    container_id: str
    pid: int


class WorkspaceResolver:
    def __init__(
        self,
        docker_binary: str = "/usr/bin/docker",
        proc_root: Path = Path("/proc"),
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._docker_binary = docker_binary
        self._proc_root = proc_root
        self._environ = dict(os.environ if environ is None else environ)

    def resolve(self, operation: StorageOperationSpec) -> ResolvedWorkspace:
        if operation.workspace.mode is WorkspaceMode.PLAIN_VOLUME:
            return self._resolve_plain_volume(operation)
        if operation.workspace.mode is WorkspaceMode.ENCRYPTED_RUNNING:
            return self._resolve_encrypted_running(operation)
        return self._resolve_encrypted_bootstrap(operation)

    def _resolve_plain_volume(self, operation: StorageOperationSpec) -> DockerVolumeWorkspace:
        workspace = DockerVolumeWorkspace(
            image=self._helper_image(),
            volume_name=operation.workspace.volume_name,
            path=_map_requested_path(
                operation.workspace.requested_path,
                operation.workspace.volume_path,
                PurePosixPath("/workspace"),
            ),
            read_only=operation.action is StorageAction.BACKUP,
        )
        if operation.action is StorageAction.RESTORE:
            self._require_empty_plain_restore_target(workspace)
        return workspace

    def _resolve_encrypted_running(self, operation: StorageOperationSpec) -> DockerUserNamespaceWorkspace:
        container_name = operation.workspace.container_name
        if not container_name:
            raise WorkspaceResolutionError("encrypted workspace is missing container_name")

        identity, inspection = self._inspect_running_container(container_name)
        self._verify_cipher_volume_mount(inspection, operation.workspace.volume_name)
        self._verify_container_cgroup(identity)
        self._verify_gocryptfs_mount(identity.pid, operation.workspace.volume_path)

        requested_path = _path_within_volume(
            operation.workspace.requested_path,
            operation.workspace.volume_path,
        )
        visible_path = (
            PurePosixPath(str(self._proc_root))
            / str(identity.pid)
            / "root"
            / requested_path.relative_to("/")
        )
        image = self._helper_image()
        self._validate_encrypted_path(
            image=image,
            pid=identity.pid,
            visible_path=visible_path,
            requested_path=requested_path,
            action=operation.action,
        )

        confirmed_identity, _ = self._inspect_running_container(container_name)
        if confirmed_identity != identity:
            raise WorkspaceResolutionError("rental container changed while resolving encrypted workspace")
        return DockerUserNamespaceWorkspace(
            image=image,
            container_name=container_name,
            container_id=identity.container_id,
            pid=identity.pid,
            path=visible_path,
            read_only=operation.action is StorageAction.BACKUP,
        )

    def _resolve_encrypted_bootstrap(self, operation: StorageOperationSpec) -> DockerEncryptedVolumeWorkspace:
        passphrase = operation.workspace.volume_passphrase
        if not passphrase:
            raise WorkspaceResolutionError("encrypted bootstrap workspace is missing volume passphrase")
        return DockerEncryptedVolumeWorkspace(
            image=self._helper_image(),
            volume_name=operation.workspace.volume_name,
            path=_map_requested_path(
                operation.workspace.requested_path,
                operation.workspace.volume_path,
                PurePosixPath("/workspace"),
            ),
            volume_passphrase=passphrase,
        )

    def _helper_image(self) -> str:
        return self._environ.get("LIUM_STORAGE_HELPER_IMAGE") or self._current_executor_image()

    def _current_executor_image(self) -> str:
        container_id = self._environ.get("HOSTNAME")
        if not container_id:
            raise WorkspaceResolutionError(
                "storage helper requires LIUM_STORAGE_HELPER_IMAGE or the current executor image"
            )
        result = self._run_docker(
            ["inspect", "-f", "{{.Image}}", container_id],
            "resolve current executor image",
        )
        image = result.stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise WorkspaceResolutionError("current executor image could not be resolved")
        return image

    def _require_empty_plain_restore_target(self, workspace: DockerVolumeWorkspace) -> None:
        check_script = (
            'target="$1"; '
            'if [ ! -e "$target" ]; then exit 0; fi; '
            'if [ ! -d "$target" ]; then exit 20; fi; '
            'if find "$target" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then exit 21; fi'
        )
        result = subprocess.run(
            [
                self._docker_binary,
                "run",
                "--rm",
                "--log-driver",
                "none",
                "-v",
                f"{workspace.volume_name}:/workspace:rw",
                "--entrypoint",
                "sh",
                workspace.image,
                "-c",
                check_script,
                "sh",
                str(workspace.path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise WorkspaceResolutionError(f"restore target must be new or empty: {workspace.path}")

    def _validate_encrypted_path(
        self,
        *,
        image: str,
        pid: int,
        visible_path: PurePosixPath,
        requested_path: PurePosixPath,
        action: StorageAction,
    ) -> None:
        if action is StorageAction.BACKUP:
            check_script = 'test -d "$1"'
            error = f"backup source is not a directory: {requested_path}"
        else:
            check_script = (
                'target="$1"; '
                'if [ ! -e "$target" ]; then exit 0; fi; '
                'if [ ! -d "$target" ]; then exit 20; fi; '
                'if find "$target" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then exit 21; fi'
            )
            error = f"restore target must be new or empty: {requested_path}"

        result = subprocess.run(
            [
                self._docker_binary,
                "run",
                "--rm",
                "--log-driver",
                "none",
                "--pid",
                "host",
                "--privileged",
                "--security-opt",
                "label=disable",
                "--read-only",
                "--network",
                "none",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=67108864",
                "--entrypoint",
                "/usr/bin/nsenter",
                image,
                "-t",
                str(pid),
                "-U",
                "--",
                "/bin/sh",
                "-c",
                check_script,
                "sh",
                str(visible_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise WorkspaceResolutionError(error)

    def _inspect_running_container(self, container_name: str) -> tuple[ContainerIdentity, Mapping[str, object]]:
        result = self._run_docker(["inspect", container_name], "inspect rental container")
        try:
            payload = json.loads(result.stdout)
            inspection = payload[0]
            state = inspection["State"]
            container_id = inspection["Id"]
            pid = state["Pid"]
            running = state["Running"]
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as error:
            raise WorkspaceResolutionError("docker inspect returned an invalid rental description") from error

        if running is not True or not isinstance(pid, int) or pid <= 0:
            raise WorkspaceResolutionError(f"rental container {container_name} is not running")
        if not isinstance(container_id, str) or not re.fullmatch(r"[0-9a-f]{64}", container_id):
            raise WorkspaceResolutionError("docker inspect returned an invalid container ID")
        if not isinstance(inspection, Mapping):
            raise WorkspaceResolutionError("docker inspect returned an invalid container object")
        return ContainerIdentity(container_id=container_id, pid=pid), inspection

    def _verify_cipher_volume_mount(self, inspection: Mapping[str, object], volume_name: str) -> None:
        mounts = inspection.get("Mounts")
        if not isinstance(mounts, Sequence):
            raise WorkspaceResolutionError("docker inspect did not include rental mounts")
        for mount in mounts:
            if not isinstance(mount, Mapping):
                continue
            if mount.get("Name") == volume_name and mount.get("Destination") == "/lium-cipher":
                return
        raise WorkspaceResolutionError(
            f"expected encrypted volume {volume_name} is not mounted at /lium-cipher"
        )

    def _verify_container_cgroup(self, identity: ContainerIdentity) -> None:
        cgroup_path = self._proc_root / str(identity.pid) / "cgroup"
        try:
            cgroup = cgroup_path.read_text()
        except OSError as error:
            raise WorkspaceResolutionError("cannot read rental process cgroup") from error
        if identity.container_id not in cgroup and identity.container_id[:12] not in cgroup:
            raise WorkspaceResolutionError("rental PID cgroup does not match the expected container")

    def _verify_gocryptfs_mount(self, pid: int, expected_mount: PurePosixPath) -> None:
        mountinfo_path = self._proc_root / str(pid) / "mountinfo"
        try:
            lines = mountinfo_path.read_text().splitlines()
        except OSError as error:
            raise WorkspaceResolutionError("cannot read rental mount information") from error

        expected = str(expected_mount)
        for line in lines:
            fields = line.split()
            if "-" not in fields or len(fields) < 10:
                continue
            separator = fields.index("-")
            mount_point = _unescape_mountinfo(fields[4])
            filesystem_type = fields[separator + 1]
            if mount_point == expected and filesystem_type == "fuse.gocryptfs":
                return
        raise WorkspaceResolutionError(f"{expected} is not a live fuse.gocryptfs mount")

    def _run_docker(self, arguments: list[str], label: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [self._docker_binary, *arguments],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            raise WorkspaceResolutionError(f"{label} failed: {message}")
        return result


def _map_requested_path(
    requested_path: PurePosixPath,
    volume_path: PurePosixPath,
    mounted_root: PurePosixPath,
) -> PurePosixPath:
    relative = _path_within_volume(requested_path, volume_path).relative_to(volume_path)
    return mounted_root / relative


def _path_within_volume(requested_path: PurePosixPath, volume_path: PurePosixPath) -> PurePosixPath:
    try:
        requested_path.relative_to(volume_path)
    except ValueError as error:
        raise WorkspaceResolutionError(
            f"requested path {requested_path} is outside volume path {volume_path}"
        ) from error
    return requested_path


def _unescape_mountinfo(value: str) -> str:
    replacements = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
    for escaped, unescaped in replacements.items():
        value = value.replace(escaped, unescaped)
    return value
