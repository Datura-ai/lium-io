import os
import posixpath
import subprocess
from dataclasses import dataclass


WORKSPACE_PATH = "/workspace"
LIUM_CIPHER_MOUNT = "/lium-cipher"


def normalize_workspace_path(requested_path: str | None, volume_path: str | None) -> str:
    path = os.path.expanduser((requested_path or "").rstrip("/"))
    if not path:
        return WORKSPACE_PATH

    volume_root = os.path.expanduser((volume_path or "").rstrip("/")) or "/"
    if path == WORKSPACE_PATH or path.startswith(f"{WORKSPACE_PATH}/"):
        normalized = posixpath.normpath(path)
    elif os.path.isabs(path):
        if path == volume_root:
            normalized = WORKSPACE_PATH
        elif path.startswith(f"{volume_root}/"):
            normalized = posixpath.normpath(f"{WORKSPACE_PATH}{path[len(volume_root):]}")
        else:
            normalized = posixpath.normpath(posixpath.join(WORKSPACE_PATH, path.lstrip("/")))
    else:
        normalized = posixpath.normpath(posixpath.join(WORKSPACE_PATH, path))
    if normalized != WORKSPACE_PATH and not normalized.startswith(f"{WORKSPACE_PATH}/"):
        raise ValueError("Workspace path escapes /workspace")
    return normalized


def normalize_volume_path(requested_path: str | None, volume_path: str | None) -> str:
    path = os.path.expanduser((requested_path or "").rstrip("/"))
    volume_root = os.path.expanduser((volume_path or "").rstrip("/")) or "/"
    if not path:
        return volume_root
    if path == volume_root or path.startswith(f"{volume_root}/"):
        normalized = posixpath.normpath(path)
    elif os.path.isabs(path):
        normalized = posixpath.normpath(posixpath.join(volume_root, path.lstrip("/")))
    else:
        normalized = posixpath.normpath(posixpath.join(volume_root, path))
    if normalized != volume_root and not normalized.startswith(f"{volume_root}/"):
        raise ValueError(f"Path escapes volume root {volume_root}")
    return normalized


def require_container_running(container_name: str) -> None:
    result = subprocess.run(
        ["/usr/bin/docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise RuntimeError(f"Container {container_name} is not running; backup and restore require a running pod")


def _running_containers_for_volume(volume_name: str, *, all_containers: bool = False) -> list[str]:
    command = ["/usr/bin/docker", "ps"]
    if all_containers:
        command.append("-a")
    command.extend(
        [
            f"--filter=volume={volume_name}",
            "--format",
            "{{.Names}}",
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _container_volume_mount_destination(container_name: str, volume_name: str) -> str | None:
    inspect_format = '{{range .Mounts}}{{.Name}}\t{{.Destination}}\n{{end}}'
    result = subprocess.run(
        ["/usr/bin/docker", "inspect", "-f", inspect_format, container_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0] == volume_name:
            return parts[1]
    return None


def _volume_has_gocryptfs_conf(volume_name: str) -> bool:
    result = subprocess.run(
        [
            "/usr/bin/docker",
            "run",
            "--rm",
            "-v",
            f"{volume_name}:/probe:ro",
            "--entrypoint",
            "test",
            "daturaai/aws-cli",
            "-f",
            "/probe/gocryptfs.conf",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    raise RuntimeError(
        f"Failed to probe gocryptfs.conf on volume {volume_name} "
        f"(exit={result.returncode}, stderr={stderr!r}, stdout={stdout!r})"
    )


@dataclass(frozen=True)
class VolumeAccess:
    volume_name: str
    volume_path: str
    encrypted: bool = False
    container_name: str | None = None

    def normalized_path(self, requested_path: str | None) -> str:
        if self.encrypted:
            return normalize_volume_path(requested_path, self.volume_path)
        return normalize_workspace_path(requested_path, self.volume_path)

    def docker_run_args(self) -> list[str]:
        if self.encrypted:
            raise ValueError("Encrypted workspace commands must run inside the rental container")
        return ["-v", f"{self.volume_name}:{WORKSPACE_PATH}"]

    def docker_exec_args(self, entrypoint: str, interactive: bool = False) -> list[str]:
        if not self.encrypted or not self.container_name:
            raise ValueError("docker exec requires an encrypted rental container_name")
        command = ["/usr/bin/docker", "exec"]
        if interactive:
            command.append("-i")
        command.extend([self.container_name, entrypoint])
        return command


def detect_volume_access(volume_name: str, volume_path: str) -> VolumeAccess:
    mounted: list[tuple[str, str]] = []
    for container_name in _running_containers_for_volume(volume_name):
        destination = _container_volume_mount_destination(container_name, volume_name)
        if destination:
            mounted.append((container_name, destination))

    encrypted = [
        container_name
        for container_name, destination in mounted
        if destination == LIUM_CIPHER_MOUNT
    ]
    if len(encrypted) > 1:
        raise RuntimeError(f"Volume {volume_name} is mounted encrypted in multiple running containers")
    if encrypted:
        container_name = encrypted[0]
        require_container_running(container_name)
        return VolumeAccess(
            volume_name=volume_name,
            volume_path=volume_path,
            encrypted=True,
            container_name=container_name,
        )

    for container_name in _running_containers_for_volume(volume_name, all_containers=True):
        destination = _container_volume_mount_destination(container_name, volume_name)
        if destination == LIUM_CIPHER_MOUNT:
            raise RuntimeError(
                f"Volume {volume_name} is encrypted (gocryptfs mount at {LIUM_CIPHER_MOUNT} "
                f"on container {container_name}) but no running container mounts it at "
                f"{LIUM_CIPHER_MOUNT}; refusing plain backup/restore of ciphertext"
            )

    if _volume_has_gocryptfs_conf(volume_name):
        raise RuntimeError(
            f"Volume {volume_name} contains gocryptfs.conf but no running container "
            f"mounts it at {LIUM_CIPHER_MOUNT}; refusing plain backup/restore of ciphertext"
        )

    return VolumeAccess(volume_name=volume_name, volume_path=volume_path, encrypted=False)
