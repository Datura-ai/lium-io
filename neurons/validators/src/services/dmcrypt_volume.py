"""dm-crypt (LUKS2) pod volumes: the shell the validator runs on the executor over SSH.

A pod volume is a loop file `<DockerRootDir>/lium-crypt/<pod_id>.img` holding LUKS2, opened as
`/dev/mapper/lium-<pod_id>` with ext4 on it, and handed to the pod as a Docker `local` volume over that
device. A start without the mapping fails at the mount (fail closed): it can never fall back to an empty
directory on the host disk. The key is HKDF(VOLUME_MASTER_SECRET, pod_id) and reaches cryptsetup only
on the SSH channel's stdin, never in argv, the environment or the volume options.

Everything runs in the host's mount namespace (`nsenter -t 1 -m`, the executor runs with `pid: host`
and privileged): dockerd mounts the device in the host's /dev, and the host's e2fsprogs must make the
ext4 its own kernel and tools mount and resize. So the host needs cryptsetup and friends, which
`DMCRYPT_CAPABILITY_PROBE` checks; the executor image needs nothing new.
"""

from __future__ import annotations

import re
import shlex

DMCRYPT_METHOD = "dm-crypt"
GOCRYPTFS_METHOD = "gocryptfs"

_POD_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_DEVICE_RE = re.compile(r"/dev/mapper/lium-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")

_PATH = "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Kept identical to machine_scrape.DMCRYPT_CAPABILITY_PROBE (the scrape runs alone on the executor and
# cannot import this module; a test holds the two together).
DMCRYPT_CAPABILITY_PROBE = (
    f"{_PATH}; "
    "for t in cryptsetup dmsetup losetup udevadm mkfs.ext4 fallocate truncate; do "
    "command -v $t >/dev/null || exit 1; done; "
    "cryptsetup --help | grep -q perf-no_read_workqueue || exit 1; "
    "[ -e /dev/loop-control ] || exit 1; "
    "[ -d /sys/module/dm_crypt ] || modprobe -nq dm_crypt"
)

# $1 op, $2 backing dir, $3 pod id, then create's size and sparse flag. The key is the first stdin line.
_SCRIPT = rf"""set -eu
{_PATH}
# libdevmapper makes the node itself: its udev handshake uses SysV semaphores in the executor's IPC
# namespace, which the host's udevd never signals
export DM_DISABLE_UDEV=1
op=$1 dir=$2 pod=$3
f="$dir/$pod.img" n="lium-$pod"
loop_of() {{ losetup -j "$f" | cut -d: -f1 | head -n1; }}
mapped() {{ dmsetup info "$n" >/dev/null 2>&1; }}
# dockerd resolves the device in the host /dev, which udev fills asynchronously: wait before any start
settle() {{ udevadm settle --timeout=10 || true; test -b "/dev/mapper/$n"; }}
close() {{
  if mapped; then cryptsetup close "$n"; fi
  l=$(loop_of); [ -z "$l" ] || losetup -d "$l"
}}
teardown() {{ close; rm -f "$f"; }}
case $op in
create)
  size=$4 sparse=$5
  IFS= read -r k
  [ ! -e "$f" ] || {{ echo "backing file exists" >&2; exit 3; }}
  mkdir -p "$dir" && chmod 700 "$dir"
  trap 'teardown || true' EXIT
  # a sparse file only for a full-node rental, like the vloopback volumes: a partial one stays
  # preallocated so the fresh sizing (df + declared sizes) does not count its free space twice
  if [ "$sparse" = 1 ]; then truncate -s "$size" "$f"; d=--allow-discards; else fallocate -l "$size" "$f"; d=; fi
  l=$(losetup -f --show --direct-io=on "$f")
  printf %s "$k" | cryptsetup luksFormat -q --type luks2 --cipher aes-xts-plain64 --key-size 512 \
    --sector-size 4096 --pbkdf pbkdf2 --pbkdf-force-iterations 1000 --label "$n" --key-file - "$l"
  # --persistent keeps the flags in the header, so every later open gets them
  printf %s "$k" | cryptsetup open $d --perf-no_read_workqueue --perf-no_write_workqueue --persistent \
    --key-file - "$l" "$n"
  settle
  mkfs.ext4 -q -m 0 "/dev/mapper/$n"
  trap - EXIT ;;
open)
  IFS= read -r k
  if ! mapped; then
    [ -f "$f" ] || {{ echo "no backing file for $n" >&2; exit 4; }}
    l=$(loop_of); [ -n "$l" ] || l=$(losetup -f --show --direct-io=on "$f")
    printf %s "$k" | cryptsetup open --key-file - "$l" "$n"
  fi
  settle ;;
close)
  close ;;
delete)
  teardown ;;
*) exit 2 ;;
esac
"""


def is_dmcrypt_pod_id(pod_id: str) -> bool:
    return bool(_POD_ID_RE.fullmatch(pod_id))


def _pod_id(pod_id: str) -> str:
    if not _POD_ID_RE.fullmatch(pod_id):
        raise ValueError(f"not a pod id: {pod_id!r}")
    return pod_id


def _host_command(op: str, pod_id: str, *args: str) -> str:
    # the backing dir sits on the Docker data root's filesystem, where the vloopback volumes and the
    # fresh sizing's df are
    host = shlex.join(["nsenter", "-t", "1", "-m", "--", "sh", "-c", _SCRIPT, "lium-dmcrypt", op])
    return (
        "root=\"$(/usr/bin/docker info --format '{{.DockerRootDir}}')\" && [ -n \"$root\" ] && "
        f"{host} \"$root/lium-crypt\" {shlex.join([_pod_id(pod_id), *args])}"
    )


def dmcrypt_create_command(pod_id: str, size_gb: int, *, sparse: bool) -> str:
    """Make, open and format the volume; the key is the stdin line. A failure leaves nothing behind."""
    if size_gb <= 0:
        raise ValueError("a dm-crypt volume needs a size")
    return _host_command("create", pod_id, f"{int(size_gb)}G", "1" if sparse else "0")


def dmcrypt_open_command(pod_id: str) -> str:
    """Open the volume unless it is open (after a host reboot it is not); the key is the stdin line."""
    return _host_command("open", pod_id)


def dmcrypt_close_command(pod_id: str) -> str:
    """Close the mapping and detach the loop device, keeping the file (a stopped pod); nothing to do is
    success. A mapping still mounted refuses the close."""
    return _host_command("close", pod_id)


def dmcrypt_delete_command(pod_id: str) -> str:
    """Close the mapping, detach the loop device and remove the file; nothing to do is success. A
    mapping still mounted refuses the close and keeps its file."""
    return _host_command("delete", pod_id)


def dmcrypt_capability_command() -> str:
    return shlex.join(["nsenter", "-t", "1", "-m", "--", "sh", "-c", DMCRYPT_CAPABILITY_PROBE])


def dmcrypt_backing_bytes_command(root_var: str = '"$root"') -> str:
    """Apparent size of every backing file, for the fresh sizing next to the vloopback declared sizes;
    prints nothing when there are none."""
    return f"nsenter -t 1 -m -- du -sb --apparent-size {root_var}/lium-crypt 2>/dev/null | cut -f1"


def mounted_volume_devices_command(container_name: str) -> str:
    """The `device` option of each named volume the container mounts, one per line."""
    name = shlex.quote(container_name)
    return (
        f"for v in $(/usr/bin/docker inspect -f "
        "'{{range .Mounts}}{{if eq .Type \"volume\"}}{{.Name}} {{end}}{{end}}' "
        f"{name}); do /usr/bin/docker volume inspect -f '{{{{index .Options \"device\"}}}}' \"$v\"; done"
    )


def volume_devices_command(volume_names: list[str]) -> str:
    names = " ".join(shlex.quote(name) for name in volume_names)
    return f"/usr/bin/docker volume inspect -f '{{{{index .Options \"device\"}}}}' {names}"


def dmcrypt_pod_ids(device_lines: str) -> list[str]:
    """Pod ids of the dm-crypt volumes among `device` option lines (other volumes print `<no value>`)."""
    return [m.group(1) for line in device_lines.splitlines() if (m := _DEVICE_RE.fullmatch(line.strip()))]


def dmcrypt_device(pod_id: str) -> str:
    return f"/dev/mapper/lium-{_pod_id(pod_id)}"


def is_closed_dmcrypt_mapping_error(error_text: str) -> bool:
    """dockerd's start error for a dm-crypt volume whose mapping is closed (a host reboot)."""
    return "failed to mount local volume" in error_text and "/dev/mapper/lium-" in error_text
