"""Grow a pod's loopback volume to the size the renter ordered once the node has the disk for it.

The fresh sizing (DockerService.resolve_volume_sizing) grants less than the order when the node is short of free
disk, e.g. while another renter's stopped pod still keeps its volume. The volume is labelled with the order at
create; each cleanup pass grows a live pod's volume towards it as far as free disk allows, online (xfs_growfs).
"""

import logging
import shlex

import asyncssh

from core.utils import _m

logger = logging.getLogger(__name__)

ORDER_LABEL = "io.lium.volume-order-gb"
GROW_HEADROOM_BYTES = 10 * 1024**3  # resolve_volume_sizing's headroom: never fill the host's disk
_GB = 1024**3

# Runs in the host's mount namespace: the volume's mount (under the plugin's propagated-mount) and backing file are
# host paths. Grows only, by whole
# 4 KiB sectors (the v3 plugin's loop sector size), never past the order or into the headroom. Prints the outcome and,
# when it read one, the volume's size in bytes after it.
GROW_SCRIPT = r"""
set -eu
mp=$1 order=$2 headroom=$3 sparse=$4
dev=$(findmnt -n -o SOURCE --target "$mp")
case "$dev" in /dev/loop*) ;; *) echo not-loop; exit 0 ;; esac
file=$(losetup -n -O BACK-FILE "$dev")
# the plugin opens it from its own rootfs, where the host's / is /srv
[ -e "$file" ] || file=${file#/srv}
cur=$(stat -c %s "$file")
if [ "$cur" -ge "$order" ]; then echo full "$cur"; exit 0; fi
command -v xfs_growfs >/dev/null 2>&1 || { echo no-xfs-growfs "$cur"; exit 0; }
avail=$(df -P -B1 "$(dirname "$file")" | awk 'NR==2 {print $4}')
# allocated, not apparent, size: a sparse file's unwritten space is already in avail (counting it twice grows the
# volume every pass); for a preallocated file the two are the same
alloc=$(( $(stat -c %b "$file") * $(stat -c %B "$file") ))
target=$((alloc + avail - headroom))
if [ "$target" -gt "$order" ]; then target=$order; fi
target=$((target / 4096 * 4096))
if [ "$target" -le "$cur" ]; then echo no-room "$cur"; exit 0; fi
if [ "$sparse" = true ]; then truncate -s "$target" "$file"; else fallocate -o "$cur" -l $((target - cur)) "$file"; fi
losetup -c "$dev"
xfs_growfs "$mp" >/dev/null
echo grown "$target"
"""


def order_labels(ordered_gb: int | None, granted_gb: int | None) -> dict[str, str]:
    """The label a volume granted short of its order carries, so a later pass can grow it; none when it got it all."""
    if ordered_gb is None or granted_gb is None or granted_gb >= ordered_gb:
        return {}
    return {ORDER_LABEL: str(ordered_gb)}


async def grow_short_volumes(
    ssh_client: asyncssh.SSHClientConnection, live_pod_ids: set[str]
) -> dict[str, int]:
    """Grow each live pod's short volume as far as free disk allows; pod id -> its volume's size in GB after the pass,
    which the backend records. Never raises: a volume not grown this pass is tried again on the next."""
    try:
        listing = await ssh_client.run(f"/usr/bin/docker volume ls -q --filter label={ORDER_LABEL}")
        names = [
            name
            for name in (listing.stdout or "").split()
            if listing.exit_status == 0 and name.removeprefix("volume_") in live_pod_ids
        ]
        if not names:
            return {}
        fields = f'{{{{.Name}}}}|{{{{index .Labels "{ORDER_LABEL}"}}}}|{{{{.Driver}}}}|{{{{index .Options "sparse"}}}}'
        inspect = await ssh_client.run(
            f"/usr/bin/docker volume inspect {' '.join(shlex.quote(n) for n in names)} --format {shlex.quote(fields)}"
        )
        if inspect.exit_status != 0:
            raise RuntimeError(f"docker volume inspect failed: {inspect.stderr}")
        info = await ssh_client.run("/usr/bin/docker info -f '{{.DockerRootDir}}'")
        docker_root = (info.stdout or "").strip()
        if info.exit_status != 0 or not docker_root:
            raise RuntimeError(f"docker info failed: {info.stderr}")
    except Exception as exc:  # noqa: BLE001 - a later pass retries
        logger.warning(_m("volume_grow_list_failed", extra={"error": str(exc)}))
        return {}

    sizes_gb: dict[str, int] = {}
    plugin_ids: dict[str, str] = {}
    for line in (inspect.stdout or "").splitlines():
        name, order_gb, driver, sparse = (line.strip().split("|") + ["", "", "", ""])[:4]
        if not order_gb.isdigit() or not driver.startswith("vloopback"):
            continue
        if driver not in plugin_ids:
            try:
                plugin = await ssh_client.run(
                    f"/usr/bin/docker plugin inspect -f '{{{{.Id}}}}' {shlex.quote(driver)}"
                )
                plugin_ids[driver] = (
                    (plugin.stdout or "").strip() if plugin.exit_status == 0 else ""
                )
            except Exception:  # noqa: BLE001 - a later pass retries
                plugin_ids[driver] = ""
        if not plugin_ids[driver]:
            logger.warning(
                _m(
                    "volume_grow_failed",
                    extra={"volume_name": name, "error": f"no plugin id for {driver}"},
                )
            )
            continue
        # docker reports the plugin-internal /mnt/<volume>; on the host the mount is under propagated-mount
        mountpoint = f"{docker_root}/plugins/{plugin_ids[driver]}/propagated-mount/{name}"
        args = [mountpoint, str(int(order_gb) * _GB), str(GROW_HEADROOM_BYTES), sparse or "false"]
        command = (
            "nsenter -t 1 -m -- sh -c "
            + shlex.quote(GROW_SCRIPT)
            + " grow "
            + " ".join(map(shlex.quote, args))
        )
        try:
            result = await ssh_client.run(command)
        except Exception as exc:  # noqa: BLE001
            logger.warning(_m("volume_grow_failed", extra={"volume_name": name, "error": str(exc)}))
            continue
        outcome = (
            result.stdout or ""
        ).strip() or f"exit {result.exit_status}: {(result.stderr or '').strip()[-300:]}"
        logger.info(
            _m(
                "volume_grow",
                extra={"volume_name": name, "order_gb": int(order_gb), "outcome": outcome},
            )
        )
        size_bytes = outcome.split()[-1]
        if size_bytes.isdigit():
            sizes_gb[name.removeprefix("volume_")] = int(size_bytes) // _GB
    return sizes_gb
