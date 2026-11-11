"""Grow a pod's loopback volume to the size the renter ordered once the node has the disk for it.

The fresh sizing (DockerService.resolve_volume_sizing) grants less than the order when the node is short of free
disk, e.g. while another renter's stopped pod still keeps its volume. The volume is labelled with the order at
create; each cleanup pass grows a live pod's volume towards it as far as free disk allows, online (xfs_growfs).
"""

import asyncio
import logging
import shlex
from dataclasses import dataclass, field

import asyncssh

from core.utils import _m
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse

logger = logging.getLogger(__name__)

ORDER_LABEL = "io.lium.volume-order-gb"
GROW_HEADROOM_BYTES = 10 * 1024**3  # resolve_volume_sizing's headroom: never fill the host's disk
_GB = 1024**3
GROW_PASS_MAX_SECONDS = (
    180  # the whole pass; the caller also keeps it under the task's own deadline
)
SSH_CALL_TIMEOUT_SECONDS = 30  # each docker read
RESIZE_TIMEOUT_SECONDS = (
    120  # the host script, killed past this on the host (a stalled fallocate or xfs_growfs)
)

# Runs in the host's mount namespace: the volume's mount (under the plugin's propagated-mount) and backing file are
# host paths. Grows only, by whole 4 KiB sectors (the v3 plugin's loop sector size), never past the order or into the
# headroom. Prints the outcome and, when it read one, the volume's filesystem size in bytes after it (what the renter
# can use, not the file's size); an unreadable filesystem size prints no size. Disk a preallocated grow took is printed
# first on its own line (`allocated <bytes>`), so a grow whose xfs_growfs then fails still reports what it used.
GROW_SCRIPT = r"""
set -eu
mp=$1 order=$2 headroom=$3 sparse=$4
dev=$(findmnt -n -o SOURCE --target "$mp")
case "$dev" in /dev/loop*) ;; *) echo not-loop; exit 0 ;; esac
file=$(losetup -n -O BACK-FILE "$dev")
# the plugin opens it from its own rootfs, where the host's / is /srv
[ -e "$file" ] || file=${file#/srv}
# one pass at a time per host (a node can sit in two pipelines at once): another pass's stale target must never
# truncate this one's grow; bounded, a pass that cannot get it retries on the next
lock=${VOLUME_GROW_LOCK:-/run/lium-volume-grow.lock}
if command -v flock >/dev/null 2>&1; then
  exec 9>"$lock"
  flock -w "${VOLUME_GROW_LOCK_WAIT:-30}" 9 || { echo locked; exit 0; }
fi
cur=$(stat -c %s "$file")
# the filesystem's size (xfs data section); empty when it cannot be read
fs_size() { s=$(xfs_info "$mp" 2>/dev/null | awk '/^data/ {for (i = 1; i <= NF; i++) {
  if ($i ~ /^bsize=/) b = substr($i, 7); if ($i ~ /^blocks=/) n = substr($i, 8) + 0 }; printf "%.0f", n * b; exit}') || s=
  [ -n "$s" ] && [ "$s" -gt 0 ] && echo "$s" || true; }
fs=$cur
if command -v xfs_growfs >/dev/null 2>&1; then
  # never the file's size in its place: it would hide a filesystem an earlier pass failed to grow
  fs=$(fs_size)
  [ -n "$fs" ] || { echo fs-size-unread; exit 0; }
  # an earlier pass extended the file but its xfs_growfs failed: finish that grow before anything else
  if [ "$fs" -lt "$cur" ]; then
    losetup -c "$dev"; xfs_growfs "$mp" >/dev/null; fs=$(fs_size)
    [ -n "$fs" ] || { echo fs-size-unread; exit 0; }
  fi
fi
if [ "$cur" -ge "$order" ]; then echo full "$fs"; exit 0; fi
command -v xfs_growfs >/dev/null 2>&1 || { echo no-xfs-growfs "$cur"; exit 0; }
# allocated, not apparent, size: a sparse file's unwritten space is already in avail (counting it twice grows the
# volume every pass); for a preallocated file the two are the same. Read on both sides of avail and take the smaller:
# a renter write in between raises the later read, a discard (fstrim punches holes in the file) lowers it, and either
# way the smaller read never counts bytes that avail also counts
alloc_bytes() { echo $(( $(stat -c %b "$file") * $(stat -c %B "$file") )); }
alloc=$(alloc_bytes)
avail=$(df -P -B1 "$(dirname "$file")" | awk 'NR==2 {print $4}')
after=$(alloc_bytes)
if [ "$after" -lt "$alloc" ]; then alloc=$after; fi
target=$((alloc + avail - headroom))
if [ "$target" -gt "$order" ]; then target=$order; fi
target=$((target / 4096 * 4096))
if [ "$target" -le "$cur" ]; then echo no-room "$fs"; exit 0; fi
if [ "$sparse" = true ]; then
  truncate -s ">$target" "$file"
else
  fallocate -o "$cur" -l $((target - cur)) "$file"
  echo allocated $((target - cur))
fi
losetup -c "$dev"
xfs_growfs "$mp" >/dev/null
echo grown "$(fs_size)"
"""


def order_labels(ordered_gb: int | None, granted_gb: int | None) -> dict[str, str]:
    """The label a volume granted short of its order carries, so a later pass can grow it; none when it got it all."""
    if ordered_gb is None or granted_gb is None or granted_gb >= ordered_gb:
        return {}
    return {ORDER_LABEL: str(ordered_gb)}


@dataclass
class GrowPass:
    # pod id -> its volume's filesystem size in GB after the pass, which the backend records
    sizes_gb: dict[str, int] = field(default_factory=dict)
    # host disk the pass's preallocated grows took, failed xfs_growfs included: the cycle's free disk is less by this
    allocated_bytes: int = 0
    # a grow script was started: it may have taken disk even when its report was lost (a timeout)
    attempted: bool = False
    # the orders of grows whose report never came back (cancelled, timed out or the call failed): each may still
    # take up to its order on the host after the pass, so the cycle counts it as taken
    unreported_bytes: int = 0


def pod_ids_at_endpoint(
    rented_data: RentedExecutorsResponse | None, address: str, port: int | str
) -> set[str]:
    """The pods the snapshot lists at this node's endpoint, under any executor id there: the only volumes a pass
    here may grow. A pod listed on another node is a retained copy on this one and never grows."""
    if not rented_data:
        return set()
    return {
        pod.pod_id
        for executor in rented_data.executors.values()
        if (executor.executor_ip_address, str(executor.executor_ip_port)) == (address, str(port))
        for pod in executor.pods
    }


async def host_free_bytes(
    ssh_client: asyncssh.SSHClientConnection, timeout_seconds: float = SSH_CALL_TIMEOUT_SECONDS
) -> int | None:
    """Free bytes on the filesystem under docker's root, where the volumes' backing files live, read through PID 1's
    root like the scrape's hard_disk (this session's own `/` is the executor container's overlay); None when unread
    or out of time."""
    if timeout_seconds <= 0:
        return None
    try:
        async with asyncio.timeout(timeout_seconds):
            result = await ssh_client.run(
                "r=$(/usr/bin/docker info -f '{{.DockerRootDir}}') && h=/proc/1/root$r && { [ -d \"$h\" ] || h=$r; } "
                "&& df -P -B1 \"$h\" | awk 'NR==2 {print $4}'",
                timeout=timeout_seconds,
            )
    except Exception:  # noqa: BLE001 - unread: the caller falls back to what the grows reported
        return None
    free = (result.stdout or "").strip()
    return int(free) if result.exit_status == 0 and free.isdigit() else None


async def credit_volumes_on_host(
    ssh_client: asyncssh.SSHClientConnection, timeout_seconds: float = SSH_CALL_TIMEOUT_SECONDS
) -> dict[str, int | None] | None:
    """The volumes the scrape counts as free disk (release_filler_cache) still on the host, each with the KB it holds
    now: an Engy cache emptied or removed since the scrape (the cleanup's sweep, a customer create's eviction) is free
    in a raw read already. A filler run volume maps to None: preallocated, it holds its scraped size while it exists.
    None when unread, out of time or any line is malformed."""
    if timeout_seconds <= 0:
        return None
    try:
        async with asyncio.timeout(timeout_seconds):
            result = await ssh_client.run(
                "d=/usr/bin/docker; f=$($d volume ls -q --filter label=io.lium.workload=filler) || exit 1;"
                ' e=$($d volume ls -q --filter name=engy_cache_) || exit 1; for v in $f; do echo "F $v"; done;'
                " for v in $e; do m=$($d volume inspect -f '{{.Mountpoint}}' \"$v\") || exit 1;"
                ' h=/proc/1/root$m; [ -d "$h" ] || h=$m; s=$(du -sk "$h") || exit 1; set -- $s; echo "E $1 $v"; done',
                timeout=timeout_seconds,
            )
    except Exception:  # noqa: BLE001 - unread: the caller gives no credit
        return None
    if result.exit_status != 0 or not isinstance(result.stdout, str):
        return None
    volumes: dict[str, int | None] = {}
    for line in result.stdout.splitlines():
        kind, _, rest = line.partition(" ")
        size, _, name = rest.partition(" ")
        if kind == "F" and rest:
            volumes[rest] = None
        elif kind == "E" and size.isdigit() and name:
            volumes[name] = int(size)
        else:
            return None
    return volumes


async def grow_short_volumes(
    ssh_client: asyncssh.SSHClientConnection,
    live_pod_ids: set[str],
    budget_seconds: float = GROW_PASS_MAX_SECONDS,
) -> GrowPass:
    """Grow each live pod's short volume as far as free disk allows. Never raises and never runs past
    `budget_seconds`: out of time, it returns what it has, and a volume not grown this pass is tried again on the
    next."""
    grow = GrowPass()
    try:
        async with asyncio.timeout(budget_seconds):
            await _grow_short_volumes(ssh_client, live_pod_ids, grow)
    except TimeoutError:
        logger.warning(
            _m(
                "volume_grow_out_of_time",
                extra={"budget_seconds": budget_seconds, "sized": len(grow.sizes_gb)},
            )
        )
    return grow


async def _grow_short_volumes(
    ssh_client: asyncssh.SSHClientConnection, live_pod_ids: set[str], grow: GrowPass
) -> None:
    try:
        listing = await ssh_client.run(
            f"/usr/bin/docker volume ls -q --filter label={ORDER_LABEL}",
            timeout=SSH_CALL_TIMEOUT_SECONDS,
        )
        names = [
            name
            for name in (listing.stdout or "").split()
            if listing.exit_status == 0 and name.removeprefix("volume_") in live_pod_ids
        ]
        if not names:
            return
        fields = f'{{{{.Name}}}}|{{{{index .Labels "{ORDER_LABEL}"}}}}|{{{{.Driver}}}}|{{{{index .Options "sparse"}}}}'
        inspect = await ssh_client.run(
            f"/usr/bin/docker volume inspect {' '.join(shlex.quote(n) for n in names)} --format {shlex.quote(fields)}",
            timeout=SSH_CALL_TIMEOUT_SECONDS,
        )
        if inspect.exit_status != 0:
            raise RuntimeError(f"docker volume inspect failed: {inspect.stderr}")
        info = await ssh_client.run(
            "/usr/bin/docker info -f '{{.DockerRootDir}}'", timeout=SSH_CALL_TIMEOUT_SECONDS
        )
        docker_root = (info.stdout or "").strip()
        if info.exit_status != 0 or not docker_root:
            raise RuntimeError(f"docker info failed: {info.stderr}")
    except Exception as exc:  # noqa: BLE001 - a later pass retries
        logger.warning(_m("volume_grow_list_failed", extra={"error": str(exc)}))
        return

    plugin_ids: dict[str, str] = {}
    for line in (inspect.stdout or "").splitlines():
        name, order_gb, driver, sparse = (line.strip().split("|") + ["", "", "", ""])[:4]
        if not order_gb.isdigit() or not driver.startswith("vloopback"):
            continue
        if driver not in plugin_ids:
            try:
                plugin = await ssh_client.run(
                    f"/usr/bin/docker plugin inspect -f '{{{{.Id}}}}' {shlex.quote(driver)}",
                    timeout=SSH_CALL_TIMEOUT_SECONDS,
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
            f"timeout -k 5 {RESIZE_TIMEOUT_SECONDS} nsenter -t 1 -m -- sh -c "
            + shlex.quote(GROW_SCRIPT)
            + " grow "
            + " ".join(map(shlex.quote, args))
        )
        grow.attempted = True
        grow.unreported_bytes += int(order_gb) * _GB  # until its report comes back (a cancel leaves it counted)
        try:
            result = await ssh_client.run(command, timeout=RESIZE_TIMEOUT_SECONDS + 15)
        except Exception as exc:  # noqa: BLE001
            logger.warning(_m("volume_grow_failed", extra={"volume_name": name, "error": str(exc)}))
            continue
        grow.unreported_bytes -= int(order_gb) * _GB
        lines = (result.stdout or "").strip().splitlines()
        allocated = [line.split()[-1] for line in lines if line.startswith("allocated ")]
        grow.allocated_bytes += sum(int(n) for n in allocated if n.isdigit())
        outcome = " ".join(line for line in lines if not line.startswith("allocated ")).strip()
        if result.exit_status != 0 or not outcome:
            # a failed script prints no size: never read one from what it printed before failing
            outcome = f"exit {result.exit_status}: {(result.stderr or '').strip()[-300:]}"
        logger.info(
            _m(
                "volume_grow",
                extra={"volume_name": name, "order_gb": int(order_gb), "outcome": outcome},
            )
        )
        size_bytes = outcome.split()[-1] if result.exit_status == 0 else ""
        if size_bytes.isdigit():
            grow.sizes_gb[name.removeprefix("volume_")] = int(size_bytes) // _GB
