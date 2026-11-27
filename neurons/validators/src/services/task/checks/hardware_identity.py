from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time

from services.nvidia_devices import read_kernel_gpu_view

from ..messages import HardwareIdentityMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from .banned_provider import has_live_rental, kernel_gpu_view, reported_gpu_uuids

logger = logging.getLogger(__name__)

# /sys/class/dmi/id is a symlink to this directory; mountinfo names mounts by the resolved path.
DMI_ID_PATH = "/sys/devices/virtual/dmi/id"
# root-only (0400) attributes: a non-root read is empty, and an empty identity is never compared
DMI_IDENTITY_ATTRIBUTES = ("product_uuid", "board_serial")
DMI_FILES = " ".join(
    f"{DMI_ID_PATH}/{name}" for name in ("sys_vendor", "product_name", *DMI_IDENTITY_ATTRIBUTES)
)
HW_ID_FS_SEPARATOR = "--hw-id-fs--"
HW_ID_VALUES_SEPARATOR = "--hw-id-values--"
HW_ID_BOOT_SEPARATOR = "--hw-id-boot--"
HW_ID_BOOT_FS_SEPARATOR = "--hw-id-boot-fs--"
HW_ID_BOOT_VALUE_SEPARATOR = "--hw-id-boot-value--"
BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
# every mount point that could cover the boot id file, the file itself included
BOOT_ID_COVERING_MOUNTS = ("/proc", "/proc/sys", "/proc/sys/kernel", "/proc/sys/kernel/random", BOOT_ID_PATH)
BOOT_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# a crash-looping host adds one id per boot; at this many ids the oldest go (a return to one needs this many boots)
BOOT_HISTORY_MAX = 1000
# One round-trip, the same shape as nvidia_devices.KERNEL_GPU_VIEW_CMD: mounts on any dmi path,
# the filesystem each DMI file really lives on (`stat -f %T`: "sysfs"), its kind (no -L, so a
# symlink shows), then the identity values.
HW_ID_CMD = (
    "grep -F '/dmi' /proc/self/mountinfo 2>/dev/null; "
    f"echo {HW_ID_FS_SEPARATOR}; stat -f -c '%n %T' {DMI_FILES} 2>/dev/null; "
    f"stat -c '%n|%F' {DMI_FILES} 2>/dev/null; "
    f"echo {HW_ID_VALUES_SEPARATOR}; "
    + "; ".join(
        f"printf '%s|' \"$(head -c 128 {DMI_ID_PATH}/{name} 2>/dev/null)\"" for name in DMI_IDENTITY_ATTRIBUTES
    )
    + f"; echo; echo {HW_ID_BOOT_SEPARATOR}; awk '"
    + "||".join(f'$5=="{m}"' for m in BOOT_ID_COVERING_MOUNTS)
    + f"' /proc/self/mountinfo 2>/dev/null; echo {HW_ID_BOOT_FS_SEPARATOR}; "
    f"stat -f -c '%T' {BOOT_ID_PATH} 2>/dev/null; stat -c '%F' {BOOT_ID_PATH} 2>/dev/null; "
    f"echo {HW_ID_BOOT_VALUE_SEPARATOR}; head -c 64 {BOOT_ID_PATH} 2>/dev/null"
)
HW_ID_READ_TIMEOUT_SECONDS = 15
# sysfs reports a 4096-byte size for its attributes, so `stat -c %F` reads "regular file"
DMI_FILE_KINDS = ("regular file", "regular empty file")
FINDINGS_NAMED = 8  # the peer writes mountinfo, so the event names at most this many
HW_ID_KEY_PREFIX = "hardware_identity"
HW_ID_TTL_SECONDS = 30 * 24 * 3600
# two machines behind one node, a standing finding; the plain record overwrites itself and would pass the
# next check
HW_ID_HOLD_KEY_PREFIX = "hardware_identity_hold"
DMI_HISTORY_MAX = 64


def dmi_overlay_findings(mountinfo: str, file_fs: str) -> list[str]:
    """Mounts on a DMI path and DMI files that sysfs does not serve.

    Docker gives the container its own sysfs and mounts nothing under dmi, so any mount there is
    an overlay (a bind of a regular file over product_uuid, a tmpfs over the directory). A file
    written into an overlaid directory shows a filesystem other than sysfs in `stat -f`; a symlink
    to another file shows a kind other than a regular file in `stat %F`.
    """
    found: list[str] = []
    for line in mountinfo.splitlines():
        fields = line.split()
        # mountinfo: id parent major:minor root mount_point options [optional…] - fstype source super
        if len(fields) < 7 or "-" not in fields[6:]:
            continue
        mount_point, fstype = fields[4], fields[fields.index("-", 6) + 1]
        if mount_point.startswith("/sys/") and "/dmi" in mount_point:
            found.append(f"{mount_point} {fstype}")
    for line in file_fs.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            path, _, kind = line.partition("|")
            if kind not in DMI_FILE_KINDS:
                found.append(f"{path} {kind}")
        else:
            path, _, fstype = line.rpartition(" ")
            if fstype != "sysfs":
                found.append(f"{path} {fstype}")
    return found[:FINDINGS_NAMED]


def gpu_sources_disagree(reported: list[str], kernel: list[str] | None) -> bool:
    """NVML (the scrape) names a card the kernel's procfs list does not; unknown when the kernel list is unreadable."""
    return kernel is not None and not set(reported) <= set(kernel)


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()[:16]


def trusted_boot_id(section: str) -> str:
    """The kernel's boot id when procfs serves it, else "" (no reboot exception, nothing remembered).

    Every mount covering the file must be procfs mapping the path to the same procfs file: /proc itself, or a
    read-only bind of the matching subtree (Docker's /proc/sys). A mount over the file or a parent of another
    filesystem, or a procfs bind of a different subtree, replaces its contents.
    """
    mounts, _, rest = section.partition(HW_ID_BOOT_FS_SEPARATOR)
    file_fs, _, value = rest.partition(HW_ID_BOOT_VALUE_SEPARATOR)
    if not mounts.strip():
        return ""  # /proc is always a mount: no line means the listing failed
    for line in mounts.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 7 or "-" not in fields[6:]:
            return ""
        root, mount_point, fstype = fields[3], fields[4], fields[fields.index("-", 6) + 1]
        if mount_point not in BOOT_ID_COVERING_MOUNTS:
            return ""
        if fstype != "proc" or root.rstrip("/") + BOOT_ID_PATH[len(mount_point) :] != BOOT_ID_PATH[len("/proc") :]:
            return ""
    if [line.strip() for line in file_fs.splitlines() if line.strip()][:1] != ["proc"]:
        return ""
    if not any(line.strip() in DMI_FILE_KINDS for line in file_fs.splitlines()):
        return ""
    value = value.strip()
    return value if BOOT_ID_RE.fullmatch(value) else ""


def boot_history(previous: dict | None) -> list[str]:
    """Trusted boot ids this node was seen with; a record from before the history seeds it with its one boot id."""
    if not previous:
        return []
    return list(previous.get("boots") or ([previous["boot"]] if previous.get("boot") else []))


def dmi_history(previous: dict | None) -> list[str]:
    """Digests of the DMI identities this node was seen with; a record from before the history seeds it."""
    if not previous:
        return []
    return list(previous.get("dmis") or ([digest([previous["dmi"]])] if previous.get("dmi") else []))


def dmi_identity_drifted(previous: dict | None, gpus: list[str], dmi_identity: str, boot_id: str = "") -> bool:
    """DMI identity changed between checks while the GPU set stayed: the board did not change, the overlay did.

    A trusted boot id never seen on this node excuses the change (a hypervisor may assign a new product_uuid on a
    restart). A boot id already in the history does not, so two machines alternating behind one node are flagged
    on the return. The history always holds the previous boot id, so "not in it" also means "differs from it".
    With no history to compare against, nothing is excused.
    """
    # a record with no boot history (written before it existed) keeps the old rule: the change is flagged
    history = boot_history(previous)
    restarted = bool(boot_id and history and boot_id not in history)
    return bool(
        previous
        and dmi_identity
        and previous.get("dmi")
        and previous.get("gpus") == gpus
        and previous.get("dmi") != dmi_identity
        and not restarted
    )


class HardwareIdentityCheck:
    """A host identity the provider cannot set from inside the machine (DAH-4547).

    DMI (`/sys/class/dmi/id`) is text a script can overlay (provider 5E5GpE1b's dmi-cloak); the GPU
    UUID the kernel driver reads from the card is not. The node is flagged when DMI files are not
    served by sysfs, when NVML names a GPU the kernel does not list, or when DMI identity
    (product_uuid, board_serial) changes between checks while the GPU set stays the same. The
    fingerprint (a hash of the kernel's GPU UUIDs, NVML's when the kernel list is unreadable) and
    the DMI hash are logged on every event.

    A finding fails the node. An unreadable host, SSH error or Redis outage decides nothing. A drift back to an
    identity already seen under the same GPUs is two machines answering for one node: it holds the node for
    HW_ID_TTL_SECONDS, whatever the checks in between read (a node that switched machines for one check a few
    times a day and passed every check in between).
    """

    check_id = "host.validate.hardware_identity"
    fatal = True
    # it overwrites the stored identity before it returns: a replay would compare against what it just wrote
    replays_after_session_drop = False

    async def run(self, ctx: Context) -> CheckResult:
        reported = sorted(set(reported_gpu_uuids(ctx)))
        kernel_uuids, _ = await kernel_gpu_view(ctx)
        kernel = sorted(set(kernel_uuids)) if kernel_uuids is not None else None
        gpus = kernel or reported

        mountinfo = file_fs = values = boot_id = ""
        if ctx.ssh is not None:
            try:
                res = await asyncio.wait_for(ctx.ssh.run(HW_ID_CMD), timeout=HW_ID_READ_TIMEOUT_SECONDS)
                mountinfo, _, rest = (res.stdout or "").partition(HW_ID_FS_SEPARATOR)
                file_fs, _, values = rest.partition(HW_ID_VALUES_SEPARATOR)
                values, _, boot_section = values.partition(HW_ID_BOOT_SEPARATOR)
                boot_id = trusted_boot_id(boot_section)
            except Exception as exc:
                logger.warning("hardware identity read failed: %s", repr(exc)[:500], extra=ctx.default_extra)
        overlays = dmi_overlay_findings(mountinfo, file_fs)
        # "<product_uuid>|<board_serial>|"; both unreadable reads as no identity
        dmi_identity = values.strip() if values.strip().strip("|") else ""

        previous = hold = None
        drifted = False
        key = f"{HW_ID_KEY_PREFIX}:{ctx.executor.uuid}"
        hold_key = f"{HW_ID_HOLD_KEY_PREFIX}:{ctx.executor.uuid}"
        try:
            raw = await ctx.services.redis.get(key)
            previous = json.loads(raw) if raw else None
            raw_hold = await ctx.services.redis.get(hold_key)
            hold = json.loads(raw_hold) if raw_hold else None
            drifted = dmi_identity_drifted(previous, gpus, dmi_identity, boot_id)
            if gpus and dmi_identity and not overlays:
                boots = boot_history(previous)
                if boot_id and boot_id not in boots:
                    boots = [*boots, boot_id][-BOOT_HISTORY_MAX:]
                dmis = dmi_history(previous)
                identity = digest([dmi_identity])
                if drifted and identity in dmis and hold is None:
                    hold = {"since": int(time.time()), "dmi_identity_hashes": [dmis[-1], identity]}
                    await ctx.services.redis.set(hold_key, json.dumps(hold), ex=HW_ID_TTL_SECONDS)
                if identity not in dmis:
                    dmis = [*dmis, identity][-DMI_HISTORY_MAX:]
                await ctx.services.redis.set(
                    key,
                    json.dumps({"gpus": gpus, "dmi": dmi_identity, "boots": boots, "dmis": dmis}),
                    ex=HW_ID_TTL_SECONDS,
                )
        except Exception:
            logger.warning("hardware identity record unavailable", exc_info=True, extra=ctx.default_extra)

        host_kernel = None
        if gpu_sources_disagree(reported, kernel) and ctx.ssh is not None:
            # a container handed some of the host's cards lists only those; the host's own list is the full one
            host_uuids, _ = await read_kernel_gpu_view(ctx.ssh, host_view=True)
            host_kernel = sorted(set(host_uuids)) if host_uuids else None

        findings = []
        if overlays:
            findings.append("dmi_overlaid")
        if gpu_sources_disagree(reported, host_kernel or kernel):
            findings.append("gpu_sources_disagree")
        if drifted:
            findings.append("dmi_changed_under_same_gpus")
        if hold is not None:
            findings.append("machines_alternate_behind_node")

        what = {
            "gpu_fingerprint": digest(gpus) if gpus else None,
            "dmi_identity_hash": digest([dmi_identity]) if dmi_identity else None,
            "previous_dmi_identity_hash": digest([previous["dmi"]])
            if previous and previous.get("dmi")
            else None,
            "kernel_gpu_uuids": kernel,
            "host_kernel_gpu_uuids": host_kernel,
            "reported_gpu_uuids": reported,
            "dmi_overlays": overlays,
            "findings": findings,
            "alternation_hold": hold,
        }
        if not findings:
            return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))

        event = render_message(Msg.SPOOFED, ctx=ctx, check_id=self.check_id, what=what)
        # like the ban checks: a node with a live pod keeps being verified (TenantEnforcementCheck still runs);
        # the flag zeroes its score in calculate_scores
        return CheckResult(
            passed=False,
            event=event,
            fatal=not has_live_rental(ctx),
            updates={"hardware_identity_passed": False},
        )
