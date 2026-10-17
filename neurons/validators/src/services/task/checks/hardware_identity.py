from __future__ import annotations

import asyncio
import hashlib
import json
import logging

from core.config import settings

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
)
HW_ID_READ_TIMEOUT_SECONDS = 15
# sysfs reports a 4096-byte size for its attributes, so `stat -c %F` reads "regular file"
DMI_FILE_KINDS = ("regular file", "regular empty file")
FINDINGS_NAMED = 8  # the peer writes mountinfo, so the event names at most this many
HW_ID_KEY_PREFIX = "hardware_identity"
HW_ID_TTL_SECONDS = 30 * 24 * 3600


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
    """NVML (the scrape) and the kernel's procfs list name different cards; unknown when the kernel list is unreadable."""
    return kernel is not None and set(reported) != set(kernel)


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()[:16]


def dmi_identity_drifted(previous: dict | None, gpus: list[str], dmi_identity: str) -> bool:
    """DMI identity changed between checks while the GPU set stayed: the board did not change, the overlay did."""
    return bool(
        previous
        and dmi_identity
        and previous.get("dmi")
        and previous.get("gpus") == gpus
        and previous.get("dmi") != dmi_identity
    )


class HardwareIdentityCheck:
    """A host identity the provider cannot set from inside the machine (DAH-4547).

    DMI (`/sys/class/dmi/id`) is text a script can overlay (provider 5E5GpE1b's dmi-cloak); the GPU
    UUID the kernel driver reads from the card is not. The node is flagged when DMI files are not
    served by sysfs, when NVML and the kernel name different GPUs, or when DMI identity
    (product_uuid, board_serial) changes between checks while the GPU set stays the same. The
    fingerprint (a hash of the kernel's GPU UUIDs, NVML's when the kernel list is unreadable) and
    the DMI hash are logged on every event.

    Shadow by default: a finding is a warning event and the node passes. Under
    HARDWARE_FINGERPRINT_ENFORCEMENT_ENABLED it fails the node. An unreadable host, SSH error or
    Redis outage decides nothing.
    """

    check_id = "host.validate.hardware_identity"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        reported = sorted(set(reported_gpu_uuids(ctx)))
        kernel_uuids, _ = await kernel_gpu_view(ctx)
        kernel = sorted(set(kernel_uuids)) if kernel_uuids is not None else None
        gpus = kernel or reported

        mountinfo = file_fs = values = ""
        if ctx.ssh is not None:
            try:
                res = await asyncio.wait_for(ctx.ssh.run(HW_ID_CMD), timeout=HW_ID_READ_TIMEOUT_SECONDS)
                mountinfo, _, rest = (res.stdout or "").partition(HW_ID_FS_SEPARATOR)
                file_fs, _, values = rest.partition(HW_ID_VALUES_SEPARATOR)
            except Exception as exc:
                logger.warning("hardware identity read failed: %s", repr(exc)[:500], extra=ctx.default_extra)
        overlays = dmi_overlay_findings(mountinfo, file_fs)
        # "<product_uuid>|<board_serial>|"; both unreadable reads as no identity
        dmi_identity = values.strip() if values.strip().strip("|") else ""

        previous = None
        key = f"{HW_ID_KEY_PREFIX}:{ctx.executor.uuid}"
        try:
            raw = await ctx.services.redis.get(key)
            previous = json.loads(raw) if raw else None
            if gpus and dmi_identity and not overlays:
                await ctx.services.redis.set(
                    key, json.dumps({"gpus": gpus, "dmi": dmi_identity}), ex=HW_ID_TTL_SECONDS
                )
        except Exception:
            logger.warning("hardware identity record unavailable", exc_info=True, extra=ctx.default_extra)

        findings = []
        if overlays:
            findings.append("dmi_overlaid")
        if gpu_sources_disagree(reported, kernel):
            findings.append("gpu_sources_disagree")
        if dmi_identity_drifted(previous, gpus, dmi_identity):
            findings.append("dmi_changed_under_same_gpus")

        what = {
            "gpu_fingerprint": digest(gpus) if gpus else None,
            "dmi_identity_hash": digest([dmi_identity]) if dmi_identity else None,
            "previous_dmi_identity_hash": digest([previous["dmi"]])
            if previous and previous.get("dmi")
            else None,
            "kernel_gpu_uuids": kernel,
            "reported_gpu_uuids": reported,
            "dmi_overlays": overlays,
            "findings": findings,
            "enforced": settings.HARDWARE_FINGERPRINT_ENFORCEMENT_ENABLED,
        }
        if not findings:
            return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))

        enforce = settings.HARDWARE_FINGERPRINT_ENFORCEMENT_ENABLED
        event = render_message(
            Msg.SPOOFED,
            ctx=ctx,
            check_id=self.check_id,
            impact=None if enforce else "Shadow observation only: score was NOT changed",
            what=what,
        )
        # like the ban checks: a node with a live pod keeps being verified (TenantEnforcementCheck still runs)
        return CheckResult(passed=not enforce, event=event, fatal=not has_live_rental(ctx))
