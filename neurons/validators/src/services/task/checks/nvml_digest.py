from __future__ import annotations

import asyncio
import logging
import re

from core.utils import _m, get_extra_info
from services.const import LIB_NVIDIA_ML_DIGESTS, LIB_NVIDIA_ML_DIGESTS_AARCH64

from ..messages import NvmlDigestMessages as Msg, render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# The validator's own read of the library the scrape hashes, picked the same way (machine_scrape.get_libnvidia_ml_path:
# the first `find /usr` hit with x86_64 or lib64 in its path, else the first hit). It proves itself complete: every
# step must exit 0 and the file's inode, size and mtime must be the same before and after hashing, so a failed read or
# a file replaced mid-read never prints the DIGEST line.
LIBNVIDIA_ML_DIGEST_CMD = (
    "p=$(find /usr -name 'libnvidia-ml.so.1' 2>/dev/null | grep -E 'x86_64|lib64' | head -n 1); "
    "[ -n \"$p\" ] || p=$(find /usr -name 'libnvidia-ml.so.1' 2>/dev/null | head -n 1); "
    "[ -n \"$p\" ] || exit 3; "
    "a=$(stat -Lc '%i %s %y' \"$p\") && m=$(md5sum < \"$p\") && s=$(sha256sum < \"$p\") "
    "&& b=$(stat -Lc '%i %s %y' \"$p\") || exit 4; "
    "[ \"$a\" = \"$b\" ] || exit 5; "
    "set -- $a; echo \"DIGEST ${m%% *}:${s%% *} $2\""
)
READ_TIMEOUT_SECONDS = 30
_DIGEST_LINE_RE = re.compile(r"DIGEST ([0-9a-f]{32}:[0-9a-f]{64}) ([1-9][0-9]*)")

class NvmlDigestCheck:
    """Detect tampered NVIDIA driver stacks by hashing libnvidia-ml.

    The old flow treated mismatched MD5 sums as proof of environment spoofing. That
    safeguard protects the marketplace from miners that LD_PRELOAD a fake NVML, so we
    carry it forward verbatim.
    """

    check_id = "gpu.validate.nvml_digest"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs
        # The shared-config map and its backend resolver cover x86_64 installers only (DAH-5053).
        is_aarch64 = specs.get("machine_arch") == "aarch64"
        if is_aarch64:
            digest_map = LIB_NVIDIA_ML_DIGESTS_AARCH64
        else:
            digest_map = ctx.config.nvml_digest_map or LIB_NVIDIA_ML_DIGESTS
        gpu_info = specs.get("gpu", {})
        driver_version = gpu_info.get("driver") or ""
        lib_digest = specs.get("md5_checksums", {}).get("libnvidia_ml", "") or ""

        expected_digest = digest_map.get(driver_version)

        if not driver_version:
            # The driver string comes from the node's own scrape: reporting none fails the check.
            return self._log_missing_driver_and_fail(ctx, lib_digest)

        if expected_digest is None:
            # Reactive verification (DAH-2451): ask the backend to resolve a driver we
            # have never seen against NVIDIA's official installer. Skip versions already
            # confirmed as spoofs so we don't re-report a known-invalid driver.
            invalid_drivers = ctx.config.nvml_invalid_drivers or []
            is_confirmed_spoof = driver_version in invalid_drivers
            if not is_confirmed_spoof and not is_aarch64:
                await ctx.services.backend.report_unknown_driver(driver_version)
            event = render_message(
                Msg.DRIVER_UNKNOWN,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "driver": driver_version,
                    "actual_md5": lib_digest,
                },
            )
            # DAH-2742: an unrecognized driver only means the digest map has not caught up
            # with a newer NVIDIA release, so it keeps its verification and waits for the
            # stale-executor sweep. A confirmed spoof is a hard fact — instant reset, like
            # DIGEST_MISMATCH.
            updates = {"clear_verified_job_info": True} if is_confirmed_spoof else {}
            return CheckResult(passed=False, event=event, updates=updates)

        # Known driver: the validator's own read decides (DAH-7050). The scrape's digest cannot tell a failed or
        # partial read from a complete read of a wrong file, so one bad scrape read reset honest nodes at once. The
        # own read proves itself complete: complete and wrong resets, complete and right passes, anything else is
        # unknown (the cycle fails, the verification stays).
        actual_digest = await self._read_digest(ctx)
        what = {
            "driver": driver_version,
            "expected_md5": expected_digest,
            "actual_md5": actual_digest,
            "scrape_md5": lib_digest,
        }
        if actual_digest == expected_digest:
            event = render_message(Msg.DIGEST_OK, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=True, event=event)
        if actual_digest is None:
            event = render_message(Msg.DIGEST_UNREAD, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=False, event=event)
        event = render_message(Msg.DIGEST_MISMATCH, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=False, event=event, updates={"clear_verified_job_info": True})

    async def _read_digest(self, ctx: Context) -> str | None:
        """The library's "<md5>:<sha256>" from a complete read; None when no complete read came back (unknown)."""
        if ctx.ssh is None:
            return None
        try:
            res = await asyncio.wait_for(ctx.ssh.run(LIBNVIDIA_ML_DIGEST_CMD), timeout=READ_TIMEOUT_SECONDS)
        except Exception as exc:
            logger.info(_m("NVML digest read failed: %s", extra=get_extra_info(ctx.default_extra)), repr(exc))
            return None
        found = _DIGEST_LINE_RE.fullmatch((res.stdout or "").strip()) if res.exit_status == 0 else None
        if found is None:
            logger.info(
                _m("NVML digest read incomplete: exit %s", extra=get_extra_info(ctx.default_extra)), res.exit_status
            )
            return None
        return found.group(1)

    def _log_missing_driver_and_fail(self, ctx: Context, lib_digest: str) -> CheckResult:
        """Log NVML_DRIVER_MISSING and return a failing result."""
        logger.warning(_m("NVML_DRIVER_MISSING", extra=get_extra_info(ctx.default_extra)))
        event = render_message(
            Msg.DRIVER_MISSING,
            ctx=ctx,
            check_id=self.check_id,
            what={"libnvidia_ml": lib_digest},
        )
        return CheckResult(passed=False, event=event)
