from __future__ import annotations

import asyncio
import logging
import re

from core.utils import _m, get_extra_info
from services.const import LIB_NVIDIA_ML_DIGESTS, LIB_NVIDIA_ML_DIGESTS_AARCH64

from ..messages import NvmlDigestMessages as Msg, render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# The second read hashes the library the scrape hashed, picked the same way (machine_scrape.get_libnvidia_ml_path:
# the first `find /usr` hit with x86_64 or lib64 in its path, else the first hit), as "<md5>:<sha256>".
LIBNVIDIA_ML_DIGEST_CMD = (
    "p=$(find /usr -name 'libnvidia-ml.so.1' 2>/dev/null | grep -E 'x86_64|lib64' | head -n 1); "
    "[ -n \"$p\" ] || p=$(find /usr -name 'libnvidia-ml.so.1' 2>/dev/null | head -n 1); "
    "[ -n \"$p\" ] && echo \"$(md5sum < \"$p\" | cut -d' ' -f1):$(sha256sum < \"$p\" | cut -d' ' -f1)\""
)
SECOND_READ_TIMEOUT_SECONDS = 15
_DIGEST_RE = re.compile(r"[0-9a-f]{32}:[0-9a-f]{64}")


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

        if driver_version and expected_digest != lib_digest:
            # Check if driver version is unknown (not in our map)
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
            else:
                # Driver is known but digest doesn't match - possible tampering. One read is not proof: a read that
                # failed or came back partial (an all-zero digest, one that never repeats) looks the same, and the
                # reset drops a rented node at once (DAH-7050). Only a second read of its own that is also wrong
                # confirms it; one that matches or cannot be read leaves the node's verification as it is.
                second_digest = await self._read_digest_again(ctx)
                what = {
                    "driver": driver_version,
                    "expected_md5": expected_digest,
                    "actual_md5": lib_digest,
                    "second_read_md5": second_digest,
                }
                confirmed = second_digest is not None and second_digest != expected_digest
                event = render_message(
                    Msg.DIGEST_MISMATCH if confirmed else Msg.DIGEST_UNCONFIRMED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what=what,
                )
                updates = {"clear_verified_job_info": True} if confirmed else {}
            return CheckResult(
                passed=False,
                event=event,
                updates=updates,
            )

        event = render_message(
            Msg.DIGEST_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "driver": driver_version,
                "libnvidia_ml": lib_digest,
            },
        )
        return CheckResult(passed=True, event=event)

    async def _read_digest_again(self, ctx: Context) -> str | None:
        """The library's digest read by the validator itself; None when it cannot be read (unknown, never a fault)."""
        if ctx.ssh is None:
            return None
        try:
            res = await asyncio.wait_for(ctx.ssh.run(LIBNVIDIA_ML_DIGEST_CMD), timeout=SECOND_READ_TIMEOUT_SECONDS)
        except Exception as exc:
            logger.info(_m("NVML digest second read failed: %s", extra=get_extra_info(ctx.default_extra)), repr(exc))
            return None
        digest = (res.stdout or "").strip()
        return digest if res.exit_status == 0 and _DIGEST_RE.fullmatch(digest) else None

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
