from __future__ import annotations

import logging

from core.utils import _m, get_extra_info
from services.const import LIB_NVIDIA_ML_DIGESTS, LIB_NVIDIA_ML_DIGESTS_AARCH64

from ..messages import NvmlDigestMessages as Msg, render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)


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
                # Driver is known but digest doesn't match - possible tampering
                event = render_message(
                    Msg.DIGEST_MISMATCH,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={
                        "driver": driver_version,
                        "expected_md5": expected_digest,
                        "actual_md5": lib_digest,
                    },
                )
                updates = {"clear_verified_job_info": True}
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
