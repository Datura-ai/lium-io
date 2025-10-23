from __future__ import annotations

from services.const import LIB_NVIDIA_ML_DIGESTS

from ..messages import NvmlDigestMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class NvmlDigestCheck:
    """Detect tampered NVIDIA driver stacks by hashing libnvidia-ml.

    The old flow treated mismatched MD5 sums as proof of environment spoofing. That
    safeguard protects the marketplace from miners that LD_PRELOAD a fake NVML, so we
    carry it forward verbatim.
    """

    check_id = "gpu.validate.nvml_digest"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        digest_map = ctx.config.nvml_digest_map or LIB_NVIDIA_ML_DIGESTS
        specs = ctx.state.specs
        gpu_info = specs.get("gpu", {})
        driver_version = gpu_info.get("driver") or ""
        lib_digest = specs.get("md5_checksums", {}).get("libnvidia_ml", "") or ""

        if driver_version and digest_map.get(driver_version) != lib_digest:
            event = render_message(
                Msg.DIGEST_MISMATCH,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "driver": driver_version,
                    "expected_md5": digest_map.get(driver_version),
                    "actual_md5": lib_digest,
                },
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True},
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
