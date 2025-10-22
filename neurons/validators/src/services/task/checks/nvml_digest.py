from __future__ import annotations

from services.const import LIB_NVIDIA_ML_DIGESTS

from ..models import build_msg
from ..pipeline import CheckResult, Context


class NvmlDigestCheck:
    check_id = "gpu.validate.nvml_digest"
    fatal = True

    def __init__(self, *, digest_map: dict[str, str] | None = None):
        self.digest_map = digest_map or LIB_NVIDIA_ML_DIGESTS

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.specs or {}
        gpu_info = specs.get("gpu", {}) or {}
        driver_version = gpu_info.get("driver") or ""
        lib_digest = specs.get("md5_checksums", {}).get("libnvidia_ml", "") or ""

        if driver_version and self.digest_map.get(driver_version) != lib_digest:
            event = build_msg(
                event="NVML library digest mismatch",
                reason="NVML_DIGEST_MISMATCH",
                severity="error",
                category="env",
                impact="Score set to 0; previous verification cleared",
                remediation=(
                    "Reinstall the NVIDIA driver matching this version and ensure libnvidia-ml "
                    "is not tampered. Avoid LD_PRELOAD or custom NVML libraries."
                ),
                what={
                    "driver": driver_version,
                    "expected_md5": self.digest_map.get(driver_version),
                    "actual_md5": lib_digest,
                },
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True},
            )

        event = build_msg(
            event="NVML library digest verified",
            reason="NVML_DIGEST_OK",
            severity="info",
            category="env",
            impact="Proceed",
            what={
                "driver": driver_version,
                "libnvidia_ml": lib_digest,
            },
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=True, event=event)
