"""GPU validation check."""

import logging

from preflight.base import PreflightCheck, CheckResult, CheckStatus
from preflight.utils import get_gpu_info
from preflight.constants import GPU_MODEL_RATES

logger = logging.getLogger(__name__)


class GPUCheck(PreflightCheck):
    """
    Validates GPU configuration and availability.

    This check:
    1. Validates GPU model is supported
    2. Validates GPU count is within limits
    3. Checks GPU utilization is low (GPUs are idle)
    4. Validates GPU UUIDs are unique
    """

    @property
    def name(self) -> str:
        return "GPU Configuration"

    async def run(self, context: dict) -> CheckResult:
        """Run the GPU validation checks."""
        try:
            # Extract gpu config from context
            gpu_config = context.get('gpu', {})
            max_gpu_count = gpu_config.get('max_gpu_count')
            gpu_utilization_limit = gpu_config.get('gpu_utilization_limit')
            gpu_memory_utilization_limit = gpu_config.get('gpu_memory_utilization_limit')

            # Get GPU information
            logger.debug("Getting GPU information...")
            gpu_info = get_gpu_info(include_utilization=True)

            if not gpu_info:
                return CheckResult(
                    name=self.name,
                    status=CheckStatus.FAILED,
                    message="No GPUs detected. Ensure NVIDIA GPUs are installed and drivers are properly configured"
                )

            gpu_model = gpu_info["gpu_model"]
            gpu_count = gpu_info["gpu_count"]
            gpu_details = gpu_info["gpu_details"]
            gpu_uuids = gpu_info["gpu_uuids"]

            logger.debug(f"GPU model: {gpu_model}, count: {gpu_count}")

            # Check 1: GPU model support
            if gpu_model not in GPU_MODEL_RATES:
                return CheckResult(
                    name=self.name,
                    status=CheckStatus.FAILED,
                    message=f"GPU model '{gpu_model}' is not supported. Supported models: {', '.join(list(GPU_MODEL_RATES.keys())[:5])}..."
                )

            # Check 2: GPU count
            if gpu_count > max_gpu_count:
                return CheckResult(
                    name=self.name,
                    status=CheckStatus.FAILED,
                    message=f"GPU count ({gpu_count}) exceeds maximum allowed ({max_gpu_count})"
                )

            if len(gpu_details) != gpu_count:
                return CheckResult(
                    name=self.name,
                    status=CheckStatus.FAILED,
                    message=f"GPU count mismatch: reported {gpu_count} but detected {len(gpu_details)} GPUs"
                )

            # Check 3: GPU utilization (must be idle)
            high_util_gpus = []
            for idx, detail in enumerate(gpu_details):
                gpu_util = detail.get("utilization", 0)
                mem_util = detail.get("memory_utilization", 0)

                if gpu_util >= gpu_utilization_limit or mem_util > gpu_memory_utilization_limit:
                    high_util_gpus.append({
                        "gpu_index": idx,
                        "gpu_utilization": gpu_util,
                        "memory_utilization": mem_util
                    })

            if high_util_gpus:
                return CheckResult(
                    name=self.name,
                    status=CheckStatus.FAILED,
                    message=f"GPUs must be idle. {len(high_util_gpus)} GPU(s) have high utilization (>={gpu_utilization_limit}%). Stop running processes"
                )

            # Check 4: GPU UUIDs (uniqueness)
            if not gpu_uuids:
                return CheckResult(
                    name=self.name,
                    status=CheckStatus.FAILED,
                    message="GPU UUIDs not detected. This is required for GPU tracking"
                )

            uuid_list = gpu_uuids.split(',')
            unique_uuids = set(uuid_list)

            if len(unique_uuids) != len(uuid_list):
                return CheckResult(
                    name=self.name,
                    status=CheckStatus.FAILED,
                    message="Duplicate GPU UUIDs detected. Each GPU must have a unique UUID"
                )

            # All checks passed
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASSED,
                message=f"GPU validation passed: {gpu_count}x {gpu_model}"
            )

        except Exception as e:
            logger.error(f"GPU validation error: {e}", exc_info=True)
            return CheckResult(
                name=self.name,
                status=CheckStatus.FAILED,
                message=f"GPU validation encountered unexpected error: {str(e)}"
            )
