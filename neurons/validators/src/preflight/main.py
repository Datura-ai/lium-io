#!/usr/bin/env python3
"""
Preflight validation main entry point.
Runs preflight checks and outputs results.
"""

import asyncio
import json
import sys
import logging
import argparse

from preflight.checks import GPUCheck, MatrixValidationCheck, VerifyXCheck
from preflight.base import CheckStatus
from preflight.constants import (
    MEMORY_ALLOCATION_PERCENTAGE,
    MEMORY_MIN_TEST_GB,
    MEMORY_MAX_TEST_GB,
    STORAGE_MIN_AVAILABLE_GB,
    STORAGE_THROUGHPUT_TEST_GB,
    NETWORK_TIMEOUT_SECONDS,
    MAX_GPU_COUNT,
    GPU_UTILIZATION_LIMIT,
    GPU_MEMORY_UTILIZATION_LIMIT,
)

logger = logging.getLogger(__name__)


def create_default_context():
    """Create default context with all configuration."""
    return {
        'verifyx': {
            'memory_allocation_percentage': MEMORY_ALLOCATION_PERCENTAGE,
            'memory_min_test_gb': MEMORY_MIN_TEST_GB,
            'memory_max_test_gb': MEMORY_MAX_TEST_GB,
            'storage_min_available_gb': STORAGE_MIN_AVAILABLE_GB,
            'storage_throughput_test_gb': STORAGE_THROUGHPUT_TEST_GB,
            'network_timeout_seconds': NETWORK_TIMEOUT_SECONDS,
            'lib_path': '/usr/lib/libverifyx.so',
        },
        'gpu': {
            'max_gpu_count': MAX_GPU_COUNT,
            'gpu_utilization_limit': GPU_UTILIZATION_LIMIT,
            'gpu_memory_utilization_limit': GPU_MEMORY_UTILIZATION_LIMIT,
        }
    }


async def main(context):
    """Run all preflight checks.

    Args:
        context: Context dict with configuration for checks
    """

    logger.debug("Starting preflight validation checks...")
    logger.debug(f"Context: {json.dumps(context, indent=2)}")

    # Initialize checks
    checks = [
        GPUCheck(),
        MatrixValidationCheck(),
        VerifyXCheck()
    ]

    # Run each check with context
    for check in checks:
        logger.debug(f"Running check: {check.name}")
        result = await check.run(context)

        if result.status == CheckStatus.PASSED:
            logger.debug(f"✓ {result.name}: PASSED - {result.message}")
        else:  # FAILED
            logger.debug(f"✗ {result.name}: FAILED - {result.message}")
            output = {
                "passed": False,
                "message": f"{result.name}: {result.message}"
            }
            print(json.dumps(output, indent=2))
            sys.exit(1)

    # All checks passed
    output = {"passed": True}
    print(json.dumps(output, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Preflight validation checks")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # VerifyX configuration arguments
    parser.add_argument("--verifyx-memory-allocation-percentage", type=int,
                       help="Percentage of RAM to allocate for testing (default: from constants)")
    parser.add_argument("--verifyx-memory-min-test-gb", type=int,
                       help="Minimum RAM required in GB (default: from constants)")
    parser.add_argument("--verifyx-memory-max-test-gb", type=int,
                       help="Maximum RAM to test in GB (default: from constants)")
    parser.add_argument("--verifyx-storage-min-available-gb", type=int,
                       help="Minimum storage space required in GB (default: from constants)")
    parser.add_argument("--verifyx-storage-throughput-test-gb", type=int,
                       help="Size of data to test storage throughput in GB (default: from constants)")
    parser.add_argument("--verifyx-network-timeout-seconds", type=int,
                       help="Timeout for network tests in seconds (default: from constants)")
    parser.add_argument("--verifyx-lib-path", type=str,
                       help="Path to libverifyx.so (default: /usr/lib/libverifyx.so)")

    # GPU check configuration arguments
    parser.add_argument("--gpu-max-count", type=int,
                       help="Maximum allowed GPU count (default: from constants)")
    parser.add_argument("--gpu-utilization-limit", type=int,
                       help="GPU utilization limit percentage (default: from constants)")
    parser.add_argument("--gpu-memory-utilization-limit", type=int,
                       help="GPU memory utilization limit percentage (default: from constants)")

    args = parser.parse_args()

    # Setup logging based on debug flag
    # Default: WARNING (silent unless there are warnings/errors)
    # Debug: DEBUG (show all logs)
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Start with default context
    context = create_default_context()

    # Override with command line arguments if provided
    if args.verifyx_memory_allocation_percentage is not None:
        context['verifyx']['memory_allocation_percentage'] = args.verifyx_memory_allocation_percentage
    if args.verifyx_memory_min_test_gb is not None:
        context['verifyx']['memory_min_test_gb'] = args.verifyx_memory_min_test_gb
    if args.verifyx_memory_max_test_gb is not None:
        context['verifyx']['memory_max_test_gb'] = args.verifyx_memory_max_test_gb
    if args.verifyx_storage_min_available_gb is not None:
        context['verifyx']['storage_min_available_gb'] = args.verifyx_storage_min_available_gb
    if args.verifyx_storage_throughput_test_gb is not None:
        context['verifyx']['storage_throughput_test_gb'] = args.verifyx_storage_throughput_test_gb
    if args.verifyx_network_timeout_seconds is not None:
        context['verifyx']['network_timeout_seconds'] = args.verifyx_network_timeout_seconds
    if args.verifyx_lib_path is not None:
        context['verifyx']['lib_path'] = args.verifyx_lib_path

    if args.gpu_max_count is not None:
        context['gpu']['max_gpu_count'] = args.gpu_max_count
    if args.gpu_utilization_limit is not None:
        context['gpu']['gpu_utilization_limit'] = args.gpu_utilization_limit
    if args.gpu_memory_utilization_limit is not None:
        context['gpu']['gpu_memory_utilization_limit'] = args.gpu_memory_utilization_limit

    asyncio.run(main(context))
