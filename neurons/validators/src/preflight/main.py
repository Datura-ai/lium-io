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

logger = logging.getLogger(__name__)


async def main():
    """Run all preflight checks."""

    logger.debug("Starting preflight validation checks...")

    # Initialize checks
    checks = [
        GPUCheck(),
        MatrixValidationCheck(),
        VerifyXCheck()
    ]

    # Run each check
    for check in checks:
        logger.debug(f"Running check: {check.name}")
        result = await check.run()

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
    args = parser.parse_args()

    # Setup logging based on debug flag
    # Default: WARNING (silent unless there are warnings/errors)
    # Debug: DEBUG (show all logs)
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
