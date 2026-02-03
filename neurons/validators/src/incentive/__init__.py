"""Incentive module for pluggable reward calculation algorithms.

This module provides a factory pattern for creating different incentive algorithms
that can be selected via configuration.
"""

from incentive.base import BaseIncentive
from incentive.config import IncentiveConfig
from incentive.default import DefaultIncentive
from incentive.factory import IncentiveFactory

# Register default incentive algorithm
IncentiveFactory.register("default", DefaultIncentive)

__all__ = [
    "BaseIncentive",
    "IncentiveConfig",
    "IncentiveFactory",
    "DefaultIncentive",
]
