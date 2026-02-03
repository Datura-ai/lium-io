"""Incentive module for pluggable reward calculation algorithms.

Avoid eager imports here to prevent circular import issues during settings init.
"""

from typing import TYPE_CHECKING

from incentive.config import IncentiveConfig

if TYPE_CHECKING:
    from incentive.base import BaseIncentive
    from incentive.default import DefaultIncentive
    from incentive.factory import IncentiveFactory

__all__ = [
    "BaseIncentive",
    "IncentiveConfig",
    "IncentiveFactory",
    "DefaultIncentive",
    "register_default_incentives",
]


def register_default_incentives() -> None:
    """Register built-in incentives without importing them at module import time."""
    from incentive.factory import IncentiveFactory
    from incentive.default import DefaultIncentive

    IncentiveFactory.register("default", DefaultIncentive)


def __getattr__(name: str):
    if name == "BaseIncentive":
        from incentive.base import BaseIncentive

        return BaseIncentive
    if name == "DefaultIncentive":
        from incentive.default import DefaultIncentive

        return DefaultIncentive
    if name == "IncentiveFactory":
        from incentive.factory import IncentiveFactory

        return IncentiveFactory
    raise AttributeError(f"module 'incentive' has no attribute {name!r}")
