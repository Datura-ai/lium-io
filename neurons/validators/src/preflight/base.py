"""Base classes for preflight validation checks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class CheckStatus(Enum):
    """Status of a validation check."""
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    status: CheckStatus
    message: str


class PreflightCheck(ABC):
    """
    Base class for all preflight validation checks.

    Each check is self-contained and runs on the local machine.
    It gathers its own data and validates it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this check."""
        pass

    @abstractmethod
    async def run(self) -> CheckResult:
        """
        Run the validation check.

        Returns:
            CheckResult with status, message, and optional details
        """
        pass
