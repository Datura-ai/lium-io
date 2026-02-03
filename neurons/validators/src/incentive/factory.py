"""Factory for creating incentive algorithm instances."""

from core.utils import get_logger, _m
from incentive.base import BaseIncentive
from incentive.config import IncentiveConfig
from incentive.default import DefaultIncentive
from incentive.rental_price import RentalPriceIncentive
from services.redis_service import RedisService

logger = get_logger(__name__)


class IncentiveFactory:
    """Factory for creating incentive algorithm instances.

    Uses a registry pattern to allow dynamic registration of new algorithms.
    """

    _registry: dict[str, type[BaseIncentive]] = {}

    @classmethod
    def register(cls, name: str, incentive_class: type[BaseIncentive]) -> None:
        """Register a new incentive algorithm.

        Args:
            name: Algorithm name to register
            incentive_class: Incentive class to register

        Raises:
            ValueError: If name is already registered
        """
        if name in cls._registry:
            logger.warning(
                _m(
                    "Incentive algorithm already registered, overwriting",
                    extra={"algorithm_name": name},
                )
            )
        cls._registry[name] = incentive_class
        logger.info(
            _m(
                "Registered incentive algorithm",
                extra={"algorithm_name": name},
            )
        )

    @classmethod
    def create(
        cls,
        config: IncentiveConfig,
        redis_service: RedisService,
    ) -> BaseIncentive:
        """Create an incentive algorithm instance based on configuration.

        Args:
            config: Incentive configuration
            redis_service: Redis service instance

        Returns:
            Instantiated incentive algorithm

        Raises:
            ValueError: If algorithm name not found in registry
        """
        algorithm = config.algorithm
        if algorithm not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown incentive algorithm: {algorithm}. "
                f"Available algorithms: {available}"
            )

        incentive_class = cls._registry[algorithm]
        logger.info(
            _m(
                "Creating incentive algorithm",
                extra={"algorithm_name": algorithm},
            )
        )
        return incentive_class(config, redis_service)


# Register algorithms
IncentiveFactory.register("default", DefaultIncentive)
IncentiveFactory.register("rental_price", RentalPriceIncentive)
