"""Configuration models for incentive algorithms."""

from pydantic import BaseModel, Field, field_validator


class IncentiveConfig(BaseModel):
    """Configuration for incentive algorithm selection and parameters.

    Attributes:
        algorithm: Algorithm name (default: "default", options: "default", "rental_price")
        eligible_gpu_types: GPU types eligible for rental incentives
        max_unrented_gpus: Maximum unrented GPUs before cap dilution
        rental_prices_per_hour: Rental prices per GPU type in USD/hour
    """

    algorithm: str = Field(
        default="default",
        description="Incentive algorithm to use"
    )

    eligible_gpu_types: list[str] = Field(
        default_factory=list,
        description="GPU types eligible for rental incentives"
    )

    max_unrented_gpus: int = Field(
        default=1000,
        description="Maximum unrented GPUs before cap dilution"
    )

    rental_prices_per_hour: dict[str, float] = Field(
        default_factory=dict,
        description="Rental prices per GPU type in USD/hour"
    )

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Validate that algorithm is one of the supported values."""
        supported = ["default", "rental_price"]
        if v not in supported:
            raise ValueError(
                f"Algorithm must be one of {supported}, got: {v}"
            )
        return v

    @field_validator("max_unrented_gpus")
    @classmethod
    def validate_max_unrented_gpus(cls, v: int) -> int:
        """Validate that max_unrented_gpus is positive."""
        if v <= 0:
            raise ValueError(
                f"max_unrented_gpus must be positive, got: {v}"
            )
        return v

    @field_validator("rental_prices_per_hour")
    @classmethod
    def validate_rental_prices(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate rental prices are non-negative."""
        for gpu_type, price in v.items():
            if price < 0:
                raise ValueError(
                    f"Rental price for {gpu_type} must be non-negative, got: {price}"
                )
        return v
