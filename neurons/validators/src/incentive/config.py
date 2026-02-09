"""Configuration models for incentive algorithms.

This module defines per-GPU-type caps for the rental price incentive system.
Different GPU types have different cap values based on expected supply/demand dynamics.
High-end GPUs (B300, B200, H200) have lower caps due to scarcity, while mid-range
GPUs have higher caps to accommodate larger deployments.
"""

from pydantic import BaseModel, Field, field_validator

from services.const import MACHINE_PRICES


# Maximum unrented GPUs per GPU type before cap dilution is applied
# High-end GPUs have lower caps (12-24) due to scarcity and high value
# Mid-tier GPUs have moderate caps (24-48) for balanced incentives
# Lower-tier GPUs have higher caps (48-96) to accommodate larger deployments
MAX_UNRENTED_GPUS_BY_TYPE = {
    "NVIDIA B300 SXM6 AC": 8,
    "NVIDIA B200": 8,
    "NVIDIA H200": 8,
    "NVIDIA H200 NVL": 8,
    "NVIDIA H100 80GB HBM3": 8,
    "NVIDIA H100 NVL": 8,
    "NVIDIA H100 PCIe": 8,
    "NVIDIA H800 80GB HBM3": 0,
    "NVIDIA H800 NVL": 0,
    "NVIDIA H800 PCIe": 0,
    "NVIDIA GeForce RTX 5090": 0,
    "NVIDIA GeForce RTX 4090": 8,
    "NVIDIA GeForce RTX 4090 D": 8,
    "NVIDIA RTX 4000 Ada Generation": 0,
    "NVIDIA RTX 6000 Ada Generation": 0,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 0,
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 0,
    "NVIDIA L4": 0,
    "NVIDIA L40S": 0,
    "NVIDIA L40": 0,
    "NVIDIA RTX 2000 Ada Generation": 0,
    "NVIDIA A100 80GB PCIe": 8,
    "NVIDIA A100-SXM4-80GB": 8,
    "NVIDIA RTX A6000": 8,
    "NVIDIA RTX A5000": 0,
    "NVIDIA RTX A4500": 0,
    "NVIDIA RTX A4000": 0,
    "NVIDIA A40": 0,
    "NVIDIA A30": 0,
    "NVIDIA GeForce RTX 3090": 0,
}


class IncentiveConfig(BaseModel):
    """Configuration for incentive algorithm selection and parameters.

    Attributes:
        algorithm: Algorithm name (default: "default", options: "default", "rental_price")
        rental_incentive_gpu_types: GPU types eligible for rental incentives
        max_unrented_gpus: Maximum unrented GPUs per GPU type before cap dilution
        rental_prices_per_hour: Rental prices per GPU type in USD/hour
    """

    algorithm: str = Field(
        default="default",
        description="Incentive algorithm to use"
    )

    rental_incentive_gpu_types: list[str] = Field(
        default=[
            gpu_type for gpu_type, cap in MAX_UNRENTED_GPUS_BY_TYPE.items() if cap > 0
        ],
        description="GPU types eligible for rental price incentives (excludes types with 0 cap)"
    )

    max_unrented_gpus: dict[str, int] = Field(
        default=MAX_UNRENTED_GPUS_BY_TYPE,
        description="Maximum unrented GPUs per GPU type before cap dilution"
    )

    rental_prices_per_hour: dict[str, float] = Field(
        default=MACHINE_PRICES,
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
    def validate_max_unrented_gpus(cls, v: dict[str, int]) -> dict[str, int]:
        """Validate that max_unrented_gpus values are non-negative."""
        # Validate all values are non-negative integers
        for gpu_type, cap in v.items():
            if cap < 0:
                raise ValueError(
                    f"max_unrented_gpus for {gpu_type} must be non-negative, got: {cap}"
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
