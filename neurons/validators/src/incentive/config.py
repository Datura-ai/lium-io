"""Configuration models for incentive algorithms.

This module defines per-GPU-type caps for the rental price incentive system.
Different GPU types have different cap values based on expected supply/demand dynamics.
High-end GPUs (B300, B200, H200) have lower caps due to scarcity, while mid-range
GPUs have higher caps to accommodate larger deployments.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

from services.const import MACHINE_PRICES


@dataclass(frozen=True)
class DefaultPrice:
    """Sentinel: resolve to MACHINE_PRICES[gpu_model] * multiplier."""
    multiplier: float = 1.0


DEFAULT_PRICE = DefaultPrice()


# Maximum unrented GPUs per GPU type before cap dilution is applied.
# Values can be either:
#   * `int` — aggregate cap across all counts of the base model (legacy shape).
#   * `dict[int, int]` — per-`gpu_count_bucket` caps. Each executor is placed in
#     the bucket matching its `gpu_splitting_min_count` when GPU splitting is
#     enabled, otherwise its `gpu_count`. Buckets are independent: the cap
#     multiplier is computed separately per `(base_model, bucket)`.
# Families migrated to per-count caps use `{1: 1, 8: 8}` — one single-GPU budget
# and one full-chassis (8×) budget, matching `GPU_COUNT_CUSTOM_PRICES` eligibility.
MAX_UNRENTED_GPUS_BY_TYPE: dict[str, int | dict[int, int]] = {
    "B300": 0,
    "B200": {1: 1, 8: 8},
    "H200": {1: 1, 8: 8},
    "H100": {1: 1, 8: 8},
    "RTX 4090": {1: 1, 8: 8},
    "A100": {1: 1, 8: 8},
    "RTX A6000": {1: 1, 8: 8},
    "RTX 3090": {1: 1, 8: 8},
    "H800": 0,
    "RTX 5090": 0,
    "RTX 4000 Ada Generation": 0,
    "RTX 6000 Ada Generation": 0,
    "RTX PRO 6000": 0,
    "L4": 0,
    "L40S": 0,
    "RTX 2000 Ada Generation": 0,
    "RTX A5000": 0,
    "RTX A4500": 0,
    "RTX A4000": 0,
    "A40": 0,
    "A30": 0,
}
# Per-(gpu_model, gpu_count) hourly prices in USD.
# Keys are full NVIDIA GPU names; values are dicts of {count_str: price_or_default}.
# Use DEFAULT_PRICE sentinel to fall back to MACHINE_PRICES.
# Price of 0 means the (gpu_model, gpu_count) combo is not eligible for rental incentive.
# Resolution order: specific GPU name > "*"; specific count > "*".
D = DEFAULT_PRICE
D12 = DefaultPrice(1.2)
GPU_COUNT_CUSTOM_PRICES: dict[str, dict[str, float | DefaultPrice]] = {
    "*": {"*": 0, "1": D, "8": D},
    # B200
    "NVIDIA B200": {"*": 0, "1": D12, "8": D12},
    # H100
    "NVIDIA H100 80GB HBM3": {"*": 0, "1": D12, "8": D12},
    "NVIDIA H100 NVL": {"*": 0, "1": D12, "8": D12},
    "NVIDIA H100 PCIe": {"*": 0, "1": D12, "8": D12},
    # A100
    "NVIDIA A100 80GB PCIe": {"*": 0, "1": D12, "8": D12},
    "NVIDIA A100-SXM4-80GB": {"*": 0, "1": D12, "8": D12},
    # RTX A6000
    "NVIDIA RTX A6000": {"*": 0, "1": D12, "8": D12},
}


BASE_GPU_MAP = {
    "NVIDIA B300 SXM6 AC": "B300",
    "NVIDIA B200": "B200",
    "NVIDIA H200": "H200",
    "NVIDIA H200 NVL": "H200",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA H100 NVL": "H100",
    "NVIDIA H100 PCIe": "H100",
    "NVIDIA H800 80GB HBM3": "H800",
    "NVIDIA H800 NVL": "H800",
    "NVIDIA H800 PCIe": "H800",
    "NVIDIA GeForce RTX 5090": "RTX 5090",
    "NVIDIA GeForce RTX 4090": "RTX 4090",
    "NVIDIA GeForce RTX 4090 D": "RTX 4090",
    "NVIDIA RTX 4000 Ada Generation": "RTX 4000 Ada Generation",
    "NVIDIA RTX 6000 Ada Generation": "RTX 6000 Ada Generation",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX PRO 6000",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": "RTX PRO 6000",
    "NVIDIA L4": "L4",
    "NVIDIA L40S": "L40S",
    "NVIDIA L40": "L40",
    "NVIDIA RTX 2000 Ada Generation": "RTX 2000 Ada Generation",
    "NVIDIA A100 80GB PCIe": "A100",
    "NVIDIA A100-SXM4-80GB": "A100",
    "NVIDIA RTX A6000": "RTX A6000",
    "NVIDIA RTX A5000": "RTX A5000",
    "NVIDIA RTX A4500": "RTX A4500",
    "NVIDIA RTX A4000": "RTX A4000",
    "NVIDIA A40": "A40",
    "NVIDIA A30": "A30",
    "NVIDIA GeForce RTX 3090": "RTX 3090",
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
        default="rental_price",
        description="Incentive algorithm to use"
    )

    rental_incentive_gpu_types: list[str] = Field(
        default=[
            gpu_type
            for gpu_type, cap in MAX_UNRENTED_GPUS_BY_TYPE.items()
            if (isinstance(cap, int) and cap > 0)
            or (isinstance(cap, dict) and any(v > 0 for v in cap.values()))
        ],
        description="GPU types eligible for rental price incentives (excludes types with 0 cap)"
    )

    max_unrented_gpus: dict[str, int | dict[int, int]] = Field(
        default=MAX_UNRENTED_GPUS_BY_TYPE,
        description=(
            "Maximum unrented GPUs per GPU type before cap dilution. "
            "int = aggregate cap across all counts (legacy). "
            "dict[int, int] = per-`gpu_count_bucket` caps."
        ),
    )

    rental_prices_per_hour: dict[str, float] = Field(
        default=MACHINE_PRICES,
        description="Default rental prices per GPU type in USD/hour"
    )

    gpu_count_custom_prices: dict[str, dict[str, float | DefaultPrice]] = Field(
        default=GPU_COUNT_CUSTOM_PRICES,
        description="Per-(gpu_model, gpu_count) hourly prices. Use DEFAULT_PRICE for MACHINE_PRICES fallback."
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

    @field_validator("max_unrented_gpus", mode="before")
    @classmethod
    def validate_max_unrented_gpus(
        cls,
        v: dict[str, int | dict[int, int]],
    ) -> dict[str, int | dict[int, int]]:
        """Validate max_unrented_gpus entries.

        Each value must be either a non-negative `int` (aggregate cap) or a
        `dict[int, int]` mapping positive gpu_count buckets to non-negative
        caps. Any other shape is rejected with a ValueError naming the offender.
        """
        for gpu_type, cap in v.items():
            if isinstance(cap, bool):
                raise ValueError(
                    f"max_unrented_gpus for {gpu_type} must be int or dict[int, int], "
                    f"got bool: {cap!r}"
                )
            if isinstance(cap, int):
                if cap < 0:
                    raise ValueError(
                        f"max_unrented_gpus for {gpu_type} must be non-negative, got: {cap}"
                    )
                continue
            if isinstance(cap, dict):
                for bucket, bucket_cap in cap.items():
                    if not isinstance(bucket, int) or isinstance(bucket, bool) or bucket <= 0:
                        raise ValueError(
                            f"max_unrented_gpus[{gpu_type!r}] bucket key must be positive int, "
                            f"got {bucket!r}"
                        )
                    if (
                        not isinstance(bucket_cap, int)
                        or isinstance(bucket_cap, bool)
                        or bucket_cap < 0
                    ):
                        raise ValueError(
                            f"max_unrented_gpus[{gpu_type!r}][{bucket}] cap must be "
                            f"non-negative int, got {bucket_cap!r}"
                        )
                continue
            raise ValueError(
                f"max_unrented_gpus for {gpu_type} must be int or dict[int, int], got: {cap!r}"
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
