from pydantic import BaseModel, ConfigDict, Field

# Default cache-template docker images (full backend DOCKER_IMAGES entries). Used as the
# packaged fallback when a shared-config payload omits the field (e.g. an older backend);
# in production the backend's DOCKER_IMAGES list is the single source of truth (DAH-2380).
# Entries stay untyped dicts (like gpu_architectures) so the backend can add or change
# metadata keys without a lium-core release; consumers read only the keys they need.
# fmt: off
DEFAULT_DOCKER_IMAGES: tuple[dict, ...] = (
    {"image": "daturaai/pytorch", "tag": "2.12.0-py3.12-cuda12.8-devel-ubuntu24.04-dind", "pytorch": 2.12, "python": 3.12, "cuda": 12.8, "ubuntu": 24.04, "size": 12022823632},
    {"image": "daturaai/pytorch", "tag": "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind", "pytorch": 2.12, "python": 3.12, "cuda": 13.0, "ubuntu": 24.04, "size": 12022823632},
)
# fmt: on


class SharedConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Dicts
    machine_prices: dict[str, float]
    # raw Vast.ai p90 $/GPU-hour per machine_prices key; partial (GPUs without market data omitted)
    machine_prices_p90: dict[str, float] = Field(default_factory=dict)
    required_deposit_amount: dict[str, float]
    gpu_architectures: dict[str, dict]
    driver_cuda_map: dict[int, float]

    # Scalars
    machine_max_price_rate: float
    machine_min_price_rate: float
    # unrented incentive cutoff multiplier: an unrented executor priced above
    # machine_prices_p90[gpu] * this rate forfeits the unrented incentive (DAH-2250)
    soft_limit_price_rate: float = Field(default=1.1)
    rental_fees_rate: float
    collateral_days: int
    collateral_contract_address: str
    bittensor_netuid: int
    volume_gb_hour_price_usd: float
    max_initial_port_count: int
    total_burn_emission: float
    require_storage_limit_supported: bool = False

    # Lists
    # default cache-template docker images, served verbatim from the backend
    # DOCKER_IMAGES list; the validator derives repo:tag refs from image/tag to
    # pre-fetch Docker Hub digests (DAH-2380)
    default_docker_images: tuple[dict, ...] = DEFAULT_DOCKER_IMAGES
