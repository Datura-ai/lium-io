from pydantic import BaseModel, ConfigDict, Field


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
    rental_fees_rate: float
    collateral_days: int
    collateral_contract_address: str
    bittensor_netuid: int
    volume_gb_hour_price_usd: float
    max_initial_port_count: int
    total_burn_emission: float
    require_storage_limit_supported: bool = False
