from pydantic import BaseModel, ConfigDict


class SharedConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Dicts
    machine_prices: dict[str, float]
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
