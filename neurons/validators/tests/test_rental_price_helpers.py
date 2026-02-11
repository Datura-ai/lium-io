from core.config import settings
from services.const import TEMPO, SECONDS_PER_BLOCK, FIXED_RATIO, TOTAL_BURN_EMISSION


BASE_JOB_SCORE = 1.0


def expected_executor_score(
    *,
    gpu_model: str,
    gpu_count: int,
    total_gpu_count: int,
    portion: float,
    is_rented: bool,
    rental_incentive_gpu_types: list[str],
    sysbox_runtime: bool,
    collateral_deposited: bool,
    uptime_minutes: float,
) -> float:
    if not is_rented:
        return 0.0

    if total_gpu_count == 0:
        return 0.0

    score = BASE_JOB_SCORE * portion * gpu_count / total_gpu_count

    if not sysbox_runtime:
        score *= 1 - settings.PORTION_FOR_SYSBOX

    if not collateral_deposited:
        uptime_ratio = min(1.0, uptime_minutes / settings.UPTIME_REQUIRED_MINUTES)
        penalty_reduction = settings.PORTION_FOR_UPTIME * uptime_ratio
        score *= 1 - settings.PORTION_FOR_UPTIME + penalty_reduction

    return score


def expected_emission_splits(
    *,
    unrented_gpu_counts: dict[str, int],
    rental_prices: dict[str, float],
    max_unrented_gpus: dict[str, int],
    tao_price: float,
    alpha_rate: float,
    base_gpu_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Calculate expected emission splits with base model grouping support.

    Args:
        unrented_gpu_counts: GPU counts by GPU type (e.g., {"H200": 5, "H200 NVL": 3})
        rental_prices: Rental prices per GPU type
        max_unrented_gpus: Max caps per base model (e.g., {"H200": 8})
        tao_price: TAO price in USD
        alpha_rate: Alpha rate for subnet emission
        base_gpu_map: Optional mapping from GPU variants to base models.
                     If provided, variants are grouped under base models for cap calculation.

    Returns:
        Dict with rental_share, burn_share, and mining_share
    """
    total_rental_cost = 0.0

    if base_gpu_map:
        # Group counts by base model
        base_model_counts = {}
        for gpu_type, count in unrented_gpu_counts.items():
            base_model = base_gpu_map.get(gpu_type, gpu_type)
            if base_model not in base_model_counts:
                base_model_counts[base_model] = {}
            base_model_counts[base_model][gpu_type] = count

        # Calculate total rental cost with base model grouping
        for base_model, gpu_types in base_model_counts.items():
            unrented_count = sum(gpu_types.values())
            max_cap = max_unrented_gpus.get(base_model, 0)

            if max_cap == 0:
                continue

            for gpu_type, count in gpu_types.items():
                hourly_rate = rental_prices.get(gpu_type, 0.0)
                if unrented_count > 0:
                    effective_rate = hourly_rate * min(unrented_count, max_cap) / unrented_count
                else:
                    effective_rate = 0
                total_rental_cost += count * effective_rate
    else:
        # Legacy behavior: no base model grouping
        for gpu_type, count in unrented_gpu_counts.items():
            hourly_rate = rental_prices.get(gpu_type, 0.0)
            max_cap = max_unrented_gpus.get(gpu_type, 0)

            # Handle case where GPU type has no defined cap
            if max_cap == 0:
                continue

            if count > max_cap and count > 0:
                effective_rate = hourly_rate * max_cap / count
            else:
                effective_rate = hourly_rate
            total_rental_cost += count * effective_rate

    epoch_subnet_emission = TEMPO * tao_price * alpha_rate

    if total_rental_cost == 0 or epoch_subnet_emission == 0:
        rental_share = 0.0
    else:
        rental_share = (
            total_rental_cost * TEMPO * SECONDS_PER_BLOCK / 3600 / FIXED_RATIO / epoch_subnet_emission
        )

    rental_share = min(rental_share, TOTAL_BURN_EMISSION)

    return {
        "rental_share": rental_share,
        "burn_share": TOTAL_BURN_EMISSION - rental_share,
        "mining_share": 1 - TOTAL_BURN_EMISSION,
    }


def expected_miner_rental_value(
    *,
    miner_results: list,
    rental_incentive_gpu_types: list[str],
    rental_prices: dict[str, float],
    max_unrented_gpus: dict[str, int],
    total_unrented_counts: dict[str, int],
    base_gpu_map: dict[str, str] | None = None,
) -> float:
    """Calculate expected miner rental value with base model grouping support.

    Args:
        miner_results: List of job results for a miner
        rental_incentive_gpu_types: List of eligible base models
        rental_prices: Rental prices per GPU type (variant-specific)
        max_unrented_gpus: Max caps per base model
        total_unrented_counts: Total unrented counts per GPU type (variant-specific)
        base_gpu_map: Optional mapping from GPU variants to base models

    Returns:
        Total rental value for the miner
    """
    if base_gpu_map:
        # First, group total counts by base model
        base_model_totals = {}
        for gpu_type, count in total_unrented_counts.items():
            base_model = base_gpu_map.get(gpu_type, gpu_type)
            if base_model not in base_model_totals:
                base_model_totals[base_model] = {}
            base_model_totals[base_model][gpu_type] = count

        # Calculate effective rates per GPU type
        effective_rates = {}
        for base_model, gpu_types in base_model_totals.items():
            total_count = sum(gpu_types.values())
            max_cap = max_unrented_gpus.get(base_model, 0)

            for gpu_type in gpu_types.keys():
                hourly_rate = rental_prices.get(gpu_type, 0.0)
                if total_count > 0:
                    effective_rates[gpu_type] = hourly_rate * min(total_count, max_cap) / total_count
                else:
                    effective_rates[gpu_type] = 0

        # Calculate miner's rental value using effective rates
        rental_value = 0.0
        for result in miner_results:
            gpu_model = result.gpu_model
            base_model = base_gpu_map.get(gpu_model, gpu_model)

            if result.is_rented or base_model not in rental_incentive_gpu_types:
                continue

            max_cap = max_unrented_gpus.get(base_model, 0)
            if max_cap == 0:
                continue

            effective_rate = effective_rates.get(gpu_model, 0)
            rental_value += result.gpu_count * effective_rate

        return rental_value
    else:
        # Legacy behavior: no base model grouping
        rental_value = 0.0

        for result in miner_results:
            gpu_model = result.gpu_model
            if result.is_rented or gpu_model not in rental_incentive_gpu_types:
                continue

            hourly_rate = rental_prices.get(gpu_model, 0.0)
            total_count = total_unrented_counts.get(gpu_model, 0)
            max_cap = max_unrented_gpus.get(gpu_model, 0)

            # Handle case where GPU type has no defined cap
            if max_cap == 0:
                continue

            if total_count > max_cap and total_count > 0:
                effective_rate = hourly_rate * max_cap / total_count
            else:
                effective_rate = hourly_rate

            rental_value += result.gpu_count * effective_rate

        return rental_value


def expected_final_weight(
    *,
    miner_mining_score: float,
    miner_rental_value: float,
    total_mining_score: float,
    total_rental_value: float,
    mining_share: float,
    rental_share: float,
    is_burner: bool,
    burn_share: float,
    num_burners: int,
) -> float:
    if is_burner:
        return burn_share / num_burners

    if total_mining_score > 0:
        mining_portion = mining_share * miner_mining_score / total_mining_score
    else:
        mining_portion = 0.0

    if total_rental_value > 0:
        rental_portion = rental_share * miner_rental_value / total_rental_value
    else:
        rental_portion = 0.0

    return mining_portion + rental_portion
