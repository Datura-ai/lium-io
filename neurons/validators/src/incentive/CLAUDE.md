# Incentive Module

Pluggable reward calculation system that determines how emission is distributed across miners each epoch.

## Architecture

```
BaseIncentive (ABC)              # base class with two-pass processing pipeline
  ├── DefaultIncentive           # mining-only scoring (score * gpu_portion * multipliers)
  │     └── RentalPriceIncentive # extends default: unrented eligible GPUs get rental pool instead of mining pool
  │
IncentiveFactory                 # registry pattern, creates algorithm by name from IncentiveConfig
BurnService                      # distributes burn emission across burner nodes (old/new logic)
PriceProvider                    # fetches TAO price (CoinGecko/Coinbase/CryptoCompare) + alpha rate from subtensor
IncentiveConfig                  # pydantic config: algorithm name, GPU caps, rental prices
```

## Entry Point

`core/validator.py:365` — each epoch the validator:
1. Creates incentive via `IncentiveFactory.create(config, redis, jobs_results, gpu_count_map)`
2. Calls `incentive.calculate_mining_scores()` — two-pass pipeline over all job results
3. Calls `incentive.calculate_final_weights(miners, last_block)` — returns cycle scores with burn applied
4. Accumulates cycle scores into `validator.miner_scores`

## Two-Pass Pipeline (`calculate_mining_scores`)

```
Pass 1 (_pre_process_job_result):  calculate mining_score per executor, aggregate totals
        _on_finish_pre_process:    finalize aggregated metrics (rental share calculation)
Pass 2 (_post_process_job_result): calculate incentive per executor using totals from pass 1
```

## Algorithms

### `default` — DefaultIncentive

Pure mining scoring. All emission goes to mining pool (9%) + burn pool (91%).

**Mining score formula:**
```
mining_score = score * gpu_portion * gpu_count / total_gpu_count * sysbox_multiplier * uptime_multiplier
```
- `score` — job verification score (0 or 1)
- `gpu_portion` — portion per GPU type from Redis
- `sysbox_multiplier` — 1.0 if sysbox, else (1 - PORTION_FOR_SYSBOX)
- `uptime_multiplier` — 1.0 if collateral deposited, else ramp from (1 - PORTION_FOR_UPTIME) to 1.0

**Incentive per executor:**
```
incentive = mining_share * mining_score / total_mining_score
```

### `rental_price` — RentalPriceIncentive (current default)

Extends DefaultIncentive. Unrented GPUs from eligible types get a separate rental emission pool.

**Phase 1 (pre_process):** Unrented eligible GPUs excluded from mining pool (`mining_score = 0`).
Eligibility: GPU base model in `rental_incentive_gpu_types` AND not rented AND score > 0.

**Phase 2 (on_finish_pre_process):** Calculate rental share:
```
gpu_count_multiplier = get_gpu_count_multiplier(base_model, gpu_count, config.gpu_count_multipliers)
weighted_count = gpu_count * gpu_count_multiplier          # multiplier-weighted GPU count
unrented_count = sum(weighted_count for all unrented executors of base_model)
unrented_cap_multiplier = min(unrented_count, max_cap) / unrented_count
effective_rate[gpu_type] = hourly_rate * unrented_cap_multiplier
total_rental_cost = sum(weighted_count * effective_rate for each gpu_type)
rental_cost_per_epoch = total_rental_cost * (TEMPO * SECONDS_PER_BLOCK) / 3600
rental_share = rental_cost_per_epoch / FIXED_RATIO / (TEMPO * tao_price * alpha_rate)
rental_share = min(rental_share, 0.91)  # capped at TOTAL_BURN_EMISSION
burn_share = 0.91 - rental_share
```

**Phase 3 (post_process):** Rental incentive per executor:
```
effective_rate[executor] = hourly_rate * unrented_cap_multiplier * gpu_count_multiplier
incentive = rental_share * gpu_count * effective_rate[executor] / total_rental_cost
```
Non-eligible GPUs use default mining incentive logic.

## Emission Split

```
Total emission = 1.0
  ├── Burn pool:    burn_share (default: 0.91, reduced by rental_share in rental_price algo)
  ├── Rental pool:  rental_share (0 in default algo, dynamic in rental_price algo)
  └── Mining pool:  mining_share = 1 - 0.91 = 0.09 (always fixed)
```

## Key Constants (`services/const.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `TOTAL_BURN_EMISSION` | 0.91 | Total burn + rental share |
| `BURNER_EMISSION` | 0.01 | Per-burner emission (old logic) |
| `TEMPO` | 360 | Blocks per epoch |
| `SECONDS_PER_BLOCK` | 12 | Seconds per block |
| `FIXED_RATIO` | 0.41 | Rental emission calculation constant |

## Config (`config.py`)

- `BASE_GPU_MAP` — maps full NVIDIA GPU names to base model names (e.g. "NVIDIA H100 80GB HBM3" -> "H100")
- `MAX_UNRENTED_GPUS_BY_TYPE` — per-base-model cap before dilution applies (0 = not eligible for rental incentive)
- `GPU_COUNT_MULTIPLIERS` — rental incentive multiplier by `(base_gpu_model, gpu_count)`. Resolution: specific GPU name > `"*"` fallback; specific count > `"*"` fallback. Controls how much weight an executor with N GPUs gets in the rental pool. Resolver function: `utils.get_gpu_count_multiplier()`
- `MACHINE_PRICES` — hourly USD rate per full GPU name (from `services/const.py`)
- `IncentiveConfig.algorithm` — `"default"` or `"rental_price"` (current default: `"rental_price"`)

## BurnService

Two modes controlled by `settings.ENABLE_NEW_BURN_LOGIC`:
- **New logic:** All burners in `settings.NEW_BURNERS` get equal share of burn_share
- **Old logic:** One random main burner (seeded by block number) gets more, others share remainder

## PriceProvider

Fetches TAO/USD price from 3 providers (CoinGecko, Coinbase, CryptoCompare) with retry.
Queries alpha rate from subtensor. 15-minute TTL cache. Fallback defaults: TAO=$200, alpha=0.001.

## Tests

- `tests/test_incentive_flow.py` — default algorithm tests
- `tests/test_rental_price_incentive_flow.py` — rental price algorithm tests
- `tests/test_gpu_count_multiplier.py` — GPU count multiplier resolution logic tests
- `tests/test_score_calculator.py` — score calculation tests
- `tests/prod_snapshot/` — snapshot-based integration tests against production data

## File Map

| File | Class/Purpose |
|------|--------------|
| `base.py` | `BaseIncentive` — ABC with two-pass pipeline |
| `default.py` | `DefaultIncentive` — mining-only scoring |
| `rental_price.py` | `RentalPriceIncentive` — rental pool extension |
| `factory.py` | `IncentiveFactory` — registry + creation |
| `config.py` | `IncentiveConfig`, `BASE_GPU_MAP`, `MAX_UNRENTED_GPUS_BY_TYPE` |
| `burn_service.py` | `BurnService` — burn emission distribution |
| `price_provider.py` | `PriceProvider` — TAO price + alpha rate with cache |
| `utils.py` | `get_gpu_count_multiplier()` — resolve multiplier by (gpu_model, count); `log_for_monitoring()` — structured logging |
