minor
`RentedExecutorsResponse.lium_filler_containers_by_executor` (optional, default empty): executor_id → the filler container names (a subset of `all_filler_containers_by_executor`) whose filler Lium started, never the provider's default job. Empty means not known (an older backend), so the validator can tell a Lium filler from the provider's default job only when the backend sends it.
