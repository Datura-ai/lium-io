# Anchor occupancy floor — a capped bucket in the validator's unrented pool (design note, DAH-3673)

Status: DRAFT for the ticket owner's approval (PR_PROCESS §1b: incentive math needs a human-approved note before code). No code, no config change in this PR. Code references are `path:line` at lium-io `61020313` (main, 19 Sep 2026); paths are relative to `neurons/validators/src/` unless they start with `lium_protocol/` (repo root) or `lium_core/` (`packages/lium-core/src/`). The floor percentage F is an open owner decision (OD-19); every "F = 60" below is a worked example, not the plan. Programme text: growth-strategy §5.3, supply-partnerships §2, gmv-bridge §3 row 1 / D1.

## 1. Goal

An **accepted anchor node** (8× H100 / H200 / B200 / B300 / RTX PRO 6000, whole node, secure tier) is paid, for its **first 90 days**, as if at least **F %** of its GPU-hours were rented at its **listed price**. F is the owner's OD-19; **default F = 0: the mechanism is present, the floor is off**, and every node is scored exactly as today.

Per UTC day, with F in percent: `top_up = max(0, F/100 × listed_price × gpu_count × 24 h − rental_revenue_that_day)`.

## 2. Non-goals

- No change to the 95/5 rental share or to any renter price; the validator never touches billing.
- No new token flow. The top-up is paid in emission from the **existing unrented (rental-share) pool** as a capped bucket that **replaces** the node's ordinary idle pay while the floor is active — never in addition to it.
- No change to the mining (rented) pool, to `PENALTY_DAYS` / `PenaltyService` in the backend, to the spot tier, or to GPU-split scoring.

## 3. How the unrented pool pays today (what the note builds on)

- Every 75 blocks (15 min, `core/config.py:107`) the validator scores every executor (`core/validator.py:543-550`). A validated idle node of an eligible model leaves the mining pool for the rental-share pool (`incentive/rental_price.py:929-1016`).
- Each idle node contributes `gpu_count × hourly_rate × sysbox × driver` USD/h to its `(base_model, gpu_count_bucket)` (`incentive/rental_price.py:644-686`); `hourly_rate` comes from `RENTAL_PRICES_PER_HOUR` (`incentive/config.py:31`: lium-core `machine_prices` plus two pins today; the 30-day paid median once the P184/P186 PRs lium-io#1401 and #1407, both open, land). A bucket over its GPU cap is diluted pro-rata (`incentive/config.py:54`, B300 8× cap = 32 GPUs = 4 nodes; `incentive/rental_price.py:763-767`).
- The sum is `total_rental_cost` (USD/h, `incentive/rental_price.py:773`); `rental_share = total_rental_cost × epoch_h / FIXED_RATIO / epoch_emission_usd` (`incentive/rental_price.py:1176-1179`, `services/const.py:142-144`), capped at `total_burn_emission` (0.91, `incentive/rental_price.py:781`; `lium_core/shared_config/defaults.py:125`). What the pool does not claim is burned (`incentive/rental_price.py:797`, `incentive/default.py:264`); the mining pool is the other 9 % and nothing here touches it.
- Per node: `incentive = rental_share × gpu_count × effective_rate / total_rental_cost` (`incentive/rental_price.py:836-848`), summed per hotkey (`:862`) into `cycle_scores` (`incentive/default.py:274`), accumulated into `miner_scores` and normalised at `set_weights` once per tempo (`core/validator.py:610-626`, `:269-273`).
- Rental state comes from the backend feed `GET /internal/executors/rented` (`clients/backend_client.py:245-253`; `RentedExecutorsResponse` at `protocol/vc_protocol/compute_requests.py:98` and `lium_protocol/lium_protocol/http.py:54`): `is_rented` (`services/task/result_handler.py:218`), `spot_executor_ids` (`protocol/vc_protocol/compute_requests.py:119`), `new_rentals_paused_executor_ids` (`:120`). Precedent for money on a backend feed: the referral EMA feed (`clients/referral_feed_client.py:29`, `incentive/default.py:282-308`; `REFERRAL_EMISSION_SHARE` default 0, `core/config.py:190`).
- The validator **does not know billed revenue per executor**; it knows, per 15-min cycle, whether the node is rented. The backend knows billed USD per executor per UTC day (`billinghistory`, `rental_snapshots`).

## 4. Mechanism

### 4.1 Allow-list: backend table, served in the rented feed

Proposed: a backend table `anchor_executor` (`executor_id`, `miner_hotkey`, `accepted_at`, `strikes`, `paused_reason`, `revoked_at`), written by an admin command (`admin-anchor-accept <executor_id>`, the shape of `admin-reserve-node` in reserved-capacity v1), and served as a new additive field on the feed the validator already reads every cycle:

```
RentedExecutorsResponse.anchor_executors: dict[str, AnchorExecutor] = {}   # executor_id → row
AnchorExecutor: miner_hotkey, accepted_at, strikes, paused_reason | None, revoked_at | None
```

Both copies of the model change (`protocol/vc_protocol/compute_requests.py:98`, `lium_protocol/lium_protocol/http.py:54`). Empty default = nobody is an anchor: an older backend, a failed fetch or a missing field fail closed, as `spot_executor_ids` does. Keyed by executor UUID, not hotkey — a hotkey carries non-anchor nodes too, and the UUID's GPU set is already permanent (DAH-3457).

`floor_active(node, now) = node.uuid ∈ anchor_executors ∧ accepted_at ≤ now < accepted_at + ANCHOR_FLOOR_DAYS ∧ revoked_at is None ∧ paused_reason is None ∧ ANCHOR_FLOOR_PCT > 0 ∧ ANCHOR_BUCKET_MAX_SHARE > 0`.

### 4.2 Routing: the anchor bucket replaces the ordinary bucket

In `_pre_process_job_result` (`incentive/rental_price.py:644`), after every existing gate has run (`eligible_for_rental_share` is already the result of the banned / spot / Discord / paused / own-default-job exclusions at `incentive/rental_price.py:480-497` and the soft-price, disk, flagship-capability and power-cap gates at `:936-991`): if `floor_active`, the node is **not** added to `unrented_count_by_bucket` / `_weighted_rate_sum_by_bucket`; it is added to one new accumulator

```
anchor_cost_per_h += gpu_count × anchor_rate × sysbox_multiplier × driver_multiplier      # only in a paying cycle (4.3)
```

So 20 anchor nodes never dilute the ordinary B300 8× bucket (cap 32 GPUs, `incentive/config.py:55`) for existing providers, and an anchor node never earns from both buckets. A rented anchor node is scored in the mining pool exactly as today (`incentive/default.py:148`).

### 4.3 The floor per UTC day, in the validator's own unit (cycles)

The validator's primitive is the 15-min cycle: 96 per UTC day. A day's floor budget is `budget_cycles = floor(F/100 × 96)` (57 at F = 60). Per anchor node and UTC day the validator keeps two Redis counters, `rented_cycles` and `floor_cycles` (key `anchor_day:{executor_id}:{YYYY-MM-DD}`, TTL 48 h; Redis already holds per-executor state, `services/redis_service.py:401`, and the incentive snapshot, `:582`). Each cycle:

- rented → `rented_cycles += 1`; mining pool as today.
- idle and `rented_cycles + floor_cycles < budget_cycles` → **paying cycle**: `floor_cycles += 1`; the node enters the anchor bucket at `anchor_rate`.
- idle otherwise → incentive 0 for this cycle with a proposed new append-only reason `ANCHOR_FLOOR_DAY_MET` (`incentive/miner_incentive_log.py:63`), so the provider's incentive log says why.

Over a day this pays `anchor_rate × gpu_count × 0.25 h × min(idle_cycles, max(0, budget − rented_cycles))` = `F/100 × P × n × 24 − rented_h × P × n` when the node was idle long enough — the ticket's formula with `rental_revenue ≈ rented_hours × listed price`. Two deviations, both bounded by the day's rented revenue: a rental ending mid-cycle counts as a rented cycle; a node idle in the morning and rented all afternoon ends the day above the floor (weights are set, nothing is clawed back). The exact `billed_usd` version needs the backend (open question 3).

### 4.4 The rate: listed price, never above it

`anchor_rate = min(price_per_gpu, ceiling)`, where `price_per_gpu` is the node's listed ask (`executor_info.price_per_gpu`, the same field the soft limit reads at `incentive/rental_price.py:254`) and `ceiling = machine_prices_p90[gpu_model] × SOFT_LIMIT_PRICE_RATE` (`incentive/rental_price.py:40`, `lium_core/shared_config/model.py:22`; the shared-config twin `soft_limit_price_rate` is `model.py:41`), falling back to `machine_prices[gpu_model] × machine_max_price_rate` (the listing ceiling already enforced at `services/task/score_calculator.py:55-57`). If `price_per_gpu` is None the rate is `hourly_rate` (the pinned median, `incentive/rental_price.py:650`).

The floor therefore never pays above the node's own rental rate (SO §74 / P156: idle pay ≤ rental rates — here the idle pay *is* the node's listed rate, for at most F × 24 h a day). **Price floor**: the programme requires the anchor's listed price ≥ the model's 30-day paid median, recomputed weekly. That is a listing rule the backend enforces at acceptance (`price_per_gpu ≥ machine_prices[gpu_model]`, which #1407 (open) makes the 30-day paid median; #1401 keeps `RENTAL_PRICES_PER_HOUR` on the same medians). The validator never raises a below-median listing to the median — it pays the listed price; the backend refuses the acceptance.

### 4.5 The cap: X % of the unrented pool, pro-rata when it binds

Cap unit = the unrented pool's ceiling in USD/h, the inverse of `_calculate_rental_share` (`incentive/rental_price.py:1131-1179`):

```
pool_ceiling_per_h = total_burn_emission × epoch_subnet_emission × FIXED_RATIO / (TEMPO × SECONDS_PER_BLOCK / 3600)
anchor_cap_per_h   = ANCHOR_BUCKET_MAX_SHARE × pool_ceiling_per_h
anchor_cap_multiplier = min(1, anchor_cap_per_h / anchor_cost_per_h)        # applied to every anchor node's effective_rate
```

computed in `_on_finish_pre_process` (`incentive/rental_price.py:754`) the way `cap_multiplier_by_bucket` is today (`:763-767`), with one ordering change: `epoch_subnet_emission` is only known after the TAO/alpha price fetch inside `_calculate_rental_share` (`:1149-1150`), so that fetch moves ahead of the cost sum (or the anchor cap is applied and `rental_share` recomputed once after it). Then `total_rental_cost += anchor_cost_per_h × anchor_cap_multiplier` (`:773`) and each paying anchor node's `incentive = rental_share × gpu_count × anchor_rate × anchor_cap_multiplier × sysbox × driver / total_rental_cost` — the formula at `incentive/rental_price.py:845` with `effective_rate` swapped. `rental_share ≤ total_burn_emission` (`:781`) still holds above everything: the bucket can only spend burn, never the mining pool. When the cap binds every anchor node is diluted by the same multiplier (pro-rata); its `floor_cycles` still count, so a diluted day is not re-paid.

Base = the ceiling, not today's idle spend: idle nodes are paid ≈ $4.3k/day (validator ledger, 18 Sep) — 25 % of that could not fund one idle 8× B300 day. The ceiling is ≈ $47k/day (§6), the "unrented pool" of the programme text.

### 4.6 The 90-day clock, eligibility and exit

- **Start** = `accepted_at`, stamped by the admin command, which refuses an executor that is not verified and online. Unverified cycles inside the window earn nothing (not `is_successful`); the clock still runs.
- **Eligibility** (backend, at acceptance): provider committed ≥ 4 nodes of 8× flagship within 60 days (a row on a provider-level `anchor_provider` table); node is 8× of a model in `FLAGSHIP_CAPABILITY_BASE_MODELS ∪ {H100, RTX PRO 6000}` (`incentive/rental_price.py:51`); verified; sysbox (already required for idle pay, `core/config.py:142`); not spot; splitting not enabled (`gpu_splitting_min_count < gpu_count`, the test at `incentive/rental_price.py:365`).
- **Exit — two provider-caused fault closes**: backend counts `rental_history.close_reason` in the fault set gmv-bridge §1 uses (`executor_offline`, `broken_by_provider`, `rent_failed`, `undeploy_failed`) on the node since `accepted_at`; the second sets `revoked_at` (and `strikes = 2` for the log line). The validator only reads `revoked_at`.
- **Pause — reserved pin**: while `executor.reserved_for_user_id` is set (reserved-capacity v1 row 3) the backend serves `paused_reason = "reserved_pin"`; no floor pay; the clock keeps running (the contract pays). Ordinary idle treatment of a pinned node is that note's question, not this one's.
- **End**: `accepted_at + ANCHOR_FLOOR_DAYS`; the node then scores as an ordinary node in its bucket. Nothing to migrate — the routing condition is evaluated every cycle.

### 4.7 Interaction with existing rules (unchanged)

| Rule | Where | Effect on the floor |
|---|---|---|
| Idle pay ≤ rental rates (SO §74 / P156; pins in #1401, not merged) | `incentive/config.py:31` | floor rate = the node's own listed price ≤ its rental rate; pins untouched |
| Spot tier earns nothing | `incentive/rental_price.py:489`, `incentive/miner_incentive_log.py:190` | an anchor moved to spot is excluded before routing; no floor |
| GPU splitting (DAH-2467/2528) | `incentive/rental_price.py:534`, `:500` | split-enabled nodes are not accepted; a split remainder never reaches the anchor bucket |
| Soft price / disk / flagship capability / power cap gates | `incentive/rental_price.py:936-991`, flags `core/config.py:407-428` | run first; a gated node has no floor cycle |
| Penalties (`PENALTY_DAYS`, `PenaltyService`) | backend | untouched; a strike ends the floor, it does not claw back emission |
| Referral pool from residual burn | `incentive/default.py:282` | unchanged; it is applied after the rental share, so a bigger anchor bucket leaves less residual burn for referral, never less for miners |
| Estimates / `/incentive-snapshot` | `incentive/rental_price.py:1199`, `:113-125` | `RentalShareState` gains `anchor: {node_count, cost_per_h, cap_per_h, cap_multiplier}`; a hypothetical executor is never an anchor, so estimates are unchanged |

## 5. Config surface (proposed; validator `Settings`, next to `REFERRAL_EMISSION_SHARE`, `core/config.py:190`)

| Name | Default | Meaning |
|---|---|---|
| `ANCHOR_FLOOR_PCT` (proposed) | `0` | F, percent of GPU-hours guaranteed per UTC day (0–100). 0 = off: no routing change anywhere. OD-19. |
| `ANCHOR_FLOOR_DAYS` (proposed) | `90` | window from `accepted_at` |
| `ANCHOR_BUCKET_MAX_SHARE` (proposed) | `0.0` | X, the anchor bucket's ceiling as a fraction of the unrented pool ceiling (§4.5). 0 = off. Owner's funding-share decision. |
| allow-list (proposed) | `RentedExecutorsResponse.anchor_executors`, default `{}` | backend table + admin command; no validator-side list |

Both proposed settings `ANCHOR_FLOOR_PCT > 0` and `ANCHOR_BUCKET_MAX_SHARE > 0` are needed for any anchor pay; set on both validator hotkeys (DAH-3394) in one deploy.

## 6. Cost model (gmv-bridge §1, §3 row 1, D1)

- Pool ceiling: ≈ 5,900 α/day × $21.73 × 0.41 (`FIXED_RATIO`) × 0.91 (`total_burn_emission`) ≈ **$47k/day**; ≈ $4.3k/day of it is paid to idle nodes today, the rest is burned.
- **Expected ≈ $0/wk** at F = 60 (worked example; F is OD-19): B300 / B200 / H200 GPUs rent 94 / 84 / 75 % of the time today (bridge §1, type level), i.e. 18–22 rented hours a day against a 14.4-hour budget, so `budget − rented_cycles ≤ 0` on almost every day.
- **Worst case**, 20 accepted 8× B300 nodes idle all day at $8.06: 160 GPUs × 14.4 h × $8.06 ≈ **$18.6k/day ≈ $130k/wk** (D1) = 39 % of the ceiling.
- **What X caps it to**: X = 0.25 → ≈ $12k/day ≈ **$84k/wk**, the worst case pro-rates to 64 % of the floor; X = 0.40 covers the worst case in full; X = 0 pays nothing.
- Per node the bound is `F/100 × 24 × listed × 8` a day ($928 at $8.06, $737 at the $6.40 pin), whatever the pool does.

## 7. Rollout and observability

1. Backend: table, admin command, feed field (additive; deploy order with the validator is free — the default is `{}`).
2. Validator, staging (proposed sequence): `ANCHOR_FLOOR_PCT=0`, one test hotkey's executor accepted → identical scores to main for a full day (the routing condition is false); then `ANCHOR_FLOOR_PCT=60 ANCHOR_BUCKET_MAX_SHARE=0.25` on staging only, read the log lines and the ledger rows below.
3. Validator, prod (proposed): `ANCHOR_FLOOR_PCT=0 ANCHOR_BUCKET_MAX_SHARE=0` — a no-op deploy. F and X are set only when OD-19 is answered.
4. Observability (proposed): one structured log line per anchor node per cycle, `Anchor_breakdown | 8xB300 [uuid8] | day 2026-10-01 rented 11 floor 3 budget 57 | $8.06 × 1.00 × 8 = $64.48/h`, next to `Rental_breakdown` (`incentive/utils.py:132`); `incentive_source = "anchor_floor"`, `incentive_formula_version = "anchor_floor_v1"` and the anchor inputs in `incentive_formula_inputs` (`services/task/models.py:152-184`), published to the backend ledger as today (`services/miner_service.py:901-920`) → a Grafana series "Anchor floor paid — last 24h (USD, validator ledger)" plus a per-node daily table on the dashboard that shows idle pay; `ANCHOR_FLOOR_DAY_MET` in the provider's incentive log.

## 8. Open questions for the ticket owner (proposed answer after each)

1. Allow-list transport: a field on `GET /internal/executors/rented` vs a list in `IncentiveConfig`? — **Feed.** The backend owns acceptance, strikes and pins; the validator stays stateless about lifecycle; fail-closed like `spot_executor_ids`.
2. F, days, X in validator `Settings` (env, the `REFERRAL_EMISSION_SHARE` precedent) or in `SharedConfig` (the `total_burn_emission` precedent, one value for both hotkeys)? — **Settings for the first PR**; no lium-core release needed. Move to shared config if the two hotkeys ever need a change without a redeploy.
3. Daily budget state in validator Redis (§4.3) vs a backend-computed T+1 deficit from billed USD served in the feed? — **Redis.** It uses the same `is_rented` the cycle already scores on and costs no feed round-trip; the backend ledger audits it. The T+1 version is exact but a day late and would make the feed carry money.
4. Floor rate = listed `price_per_gpu` (capped at the p90 soft limit) or the pinned median `hourly_rate`? — **Listed, capped.** The programme says "at its listed price"; §74 holds because the listed price is the node's rental rate.
5. Beyond the day's budget: 0 (floor-only) or fall back to ordinary idle pay? — **0.** It keeps anchors out of the ordinary buckets (20 anchors would dilute the B300 8× bucket 5×) and keeps the per-node bound at `F/100 × 24 × listed × 8`.
6. Cap base = the pool ceiling (§4.5) or today's idle spend? — **Ceiling**, X = 0.25 recommended (§6).
7. Split-enabled anchors: excluded in v1, or per-GPU occupancy (`rented_gpu_count / gpu_count`)? — **Excluded**; the DAH-2546 capability gate is met by NCU or TDX (the flag is shadow today, `core/config.py:419`).
8. Two validator hotkeys keep independent counters from the same feed; drift is at most one cycle. Acceptable? — **Yes**; stake-weighted consensus averages the two weight vectors.
9. Strikes: backend from `rental_history.close_reason`, validator reads `revoked_at` and shows `strikes` in the incentive log? — **Yes**; one source of truth for "provider-caused".
10. Does the clock pause during a reserved pin? — **No**; the node is earning under contract.

## 9. Owner decisions referenced

- **OD-19** — F (recommended 60; offer sheet version A). **DEFAULT: 0 = no floor** (offer sheet version B goes out) until the owner answers.
- **Funding share cap X** (`ANCHOR_BUCKET_MAX_SHARE`; recommended 0.25). **DEFAULT: 0** until answered.

Both defaults keep every score identical to today; the code that follows this note ships with them.
