"""Scoring harness: one fixed fleet through the real RentalPriceIncentive cycle, per flag set.

Every case the port budget for split nodes touches (within budget, over budget, no port backed,
whole idle, part rented, fully rented, not splitting, excluded for another reason, a later idle
gate, a GPU model outside the unrented program, each cap tier and the over-cap tier move) runs
through `calculate_mining_scores` with the real `_calculate_rental_share` (fixed TAO price and
alpha rate). The per-hotkey incentives and cycle totals must equal the golden values in
fixtures/port_budget_score_parity.json, recorded when the budget gate ran as its own step before
the idle-gate chain was shared. PORT_BUDGET_HARNESS_OUT=<dir> also writes the full per-executor
output of each flag set there, to compare two trees.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.config import IncentiveConfig
from incentive.rental_price import RentalPriceIncentive
from services.const import DEFAULT_JOB_OWNER_MINER
from services.task_service import JobResult

from core.config import settings, shared_client

H200 = "NVIDIA H200"
B300 = "NVIDIA B300 SXM6 AC"
H100 = "NVIDIA H100 80GB HBM3"
RTX4090 = "NVIDIA GeForce RTX 4090"
H800 = "NVIDIA H800 80GB HBM3"  # no unrented cap: outside the unrented program
RTX5080 = "NVIDIA GeForce RTX 5080"
GPU_PORTION = {H200: 0.30, B300: 0.25, H100: 0.20, RTX4090: 0.10, H800: 0.10, RTX5080: 0.05}
SOFT_PRICE_P90 = {H200: 2.0}
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "port_budget_score_parity.json"

FLAG_SETS = {
    "budget_off_floor_off": {"budget": False, "floor": False},
    "budget_on_floor_off": {"budget": True, "floor": False},
    "budget_off_floor_on": {"budget": False, "floor": True},
    "budget_on_floor_on": {"budget": True, "floor": True},
}


def _job(
    uuid: str,
    gpu_model: str,
    gpu_count: int,
    *,
    ports: int | None = 30,
    rented: int | None = None,
    split_min: int | None = None,
    **fields,
) -> JobResult:
    spec: dict = {} if ports is None else {"available_port_count": ports}
    price_per_gpu: float | None = fields.pop("price_per_gpu", None)
    values: dict = {
        "spec": spec,
        "executor_info": ExecutorSSHInfo(
            uuid=uuid,
            address="10.0.0.1",
            port=8080,
            ssh_username="root",
            ssh_port=22,
            python_path="/usr/bin/python3",
            root_dir="/tmp",
            price_per_gpu=price_per_gpu,
        ),
        "score": 1.0,
        "job_score": 1.0,
        "job_batch_id": "batch",
        "log_status": "success",
        "log_text": "ok",
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "is_rented": rented is not None,
        "rented_gpu_count": rented,
        "supports_gpu_splitting": split_min is not None,
        "gpu_splitting_min_count": split_min,
        "collateral_deposited": True,
        "sysbox_runtime": True,
    }
    values.update(fields)
    return JobResult(**values)


def _fleet() -> dict[str, list[JobResult]]:
    return {
        # H200: 1x cap 10, 8x cap 64
        "hk-h200-plain-idle-1": [_job("e01", H200, 1, ports=12)],
        "hk-h200-plain-idle-8": [_job("e02", H200, 8, ports=30)],
        "hk-h200-plain-rented-8": [_job("e03", H200, 8, ports=30, rented=8)],
        "hk-h200-split-idle-within": [_job("e04", H200, 8, ports=24, split_min=1)],
        "hk-h200-split-idle-over": [_job("e05", H200, 8, ports=6, split_min=1)],
        "hk-h200-split-idle-zero": [_job("e06", H200, 8, ports=2, split_min=1)],
        "hk-h200-split-idle-min2-over": [_job("e07", H200, 8, ports=9, split_min=2)],
        "hk-h200-split-idle-min8": [_job("e08", H200, 8, ports=2, split_min=8)],
        "hk-h200-split-idle-no-port-count": [_job("e09", H200, 8, ports=None, split_min=1)],
        "hk-h200-split-part-within": [_job("e10", H200, 8, ports=12, rented=4, split_min=1)],
        "hk-h200-split-part-over": [_job("e11", H200, 8, ports=5, rented=4, split_min=1)],
        "hk-h200-split-part-zero": [_job("e12", H200, 8, ports=2, rented=4, split_min=1)],
        "hk-h200-split-full-rented": [_job("e13", H200, 8, ports=0, rented=8, split_min=1)],
        "hk-h200-split-idle-over-spot": [_job("e14", H200, 8, ports=6, split_min=1, is_spot=True)],
        "hk-h200-split-idle-zero-spot": [_job("e24", H200, 8, ports=2, split_min=1, is_spot=True)],
        "hk-h200-split-idle-zero-no-sysbox": [
            _job("e25", H200, 8, ports=2, split_min=1, sysbox_runtime=False)
        ],
        "hk-h200-split-part-zero-paused": [
            _job("e26", H200, 8, ports=1, rented=6, split_min=1, is_new_rentals_paused=True)
        ],
        "hk-h200-split-idle-over-paused": [
            _job("e15", H200, 8, ports=6, split_min=1, is_new_rentals_paused=True)
        ],
        "hk-h200-split-idle-over-banned": [
            _job("e16", H200, 8, ports=6, split_min=1, is_provider_banned=True)
        ],
        "hk-h200-split-idle-over-no-discord": [
            _job("e17", H200, 8, ports=6, split_min=1, provider_discord_connected=False)
        ],
        "hk-h200-split-part-over-own-job": [
            _job(
                "e18",
                H200,
                8,
                ports=5,
                rented=4,
                split_min=1,
                default_job_owner=DEFAULT_JOB_OWNER_MINER,
            )
        ],
        "hk-h200-split-idle-over-soft-price": [
            _job("e19", H200, 8, ports=6, split_min=1, price_per_gpu=10.0)
        ],
        "hk-h200-split-idle-over-no-sysbox": [
            _job("e20", H200, 8, ports=6, split_min=1, sysbox_runtime=False)
        ],
        "hk-h200-split-idle-over-old-driver": [
            _job("e21", H200, 8, ports=6, split_min=1, nvidia_driver_version="470.1")
        ],
        "hk-h200-split-idle-over-failed": [
            _job(
                "e22", H200, 8, ports=6, split_min=1, score=0.0, job_score=0.0, log_status="failed"
            )
        ],
        "hk-h200-split-idle-over-no-collateral": [
            _job("e23", H200, 8, ports=6, split_min=1, collateral_deposited=False)
        ],
        # B300: 1x cap 4, 8x cap 32 — seven 8x split nodes overfill the 8x tier, so the over-cap
        # tier move runs; with the budget on only the backed GPUs have to fit the 1x cap
        "hk-b300-plain-idle-1": [_job("e30", B300, 1, ports=12)],
        "hk-b300-multi": [
            _job("e31", B300, 8, ports=6, split_min=1),
            _job("e32", B300, 8, ports=9, split_min=1),
            _job("e33", B300, 8, ports=24, split_min=1),
        ],
        "hk-b300-split-idle-within-a": [_job("e34", B300, 8, ports=24, split_min=1)],
        "hk-b300-split-idle-within-b": [_job("e35", B300, 8, ports=30, split_min=1)],
        "hk-b300-split-idle-over-c": [_job("e36", B300, 8, ports=3, split_min=1)],
        "hk-b300-split-part-over": [_job("e37", B300, 8, ports=6, rented=2, split_min=1)],
        # H100: 2-GPU bundles on a part-rented node, and a plain idle yardstick
        "hk-h100-split-part-min2-over": [_job("e40", H100, 8, ports=3, rented=2, split_min=2)],
        "hk-h100-plain-idle-1": [_job("e41", H100, 1, ports=12)],
        "hk-h100-plain-rented-2": [_job("e42", H100, 2, ports=12, rented=2)],
        # RTX 4090
        "hk-4090-split-idle-over": [_job("e50", RTX4090, 8, ports=12, split_min=1)],
        "hk-4090-plain-idle-1": [_job("e51", RTX4090, 1, ports=12)],
        # outside the unrented program: idle earns nothing, rented earns in the mining pool
        "hk-h800-split-idle-over": [_job("e60", H800, 8, ports=6, split_min=1)],
        "hk-h800-split-part-over": [_job("e61", H800, 8, ports=5, rented=4, split_min=1)],
        "hk-5080-plain-rented-1": [_job("e62", RTX5080, 1, ports=12, rented=1)],
    }


def _set_setting(monkeypatch: pytest.MonkeyPatch, name: str, value: bool) -> None:
    # a tree without one of these flags runs as the baseline without that gate
    if name in type(settings).model_fields:
        monkeypatch.setattr(settings, name, value)


async def _run(monkeypatch: pytest.MonkeyPatch, flags: dict[str, bool]) -> dict:
    _set_setting(monkeypatch, "ENABLE_UNRENTED_PORT_BUDGET_FOR_SPLIT_GPUS", flags["budget"])
    _set_setting(monkeypatch, "ENABLE_UNRENTED_PORT_FLOOR_FOR_SPLIT_REMAINDER", flags["floor"])
    _set_setting(monkeypatch, "ENABLE_SPLIT_PARTIAL_RENTAL_SCORING", True)
    _set_setting(monkeypatch, "ENABLE_UNRENTED_SOFT_PRICE_LIMIT", True)
    _set_setting(monkeypatch, "ENABLE_UNRENTED_VRAM_OVER_DISK_LIMIT", False)
    _set_setting(monkeypatch, "ENABLE_UNRENTED_FLAGSHIP_CAPABILITY_LIMIT", False)
    _set_setting(monkeypatch, "ENABLE_UNRENTED_POWER_CAP_LIMIT", False)
    _set_setting(monkeypatch, "SKIP_COLLATERAL_PENALTY", False)
    monkeypatch.setattr(
        shared_client,
        "_config",
        shared_client.config.model_copy(
            update={"machine_prices_p90": SOFT_PRICE_P90, "soft_limit_price_rate": 1.1}
        ),
    )

    job_results: dict[str, list[JobResult]] = _fleet()
    total_by_model: dict[str, int] = {}
    for results in job_results.values():
        for result in results:
            total_by_model[result.gpu_model] = (
                total_by_model.get(result.gpu_model, 0) + result.gpu_count
            )
    redis_service = AsyncMock()
    redis_service.get_portion_per_gpu_type = AsyncMock(side_effect=lambda model: GPU_PORTION[model])
    redis_service.get_executor_uptime = AsyncMock(return_value=30)
    incentive = RentalPriceIncentive(IncentiveConfig(), redis_service, job_results, total_by_model)
    incentive.price_provider = AsyncMock()
    incentive.price_provider.get_tao_price = AsyncMock(return_value=300.0)
    incentive.price_provider.get_alpha_rate = AsyncMock(return_value=0.5)

    await incentive.calculate_mining_scores()

    executors: dict[str, dict] = {}
    for hotkey, results in incentive.job_results.items():
        for result in results:
            executors[f"{hotkey}/{result.executor_info.uuid}"] = {
                "gpu_count": result.gpu_count,
                "mining_score": result.mining_score,
                "eligible_for_rental_share": result.eligible_for_rental_share,
                "incentive": result.incentive,
                "incentive_rented": result.incentive_rented,
                "incentive_idle": result.incentive_idle,
                "count_bucket": result.count_bucket,
                "bucket_reassigned_from": result.bucket_reassigned_from,
                "port_unbacked_gpu_count": getattr(result, "port_unbacked_gpu_count", None),
                "zero_incentive_reasons": [str(r.reason) for r in result.zero_incentive_reasons],
            }
    return {
        "miner_incentives": dict(sorted(incentive.miner_incentives.items())),
        "rental_share": incentive.rental_share,
        "total_rental_cost": incentive.total_rental_cost,
        "total_mining_score": incentive.total_mining_score,
        "unrented_count_by_bucket": {
            f"{model}·{bucket}": count
            for (model, bucket), count in sorted(incentive.unrented_count_by_bucket.items())
        },
        "executors": executors,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("flag_set", list(FLAG_SETS))
async def test_port_budget_score_parity(monkeypatch, flag_set):
    output: dict = await _run(monkeypatch, FLAG_SETS[flag_set])
    out_dir: str | None = os.environ.get("PORT_BUDGET_HARNESS_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{flag_set}.json"), "w") as handle:
            json.dump(output, handle, indent=1, sort_keys=True, default=str)
    golden: dict = json.loads(GOLDEN_PATH.read_text())[flag_set]
    assert output["unrented_count_by_bucket"] == golden["unrented_count_by_bucket"]
    for key in ("rental_share", "total_rental_cost", "total_mining_score"):
        assert output[key] == pytest.approx(golden[key], rel=1e-12, abs=1e-15), key
    assert output["miner_incentives"].keys() == golden["miner_incentives"].keys()
    for hotkey, incentive in golden["miner_incentives"].items():
        assert output["miner_incentives"][hotkey] == pytest.approx(
            incentive, rel=1e-12, abs=1e-15
        ), hotkey
