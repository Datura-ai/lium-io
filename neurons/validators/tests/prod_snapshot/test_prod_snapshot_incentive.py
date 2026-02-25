"""Production snapshot tests for incentive algorithms.

Validates that the incentive code produces the same results as production
Loki logs. This ensures refactoring doesn't break the scoring logic.

Each snapshot lives in its own directory under fixtures/:
    fixtures/snapshot_2026-02-06/
    ├── snapshot.json          # Input data + expected outputs
    ├── results_default.csv    # Generated CSV for default algorithm
    └── results_rental_price.csv  # Generated CSV for rental_price algorithm

Usage:
    # Normal run — asserts against saved expected values:
    uv run pytest tests/prod_snapshot/ -v

    # Update mode — overwrites expected values in snapshot JSON:
    uv run pytest tests/prod_snapshot/ -v --update-snapshot
"""

import csv
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import settings
from datura.requests.miner_requests import ExecutorSSHInfo
from incentive.base import BaseIncentive
from incentive.config import IncentiveConfig
from incentive.default import DefaultIncentive
from incentive.rental_price import RentalPriceIncentive
from services.task_service import JobResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"

CSV_COLUMNS = [
    "miner_hotkey",
    "executor_id",
    "gpu_model",
    "gpu_count",
    "score",
    "job_score",
    "collateral_deposited",
    "sysbox_runtime",
    "is_rented",
    # V1 scoring fields
    "mining_score",
    "sysbox_multiplier",
    "uptime_multiplier",
    "gpu_portion",
    "total_gpu_count",
    # V2 rental price fields
    "eligible_for_rental_share",
    "max_cap",
    "total_unrented_by_gpu_type",
    "cap_dilution_applied",
    "hourly_rate",
    "unrented_cap_multiplier",
    "gpu_count_multiplier",
    "effective_rate",
    "rental_share",
    "burn_share",
    "total_rental_cost",
    # Final results
    "incentive",
    "incentive_usd_per_hour",
]


# ---------------------------------------------------------------------------
# Auto-discover snapshot directories
# ---------------------------------------------------------------------------

def _discover_snapshot_dirs() -> list[Path]:
    """Find all snapshot_* directories under fixtures/."""
    if not FIXTURES_DIR.exists():
        return []
    return sorted(
        d for d in FIXTURES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("snapshot_")
    )


SNAPSHOT_DIRS = _discover_snapshot_dirs()
SNAPSHOT_IDS = [d.name for d in SNAPSHOT_DIRS]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=SNAPSHOT_DIRS, ids=SNAPSHOT_IDS)
def snapshot_dir(request: pytest.FixtureRequest) -> Path:
    """Parametrized fixture — yields each snapshot directory."""
    return request.param


@pytest.fixture
def prod_snapshot(snapshot_dir: Path) -> dict:
    """Load production snapshot fixture."""
    snapshot_path = snapshot_dir / "snapshot.json"
    return json.loads(snapshot_path.read_text())


@pytest.fixture
def mock_redis_from_snapshot(prod_snapshot: dict) -> AsyncMock:
    """Mock redis service using snapshot data."""
    service = AsyncMock()
    gpu_portions = prod_snapshot["redis_data"]["gpu_portions"]
    executor_uptimes = prod_snapshot["redis_data"]["executor_uptimes"]

    async def get_portion(gpu_model: str) -> float:
        return gpu_portions.get(gpu_model, 0.0)

    async def get_uptime(executor_info: ExecutorSSHInfo) -> int:
        key = f"{executor_info.address}:{executor_info.port}"
        return executor_uptimes.get(key, 0)

    service.get_portion_per_gpu_type = AsyncMock(side_effect=get_portion)
    service.get_executor_uptime = AsyncMock(side_effect=get_uptime)
    return service


@pytest.fixture
def prod_settings(monkeypatch, prod_snapshot: dict) -> None:
    """Patch settings to match production config from snapshot."""
    s = prod_snapshot["settings"]
    monkeypatch.setattr(settings, "PORTION_FOR_SYSBOX", s["PORTION_FOR_SYSBOX"])
    monkeypatch.setattr(settings, "PORTION_FOR_UPTIME", s["PORTION_FOR_UPTIME"])
    monkeypatch.setattr(settings, "UPTIME_REQUIRED_MINUTES", s["UPTIME_REQUIRED_MINUTES"])
    monkeypatch.setattr(settings, "BURNERS", s["BURNERS"])
    monkeypatch.setattr(settings, "NEW_BURNERS", s["NEW_BURNERS"])
    monkeypatch.setattr(settings, "ENABLE_NEW_BURN_LOGIC", s["ENABLE_NEW_BURN_LOGIC"])
    monkeypatch.setattr(settings, "incentive", IncentiveConfig(algorithm="default"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_job_results(prod_snapshot: dict) -> dict[str, list[JobResult]]:
    """Build JobResult objects from snapshot data."""
    job_results: dict[str, list[JobResult]] = {}
    for hotkey, results in prod_snapshot["job_results"].items():
        job_results[hotkey] = []
        for r in results:
            executor_info = ExecutorSSHInfo(
                uuid=r["executor_id"],
                address=r["address"],
                port=r["port"],
                ssh_username="root",
                ssh_port=22,
                python_path="/usr/bin/python3",
                root_dir="/tmp",
            )
            job_result = JobResult(
                executor_info=executor_info,
                score=r["score"],
                job_score=r["job_score"],
                gpu_model=r["gpu_model"],
                gpu_count=r["gpu_count"],
                collateral_deposited=r["collateral_deposited"],
                sysbox_runtime=r["sysbox_runtime"],
                is_rented=r["is_rented"],
                job_batch_id=r["job_batch_id"],
                log_status=r["log_status"],
                log_text=r["log_text"],
            )
            job_results[hotkey].append(job_result)
    return job_results


def _create_incentive(
    prod_snapshot: dict,
    mock_redis: AsyncMock,
    algorithm: str = "default",
) -> BaseIncentive:
    """Create incentive instance from snapshot data."""
    config = IncentiveConfig(algorithm=algorithm)
    burn_service = AsyncMock()
    job_results = _build_job_results(prod_snapshot)
    gpu_count_map = prod_snapshot["total_gpu_model_count_map"]

    if algorithm == "default":
        return DefaultIncentive(
            config=config,
            redis_service=mock_redis,
            burn_service=burn_service,
            jobs_results=job_results,
            total_gpu_model_count_map=gpu_count_map,
        )
    if algorithm == "rental_price":
        inc = RentalPriceIncentive(
            config=config,
            redis_service=mock_redis,
            burn_service=burn_service,
            jobs_results=job_results,
            total_gpu_model_count_map=gpu_count_map,
        )
        s = prod_snapshot["settings"]
        inc.price_provider.set_mock_prices(
            tao_price=s.get("tao_price_usd", 400.0),
            alpha_rate=s.get("alpha_rate", 0.001),
        )
        return inc
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _calc_hourly_emission_usd(prod_snapshot: dict) -> float:
    """Calculate total hourly miner emission in USD.

    Mirrors compute-app SubtensorService.get_total_miner_rewards_in_usd:
      BLOCKS_PER_DAY * subnet.price * tao_price * MINER_EMISSION_PERCENTAGE / 24
    """
    BLOCKS_PER_DAY = 7200  # 24 * 3600 / 12
    MINER_EMISSION_PERCENTAGE = 0.41
    s = prod_snapshot["settings"]
    tao_price = s.get("tao_price_usd", 400.0)
    alpha_rate = s.get("alpha_rate", 0.001)
    return BLOCKS_PER_DAY * alpha_rate * tao_price * MINER_EMISSION_PERCENTAGE / 24


def _extract_actual_output(incentive: BaseIncentive) -> dict:
    """Extract actual results into the same shape as expected_output."""
    executor_mining_scores: dict[str, float] = {}
    for _hotkey, results in incentive.job_results.items():
        for r in results:
            executor_mining_scores[r.executor_info.uuid] = r.mining_score

    return {
        "executor_mining_scores": executor_mining_scores,
        "total_mining_score": incentive.total_mining_score,
        "miner_incentives": dict(incentive.miner_incentives),
        "metrics": {
            "total_executors": incentive.total_executors,
            "successful_executors": incentive.successful_executors,
            "failed_executors": incentive.failed_executors,
        },
    }


def _dump_results_csv(
    incentive: BaseIncentive,
    path: Path,
    hourly_emission_usd: float,
) -> None:
    """Write all job results to CSV for easy inspection."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for hotkey, results in incentive.job_results.items():
            for r in results:
                incentive_usd = (
                    r.incentive * hourly_emission_usd
                    if r.incentive is not None else None
                )
                writer.writerow({
                    "miner_hotkey": hotkey,
                    "executor_id": r.executor_info.uuid,
                    "gpu_model": r.gpu_model,
                    "gpu_count": r.gpu_count,
                    "gpu_count_multiplier": r.gpu_count_multiplier,
                    "score": r.score,
                    "job_score": r.job_score,
                    "collateral_deposited": r.collateral_deposited,
                    "sysbox_runtime": r.sysbox_runtime,
                    "is_rented": r.is_rented,
                    "mining_score": r.mining_score,
                    "sysbox_multiplier": r.sysbox_multiplier,
                    "uptime_multiplier": r.uptime_multiplier,
                    "gpu_portion": r.gpu_portion,
                    "total_gpu_count": r.total_gpu_count,
                    "eligible_for_rental_share": r.eligible_for_rental_share,
                    "hourly_rate": r.hourly_rate,
                    "max_cap": r.max_cap,
                    "total_unrented_by_gpu_type": r.total_unrented_by_gpu_type,
                    "cap_dilution_applied": r.cap_dilution_applied,
                    "unrented_cap_multiplier": r.unrented_cap_multiplier,
                    "effective_rate": r.effective_rate,
                    "rental_share": r.rental_share,
                    "burn_share": r.burn_share,
                    "total_rental_cost": r.total_rental_cost,
                    "incentive": r.incentive,
                    "incentive_usd_per_hour": incentive_usd,
                })


def _update_snapshot(snapshot_path: Path, algorithm: str, actual: dict) -> None:
    """Write actual results back to the snapshot JSON."""
    snapshot = json.loads(snapshot_path.read_text())
    snapshot[f"expected_output_{algorithm}"] = actual
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")


def _assert_snapshot(expected: dict, incentive: BaseIncentive) -> None:
    """Assert incentive results match expected snapshot values."""
    # Executor mining scores
    for _hotkey, results in incentive.job_results.items():
        for result in results:
            eid = result.executor_info.uuid
            exp_score = expected["executor_mining_scores"].get(eid, 0.0)
            assert result.mining_score == pytest.approx(exp_score, abs=1e-15), (
                f"Executor {eid}: got {result.mining_score}, expected {exp_score}"
            )

    # Total mining score
    assert incentive.total_mining_score == pytest.approx(
        expected["total_mining_score"], rel=1e-12
    )

    # Miner incentives
    for hotkey, exp_val in expected["miner_incentives"].items():
        actual = incentive.miner_incentives.get(hotkey, 0.0)
        assert actual == pytest.approx(exp_val, rel=1e-12), (
            f"Miner {hotkey}: got {actual}, expected {exp_val}"
        )

    # Metrics
    assert incentive.total_executors == expected["metrics"]["total_executors"]
    assert incentive.successful_executors == expected["metrics"]["successful_executors"]
    assert incentive.failed_executors == expected["metrics"]["failed_executors"]


async def _run_snapshot_test(
    snapshot_dir: Path,
    prod_snapshot: dict,
    mock_redis: AsyncMock,
    algorithm: str,
    update: bool,
) -> None:
    """Run incentive pipeline, dump CSV, assert or update snapshot."""
    snapshot_path = snapshot_dir / "snapshot.json"
    incentive = _create_incentive(prod_snapshot, mock_redis, algorithm)
    await incentive.calculate_mining_scores()

    # Dump CSV
    hourly_emission_usd = _calc_hourly_emission_usd(prod_snapshot)
    csv_path = snapshot_dir / f"results_{algorithm}.csv"
    _dump_results_csv(incentive, csv_path, hourly_emission_usd)

    # Assert or Update
    actual = _extract_actual_output(incentive)
    if update:
        _update_snapshot(snapshot_path, algorithm, actual)
        pytest.skip(f"Snapshot updated: expected_output_{algorithm}")
    else:
        expected = prod_snapshot.get(f"expected_output_{algorithm}")
        assert expected is not None, (
            f"No expected_output_{algorithm} in snapshot. "
            f"Run with --update-snapshot first."
        )
        _assert_snapshot(expected, incentive)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prod_snapshot_default(
    snapshot_dir: Path,
    prod_snapshot: dict,
    mock_redis_from_snapshot: AsyncMock,
    prod_settings: None,
    update_snapshot: bool,
):
    """Default incentive pipeline against production snapshot."""
    await _run_snapshot_test(
        snapshot_dir, prod_snapshot, mock_redis_from_snapshot, "default", update_snapshot,
    )


@pytest.mark.asyncio
async def test_prod_snapshot_rental_price(
    snapshot_dir: Path,
    prod_snapshot: dict,
    mock_redis_from_snapshot: AsyncMock,
    prod_settings: None,
    update_snapshot: bool,
):
    """Rental price incentive pipeline against production snapshot."""
    await _run_snapshot_test(
        snapshot_dir, prod_snapshot, mock_redis_from_snapshot, "rental_price", update_snapshot,
    )
