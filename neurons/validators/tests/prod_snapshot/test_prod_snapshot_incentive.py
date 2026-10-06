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

import copy
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
    "is_spot",
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
    monkeypatch.setattr(settings, "SKIP_COLLATERAL_PENALTY", False)
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
                # Decides which sysbox rule the rented executor falls under: a rental created
                # on or after SYSBOX_RENTED_CUTOFF scores 0 without sysbox, an older one only
                # loses PORTION_FOR_SYSBOX. Absent in a snapshot = the older, lenient rule.
                rental_created_at=r.get("rental_created_at"),
                rented_gpu_count=r.get("rented_gpu_count"),
                job_batch_id=r["job_batch_id"],
                log_status=r["log_status"],
                log_text=r["log_text"],
                supports_gpu_splitting=r.get("supports_gpu_splitting", False),
                gpu_splitting_min_count=r.get("gpu_splitting_min_count"),
                is_spot=r.get("is_spot", False),
                default_job_owner=r.get("default_job_owner"),
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
    executor_incentives: dict[str, float | None] = {}
    executor_gpu_counts: dict[str, int] = {}
    for _hotkey, results in incentive.job_results.items():
        for r in results:
            executor_mining_scores[r.executor_info.uuid] = r.mining_score
            # The per-executor payout the backend accounts on, and the GPU count it is
            # published with — a partially rented split node (DAH-2467) is scored as two
            # portions and must arrive here merged back into one whole-box row.
            executor_incentives[r.executor_info.uuid] = r.incentive
            executor_gpu_counts[r.executor_info.uuid] = r.gpu_count

    return {
        "executor_mining_scores": executor_mining_scores,
        "executor_incentives": executor_incentives,
        "executor_gpu_counts": executor_gpu_counts,
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
                    "score": r.score,
                    "job_score": r.job_score,
                    "collateral_deposited": r.collateral_deposited,
                    "sysbox_runtime": r.sysbox_runtime,
                    "is_rented": r.is_rented,
                    "is_spot": r.is_spot,
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


def _results_by_executor_id(incentive: BaseIncentive) -> dict[str, JobResult]:
    return {
        r.executor_info.uuid: r
        for results in incentive.job_results.values()
        for r in results
    }


def _partially_rented_split_rows(prod_snapshot: dict) -> list[dict]:
    """Snapshot rows for split-opted-in executors with only part of their GPUs rented (DAH-2467)."""
    return [
        r
        for results in prod_snapshot["job_results"].values()
        for r in results
        if r["is_rented"]
        and r.get("supports_gpu_splitting")
        and r.get("gpu_splitting_min_count")
        and 0 < (r.get("rented_gpu_count") or 0) < r["gpu_count"]
    ]


def _snapshot_without_rented_gpu_counts(prod_snapshot: dict) -> dict:
    """The same snapshot as a backend that does not report per-pod gpu_count would deliver it."""
    downgraded = copy.deepcopy(prod_snapshot)
    for results in downgraded["job_results"].values():
        for r in results:
            r.pop("rented_gpu_count", None)
    return downgraded


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

    # Per-executor incentive and published GPU count
    for _hotkey, results in incentive.job_results.items():
        for result in results:
            eid = result.executor_info.uuid
            exp_incentive = expected["executor_incentives"].get(eid)
            assert result.incentive == pytest.approx(exp_incentive, rel=1e-12), (
                f"Executor {eid} incentive: got {result.incentive}, expected {exp_incentive}"
            )
            exp_gpu_count = expected["executor_gpu_counts"].get(eid)
            assert result.gpu_count == exp_gpu_count, (
                f"Executor {eid} gpu_count: got {result.gpu_count}, expected {exp_gpu_count}"
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


async def _score_with_and_without_rented_gpu_counts(
    prod_snapshot: dict,
    mock_redis: AsyncMock,
) -> tuple[BaseIncentive, BaseIncentive]:
    """The same pool scored as (mixed pools, whole-box) — the latter is today's behavior."""
    mixed = _create_incentive(prod_snapshot, mock_redis, "rental_price")
    await mixed.calculate_mining_scores()
    whole_box = _create_incentive(
        _snapshot_without_rented_gpu_counts(prod_snapshot), mock_redis, "rental_price"
    )
    await whole_box.calculate_mining_scores()
    return mixed, whole_box


@pytest.mark.asyncio
async def test_prod_snapshot_partially_rented_split_node_is_scored_in_both_pools(
    snapshot_dir: Path,
    prod_snapshot: dict,
    mock_redis_from_snapshot: AsyncMock,
    prod_settings: None,
):
    """DAH-2467 — the same prod pool scored with and without the backend's per-pod GPU counts.

    Holds for every partially rented split node regardless of what its free GPUs end up
    earning: the mining pool pays for the rented GPUs only, the free GPUs join the unrented
    pool, and the executor still publishes as ONE row at its full GPU count.
    """
    split_rows = _partially_rented_split_rows(prod_snapshot)
    if not split_rows:
        pytest.skip("no partially rented split node in this snapshot")

    mixed, whole_box = await _score_with_and_without_rented_gpu_counts(
        prod_snapshot, mock_redis_from_snapshot
    )
    mixed_results = _results_by_executor_id(mixed)
    whole_box_results = _results_by_executor_id(whole_box)
    for row in split_rows:
        executor_id = row["executor_id"]
        mixed_result = mixed_results[executor_id]
        # Merged back into one row: machine-spec publish still sees the whole box.
        assert mixed_result.gpu_count == row["gpu_count"]
        assert mixed_result.is_rented is True
        # The mining pool now pays for the rented GPUs only.
        assert mixed_result.mining_score < whole_box_results[executor_id].mining_score

    # Every free GPU joined the unrented pool — that is what the second portion is for.
    free_gpu_count = sum(row["gpu_count"] - row["rented_gpu_count"] for row in split_rows)
    assert (
        sum(mixed.unrented_count_by_bucket.values())
        - sum(whole_box.unrented_count_by_bucket.values())
    ) == free_gpu_count


@pytest.mark.asyncio
async def test_prod_snapshot_split_node_earns_more_when_its_free_gpus_can_earn(
    snapshot_dir: Path,
    prod_snapshot: dict,
    mock_redis_from_snapshot: AsyncMock,
    prod_settings: None,
):
    """DAH-2467 — the goal of the change, on a real fleet rather than a mocked rental share.

    Scoped to split nodes whose free GPUs are actually admitted to the unrented pool, which
    under this snapshot's rules means running sysbox. Such a node must come out ahead of
    today's whole-box scoring: it gives up mining score for the rented GPUs and more than
    makes it back on the free ones.
    """
    split_rows = [r for r in _partially_rented_split_rows(prod_snapshot) if r["sysbox_runtime"]]
    if not split_rows:
        pytest.skip("no partially rented split node with sysbox in this snapshot")

    mixed, whole_box = await _score_with_and_without_rented_gpu_counts(
        prod_snapshot, mock_redis_from_snapshot
    )
    mixed_results = _results_by_executor_id(mixed)
    whole_box_results = _results_by_executor_id(whole_box)
    for row in split_rows:
        executor_id = row["executor_id"]
        assert mixed_results[executor_id].incentive > whole_box_results[executor_id].incentive, (
            f"Executor {executor_id} ({row['rented_gpu_count']} of {row['gpu_count']} GPUs "
            f"rented) earns less than it does scored whole-box"
        )
