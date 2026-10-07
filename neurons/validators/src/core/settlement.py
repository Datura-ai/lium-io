"""DAH-4001: the pure parts of delayed settlement on the validator side.

The backend holds every cycle's vector for a day and serves the settled vector per tempo; what lives here is
how a cycle's per-node rows are described, how the fallback vector is kept, and how a tempo is numbered.
"""

from collections import Counter


def tempo_index(block: int, tempo: int) -> int:
    """The tempo a block belongs to: the window the backend serves for it is one tempo of cycles a day earlier."""
    return block // tempo


def cycle_node_shares(job_results: dict[str, list]) -> list[dict]:
    """The per-node rows behind a cycle's vector: one per priced result, with its rented and idle parts and whether
    the idle part is spot-node pay (earned on the Spot tier, never withheld)."""
    rows: list[dict] = []
    for miner_hotkey, results in job_results.items():
        for result in results:
            rented = float(result.incentive_rented or 0.0)
            idle = float(result.incentive_idle or 0.0)
            if rented <= 0 and idle <= 0:
                continue
            rows.append(
                {
                    "executor_id": str(result.executor_info.uuid),
                    "hotkey": miner_hotkey,
                    "rented": rented,
                    "idle": idle,
                    "spot": bool(result.is_spot),
                }
            )
    return rows


def fallback_vector(
    cycle_scores: dict[str, float], node_shares: list[dict], burner_hotkey: str | None
) -> dict[str, float]:
    """The cycle's vector with every idle share moved to the verified burner: what the validator submits when the
    backend cannot serve a tempo. Rental and referral stay as scored; the idle of that tempo is forfeited to the
    burner rather than paid from a stale vector. With no verified burner the vector is submitted as scored."""
    if burner_hotkey is None:
        return dict(cycle_scores)
    vector = Counter(cycle_scores)
    for row in node_shares:
        moved = min(float(row["idle"]), vector.get(row["hotkey"], 0.0))
        if moved <= 0:
            continue
        vector[row["hotkey"]] -= moved
        vector[burner_hotkey] += moved
    return dict(vector)


def accumulate(into: dict[str, float], vector: dict[str, float]) -> None:
    for hotkey, score in vector.items():
        into[hotkey] = into.get(hotkey, 0.0) + score


def share_moved(live: dict[str, float], settled: dict[str, float]) -> float:
    """Half the L1 distance between the two normalized vectors: the share of emission that would move."""
    live_total = sum(live.values()) or 1.0
    settled_total = sum(settled.values()) or 1.0
    hotkeys = set(live) | set(settled)
    return (
        sum(
            abs(live.get(h, 0.0) / live_total - settled.get(h, 0.0) / settled_total)
            for h in hotkeys
        )
        / 2
    )
