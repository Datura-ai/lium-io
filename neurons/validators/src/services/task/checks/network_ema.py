NETWORK_EMA_ALPHA = 0.7


def compute_ema(prev: float | None, current: float | None, alpha: float = NETWORK_EMA_ALPHA) -> float | None:
    """Compute EMA for a network speed measurement.

    Bootstrap rule: if prev is None (first measurement), returns current as the initial EMA.
    If current is None (measurement failed), preserves prev unchanged.
    """
    if current is None:
        return prev
    if prev is None:
        return current
    return alpha * prev + (1 - alpha) * current
