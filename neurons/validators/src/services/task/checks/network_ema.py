NETWORK_EMA_ALPHA = 0.5


def compute_ema(prev: float | None, current: float | None, alpha: float = NETWORK_EMA_ALPHA) -> float | None:
    """Compute EMA for a network speed measurement.

    EMA = alpha * current + (1 - alpha) * prev

    Bootstrap rule: if prev is None (first measurement), returns current as the initial EMA.
    If current is None (measurement failed), preserves prev unchanged.
    """
    if current is None:
        return prev
    if prev is None:
        return current
    return alpha * current + (1 - alpha) * prev


def resolve_display_speeds(network: dict) -> tuple[float | None, float | None]:
    """Pick the (upload, download) speeds that renters and providers are shown.

    Most trusted first: the verifyx EMA, then the scrape EMA, then the raw measurements.
    ``0`` counts as missing, so a zeroed EMA does not mask a real measurement. Resolved here,
    once, so that no consumer has to repeat the priority order and drift from it.
    """
    upload = (
        network.get("ema_verifyx_upload_speed")
        or network.get("ema_upload_speed")
        or network.get("verifyx_upload_speed")
        or network.get("upload_speed")
        or None
    )
    download = (
        network.get("ema_verifyx_download_speed")
        or network.get("ema_download_speed")
        or network.get("verifyx_download_speed")
        or network.get("download_speed")
        or None
    )
    return upload, download
