import logging

import pytest
from neurons.validators.src.services.task.checks import verifyx
from neurons.validators.src.services.task.checks.verifyx import MAX_PACKAGE_FALLBACK_PROBES

from tests.test_verifyx_capacity_probe import (
    HOST_CAPACITY_MBPS,
    HOST_UPLOAD_MBPS,
    _cloudflare_unreachable_payload,
    _judge,
    _probe_payload,
    _run_check,
)


async def _download_emas(context_factory, payloads: list[dict]) -> list[float]:
    emas = [HOST_CAPACITY_MBPS]
    for payload in payloads:
        result = await _run_check(
            context_factory, _judge(payload), prev_ema=emas[-1], prev_upload=HOST_UPLOAD_MBPS
        )
        emas.append(result.updates["state"].specs["network"]["ema_verifyx_download_speed"])
    return emas[1:]


def _streak_logs(caplog) -> int:
    return sum(
        "VERIFYX_PACKAGE_FALLBACK_STREAK" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_past_the_cap_the_streak_is_logged_and_the_missing_capacity_reading_is_fed_as_zero(
    context_factory, caplog
):
    blocked = [_cloudflare_unreachable_payload()] * (MAX_PACKAGE_FALLBACK_PROBES + 1)

    with caplog.at_level(logging.WARNING, logger=verifyx.logger.name):
        emas = await _download_emas(context_factory, blocked)

    assert _streak_logs(caplog) == 1
    assert emas[-1] == pytest.approx(emas[-2] / 2)


@pytest.mark.asyncio
async def test_a_measured_probe_starts_the_count_again(context_factory, caplog):
    blocked = [_cloudflare_unreachable_payload()] * MAX_PACKAGE_FALLBACK_PROBES

    with caplog.at_level(logging.WARNING, logger=verifyx.logger.name):
        await _download_emas(context_factory, blocked + [_probe_payload()] + blocked)

    assert _streak_logs(caplog) == 0
