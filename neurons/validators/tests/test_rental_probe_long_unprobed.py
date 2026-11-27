import logging
import time

import asyncssh
import pytest
from neurons.validators.src.services.task.checks import rental_probe as module
from neurons.validators.src.services.task.checks.rental_probe import RentalProbeCheck
from neurons.validators.src.services.task.messages import RentalProbeMessages as Msg

from tests.test_rental_probe_check import FakeRedis, make_probe_context, probe_settings

HOUR = 3600


async def _unprobed_logs(caplog, **context) -> tuple[int, str]:
    ctx, docker, _ = make_probe_context(**context)
    return await _count_unprobed_logs_and_reason(caplog, ctx, docker)


async def _count_unprobed_logs_and_reason(caplog, ctx, docker) -> tuple[int, str]:
    caplog.clear()
    with probe_settings(), caplog.at_level(logging.WARNING, logger=module.logger.name):
        result = await RentalProbeCheck().run(ctx)
    assert docker.create_calls == []
    unprobed_log_count = sum("RENTAL_PROBE_LONG_UNPROBED" in record.getMessage() for record in caplog.records)
    return unprobed_log_count, result.event.reason_code


@pytest.mark.asyncio
@pytest.mark.real_rental_probe_proof
async def test_a_never_probed_node_without_the_image_is_not_listed(caplog):
    # unlisted until its first pass, which says more than the long-unprobed log line
    redis = FakeRedis(last_ok=None)

    assert await _unprobed_logs(caplog, image_present=False, redis=redis) == (
        0,
        Msg.NOT_YET_PROVEN.reason,
    )


@pytest.mark.asyncio
async def test_a_node_last_probed_days_ago_without_the_image_is_logged(caplog):
    redis = FakeRedis(last_ok=time.time() - 48 * HOUR)

    assert await _unprobed_logs(caplog, image_present=False, redis=redis) == (1, Msg.SKIPPED.reason)


@pytest.mark.asyncio
async def test_a_node_probed_within_four_intervals_is_not_logged(caplog):
    redis = FakeRedis(last_ok=time.time() - 7 * HOUR)

    assert await _unprobed_logs(caplog, image_present=False, redis=redis) == (0, Msg.SKIPPED.reason)


@pytest.mark.asyncio
@pytest.mark.real_rental_probe_proof
async def test_an_unreadable_image_list_on_a_never_probed_node_keeps_its_listing(caplog):
    # an image list the validator could not read is the validator's side, so the stored listing stands
    ctx, docker, _ = make_probe_context(redis=FakeRedis(last_ok=None))
    ctx.ssh.run.side_effect = asyncssh.ConnectionLost("lost")

    result = await _count_unprobed_logs_and_reason(caplog, ctx, docker)

    assert result == (0, Msg.NOT_RUN_ON_VALIDATOR.reason)


@pytest.mark.asyncio
async def test_an_unreadable_image_list_on_a_node_last_probed_days_ago_is_logged(caplog):
    ctx, docker, _ = make_probe_context(redis=FakeRedis(last_ok=time.time() - 48 * HOUR))
    ctx.ssh.run.side_effect = asyncssh.ConnectionLost("lost")

    result = await _count_unprobed_logs_and_reason(caplog, ctx, docker)

    assert result == (1, Msg.INCONCLUSIVE.reason)
