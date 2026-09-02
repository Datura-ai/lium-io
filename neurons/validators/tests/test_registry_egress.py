"""DAH-2835 — RegistryEgressCheck.

A host that cannot reach Docker Hub still passes every check we have, because none of them
touches the registry: the sysbox probe bundles its image, `docker run` carries no `--pull`,
and the cached-template check only inspects what is already local. So the host stays
verified and rentable, and every rental of a non-cached image dies at `docker_pull`.

The check probes the registry once a cycle and keeps a per-executor counter of the cycles in
a row that failed. One reachable cycle deletes the counter. The count rides
`executor.specs`; only a run of several cycles withholds the unrented incentive, and score is
never touched.
"""

import json
import time
from unittest.mock import AsyncMock, Mock

import pytest

from neurons.validators.src.services.task.checks.registry_egress import (
    COUNTER_WINDOW_SECONDS,
    RegistryEgressCheck,
    RegistryUnreachableRecord,
)
from neurons.validators.src.services.task.messages import RegistryEgressMessages as Msg

from tests.helpers import build_context_config, build_services, build_state

_DIGESTS = {"daturaai/torch:2.4.0": "sha256:aaa"}


def _ssh(stdout="401", exit_status=0, raises=False):
    ssh = AsyncMock()
    if raises:
        ssh.run = AsyncMock(side_effect=RuntimeError("ssh down"))
    else:
        ssh.run = AsyncMock(return_value=Mock(exit_status=exit_status, stdout=stdout))
    return ssh


def _redis(stored=None, get_raises=False, set_raises=False):
    down = RuntimeError("redis down")
    redis = AsyncMock()
    redis.get = AsyncMock(side_effect=down) if get_raises else AsyncMock(return_value=stored)
    redis.set = AsyncMock(side_effect=down) if set_raises else AsyncMock()
    redis.delete = AsyncMock()
    return redis


def _record(cycles: int, age_seconds: float = 0.0) -> str:
    return RegistryUnreachableRecord(
        cycles=cycles, last_seen_at=time.time() - age_seconds
    ).model_dump_json()


def _ctx(context_factory, ssh, redis=None, digests=None):
    return context_factory(
        config=build_context_config(
            default_docker_image_digests=_DIGESTS if digests is None else digests
        ),
        state=build_state(),
        services=build_services(redis=redis or _redis()),
        ssh=ssh,
    )


def _written_cycles(redis) -> int:
    return json.loads(redis.set.await_args.args[1])["cycles"]


def test_check_is_not_fatal():
    # Incentive-only enforcement: a broken registry must never zero a provider's score.
    assert RegistryEgressCheck.fatal is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["401", "200"])
async def test_registry_answers_publishes_zero_cycles(context_factory, status):
    # /v2/ answers 401 without credentials and 200 with them. Both prove egress works.
    ssh = _ssh(stdout=status)
    redis = _redis()
    ctx = _ctx(context_factory, ssh, redis)

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.REACHABLE.reason
    assert result.updates["state"].registry_unreachable_cycles == 0
    redis.delete.assert_awaited_once()
    cmd = ssh.run.await_args.args[0]
    assert "registry-1.docker.io/v2/" in cmd
    assert ssh.run.await_args.kwargs["check"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["000", "503"])
async def test_first_bad_cycle_counts_one(context_factory, status):
    # curl writes 000 when it never got a response; a 5xx is not egress a rental can start on.
    redis = _redis(stored=None)
    ctx = _ctx(context_factory, _ssh(stdout=status), redis)

    result = await RegistryEgressCheck().run(ctx)

    assert result.event.reason_code == Msg.UNREACHABLE.reason
    assert result.event.what_we_saw["http_status"] == status
    assert result.updates["state"].registry_unreachable_cycles == 1
    assert _written_cycles(redis) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("stored_cycles", "expected"), [(1, 2), (2, 3)])
async def test_bad_cycle_increments_stored_counter(context_factory, stored_cycles, expected):
    redis = _redis(stored=_record(stored_cycles))
    ctx = _ctx(context_factory, _ssh(stdout="000"), redis)

    result = await RegistryEgressCheck().run(ctx)

    assert result.updates["state"].registry_unreachable_cycles == expected
    assert _written_cycles(redis) == expected


@pytest.mark.asyncio
async def test_one_good_cycle_forgives_the_run(context_factory):
    # Two bad cycles then a good one: the counter is deleted, not decremented.
    redis = _redis(stored=_record(2))
    ctx = _ctx(context_factory, _ssh(stdout="401"), redis)

    result = await RegistryEgressCheck().run(ctx)

    assert result.updates["state"].registry_unreachable_cycles == 0
    redis.delete.assert_awaited_once()
    redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_record_starts_the_count_again(context_factory):
    # RedisService.set has no TTL, so a run nobody has observed for hours ages out by itself.
    redis = _redis(stored=_record(5, age_seconds=COUNTER_WINDOW_SECONDS + 60))
    ctx = _ctx(context_factory, _ssh(stdout="000"), redis)

    result = await RegistryEgressCheck().run(ctx)

    assert result.updates["state"].registry_unreachable_cycles == 1
    assert _written_cycles(redis) == 1


@pytest.mark.asyncio
async def test_corrupt_record_starts_the_count_again(context_factory):
    redis = _redis(stored=b"not json")
    ctx = _ctx(context_factory, _ssh(stdout="000"), redis)

    result = await RegistryEgressCheck().run(ctx)

    assert result.updates["state"].registry_unreachable_cycles == 1


@pytest.mark.asyncio
async def test_redis_read_failure_publishes_nothing(context_factory):
    # Our own Redis being down says nothing about the provider's network.
    ctx = _ctx(context_factory, _ssh(stdout="000"), _redis(get_raises=True))

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}


@pytest.mark.asyncio
async def test_redis_write_failure_publishes_nothing(context_factory):
    # A count we could not persist would restart at 1 next cycle and never reach the gate.
    ctx = _ctx(context_factory, _ssh(stdout="000"), _redis(set_raises=True))

    result = await RegistryEgressCheck().run(ctx)

    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}


@pytest.mark.asyncio
async def test_redis_delete_failure_publishes_nothing(context_factory):
    redis = _redis()
    redis.delete = AsyncMock(side_effect=RuntimeError("redis down"))
    ctx = _ctx(context_factory, _ssh(stdout="401"), redis)

    result = await RegistryEgressCheck().run(ctx)

    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}


@pytest.mark.asyncio
async def test_validator_lost_docker_hub_skips(context_factory):
    # The guard that stops a Docker Hub outage from penalising the whole fleet: an empty digest
    # snapshot means the VALIDATOR could not reach the registry this cycle, so a host that
    # cannot reach it either proves nothing. Publish nothing, touch no counter.
    ssh = _ssh(stdout="000")
    redis = _redis()
    ctx = _ctx(context_factory, ssh, redis, digests={})

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}
    ssh.run.assert_not_awaited()
    redis.get.assert_not_awaited()
    redis.set.assert_not_awaited()
    redis.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_ssh_failure_fails_open(context_factory):
    # An SSH error says nothing about the registry. Publish nothing rather than a false count.
    ctx = _ctx(context_factory, _ssh(raises=True))

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}


@pytest.mark.asyncio
async def test_unreadable_output_fails_open(context_factory):
    # curl printed something that is not a status code: unknown, not unreachable.
    ctx = _ctx(context_factory, _ssh(stdout="curl: command not found", exit_status=127))

    result = await RegistryEgressCheck().run(ctx)

    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.updates == {}
