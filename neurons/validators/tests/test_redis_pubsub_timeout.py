"""DAH-2475 regression: a quiet pubsub channel must not be torn down by the command socket timeout.

The Redis hardening added `socket_timeout` so a stuck command fails fast instead of hanging. That is
right for request/response commands, but `pubsub.listen()` is a BLOCKING read on a channel that is
idle most of the time — redis-py falls back to `socket_timeout` when the caller passes no deadline,
so the shared client aborted the subscription after that many seconds of silence.

`subscribe_mesages_from_redis` is the validator's main inbound channel (machine specs, log streams,
inspector events, score resets, GPU estimates); it reconnects on error, so the effect was a silent
reconnect loop rather than an outage — and, with a bounded pool, a slow connection leak.
"""

import inspect

import pytest
import redis.asyncio as aioredis

from services.redis_service import RedisService


def test_the_retry_can_actually_retry_an_async_call() -> None:
    # redis.retry.Retry and redis.asyncio.retry.Retry have the same signature, so passing the
    # synchronous one to an async client raises nothing and silently performs ZERO retries: its
    # `return do()` hands back the coroutine, which the caller awaits OUTSIDE the try block.
    for pool in (RedisService().redis, RedisService().pubsub_redis):
        retry = pool.connection_pool.connection_kwargs.get("retry")
        assert retry is not None
        assert inspect.iscoroutinefunction(type(retry).call_with_retry), (
            f"{type(retry).__module__}.{type(retry).__name__} cannot retry an awaitable"
        )


@pytest.mark.asyncio
async def test_a_failed_subscribe_does_not_leak_its_pooled_connection() -> None:
    # PubSub.execute_command acquires the connection BEFORE sending SUBSCRIBE, and PubSub.__del__
    # only deregisters a callback — it never returns the connection. So a SUBSCRIBE that fails after
    # connect strands one connection, and the caller cannot close a pubsub it was never handed.
    service = RedisService()
    closed: list[bool] = []

    class FailingPubSub:
        async def subscribe(self, *channel: str) -> None:
            raise ConnectionError("redis is down")

        async def aclose(self) -> None:
            closed.append(True)

    service.pubsub_redis.pubsub = lambda *args, **kwargs: FailingPubSub()

    with pytest.raises(ConnectionError):
        await service.subscribe("SOME_CHANNEL")

    assert closed == [True], "the pubsub was not closed, so its pooled connection leaked"


def test_commands_keep_a_socket_timeout() -> None:
    # The whole point of the hardening: a stuck command must not hang forever.
    service = RedisService()
    assert service.redis.connection_pool.connection_kwargs.get("socket_timeout") is not None


def test_subscriptions_do_not_inherit_the_command_socket_timeout() -> None:
    # A blocking listen() on an idle channel would be aborted by any socket_timeout.
    service = RedisService()
    pubsub_kwargs = service.pubsub_redis.connection_pool.connection_kwargs
    assert pubsub_kwargs.get("socket_timeout") is None


def test_subscriptions_use_a_separate_pool_from_commands() -> None:
    # Sharing the pool is what leaked the timeout into listen() in the first place; keeping them
    # apart also stops a subscription from eating a command connection.
    service = RedisService()
    assert service.pubsub_redis.connection_pool is not service.redis.connection_pool


def test_subscriptions_still_bound_their_connections() -> None:
    # Unbounded was the pre-hardening behaviour; a leak there is unbounded too.
    service = RedisService()
    assert isinstance(service.pubsub_redis.connection_pool, aioredis.BlockingConnectionPool)
    assert service.pubsub_redis.connection_pool.max_connections > 0
