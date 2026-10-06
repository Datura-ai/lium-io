"""Tests for the dedicated Docker SDK thread pool (DAH-2475 follow-up).

Blocking Docker SDK calls used to run in the event loop's default executor. asyncio resolves DNS
in that same pool (`loop.getaddrinfo` -> `run_in_executor`), so a wave of filler creates filled it
with minutes-long pulls and name resolution queued behind them: every new Redis connection and the
backend WebSocket handshake timed out before reaching a socket. A thread pool is FIFO, so a wider
default pool does not fix this — the Docker calls need a queue of their own.
"""

import asyncio
import threading
from pathlib import Path

import pytest
from neurons.validators.src.core.utils import widen_default_thread_pool
from neurons.validators.src.services import rental_docker_sdk
from neurons.validators.src.services.rental_docker_sdk import (
    _DOCKER_EXECUTOR_MAX_WORKERS,
    _in_docker_thread,
)


def _current_thread_name() -> str:
    return threading.current_thread().name


@pytest.mark.asyncio
async def test_docker_calls_run_in_the_dedicated_pool() -> None:
    thread_name = await _in_docker_thread(_current_thread_name)

    assert thread_name.startswith("docker-sdk")


@pytest.mark.asyncio
async def test_docker_calls_forward_keyword_arguments() -> None:
    # create_volume() and friends pass their arguments by keyword.
    result = await _in_docker_thread(dict, name="volume", driver="local")

    assert result == {"name": "volume", "driver": "local"}


@pytest.mark.asyncio
async def test_default_pool_still_runs_while_every_docker_thread_is_busy() -> None:
    # The regression itself: with a shared pool, the call below queues behind the busy Docker
    # threads — that is how a create wave starved getaddrinfo and timed out every new connection.
    loop = asyncio.get_running_loop()
    entered = threading.Semaphore(0)
    release = threading.Event()

    def _occupy_thread() -> None:
        entered.release()
        release.wait()

    busy = [asyncio.create_task(_in_docker_thread(_occupy_thread)) for _ in range(64)]
    try:
        deadline = loop.time() + 5
        saturated = 0
        while saturated < _DOCKER_EXECUTOR_MAX_WORKERS and loop.time() < deadline:
            if entered.acquire(blocking=False):
                saturated += 1
            else:
                await asyncio.sleep(0.01)
        assert saturated == _DOCKER_EXECUTOR_MAX_WORKERS, "Docker pool never filled up, test proves nothing"

        thread_name = await asyncio.wait_for(
            loop.run_in_executor(None, _current_thread_name), timeout=5
        )
        assert not thread_name.startswith("docker-sdk")
    finally:
        release.set()
        await asyncio.gather(*busy)


def test_docker_sdk_module_does_not_use_the_default_executor() -> None:
    # asyncio.to_thread() always targets the default executor, so it must not come back here.
    source = Path(rental_docker_sdk.__file__).read_text()

    assert "asyncio.to_thread" not in source


@pytest.mark.asyncio
async def test_widen_default_thread_pool_replaces_the_default_executor() -> None:
    loop = asyncio.get_running_loop()

    widen_default_thread_pool(loop)

    thread_name = await loop.run_in_executor(None, _current_thread_name)
    assert thread_name.startswith("asyncio-default")
