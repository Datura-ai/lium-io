"""DAH-6540: a rent moved to another node keeps its pod id, so a delete sent to the node it left must not cancel, or
wait for, the pod's create on the new node."""

import asyncio

import pytest
from services.docker_service import create_steps_after_reply, inflight_creates

OLD, NEW = ("miner-hk", "exec-old"), ("miner-hk", "exec-new")


@pytest.mark.asyncio
async def test_a_delete_on_the_old_node_cancels_only_the_create_there():
    with inflight_creates.track("pod-6540", OLD):
        new_create_seen = asyncio.get_running_loop().create_future()

        async def new_node_create():
            with inflight_creates.track("pod-6540", NEW):
                await asyncio.sleep(0)
                assert inflight_creates.cancel("pod-6540", OLD)
                new_create_seen.set_result(inflight_creates.is_cancelled("pod-6540"))

        await asyncio.create_task(new_node_create())
        assert inflight_creates.is_cancelled("pod-6540")  # the old node's create, read from its own task

    assert await new_create_seen is False


@pytest.mark.asyncio
async def test_a_delete_on_the_old_node_does_not_wait_for_the_new_nodes_create():
    release = asyncio.Event()

    async def new_node_create():
        with inflight_creates.track("pod-6540", NEW):
            await release.wait()

    create = asyncio.create_task(new_node_create())
    await asyncio.sleep(0)
    assert not inflight_creates.cancel("pod-6540", OLD)
    assert await inflight_creates.wait_until_done("pod-6540", 0.01, OLD)
    assert not await inflight_creates.wait_until_done("pod-6540", 0.01, NEW)
    release.set()
    await create


@pytest.mark.asyncio
async def test_a_delete_on_the_old_node_does_not_cut_the_new_nodes_steps_after_reply():
    release = asyncio.Event()
    create_steps_after_reply.start("pod-6540", NEW, release.wait())

    assert await create_steps_after_reply.wait_until_done("pod-6540", OLD, 0.01)
    release.set()
    assert await create_steps_after_reply.wait_until_done("pod-6540", NEW, 1)
