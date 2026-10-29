"""DAH-4272: the verified record (count, spec, GPU anchor, broken mark) is per (miner hotkey, executor id).

An executor id is self-declared and unique only per hotkey (the backend's uq_miner_executor), so a hotkey that
lists another's executor id writes, resets and marks only its own record.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from neurons.validators.src.services.task import pipeline_factory as pipeline_factory_module
from neurons.validators.src.services.task.pipeline_factory import PipelineFactory
from services.redis_service import (
    GPU_ANCHOR_BROKEN_KEY,
    RESET_VERIFIED_JOB_CHANNEL,
    VERIFIED_JOB_COUNT_KEY,
    RedisService,
)

B8717923 = "b8717923-0000-4000-8000-000000000000"
RUNNING_HOTKEY = "5C8CG5Jk-running-hotkey"
OTHER_HOTKEY = "5Other-hotkey"


@pytest.fixture
def service():
    service = RedisService.__new__(RedisService)
    service.redis = FakeRedis(server=FakeServer())
    service.lock = asyncio.Lock()
    service.publish = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_a_reset_and_anchor_mark_by_another_hotkey_leave_the_record_alone(service):
    await service.set_verified_job_info(RUNNING_HOTKEY, B8717923, prev_info={}, success=True, spec="B300:8", uuids="GPU-a")
    honest = await service.get_verified_job_info(RUNNING_HOTKEY, B8717923)

    await service.clear_verified_job_info(OTHER_HOTKEY, B8717923, prev_info={}, anchor_broken=True)

    assert await service.get_verified_job_info(RUNNING_HOTKEY, B8717923) == honest
    assert GPU_ANCHOR_BROKEN_KEY not in honest
    assert (await service.get_verified_job_info(OTHER_HOTKEY, B8717923))[GPU_ANCHOR_BROKEN_KEY] is True
    channel, published = service.publish.await_args.args
    assert (channel, published["miner_hotkey"], published["executor_uuid"]) == (
        RESET_VERIFIED_JOB_CHANNEL, OTHER_HOTKEY, B8717923
    )


@pytest.mark.asyncio
async def test_a_record_from_before_the_release_moves_only_to_the_listed_owner(service):
    legacy = {"count": 7, "failed": 0, "spec": "B300:8", "uuids": "GPU-a", GPU_ANCHOR_BROKEN_KEY: True}
    await service.redis.hset(VERIFIED_JOB_COUNT_KEY, B8717923, json.dumps(legacy))

    assert await service.get_verified_job_info(OTHER_HOTKEY, B8717923, listed_owner_hotkey=RUNNING_HOTKEY) == {}
    assert await service.get_verified_job_info(RUNNING_HOTKEY, B8717923) == {}

    prev = await service.get_verified_job_info(RUNNING_HOTKEY, B8717923, listed_owner_hotkey=RUNNING_HOTKEY)
    assert prev == legacy
    assert await service.redis.hget(VERIFIED_JOB_COUNT_KEY, B8717923) is None
    assert await service.get_verified_job_info(OTHER_HOTKEY, B8717923, listed_owner_hotkey=OTHER_HOTKEY) == {}
    await service.set_verified_job_info(RUNNING_HOTKEY, B8717923, prev_info=prev, success=True)

    written = json.loads(await service.redis.hget(VERIFIED_JOB_COUNT_KEY, f"{RUNNING_HOTKEY}:{B8717923}"))
    assert written == {**legacy, "count": 8}
    assert await service.get_verified_job_info(RUNNING_HOTKEY, B8717923) == written


@pytest.mark.asyncio
async def test_build_context_reads_the_record_of_the_validated_hotkey(monkeypatch):
    # Context validates a live asyncssh connection; capture the kwargs instead
    monkeypatch.setattr(pipeline_factory_module, "Context", lambda **kw: SimpleNamespace(**kw))
    factory = PipelineFactory.__new__(PipelineFactory)
    factory.redis_service = SimpleNamespace(
        get_verified_job_info=AsyncMock(return_value={}),
        is_elem_exists_in_set=AsyncMock(return_value=False),
    )
    for name in (
        "ssh_service",
        "validation_service",
        "verifyx_validation_service",
        "inspector_validation_service",
        "executor_connectivity_service",
        "backend_client",
        "pod_recovery",
        "container_cleanup",
    ):
        setattr(factory, name, MagicMock())
    executor = SimpleNamespace(
        uuid=B8717923, address="1.2.3.4", port=8000, ssh_username="root", ssh_port=22,
        root_dir="/root/app",
    )
    encrypted_files = SimpleNamespace(
        encrypt_key="k", machine_scrape_file_name="scrape", machine_scrape_source=None,
        all_keys={}, tmp_directory="/tmp/x",
    )
    miner_info = SimpleNamespace(
        job_batch_id="b", miner_hotkey=RUNNING_HOTKEY, miner_coldkey="c",
        miner_address="1.1.1.1", miner_port=1,
    )

    await factory.build_context(
        shell=SimpleNamespace(ssh_client=MagicMock()),
        miner_info=miner_info,
        executor_info=executor,
        keypair=None,
        private_key="p",
        public_key="q",
        encrypted_files=encrypted_files,
        rented_data=None,
        default_docker_image_digests={},
    )

    factory.redis_service.get_verified_job_info.assert_awaited_once_with(RUNNING_HOTKEY, B8717923, None)
