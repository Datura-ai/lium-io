import json
import asyncio
import logging
import time
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason
import redis.asyncio as aioredis
from datura.requests.miner_requests import ExecutorSSHInfo
from protocol.vc_protocol.compute_requests import ExecutorUptimeResponse, RentedMachine
from core.config import settings
from core.utils import _m
from services.const import GPU_MODEL_RATES

MACHINE_SPEC_CHANNEL = "MACHINE_SPEC_CHANNEL"
STREAMING_LOG_CHANNEL = "STREAMING_LOG_CHANNEL"
RESET_VERIFIED_JOB_CHANNEL = "RESET_VERIFIED_JOB_CHANNEL"
RENTED_MACHINE_PREFIX = "rented_machines_prefix"
PENDING_PODS_PREFIX = "pending_pods_prefix"
DUPLICATED_MACHINE_SET = "duplicated_machines"
RENTAL_SUCCEED_MACHINE_SET = "rental_succeed_machines"
AVAILABLE_PORT_MAPS_PREFIX = "available_port_maps"
VERIFIED_JOB_COUNT_KEY = "verified_job_counts"
EXECUTORS_UPTIME_PREFIX = "executors_uptime"
NORMALIZED_SCORE_CHANNEL = "normalized_score_channel"
REVENUE_PER_GPU_TYPE_SET = "revenue_per_gpu_type"
BANNED_GUIDS = "banned_guids"
PORTION_PER_GPU_TYPE_SET = "portion_per_gpu_type"

logger = logging.getLogger(__name__)


class RedisService:
    def __init__(self):
        self.redis = aioredis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}")
        self.lock = asyncio.Lock()

    async def publish(self, channel: str, message: dict):
        """Publish a message to a Redis channel."""
        await self.redis.publish(channel, json.dumps(message))

    async def subscribe(self, *channel: str):
        """Subscribe to a Redis channel."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(*channel)
        return pubsub

    async def set(self, key: str, value: str):
        """Set a key-value pair in Redis."""
        async with self.lock:
            await self.redis.set(key, value)

    async def get(self, key: str):
        """Get a value by key from Redis."""
        async with self.lock:
            return await self.redis.get(key)

    async def delete(self, key: str):
        """Remove a key from Redis."""
        async with self.lock:
            await self.redis.delete(key)

    async def sadd(self, key: str, elem: str):
        """Add an element to a set in Redis."""
        async with self.lock:
            await self.redis.sadd(key, elem)

    async def srem(self, key: str, elem: str):
        """Remove an element from a set in Redis."""
        async with self.lock:
            await self.redis.srem(key, elem)

    async def is_elem_exists_in_set(self, key: str, elem: str):
        """Check an element exists or not in a set in Redis."""
        async with self.lock:
            return await self.redis.sismember(key, elem)

    async def smembers(self, key: str):
        async with self.lock:
            return await self.redis.smembers(key)

    async def lpush(self, key: str, element: bytes):
        """Add an element to a list in Redis."""
        async with self.lock:
            await self.redis.lpush(key, element)

    async def lrange(self, key: str) -> list[bytes]:
        """Get all elements from a list in Redis in order."""
        async with self.lock:
            return await self.redis.lrange(key, 0, -1)

    async def lrem(self, key: str, element: bytes, count: int = 0):
        """Remove elements from a list in Redis."""
        async with self.lock:
            await self.redis.lrem(key, count, element)

    async def ltrim(self, key: str, max_length: int):
        """Trim the list to maintain a maximum length."""
        async with self.lock:
            await self.redis.ltrim(key, 0, max_length - 1)

    async def lpop(self, key: str) -> bytes:
        """Remove and return the first element (last inserted) from a list in Redis."""
        async with self.lock:
            return await self.redis.lpop(key)

    async def rpop(self, key: str) -> bytes:
        """Remove and return the last element (first inserted) from a list in Redis."""
        async with self.lock:
            return await self.redis.rpop(key)

    async def hset(self, key: str, field: str, value: str):
        async with self.lock:
            await self.redis.hset(key, field, value)

    async def hget(self, key: str, field: str):
        async with self.lock:
            return await self.redis.hget(key, field)

    async def hgetall(self, key: str):
        async with self.lock:
            return await self.redis.hgetall(key)

    async def hdel(self, key: str, *fields: str):
        async with self.lock:
            await self.redis.hdel(key, *fields)

    async def clear_by_pattern(self, pattern: str):
        async with self.lock:
            async for key in self.redis.scan_iter(match=pattern):
                await self.redis.delete(key.decode())

    async def add_rented_machine(self, machine: RentedMachine):
        await self.hset(RENTED_MACHINE_PREFIX, f"{machine.executor_ip_address}:{machine.executor_ip_port}", machine.model_dump_json())

    async def remove_rented_machine(self, executor: ExecutorSSHInfo):
        await self.hdel(RENTED_MACHINE_PREFIX, f"{executor.address}:{executor.port}")

    async def get_rented_machine(self, executor: ExecutorSSHInfo):
        data = await self.hget(RENTED_MACHINE_PREFIX, f"{executor.address}:{executor.port}")
        if not data:
            return None

        return json.loads(data)

    async def add_executor_uptime(self, machine: ExecutorUptimeResponse):
        await self.hset(EXECUTORS_UPTIME_PREFIX, f"{machine.executor_ip_address}:{machine.executor_ip_port}", str(machine.uptime_in_minutes))

    async def get_executor_uptime(self, executor: ExecutorSSHInfo) -> int:
        try:
            data = await self.hget(EXECUTORS_UPTIME_PREFIX, f"{executor.address}:{executor.port}")
            if not data:
                return 0
            return int(data)
        except Exception as e:
            logger.error(_m("Error getting executor uptime: {e}", extra={"error": e}), exc_info=True)
            return 0

    async def add_pending_pod(self, miner_hotkey: str, executor_id: str):
        now = int(time.time())
        await self.hset(PENDING_PODS_PREFIX, f"{miner_hotkey}:{executor_id}", json.dumps({"time": now}))

    async def remove_pending_pod(self, miner_hotkey: str, executor_id: str):
        await self.hdel(PENDING_PODS_PREFIX, f"{miner_hotkey}:{executor_id}")

    async def renting_in_progress(self, miner_hotkey: str, executor_id: str):
        data = await self.hget(PENDING_PODS_PREFIX, f"{miner_hotkey}:{executor_id}")
        if not data:
            return False

        now = int(time.time())
        data = json.loads(data)
        if now - data.get('time', 0) >= 30 * 60:  # 30 mins
            await self.remove_pending_pod(miner_hotkey, executor_id)
            return False

        return True

    async def set_verified_job_info(
        self,
        miner_hotkey: str,
        executor_id: str,
        prev_info: dict = {},
        success: bool = True,
        spec: str = '',
        uuids: str = '',
    ):
        count = prev_info.get('count', 0)
        failed = prev_info.get('failed', 0)
        prev_spec = prev_info.get('spec', '')
        prev_uuids = prev_info.get('uuids', '')

        if (success):
            count += 1
        else:
            failed += 1

        # if failed * 20 >= count:
        #     return await self.clear_verified_job_info(
        #         miner_hotkey=miner_hotkey,
        #         executor_id=executor_id,
        #         prev_info=prev_info,
        #     )

        data = {
            "count": count,
            "failed": failed,
            "spec": prev_spec if prev_spec else spec,
            "uuids": prev_uuids if prev_uuids else uuids,
        }

        await self.hset(VERIFIED_JOB_COUNT_KEY, executor_id, json.dumps(data))

    async def clear_verified_job_info(
        self,
        miner_hotkey: str,
        executor_id,
        prev_info: dict = {},
        reason: ResetVerifiedJobReason = ResetVerifiedJobReason.DEFAULT
    ):
        spec = prev_info.get('spec', '')
        uuids = prev_info.get('uuids', '')

        data = {
            "count": 0,
            "failed": 0,
            "spec": spec,
            "uuids": uuids,
        }
        await self.hset(VERIFIED_JOB_COUNT_KEY, executor_id, json.dumps(data))

        await self.publish(
            RESET_VERIFIED_JOB_CHANNEL,
            {
                "miner_hotkey": miner_hotkey,
                "executor_uuid": executor_id,
                "reason": reason.value,
            },
        )

    async def get_verified_job_info(self, executor_id: str):
        data = await self.hget(VERIFIED_JOB_COUNT_KEY, executor_id)
        if not data:
            return {}

        return json.loads(data)

    async def set_portion_per_gpu_type(self, gpu_type: str, portion: float):
        await self.hset(PORTION_PER_GPU_TYPE_SET, gpu_type, str(portion))

    async def get_portion_per_gpu_type(self, gpu_type: str):
        portion = await self.hget(PORTION_PER_GPU_TYPE_SET, gpu_type)
        if not portion:
            gpu_model_rate = GPU_MODEL_RATES.get(gpu_type, 0)
            return gpu_model_rate

        return float(portion)

    async def set_banned_guids(self, guids: list[str]):
        await self.redis.set(BANNED_GUIDS, json.dumps(guids))

    async def get_banned_guids(self) -> list[str]:
        data = await self.redis.get(BANNED_GUIDS)
        if not data:
            return []
        return json.loads(data)
