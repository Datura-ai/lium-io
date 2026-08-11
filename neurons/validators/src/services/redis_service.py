import json
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from protocol.vc_protocol.validator_requests import ResetVerifiedJobReason
import redis.asyncio as aioredis
import redis.exceptions
from redis.backoff import ExponentialBackoff
from redis.asyncio.retry import Retry  # NOT redis.retry.Retry: the sync one silently never retries
from datura.requests.miner_requests import ExecutorSSHInfo
from protocol.vc_protocol.compute_requests import ExecutorUptimeResponse, RentedMachine
from core.config import settings
from core.utils import _m
from services.const import GPU_MODEL_RATES

MACHINE_SPEC_CHANNEL = "MACHINE_SPEC_CHANNEL"
STREAMING_LOG_CHANNEL = "STREAMING_LOG_CHANNEL"
INSPECTOR_EVENT_CHANNEL = "INSPECTOR_EVENT_CHANNEL"
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
GPU_ESTIMATES_CHANNEL = "gpu_estimates_channel"
GPU_ESTIMATES_KEY = "gpu_estimates"
INCENTIVE_SNAPSHOT_KEY = "incentive_snapshot"

# Distributed lock settings
EXECUTOR_LOCK_TIMEOUT = 30  # TTL for lock auto-release (seconds)
EXECUTOR_LOCK_BLOCKING_TIMEOUT = 10  # Time to wait for lock acquisition (seconds)

# DAH-2475: connection-pool resilience. The client used to be built with no options at all, which
# meant an UNBOUNDED pool and no retries: a wave of concurrent container creates each grabbed a fresh
# connection, and the resulting thundering herd of new TCP connects timed out ("Timeout connecting to
# server", 51/min at the 2026-07-22 08:35 wave). With no retry configured, every operation caught in
# that window failed outright — which failed the creates and put healthy nodes into launch backoff.
# Bounding the pool makes a burst QUEUE on an existing connection instead of opening a new one, and
# the retry rides out a blip that lasts less than a second.
REDIS_MAX_CONNECTIONS = 64
# Subscriptions get their own small pool. `pubsub.listen()` is a BLOCKING read on channels that are
# idle most of the time, and redis-py falls back to the connection's socket_timeout when the caller
# passes no deadline — so a subscription sharing the command pool is torn down after
# REDIS_SOCKET_TIMEOUT_SECONDS of silence, which for these channels is normal. Separate pool, no
# socket_timeout: commands still fail fast, subscriptions are allowed to wait.
REDIS_PUBSUB_MAX_CONNECTIONS = 8
# Seconds a caller waits for a pooled connection once all of them are busy. The pool MUST be a
# BlockingConnectionPool for this: the default pool raises MaxConnectionsError instead of waiting,
# and it raises from get_connection() BEFORE the retry wrapper is reached, so a bounded default pool
# would convert an overload burst into exactly the hard failures this hardening exists to remove.
REDIS_POOL_WAIT_TIMEOUT_SECONDS = 10
REDIS_SOCKET_TIMEOUT_SECONDS = 10
REDIS_CONNECT_TIMEOUT_SECONDS = 5
REDIS_RETRY_ATTEMPTS = 3
# Validate a pooled connection that has been idle this long, so a silently-dropped connection is
# discovered by the health check rather than by failing a caller's operation.
REDIS_HEALTH_CHECK_INTERVAL_SECONDS = 30

logger = logging.getLogger(__name__)


class RedisService:
    def __init__(self):
        self.redis = aioredis.Redis(
            connection_pool=aioredis.BlockingConnectionPool.from_url(
                settings.get_redis_connection_url(),
                max_connections=REDIS_MAX_CONNECTIONS,
                timeout=REDIS_POOL_WAIT_TIMEOUT_SECONDS,
                socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SECONDS,
                socket_keepalive=True,
                health_check_interval=REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
                retry=Retry(ExponentialBackoff(cap=1.0, base=0.1), REDIS_RETRY_ATTEMPTS),
                retry_on_error=[redis.exceptions.ConnectionError, redis.exceptions.TimeoutError],
            )
        )
        self.pubsub_redis = aioredis.Redis(
            connection_pool=aioredis.BlockingConnectionPool.from_url(
                settings.get_redis_connection_url(),
                max_connections=REDIS_PUBSUB_MAX_CONNECTIONS,
                timeout=REDIS_POOL_WAIT_TIMEOUT_SECONDS,
                # No socket_timeout on purpose — see REDIS_PUBSUB_MAX_CONNECTIONS.
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT_SECONDS,
                socket_keepalive=True,
                retry=Retry(ExponentialBackoff(cap=1.0, base=0.1), REDIS_RETRY_ATTEMPTS),
                retry_on_error=[redis.exceptions.ConnectionError, redis.exceptions.TimeoutError],
            )
        )
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire_executor_lock(
        self,
        executor_id: str,
        timeout: int = EXECUTOR_LOCK_TIMEOUT,
        blocking_timeout: int = EXECUTOR_LOCK_BLOCKING_TIMEOUT,
    ):
        """Distributed lock for executor operations to prevent race conditions."""
        lock = aioredis.lock.Lock(
            self.redis,
            f"lock:executor:{executor_id}",
            timeout=timeout,
            blocking_timeout=blocking_timeout,
        )

        try:
            async with lock:
                yield lock
        except (redis.exceptions.LockError, redis.exceptions.LockNotOwnedError) as e:
            logger.error(
                _m(
                    f"Lock error for executor {executor_id}",
                    extra={"error": str(e)},
                )
            )
            raise

    async def publish(self, channel: str, message: dict):
        """Publish a message to a Redis channel."""
        await self.redis.publish(channel, json.dumps(message))

    async def subscribe(self, *channel: str):
        """Subscribe to a Redis channel. Caller MUST `await pubsub.aclose()` when it stops reading —
        the connection returns to the pool only then, and the pool is bounded."""
        pubsub = self.pubsub_redis.pubsub()
        try:
            await pubsub.subscribe(*channel)
        except BaseException:
            # SUBSCRIBE acquires the pooled connection before it sends, and the caller never receives
            # this pubsub, so nothing else can return the connection to the bounded pool.
            await pubsub.aclose()
            raise
        return pubsub

    async def set(self, key: str, value: str):
        """Set a key-value pair in Redis."""
        async with self.lock:
            await self.redis.set(key, value)

    async def set_with_expiration(self, key: str, value: str, ttl_seconds: int):
        """Set a key that removes itself.

        DAH-2582 needs this for the attested-identity registry: an entry has to age out so a
        node that legitimately re-registers under a new executor id after a rebuild eventually
        stops colliding with its own past, and so an unbounded key space does not accumulate one
        entry per GPU per node forever.
        """
        async with self.lock:
            await self.redis.set(key, value, ex=ttl_seconds)

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
    
    async def add_rented_pod(self, executor: ExecutorSSHInfo, pod_id: str, container_name: str):
        rented_machine = await self.get_rented_machine(executor)
        if not rented_machine:
            rented_machine = {"owner_flag": False, "containers": [{"name": container_name, "pod_id": pod_id}]}
        else:
            rented_machine["containers"].append({"name": container_name, "pod_id": pod_id})
        
        await self.hset(RENTED_MACHINE_PREFIX, f"{executor.address}:{executor.port}", json.dumps(rented_machine))

    async def remove_rented_machine(self, executor: ExecutorSSHInfo, container_name: str | None = None):
        if not container_name:
            await self.hdel(RENTED_MACHINE_PREFIX, f"{executor.address}:{executor.port}")
        else:
            rented_machine = await self.get_rented_machine(executor)
            if not rented_machine:
                return
            rented_machine["containers"] = [item for item in rented_machine["containers"] if item["name"] != container_name]
            if not rented_machine["containers"]:
                await self.hdel(RENTED_MACHINE_PREFIX, f"{executor.address}:{executor.port}")
            else:
                await self.hset(RENTED_MACHINE_PREFIX, f"{executor.address}:{executor.port}", json.dumps(rented_machine))

    async def get_rented_machine(self, executor: ExecutorSSHInfo) -> dict | None:
        data = await self.hget(RENTED_MACHINE_PREFIX, f"{executor.address}:{executor.port}")
        if not data:
            return None

        data = json.loads(data)
        
        # check if the data is new structure
        if "containers" in data:
            containers = data.get("containers", [])
            containers = [container for container in containers if container.get("name", "").strip()]
            if not containers:
                return None
            return {"owner_flag": data.get("owner_flag", False), "containers": containers}
        
        # check if the data is old structure
        container_name: str = data.get("container_name", "")
        owner_flag: bool = data.get("owner_flag", False)
        if not container_name or not container_name.strip():
            return None
        return {"owner_flag": owner_flag, "containers": [{"name": container_name}]}

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

    async def add_pending_pod(self, miner_hotkey: str, executor_id: str, pod_id: str):
        pending_pods: list[dict] = await self.get_pending_pods(miner_hotkey, executor_id)
        now = int(time.time())
        
        pending_pods.append({"time": now, "pod_id": pod_id})
        await self.hset(PENDING_PODS_PREFIX, f"{miner_hotkey}:{executor_id}", json.dumps(pending_pods))

    async def remove_pending_pod(self, miner_hotkey: str, executor_id: str, pod_id: str):
        pending_pods: list[dict] = await self.get_pending_pods(miner_hotkey, executor_id)
        pending_pods = [item for item in pending_pods if item.get('pod_id', '') != pod_id]
        if pending_pods:
            await self.hset(PENDING_PODS_PREFIX, f"{miner_hotkey}:{executor_id}", json.dumps(pending_pods))
        else:
            await self.hdel(PENDING_PODS_PREFIX, f"{miner_hotkey}:{executor_id}")
    
    async def get_pending_pods(self, miner_hotkey: str, executor_id: str) -> list[dict]:
        data = await self.hget(PENDING_PODS_PREFIX, f"{miner_hotkey}:{executor_id}")
        if not data:
            return []

        data = json.loads(data)
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            logger.warning(f"Unexpected data type in get_pending_pods: {type(data)}")
            return []
        
        now = int(time.time())
        pending_pods = []
        
        for item in data:
            if now - item.get('time', 0) >= 30 * 60:  # 30 mins
                continue
            pending_pods.append(item)
        
        return pending_pods

    async def renting_in_progress(self, miner_hotkey: str, executor_id: str, pod_id: str | None = None) -> bool:
        pending_pods: list[dict] = await self.get_pending_pods(miner_hotkey, executor_id)
        
        if pod_id:
            pending_pods = [item for item in pending_pods if item.get('pod_id', '') == pod_id]
        
        return len(pending_pods) > 0

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
        try:
            if gpu_type is None or not isinstance(gpu_type, str):
                return 0
            
            portion = await self.hget(PORTION_PER_GPU_TYPE_SET, gpu_type)
            portion = float(portion) if portion else 0
            if not portion:
                gpu_model_rate = GPU_MODEL_RATES.get(gpu_type, 0)
                return gpu_model_rate

            return portion
        except Exception as e:
            logger.error(_m("Error getting portion per gpu type.", extra={"error": str(e), "gpu_type": gpu_type}), exc_info=True)
            return 0

    async def set_banned_guids(self, guids: list[str]):
        await self.redis.set(BANNED_GUIDS, json.dumps(guids))

    async def get_banned_guids(self) -> list[str]:
        data = await self.redis.get(BANNED_GUIDS)
        if not data:
            return []
        return json.loads(data)

    async def set_gpu_estimates(self, estimates: dict) -> None:
        """Serialize and store precomputed GPU estimates under GPU_ESTIMATES_KEY."""
        serialized: dict[str, dict] = {}
        for gpu_model, data in estimates.items():
            serialized[gpu_model] = {k: v.model_dump() for k, v in data.items()}
        await self.set(GPU_ESTIMATES_KEY, json.dumps(serialized))

    async def get_gpu_estimates(self) -> dict | None:
        """Read and deserialize precomputed GPU estimates from Redis."""
        data = await self.get(GPU_ESTIMATES_KEY)
        if not data:
            return None
        return json.loads(data)

    async def set_incentive_snapshot(self, snapshot) -> None:
        """Serialize and store the incentive snapshot under INCENTIVE_SNAPSHOT_KEY."""
        await self.set(INCENTIVE_SNAPSHOT_KEY, snapshot.model_dump_json())

    async def get_incentive_snapshot(self) -> dict | None:
        """Read and deserialize the incentive snapshot from Redis."""
        data = await self.get(INCENTIVE_SNAPSHOT_KEY)
        if not data:
            return None
        return json.loads(data)
