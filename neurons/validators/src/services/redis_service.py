import json
import redis.asyncio as aioredis
from core.config import settings

MACHINE_SPEC_CHANNEL_NAME = "channel:1"


class RedisService:
    def __init__(self):
        self.redis = aioredis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}")

    async def publish(self, channel: str, message: dict):
        """Publish a message to a Redis channel."""
        await self.redis.publish(channel, json.dumps(message))

    async def subscribe(self, channel: str):
        """Subscribe to a Redis channel."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)
        return pubsub

    async def set(self, key: str, value: str):
        """Set a key-value pair in Redis."""
        await self.redis.set(key, value)

    async def get(self, key: str):
        """Get a value by key from Redis."""
        return await self.redis.get(key)
