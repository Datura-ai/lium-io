import json
import redis.asyncio as aioredis
from core.config import settings


class RedisService:
    def __init__(self):
        self.redis = aioredis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}")

    async def publish(self, channel: str, message: dict):
        """Publish a message to a Redis channel."""
        try:
            await self.redis.publish(channel, json.dumps(message))
        except Exception as e:
            print(f"Error publishing message to {channel}: {e}")

    async def subscribe(self, channel: str):
        """Subscribe to a Redis channel."""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(channel)
            return pubsub
        except Exception as e:
            print(f"Error subscribing to {channel}: {e}")

    async def set(self, key: str, value: str):
        """Set a key-value pair in Redis."""
        try:
            await self.redis.set(key, value)
        except Exception as e:
            print(f"Error setting key {key}: {e}")

    async def get(self, key: str):
        """Get a value by key from Redis."""
        try:
            return await self.redis.get(key)
        except Exception as e:
            print(f"Error getting key {key}: {e}")
            return None
