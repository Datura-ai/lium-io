import logging

logger = logging.getLogger(__name__)


def wait_for_services_sync(timeout=30):
    """Wait until Redis are working."""
    import time

    from redis import Redis
    from redis.exceptions import ConnectionError as RedisConnectionError

    from core.config import settings

    # Initialize Redis client
    redis_client = Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)

    start_time = time.time()

    logger.info("Waiting for services to be available...")

    while True:
        try:
            # Check Redis connection
            redis_client.ping()
            logger.info("Connected to Redis.")

            break  # Exit loop if both connections are successful
        except RedisConnectionError as e:
            if time.time() - start_time > timeout:
                logger.error("Timeout while waiting for services to be available.")
                raise e
            logger.warning("Waiting for services to be available...")
            time.sleep(1)
