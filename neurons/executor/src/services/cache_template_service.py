"""On-boot cache template pre-pull.

Runs inside the executor FastAPI app (started from a lifespan hook). It pulls the
default "cache template" docker image for this host's GPU as soon as the executor
container starts — instead of waiting for the validator to launch
``gpus_utility.py`` on its next verification cycle — and keeps it fresh by
re-pulling whenever the template's remote digest changes.

All blocking docker / NVML work is dispatched to a thread so the executor's
event loop (and therefore its HTTP API) is never blocked by a multi-minute pull.
"""

import asyncio

import aiohttp
import docker
import psutil
import pynvml

from core.config import settings
from core.logger import get_logger
from services.pull_lock import cache_pull_lock

logger = get_logger(__name__)

# Require this multiple of the image size in free disk before pulling.
MIN_DISK_SPACE_MULTIPLIER = 3.0
# Backoff used after errors / empty responses (never longer than the refresh).
ERROR_INTERVAL_SECONDS = 5 * 60

DEFAULT_DOCKER_IMAGE_PATH = "/executors/default-docker-image"


def _get_gpu_info() -> tuple[str, str]:
    """Return (gpu_model, driver_version) via NVML, degrading to "unknown"."""
    gpu_name = "unknown"
    driver_version = "unknown"
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to get GPU info for cache pre-pull: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return gpu_name, driver_version


async def _fetch_templates(session: aiohttp.ClientSession, url: str, params: dict) -> list[dict]:
    async with session.get(url, params=params) as response:
        if response.status != 200:
            logger.error(f"Failed to get cache templates. Status: {response.status}")
            return []
        data = await response.json()
        logger.info(f"Received {len(data) if data else 0} cache template(s)")
        return data or []


async def _remote_digest(client: "docker.DockerClient", image_ref: str) -> str | None:
    """Manifest digest the registry currently serves for image_ref, or None."""
    try:
        registry_data = await asyncio.to_thread(client.images.get_registry_data, image_ref)
        return registry_data.id
    except Exception as e:
        logger.warning(f"Could not read remote digest for {image_ref}: {e}")
        return None


async def _local_digests(client: "docker.DockerClient", image_ref: str) -> list[str]:
    """RepoDigests of the locally cached image, or [] if not present."""
    try:
        image = await asyncio.to_thread(client.images.get, image_ref)
    except docker.errors.ImageNotFound:
        return []
    except Exception as e:
        logger.warning(f"Could not read local image {image_ref}: {e}")
        return []
    return image.attrs.get("RepoDigests", []) or []


async def _cleanup_old_tags(client: "docker.DockerClient", repository: str, keep_tag: str) -> None:
    """Remove other locally cached tags of the same repository."""
    try:
        images = await asyncio.to_thread(client.images.list, repository)
    except Exception as e:
        logger.warning(f"Failed to list images for {repository}: {e}")
        return
    for image in images:
        for tag in list(image.tags):
            repo, _, tg = tag.rpartition(":")
            if repo == repository and tg and tg != keep_tag:
                try:
                    await asyncio.to_thread(client.images.remove, tag)
                    logger.info(f"Removed unused image: {tag}")
                except Exception as e:
                    logger.warning(f"Failed to remove image {tag}: {e}")


async def _ensure_template(client: "docker.DockerClient", data: dict) -> None:
    docker_image = data.get("docker_image")
    docker_image_tag = data.get("docker_image_tag")
    docker_image_size = data.get("docker_image_size") or 0

    if not docker_image or not docker_image_tag:
        logger.warning(f"Skipping malformed cache template entry: {data}")
        return

    image_ref = f"{docker_image}:{docker_image_tag}"

    # Compare remote vs local digest; skip the pull when already up to date.
    remote = await _remote_digest(client, image_ref)
    local = await _local_digests(client, image_ref)
    if local:
        # Image is already cached. Re-pull only if we can confirm the remote
        # digest changed. If the remote digest is unreadable (rate limit / auth),
        # keep the cached copy rather than re-pulling every cycle.
        if remote is None:
            logger.info(f"{image_ref} cached; remote digest unreadable, keeping local copy")
            return
        if any(remote in digest for digest in local):
            logger.info(f"Cache template {image_ref} already up to date ({remote}); skipping pull")
            return

    # Ensure enough disk headroom before pulling.
    if docker_image_size:
        required_space = int(docker_image_size * MIN_DISK_SPACE_MULTIPLIER)
        available_space = psutil.disk_usage("/").free
        if available_space < required_space:
            logger.warning(
                f"Skipping pull of {image_ref} - insufficient disk space. "
                f"Required: {required_space}, Available: {available_space}"
            )
            return

    # Hold the cross-process lock so we never pull the same image at the same
    # time as a validator-launched gpus_utility.py run.
    with cache_pull_lock() as acquired:
        if not acquired:
            logger.info(f"Another puller holds the lock; skipping {image_ref} this cycle")
            return
        logger.info(f"Pulling cache template {image_ref} (remote={remote}, local={local})")
        await asyncio.to_thread(client.images.pull, docker_image, docker_image_tag)
        logger.info(f"Successfully pulled {image_ref}")

    await _cleanup_old_tags(client, docker_image, docker_image_tag)


async def run_cache_template_prefetch() -> None:
    """Background loop: pull the GPU's cache template on boot and keep it fresh."""
    base_url = settings.COMPUTE_REST_API_URL
    if not base_url:
        logger.warning(
            "COMPUTE_REST_API_URL not set; cache template pre-pull disabled"
        )
        return

    refresh_interval = settings.CACHE_TEMPLATE_REFRESH_SECONDS
    error_interval = min(ERROR_INTERVAL_SECONDS, refresh_interval)

    gpu_model, driver_version = _get_gpu_info()
    url = f"{base_url.rstrip('/')}{DEFAULT_DOCKER_IMAGE_PATH}"
    params = {"gpu_model": gpu_model, "driver_version": driver_version}
    logger.info(
        f"Starting cache template pre-pull for gpu_model={gpu_model} "
        f"driver_version={driver_version} (refresh every {refresh_interval}s)"
    )

    try:
        client = docker.from_env()
    except Exception as e:
        logger.error(f"Cannot connect to docker; cache pre-pull disabled: {e}")
        return

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                templates = await _fetch_templates(session, url, params)
                if not templates:
                    await asyncio.sleep(error_interval)
                    continue

                for data in templates:
                    await _ensure_template(client, data)

                # Sleep once per full sweep, after all templates are processed.
                await asyncio.sleep(refresh_interval)
            except asyncio.CancelledError:
                logger.info("Cache template pre-pull cancelled")
                raise
            except aiohttp.ClientError as e:
                logger.error(f"Network error during cache pre-pull: {e}")
                await asyncio.sleep(error_interval)
            except Exception as e:
                logger.error(f"Unexpected error during cache pre-pull: {e}")
                await asyncio.sleep(error_interval)
