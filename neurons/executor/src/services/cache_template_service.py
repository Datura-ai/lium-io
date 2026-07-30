"""On-boot cache template pre-pull.

Runs inside the executor FastAPI app (started from a lifespan hook). It pulls the
default "cache template" docker image for this host's GPU as soon as the executor
container starts — instead of waiting for the validator to launch
``gpus_utility.py`` on its next verification cycle — and keeps it fresh by
re-pulling whenever the template's remote digest changes.

All blocking docker / NVML work is dispatched to a thread so the executor's
event loop (and therefore its HTTP API) is never blocked by a multi-minute pull.

DAH-2470: every branch below also records a named outcome into a
``CachePrefetchState`` document, which the validator reads over SSH when it zeroes a
node for a bad image digest. The helpers therefore return their error text instead of
only logging it — the state file is the single source a reader will have, so nothing
this module prints may be lost on the way there. Recording never changes a pull
decision: state failures are swallowed inside ``cache_prefetch_state``.
"""

import asyncio

import aiohttp
import docker
import psutil
import pynvml

from core.config import settings
from core.logger import get_logger
from services.cache_prefetch_state import STATE_PATH, CachePrefetchState, Outcome
from services.pull_lock import cache_pull_lock

logger = get_logger(__name__)

# Require this multiple of the image size in free disk before pulling.
MIN_DISK_SPACE_MULTIPLIER = 3.0
# Backoff used after errors / empty responses (never longer than the refresh).
ERROR_INTERVAL_SECONDS = 5 * 60

DEFAULT_DOCKER_IMAGE_PATH = "/executors/default-docker-image"


def _get_gpu_info() -> tuple[str, str, str | None]:
    """Return (gpu_model, driver_version, error) via NVML, degrading to "unknown"."""
    gpu_name = "unknown"
    driver_version = "unknown"
    error: str | None = None
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
        error = str(e)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return gpu_name, driver_version, error


async def _fetch_templates(
    session: aiohttp.ClientSession, url: str, params: dict
) -> tuple[list[dict], int | None, str | None]:
    """Return (templates, http_status, error) for the backend recommendation call."""
    async with session.get(url, params=params) as response:
        if response.status != 200:
            logger.error(f"Failed to get cache templates. Status: {response.status}")
            return [], response.status, f"HTTP {response.status}"
        data = await response.json()
        logger.info(f"Received {len(data) if data else 0} cache template(s)")
        return data or [], response.status, None


async def _remote_digest(
    client: "docker.DockerClient", image_ref: str
) -> tuple[str | None, str | None]:
    """Return (manifest digest the registry serves for image_ref, error)."""
    try:
        registry_data = await asyncio.to_thread(client.images.get_registry_data, image_ref)
        return registry_data.id, None
    except Exception as e:
        logger.warning(f"Could not read remote digest for {image_ref}: {e}")
        return None, str(e)


async def _local_digests(
    client: "docker.DockerClient", image_ref: str
) -> tuple[list[str], str | None]:
    """Return (RepoDigests of the locally cached image, error). Absent image is not an error."""
    try:
        image = await asyncio.to_thread(client.images.get, image_ref)
    except docker.errors.ImageNotFound:
        return [], None
    except Exception as e:
        logger.warning(f"Could not read local image {image_ref}: {e}")
        return [], str(e)
    return image.attrs.get("RepoDigests", []) or [], None


async def _cleanup_old_tags(
    client: "docker.DockerClient", repository: str, keep_tag: str
) -> str | None:
    """Remove other locally cached tags of the same repository; return the last error."""
    try:
        images = await asyncio.to_thread(client.images.list, repository)
    except Exception as e:
        logger.warning(f"Failed to list images for {repository}: {e}")
        return str(e)
    error: str | None = None
    for image in images:
        for tag in list(image.tags):
            repo, _, tg = tag.rpartition(":")
            if repo == repository and tg and tg != keep_tag:
                try:
                    await asyncio.to_thread(client.images.remove, tag)
                    logger.info(f"Removed unused image: {tag}")
                except Exception as e:
                    logger.warning(f"Failed to remove image {tag}: {e}")
                    error = str(e)
    return error


async def _ensure_template(
    client: "docker.DockerClient", data: dict, state: CachePrefetchState | None = None
) -> None:
    # A throwaway recorder keeps the body free of `if state:` guards when this is
    # called on its own (tests); the loop always passes the real one.
    state = state or CachePrefetchState(path=None)
    docker_image = data.get("docker_image")
    docker_image_tag = data.get("docker_image_tag")
    docker_image_size = data.get("docker_image_size") or 0
    # DAH-2461: bare "sha256:…" digest the backend resolved live from Docker Hub.
    # When present it is the source of truth for freshness: digest-pinned pulls are
    # content-addressed, so they bypass provider registry mirrors that keep serving
    # a stale manifest for the tag (and 403 on digests they don't have cached).
    expected_digest = data.get("docker_image_digest")

    if not docker_image or not docker_image_tag:
        logger.warning(f"Skipping malformed cache template entry: {data}")
        state.note_malformed_template(data)
        return

    image_ref = f"{docker_image}:{docker_image_tag}"

    local, local_error = await _local_digests(client, image_ref)
    state.note_local_digests(image_ref, local, error=local_error)
    state.note_expected_digest(image_ref, expected_digest)
    remote: str | None = None
    if expected_digest:
        if any(expected_digest in digest for digest in local):
            logger.info(
                f"Cache template {image_ref} already at backend digest ({expected_digest}); skipping pull"
            )
            state.record_image_outcome(image_ref, Outcome.UP_TO_DATE)
            return
    elif local:
        # Legacy fallback (backend sent no digest): only query the daemon for the
        # remote digest when the image is already cached; on a first pull there is
        # nothing to compare against, so skip the call and avoid a needless registry
        # manifest request (counts against pull limits).
        remote, remote_error = await _remote_digest(client, image_ref)
        state.note_remote_digest(image_ref, remote, error=remote_error)
        # Re-pull only if we can confirm the remote digest changed. If the remote
        # digest is unreadable (rate limit / auth), keep the cached copy rather
        # than re-pulling every cycle.
        if remote is None:
            logger.info(f"{image_ref} cached; remote digest unreadable, keeping local copy")
            # DAH-2470 leading hypothesis: this branch never retries, so a node that
            # lands here keeps a stale image indefinitely. The counter makes that
            # visible — one occurrence is noise, hundreds is the answer.
            state.record_image_outcome(
                image_ref, Outcome.REMOTE_DIGEST_UNREADABLE, error=remote_error
            )
            return
        if any(remote in digest for digest in local):
            logger.info(f"Cache template {image_ref} already up to date ({remote}); skipping pull")
            state.record_image_outcome(image_ref, Outcome.UP_TO_DATE)
            return

    # Ensure enough disk headroom before pulling.
    if docker_image_size:
        required_space = int(docker_image_size * MIN_DISK_SPACE_MULTIPLIER)
        available_space = psutil.disk_usage("/").free
        state.note_disk(image_ref, required_space, available_space)
        if available_space < required_space:
            logger.warning(
                f"Skipping pull of {image_ref} - insufficient disk space. "
                f"Required: {required_space}, Available: {available_space}"
            )
            state.record_image_outcome(
                image_ref,
                Outcome.INSUFFICIENT_DISK,
                error=f"required {required_space}, available {available_space}",
            )
            return

    # Hold the cross-process lock so we never pull the same image at the same
    # time as a validator-launched gpus_utility.py run.
    with cache_pull_lock() as acquired:
        if not acquired:
            logger.info(f"Another puller holds the lock; skipping {image_ref} this cycle")
            state.record_image_outcome(image_ref, Outcome.LOCK_HELD)
            return
        state.note_pull_attempt(image_ref)
        try:
            if expected_digest:
                pinned_ref = f"{docker_image}@{expected_digest}"
                logger.info(f"Pulling cache template {pinned_ref} (local={local})")
                image = await asyncio.to_thread(client.images.pull, pinned_ref)
                # Re-point the tag at the pinned build: a digest pull alone leaves the
                # tag (possibly poisoned by a stale mirror) untouched.
                await asyncio.to_thread(image.tag, docker_image, docker_image_tag)
                logger.info(f"Successfully pulled {image_ref} at {expected_digest}")
            else:
                logger.info(f"Pulling cache template {image_ref} (remote={remote}, local={local})")
                await asyncio.to_thread(client.images.pull, docker_image, docker_image_tag)
                logger.info(f"Successfully pulled {image_ref}")
        except Exception as e:
            # Record, then re-raise: a failed pull already aborts the sweep and backs
            # off, and DAH-2470 must not change that.
            state.note_pull_error(image_ref, e)
            state.record_image_outcome(image_ref, Outcome.PULL_FAILED, error=e)
            raise
        state.note_pull_ok(image_ref)
        state.record_image_outcome(image_ref, Outcome.PULL_OK)

    cleanup_error = await _cleanup_old_tags(client, docker_image, docker_image_tag)
    state.note_cleanup_error(image_ref, cleanup_error)


async def run_cache_template_prefetch(state_path: str | None = STATE_PATH) -> None:
    """Background loop: pull the GPU's cache template on boot and keep it fresh."""
    base_url = settings.COMPUTE_REST_API_URL
    refresh_interval = settings.CACHE_TEMPLATE_REFRESH_SECONDS
    # Published from the first line onward, so the two ways this loop can quit before
    # it ever starts are visible to the validator instead of silent.
    state = CachePrefetchState(
        path=state_path,
        backend_url=f"{base_url.rstrip('/')}{DEFAULT_DOCKER_IMAGE_PATH}" if base_url else None,
        refresh_interval_seconds=refresh_interval,
    )

    if not base_url:
        logger.warning(
            "COMPUTE_REST_API_URL not set; cache template pre-pull disabled"
        )
        state.record_loop_outcome(Outcome.PREFETCH_DISABLED_NO_BACKEND_URL)
        state.flush()
        return

    error_interval = min(ERROR_INTERVAL_SECONDS, refresh_interval)
    url = f"{base_url.rstrip('/')}{DEFAULT_DOCKER_IMAGE_PATH}"

    try:
        client = docker.from_env()
        state.note_docker(available=True)
    except Exception as e:
        logger.error(f"Cannot connect to docker; cache pre-pull disabled: {e}")
        state.note_docker(available=False, error=e)
        state.record_loop_outcome(Outcome.DOCKER_UNAVAILABLE, error=e)
        state.flush()
        return

    logger.info(f"Cache template pre-pull starting (refresh every {refresh_interval}s)")

    gpu_model = "unknown"
    driver_version = "unknown"
    # Bound every backend request so a hung server cannot wedge the prefetch loop;
    # timeouts surface as exceptions and route through the existing error backoff.
    session_timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        while True:
            try:
                state.begin_sweep()
                # Resolve GPU info off-thread, and keep retrying while it is still
                # unknown: at early boot the NVIDIA driver may not be ready yet,
                # and caching "unknown" once would wedge the loop forever.
                if gpu_model == "unknown":
                    gpu_model, driver_version, gpu_error = await asyncio.to_thread(_get_gpu_info)
                    state.note_gpu(gpu_model, driver_version, error=gpu_error)
                    if gpu_model == "unknown":
                        logger.warning("GPU not detected yet; retrying cache pre-pull shortly")
                        state.record_loop_outcome(Outcome.GPU_UNKNOWN, error=gpu_error)
                        state.flush()
                        await asyncio.sleep(error_interval)
                        continue
                    logger.info(
                        f"Cache pre-pull resolved gpu_model={gpu_model} "
                        f"driver_version={driver_version}"
                    )

                params = {"gpu_model": gpu_model, "driver_version": driver_version}
                templates, status, backend_error = await _fetch_templates(session, url, params)
                state.note_backend(
                    status=status, template_count=len(templates), error=backend_error
                )
                if not templates:
                    state.record_loop_outcome(Outcome.BACKEND_NO_TEMPLATES, error=backend_error)
                    state.flush()
                    await asyncio.sleep(error_interval)
                    continue

                for data in templates:
                    await _ensure_template(client, data, state)

                state.record_loop_outcome(Outcome.SWEEP_OK)
                # Publish before sleeping, so the document is never a full refresh
                # interval behind what the loop actually knows.
                state.flush()
                # Sleep once per full sweep, after all templates are processed.
                await asyncio.sleep(refresh_interval)
            except asyncio.CancelledError:
                logger.info("Cache template pre-pull cancelled")
                raise
            except aiohttp.ClientError as e:
                logger.error(f"Network error during cache pre-pull: {e}")
                state.note_loop_error(e)
                state.record_loop_outcome(Outcome.LOOP_ERROR, error=e)
                state.flush()
                await asyncio.sleep(error_interval)
            except Exception as e:
                logger.error(f"Unexpected error during cache pre-pull: {e}")
                state.note_loop_error(e)
                state.record_loop_outcome(Outcome.LOOP_ERROR, error=e)
                state.flush()
                await asyncio.sleep(error_interval)
