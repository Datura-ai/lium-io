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
import random
import time

import aiohttp
import docker
import psutil
import pynvml

from core.config import settings
from core.logger import get_logger
from services.cache_prefetch_state import STATE_PATH, CachePrefetchState, Outcome
from services.pre_pull_service import STREAM_READ_TIMEOUT_SECONDS, PrePuller
from services.pull_lock import cache_pull_lock

logger = get_logger(__name__)

# Require this multiple of the image size in free disk before pulling.
MIN_DISK_SPACE_MULTIPLIER = 3.0
# Backoff used after errors / empty responses (never longer than the refresh).
ERROR_INTERVAL_SECONDS = 5 * 60
# Until the first sweep completes, an error is retried after 15, 30, 60 and 120 s (each plus up
# to 15 s of jitter, so a fleet that boots together does not retry in step) before the loop falls
# back to ERROR_INTERVAL_SECONDS: a new node is verified within minutes of being added, and the
# default image is what that verification looks for.
FIRST_SWEEP_RETRY_BASE_SECONDS = 15
FIRST_SWEEP_RETRY_JITTER_SECONDS = 15
FIRST_SWEEP_FAST_RETRIES = 4
# The pre-pull sweep's budget ends this long before the next refresh (DAH-2977): a pull whose
# stream goes silent at the end of its budget still holds the cross-process pull lock for one
# read timeout, and the mandatory refresh would otherwise find the lock held and skip its pull.
SWEEP_DEADLINE_MARGIN_SECONDS = STREAM_READ_TIMEOUT_SECONDS + 30

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


class PullStreamError(Exception):
    """The daemon failed the pull inside its progress stream, after answering HTTP 200."""


def _stream_error(event: object) -> str | None:
    if not isinstance(event, dict):
        return None
    detail = event.get("errorDetail")
    message = detail.get("message") if isinstance(detail, dict) else None
    return message or event.get("error") or None


def _pull(client: "docker.DockerClient", repository: str, tag: str):
    """Blocking pull that raises the daemon's own error; returns the pulled image.

    ``images.pull`` drains the same stream and ignores an ``error`` event in it, so a failed
    pull surfaced only as the follow-up lookup's "No such image" and the real cause was lost.
    """
    for event in client.api.pull(repository, tag=tag, stream=True, decode=True):
        error = _stream_error(event)
        if error:
            raise PullStreamError(error)
    separator = "@" if tag.startswith("sha256:") else ":"
    return client.images.get(f"{repository}{separator}{tag}")


async def _cleanup_old_tags(
    client: "docker.DockerClient",
    repository: str,
    keep_tag: str,
    keep_tags: frozenset[str] = frozenset(),
) -> str | None:
    """Remove other locally cached tags of the same repository; return the last error.

    ``keep_tags`` (DAH-2977): tags of pre-pull templates, which may share the repository
    (daturaai/pytorch) and must survive a default-image refresh.
    """
    try:
        images = await asyncio.to_thread(client.images.list, repository)
    except Exception as e:
        logger.warning(f"Failed to list images for {repository}: {e}")
        return str(e)
    error: str | None = None
    for image in images:
        for tag in list(image.tags):
            repo, _, tg = tag.rpartition(":")
            if repo == repository and tg and tg != keep_tag and tg not in keep_tags:
                try:
                    await asyncio.to_thread(client.images.remove, tag)
                    logger.info(f"Removed unused image: {tag}")
                except Exception as e:
                    logger.warning(f"Failed to remove image {tag}: {e}")
                    error = str(e)
    return error


async def _ensure_template(
    client: "docker.DockerClient",
    data: dict,
    state: CachePrefetchState | None = None,
    keep_tags: frozenset[str] = frozenset(),
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
                image = await asyncio.to_thread(_pull, client, docker_image, expected_digest)
                # Re-point the tag at the pinned build: a digest pull alone leaves the
                # tag (possibly poisoned by a stale mirror) untouched.
                await asyncio.to_thread(image.tag, docker_image, docker_image_tag)
                logger.info(f"Successfully pulled {image_ref} at {expected_digest}")
            else:
                logger.info(f"Pulling cache template {image_ref} (remote={remote}, local={local})")
                await asyncio.to_thread(_pull, client, docker_image, docker_image_tag)
                logger.info(f"Successfully pulled {image_ref}")
        except Exception as e:
            # Record, then re-raise: a failed pull already aborts the sweep and backs
            # off, and DAH-2470 must not change that.
            state.note_pull_error(image_ref, e)
            state.record_image_outcome(image_ref, Outcome.PULL_FAILED, error=e)
            raise
        state.note_pull_ok(image_ref)
        state.record_image_outcome(image_ref, Outcome.PULL_OK)

    cleanup_error = await _cleanup_old_tags(client, docker_image, docker_image_tag, keep_tags)
    state.note_cleanup_error(image_ref, cleanup_error)


def _first_sweep_retry_delay(attempt: int, error_interval: float) -> float | None:
    """Delay before fast retry number ``attempt`` (0-based), or None once they are used up."""
    if attempt >= FIRST_SWEEP_FAST_RETRIES:
        return None
    base = min(error_interval, FIRST_SWEEP_RETRY_BASE_SECONDS * 2**attempt)
    return base + random.uniform(0, FIRST_SWEEP_RETRY_JITTER_SECONDS)


async def _run_pre_pull_sweep(
    pre_puller: PrePuller, entries: list[dict], protected: frozenset[str], deadline: float
) -> None:
    """One opportunistic sweep, as its own task: whatever it does never reaches the loop."""
    try:
        await pre_puller.sweep(entries, protected=protected, deadline=deadline)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Opportunistic: never let it change the default image's outcome.
        logger.warning(f"pre-pull sweep failed: {e}")


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

    # DAH-2977: off by default. When on, the backend is asked for the top-N official
    # templates too (`pre_pull: true` entries) and PrePuller warms one per sweep while idle.
    pre_puller = PrePuller(client) if settings.PRE_PULL_TEMPLATES_ENABLED else None
    # The sweep in flight, if any. The loop starts it and never awaits it (review, DAH-2977):
    # a pull stream that goes silent holds the sweep for the read timeout past its deadline
    # and eviction has no deadline at all, so awaiting it would move the default image's
    # refresh while the validator checks that digest. One sweep at a time: a refresh that
    # finds the previous one still running skips its own.
    sweep_task: asyncio.Task | None = None

    first_sweep_done = False
    fast_retries_used = 0

    def error_backoff() -> float:
        nonlocal fast_retries_used
        delay = (
            None
            if first_sweep_done
            else _first_sweep_retry_delay(fast_retries_used, error_interval)
        )
        if delay is None:
            return error_interval
        fast_retries_used += 1
        logger.info(f"First sweep not completed yet; retrying cache pre-pull in {delay:.0f}s")
        return delay

    gpu_model = "unknown"
    driver_version = "unknown"
    # Bound every backend request so a hung server cannot wedge the prefetch loop;
    # timeouts surface as exceptions and route through the existing error backoff.
    session_timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        while True:
            try:
                state.begin_sweep()
                # When the next mandatory refresh is due. With the pre-pull on, the sleep below
                # runs only up to it, so the default image is re-checked every refresh_interval
                # whatever the sweep did (review, DAH-2977).
                refresh_deadline = time.monotonic() + refresh_interval
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
                        await asyncio.sleep(error_backoff())
                        continue
                    logger.info(
                        f"Cache pre-pull resolved gpu_model={gpu_model} "
                        f"driver_version={driver_version}"
                    )

                params = {"gpu_model": gpu_model, "driver_version": driver_version}
                if pre_puller:
                    params["include_pre_pull"] = "true"
                templates, status, backend_error = await _fetch_templates(session, url, params)
                state.note_backend(
                    status=status, template_count=len(templates), error=backend_error
                )
                if not templates:
                    state.record_loop_outcome(Outcome.BACKEND_NO_TEMPLATES, error=backend_error)
                    state.flush()
                    await asyncio.sleep(error_backoff())
                    continue

                # Pre-pull entries are opportunistic and idle-only, so they never go
                # through the mandatory default-image path below; their tags are only
                # shielded from its old-tag cleanup (same repository). An entry the backend
                # marks pre_pull that is also this node's default image stays out of the
                # pre-pull set, and the puller is told the mandatory refs so one it tracked
                # from an earlier sweep stops being an eviction candidate: the default image
                # is never removed (the backend's top-N is global, the default is per
                # gpu_model, so the overlap is decided here).
                mandatory_refs = {
                    (data.get("docker_image"), data.get("docker_image_tag"))
                    for data in templates
                    if not data.get("pre_pull")
                }
                pre_pull = [
                    data
                    for data in templates
                    if data.get("pre_pull")
                    and (data.get("docker_image"), data.get("docker_image_tag"))
                    not in mandatory_refs
                ]
                keep_tags = frozenset(
                    data["docker_image_tag"] for data in pre_pull if data.get("docker_image_tag")
                )
                if pre_puller:
                    # Published before the mandatory pass, every refresh, so a sweep still
                    # running from the previous one cannot evict a ref that became this node's
                    # default since it started, not even while that ref is being checked below.
                    pre_puller.protected = frozenset(
                        f"{repo}:{tag}" for repo, tag in mandatory_refs if repo and tag
                    )
                for data in templates:
                    if not data.get("pre_pull"):
                        await _ensure_template(client, data, state, keep_tags)
                if pre_puller:
                    # The default image's outcome is what the validator reads (DAH-2470): publish
                    # it now, not after a sweep that can wait out the start jitter and one pull.
                    state.flush()
                    if sweep_task is not None and not sweep_task.done():
                        logger.info(
                            "pre-pull: the previous sweep is still running; no new sweep this refresh"
                        )
                    else:
                        # Started, not awaited. Its budget ends a read timeout (plus slack) before
                        # the refresh, so the pull lock is free again when the default image is
                        # re-checked, even when the stream went silent at the very end.
                        sweep_task = asyncio.create_task(
                            _run_pre_pull_sweep(
                                pre_puller,
                                pre_pull,
                                pre_puller.protected,
                                refresh_deadline - SWEEP_DEADLINE_MARGIN_SECONDS,
                            )
                        )

                state.record_loop_outcome(Outcome.SWEEP_OK)
                first_sweep_done = True
                # Publish before sleeping, so the document is never a full refresh
                # interval behind what the loop actually knows.
                state.flush()
                # Sleep once per full sweep. With the pre-pull on, only until the refresh
                # deadline, so the default image is re-checked every refresh_interval whether
                # or not the sweep task has finished. Flag off, the sleep is today's full
                # interval after the work, unchanged.
                if pre_puller:
                    await asyncio.sleep(max(0.0, refresh_deadline - time.monotonic()))
                else:
                    await asyncio.sleep(refresh_interval)
            except asyncio.CancelledError:
                logger.info("Cache template pre-pull cancelled")
                if sweep_task is not None:
                    sweep_task.cancel()
                raise
            except aiohttp.ClientError as e:
                logger.error(f"Network error during cache pre-pull: {e}")
                state.note_loop_error(e)
                state.record_loop_outcome(Outcome.LOOP_ERROR, error=e)
                state.flush()
                await asyncio.sleep(error_backoff())
            except Exception as e:
                logger.error(f"Unexpected error during cache pre-pull: {e}")
                state.note_loop_error(e)
                state.record_loop_outcome(Outcome.LOOP_ERROR, error=e)
                state.flush()
                await asyncio.sleep(error_backoff())
