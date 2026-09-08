"""Idle-time pre-pull of the top-N official templates (DAH-2977).

Measured on 398 rentals (5 Sep 2026): a rental whose image is already on the node
starts in p50 23 s / p90 54 s; one that has to pull it starts in p50 61 s / p90 324 s.
Only the node's default image is pre-pulled today (``cache_template_service``).

With ``PRE_PULL_TEMPLATES_ENABLED`` the cache-template loop asks the backend for the
top-N official templates as well (entries carrying ``pre_pull: true``) and hands them
to :func:`PrePuller.sweep` once per refresh sweep, which pulls AT MOST ONE missing image,
digest-pinned, and only if

* the node is idle — no ``pod_*`` container exists and no validator docker-over-ssh
  session is open (that is how a rental's own pull and ``docker run`` reach this host);
  a pull already running is cancelled the moment either appears;
* the docker root keeps ``PRE_PULL_MIN_FREE_GB`` free afterwards — least-recently-used
  pre-pulled images are evicted first to make room, nothing else is ever removed;
* it finishes within ``PRE_PULL_TIMEOUT_SECONDS``.

One pull per sweep per node plus a random start delay keeps a fleet-wide enable from
stampeding the registry. Every pull attempt ends in exactly one log line
``pre_pull image=… seconds=… outcome=…``. Bookkeeping (what we pulled, when it was last
used by a rental) lives in a small JSON file so eviction order survives restarts.
"""

import asyncio
import json
import os
import random
import time
from pathlib import Path

import docker
import psutil

from core.config import settings
from core.logger import get_logger
from services.pull_lock import cache_pull_lock

logger = get_logger(__name__)

STATE_PATH = "/var/lib/lium/pre_pull_state.json"
# Same prefix monitor.py watches: every rental container the validator creates.
RENTAL_CONTAINER_PREFIX = "pod_"
# The validator drives rental docker operations through docker-py over SSH, which runs
# this on the executor for the whole operation (visible thanks to `pid: host`).
DOCKER_OVER_SSH_MARKER = "dial-stdio"
# Compressed size → on-disk estimate; same factor cache_template_service uses.
ON_DISK_MULTIPLIER = 3.0
DISK_PATH = "/"
# How often a running pull re-checks that the node is still idle.
ACTIVITY_CHECK_SECONDS = 10
GIB = 1024**3


def rental_activity(client: "docker.DockerClient") -> str | None:
    """Why the node is not idle right now, or ``None``.

    A rental container that is not finished counts, whatever its state: one still in
    ``created`` is a rental starting. Exited/dead leftovers awaiting cleanup do not.
    """
    for container in client.containers.list(all=True):
        if (container.name or "").startswith(RENTAL_CONTAINER_PREFIX) and container.status not in ("exited", "dead"):
            return f"rental container {container.name}"
    for proc in psutil.process_iter(["cmdline"]):
        if any(DOCKER_OVER_SSH_MARKER in part for part in (proc.info.get("cmdline") or [])):
            return "docker-over-ssh session"
    return None


class PrePullState:
    """What this node pre-pulled and when a rental last used it; never raises."""

    def __init__(self, path: str | None = STATE_PATH):
        self._path = Path(path) if path else None
        self.images: dict[str, dict] = {}
        try:
            if self._path and self._path.exists():
                doc = json.loads(self._path.read_text(encoding="utf-8"))
                self.images = dict(doc.get("images") or {})
        except Exception as e:
            logger.warning(f"pre-pull state unreadable, starting empty: {e}")

    def record_present(self, image_ref: str, digest: str, size: int) -> None:
        record = self.images.setdefault(image_ref, {"pulled_at": time.time()})
        record.update({"digest": digest, "size": size})

    def touch_used(self, image_refs: set[str]) -> None:
        for image_ref in image_refs & self.images.keys():
            self.images[image_ref]["last_used_at"] = time.time()

    def forget(self, image_ref: str) -> None:
        self.images.pop(image_ref, None)

    def lru(self, skip: set[str]) -> str | None:
        """Least-recently-used tracked image not in ``skip``; used = last rental, else pull."""
        candidates = [ref for ref in self.images if ref not in skip]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda ref: self.images[ref].get("last_used_at") or self.images[ref].get("pulled_at") or 0,
        )

    def flush(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps({"schema": 1, "images": self.images}), encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception as e:
            logger.warning(f"pre-pull state not written: {e}")


def _has_digest(client: "docker.DockerClient", repo: str, digest: str) -> bool:
    try:
        client.images.get(f"{repo}@{digest}")
        return True
    except docker.errors.ImageNotFound:
        return False


def _remove_ref(client: "docker.DockerClient", ref: str) -> bool:
    """Untag one reference; the layers go with the last one. False if docker refused."""
    try:
        client.images.remove(ref)
        return True
    except docker.errors.ImageNotFound:
        return True
    except Exception as e:
        logger.info(f"pre-pull eviction kept {ref}: {e}")
        return False


def _pull_pinned(
    client: "docker.DockerClient", repo: str, tag: str, digest: str, timeout_seconds: float
) -> tuple[str, str | None]:
    """Blocking digest pull with a hard deadline and idle re-checks; (outcome, detail).

    Uses the raw ``/images/create`` stream (as the validator's rental pull does) because
    ``images.pull`` has no timeout and cannot be interrupted: closing the response is
    what makes the daemon abandon an unfinished pull.
    """
    api = client.api
    registry, _ = docker.auth.resolve_repository_name(repo)
    auth_header = docker.auth.get_config_header(api, registry)
    headers = {"X-Registry-Auth": auth_header} if auth_header else {}
    deadline = time.monotonic() + timeout_seconds
    next_check = time.monotonic() + ACTIVITY_CHECK_SECONDS
    response = api._post(
        api._url("/images/create"),
        params={"fromImage": repo, "tag": digest},
        headers=headers,
        stream=True,
        timeout=timeout_seconds,
    )
    try:
        api._raise_for_status(response)
        for event in api._stream_helper(response, decode=True) or ():
            if isinstance(event, dict) and event.get("error"):
                return "pull_failed", str(event["error"])
            now = time.monotonic()
            if now > deadline:
                return "timeout", f"exceeded {timeout_seconds:.0f}s"
            if now >= next_check:
                next_check = now + ACTIVITY_CHECK_SECONDS
                busy = rental_activity(client)
                if busy:
                    return "preempted", busy
    finally:
        response.close()
    # A digest pull leaves the tag untouched; rentals look the image up by repo:tag.
    client.images.get(f"{repo}@{digest}").tag(repo, tag)
    return "pull_ok", None


class PrePuller:
    def __init__(self, client: "docker.DockerClient", state_path: str | None = STATE_PATH):
        self.client = client
        self.state = PrePullState(state_path)
        self._first_sweep = True

    async def sweep(self, entries: list[dict]) -> None:
        """Pull at most one missing ``pre_pull`` entry, if the node is idle and has room."""
        if not entries:
            return
        if self._first_sweep:
            self._first_sweep = False
            delay = random.uniform(0, settings.PRE_PULL_START_JITTER_SECONDS)
            logger.info(f"pre-pull: first sweep in {delay:.0f}s (start jitter)")
            await asyncio.sleep(delay)

        containers = await asyncio.to_thread(self.client.containers.list, all=True)
        self.state.touch_used(
            {
                (container.attrs.get("Config") or {}).get("Image") or ""
                for container in containers
                if (container.name or "").startswith(RENTAL_CONTAINER_PREFIX)
            }
        )

        for data in entries:
            repo, tag, digest = data.get("docker_image"), data.get("docker_image_tag"), data.get("docker_image_digest")
            if not repo or not tag or not digest:
                continue  # pre-pull is digest-pinned only
            image_ref = f"{repo}:{tag}"
            size = int(data.get("docker_image_size") or 0)
            if await asyncio.to_thread(_has_digest, self.client, repo, digest):
                self.state.record_present(image_ref, digest, size)
                continue

            busy = await asyncio.to_thread(rental_activity, self.client)
            if busy:
                logger.info(f"pre-pull: node busy ({busy}); {image_ref} waits for the next sweep")
                break

            started = time.monotonic()
            room, detail = await self._make_room(image_ref, int(size * ON_DISK_MULTIPLIER))
            if not room:
                outcome = "insufficient_disk"
            else:
                with cache_pull_lock() as acquired:
                    if not acquired:
                        outcome, detail = "lock_held", "another puller holds the lock"
                    else:
                        try:
                            outcome, detail = await asyncio.to_thread(
                                _pull_pinned, self.client, repo, tag, digest, settings.PRE_PULL_TIMEOUT_SECONDS
                            )
                        except Exception as e:
                            outcome, detail = "pull_failed", str(e)
            seconds = time.monotonic() - started
            logger.info(
                f"pre_pull image={image_ref} digest={digest} seconds={seconds:.1f} outcome={outcome}"
                + (f" detail={detail}" if detail else "")
            )
            if outcome == "pull_ok":
                self.state.record_present(image_ref, digest, size)
            break  # one pull per sweep per node

        self.state.flush()

    async def _make_room(self, keep_ref: str, need_bytes: int) -> tuple[bool, str | None]:
        """Keep ``PRE_PULL_MIN_FREE_GB`` free after the pull, evicting LRU pre-pulled images first."""
        floor = settings.PRE_PULL_MIN_FREE_GB * GIB
        skip = {keep_ref}
        while True:
            free = psutil.disk_usage(DISK_PATH).free
            if free - need_bytes >= floor:
                return True, None
            victim = self.state.lru(skip)
            if victim is None:
                return False, (
                    f"free {free / GIB:.0f} GiB < need {need_bytes / GIB:.0f} GiB + floor {floor / GIB:.0f} GiB"
                )
            skip.add(victim)
            digest = self.state.images[victim].get("digest")
            removed = await asyncio.to_thread(_remove_ref, self.client, victim)
            if digest:
                digest_ref = f"{victim.rpartition(':')[0]}@{digest}"
                removed = await asyncio.to_thread(_remove_ref, self.client, digest_ref) and removed
            if removed:
                logger.info(f"pre-pull: evicted {victim} (least recently used) to keep disk headroom")
                self.state.forget(victim)
