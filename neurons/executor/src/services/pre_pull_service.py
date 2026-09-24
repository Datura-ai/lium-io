"""Idle-time pre-pull of the top-N official templates (DAH-2977).

Measured on 398 rentals (5 Sep 2026): a rental whose image is already on the node
starts in p50 23 s / p90 54 s; one that has to pull it starts in p50 61 s / p90 324 s.
Only the node's default image is pre-pulled today (``cache_template_service``).

With ``PRE_PULL_TEMPLATES_ENABLED`` the cache-template loop asks the backend for the
top-N official templates as well (entries carrying ``pre_pull: true``) and hands them
to :func:`PrePuller.sweep` once per refresh sweep, which pulls AT MOST ONE missing image,
digest-pinned, and only if

* the node is idle — no ``pod_*`` container exists (in any state), no validator docker-over-ssh
  session is open (that is how a rental's own pull and ``docker run`` reach this host) and the
  validator's VerifyX bandwidth sample is not running; a pull already running is cancelled
  within 10 s when any of them appears (2 s for the bandwidth sample);
* the docker root keeps ``PRE_PULL_MIN_FREE_GB`` free afterwards — pre-pulled images that left
  the current top-N are evicted, least recently used first, to make room; nothing else is ever
  removed, and a pull that would not fit without evicting a served image is skipped;
* it finishes within ``PRE_PULL_TIMEOUT_SECONDS`` and before the next mandatory refresh
  is due: the start delay and the pull are both capped at the refresh deadline the loop
  hands in, and a pull that would not fit waits for the next sweep. The loop starts the
  sweep as its own task and never waits for it, so the default image's refresh runs on
  time even when a sweep overruns (a silent pull stream, a slow eviction); while one
  sweep is still running the next refresh does not start another.

One pull per sweep per node plus a random start delay keeps a fleet-wide enable from
stampeding the registry. Every pull attempt ends in exactly one log line
``pre_pull image=… seconds=… outcome=…``. Bookkeeping (what we pulled, when it was last
used by a rental) lives in a small JSON file on the ``reserve_data`` volume so eviction
order survives restarts and a force-recreate of the executor container.
"""

import asyncio
import json
import os
import random
import time
from pathlib import Path

import docker
import psutil
import requests
import urllib3

from core.config import settings
from core.logger import get_logger
from services.pull_lock import cache_pull_lock

logger = get_logger(__name__)

# On the `reserve_data` named volume (compose mounts it at /var/lium-reserve), which outlives a
# force-recreate of the executor container. /var/lib/lium is the container's writable layer:
# a state file there is gone after a recreate and every image it tracked would stay on the
# host outside the eviction inventory.
STATE_PATH = "/var/lium-reserve/pre_pull_state.json"
# A pull with less than this left before the refresh deadline is not started; it waits for
# the next sweep instead of being cut off by the deadline a few seconds in.
MIN_PULL_BUDGET_SECONDS = 60
# Same prefix monitor.py watches: every rental container the validator creates.
RENTAL_CONTAINER_PREFIX = "pod_"
# The validator drives rental docker operations through docker-py over SSH, which runs
# this on the executor for the whole operation (visible thanks to `pid: host`).
DOCKER_OVER_SSH_MARKER = "dial-stdio"
# The validator's VerifyX probe, whose download sample feeds the 100 Mbps EMA gate: it runs as
# `python …/verifyx_executor.py --seed …`, over SSH or as the `/verify` subprocess
# (local_verify_service). A pull sharing the link would lower that sample.
BANDWIDTH_TEST_MARKER = "verifyx_executor.py"
# How often a running pull re-checks for the probe alone (a process scan, no docker call): a
# sample lasts tens of seconds, so the full idle check's cadence would share too much of it.
BANDWIDTH_TEST_CHECK_SECONDS = 2
# Compressed size → on-disk estimate; same factor cache_template_service uses.
ON_DISK_MULTIPLIER = 3.0
DISK_PATH = "/"
# How often a running pull re-checks that the node is still idle.
ACTIVITY_CHECK_SECONDS = 10
# Socket timeouts of the pull stream: connect, and the longest silence between two progress
# events before the pull counts as stalled. Docker streams progress many times a second while
# a layer downloads or extracts, so a minute of silence is a stuck registry, not a slow one.
CONNECT_TIMEOUT_SECONDS = 30
STREAM_READ_TIMEOUT_SECONDS = 60
GIB = 1024**3


def rental_activity(client: "docker.DockerClient") -> str | None:
    """Why the node is not idle right now, or ``None``.

    Any rental container counts until it is removed, whatever its state: ``created`` is a
    rental starting, ``exited`` may be a stopped rental that can start again. The validator
    removes an ended rental's container (orphans at its next cleanup cycle).
    """
    for container in client.containers.list(all=True):
        if (container.name or "").startswith(RENTAL_CONTAINER_PREFIX):
            return f"rental container {container.name}"
    return host_activity()


def host_activity() -> str | None:
    """Validator work on this host that a pull must not overlap, or ``None``."""
    for proc in psutil.process_iter(["cmdline"]):
        cmdline = proc.info.get("cmdline") or []
        if any(DOCKER_OVER_SSH_MARKER in part for part in cmdline):
            return "docker-over-ssh session"
        if any(part.endswith(BANDWIDTH_TEST_MARKER) for part in cmdline):
            return "validator bandwidth test"
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
    started = time.monotonic()
    deadline = started + timeout_seconds
    next_check = started + ACTIVITY_CHECK_SECONDS
    next_probe_check = started + BANDWIDTH_TEST_CHECK_SECONDS
    # The deadline and the idle check run per stream event, so the read timeout is what
    # bounds a stream that goes silent: without it a stalled registry would hold this thread
    # (and the pull lock) for the whole budget past the caller's refresh deadline.
    response = api._post(
        api._url("/images/create"),
        params={"fromImage": repo, "tag": digest},
        headers=headers,
        stream=True,
        timeout=(CONNECT_TIMEOUT_SECONDS, STREAM_READ_TIMEOUT_SECONDS),
    )
    try:
        api._raise_for_status(response)
        for event in api._stream_helper(response, decode=True) or ():
            if isinstance(event, dict) and event.get("error"):
                return "pull_failed", str(event["error"])
            now = time.monotonic()
            if now > deadline:
                return "timeout", f"exceeded {timeout_seconds:.0f}s"
            busy = None
            if now >= next_check:
                next_check = now + ACTIVITY_CHECK_SECONDS
                next_probe_check = now + BANDWIDTH_TEST_CHECK_SECONDS
                busy = rental_activity(client)
            elif now >= next_probe_check:
                next_probe_check = now + BANDWIDTH_TEST_CHECK_SECONDS
                busy = host_activity()
            if busy:
                return "preempted", busy
    except (requests.exceptions.Timeout, urllib3.exceptions.TimeoutError):
        # docker-py reads the chunked stream off `response.raw`, so a silent stream surfaces as
        # urllib3's ReadTimeoutError, not requests' (requests only remaps inside iter_content).
        return "timeout", f"no stream event for {STREAM_READ_TIMEOUT_SECONDS}s"
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
        # The mandatory refs (``repo:tag``) as of the loop's latest refresh. The loop writes it
        # every refresh, so a sweep still running from the previous one never evicts a ref that
        # became the default in between (``sweep()``'s ``protected`` is only what it saw at start).
        self.protected: frozenset[str] = frozenset()

    async def sweep(
        self,
        entries: list[dict],
        protected: frozenset[str] = frozenset(),
        deadline: float | None = None,
    ) -> None:
        """Pull at most one missing ``pre_pull`` entry, if the node is idle and has room.

        ``protected`` are the mandatory refs (``repo:tag``) the default-image path owns this
        sweep: a ref pre-pulled earlier that has since become this node's default is untracked
        here so the disk guard never evicts it.

        ``deadline`` is the ``time.monotonic()`` instant this sweep should be done by; the loop
        sets it a read timeout plus slack before its next mandatory refresh. The start jitter
        and the pull budget are both cut at it, so one sweep normally fits in one refresh
        interval, a node pulls at most one image per interval, and the pull lock is free again
        when the default image is re-checked. It is a budget, not a guarantee: a silent stream
        overruns it by the read timeout and eviction is not timed, which is why the loop runs
        this as a task it does not wait for.
        ``None`` means no cap, which only the tests use."""
        for image_ref in protected & self.state.images.keys():
            self.state.forget(image_ref)
            logger.info(f"pre-pull: {image_ref} is now a mandatory image; no longer tracked for eviction")
        await self._evict_unlisted(entries, protected)
        if not entries:
            self.state.flush()
            return
        if self._first_sweep:
            delay = random.uniform(0, settings.PRE_PULL_START_JITTER_SECONDS)
            if deadline is not None and delay > deadline - time.monotonic():
                # The drawn delay does not fit before the refresh deadline: sleep what is
                # left and draw again next sweep, so this node keeps its share of the
                # fleet-wide spread instead of starting early.
                delay = max(0.0, deadline - time.monotonic())
            else:
                self._first_sweep = False
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
                await self._retire_superseded(image_ref, repo, digest)
                self.state.record_present(image_ref, digest, size)
                continue

            busy = await asyncio.to_thread(rental_activity, self.client)
            if busy:
                logger.info(f"pre-pull: node busy ({busy}); {image_ref} waits for the next sweep")
                break

            if deadline is not None and deadline - time.monotonic() < MIN_PULL_BUDGET_SECONDS:
                # Checked before eviction, so nothing is removed for a pull that will not run.
                logger.info(
                    f"pre-pull: {max(0.0, deadline - time.monotonic()):.0f}s left before the "
                    f"refresh deadline; {image_ref} waits for the next sweep"
                )
                break

            started = time.monotonic()
            served = {f"{e.get('docker_image')}:{e.get('docker_image_tag')}" for e in entries}
            room, detail = await self._make_room(image_ref, int(size * ON_DISK_MULTIPLIER), served)
            # The budget is measured after eviction, which takes time of its own: the pull's
            # clock starts below, so this is what cuts it at the deadline.
            budget = float(settings.PRE_PULL_TIMEOUT_SECONDS)
            if deadline is not None:
                budget = max(0.0, min(budget, deadline - time.monotonic()))
                if budget < MIN_PULL_BUDGET_SECONDS:
                    # A slow eviction ate the budget; a pull that must end in under a minute
                    # would only be logged as a timeout. The room it made stays for next sweep.
                    logger.info(
                        f"pre-pull: eviction left {budget:.0f}s before the refresh deadline; "
                        f"{image_ref} waits for the next sweep"
                    )
                    break
            if not room:
                outcome = "insufficient_disk"
            else:
                with cache_pull_lock() as acquired:
                    if not acquired:
                        outcome, detail = "lock_held", "another puller holds the lock"
                    else:
                        try:
                            outcome, detail = await asyncio.to_thread(
                                _pull_pinned, self.client, repo, tag, digest, budget
                            )
                        except Exception as e:
                            outcome, detail = "pull_failed", str(e)
            seconds = time.monotonic() - started
            logger.info(
                f"pre_pull image={image_ref} digest={digest} seconds={seconds:.1f} outcome={outcome}"
                + (f" detail={detail}" if detail else "")
            )
            if outcome == "pull_ok":
                await self._retire_superseded(image_ref, repo, digest)
                self.state.record_present(image_ref, digest, size)
            break  # one pull per sweep per node

        self.state.flush()

    async def _evict_unlisted(self, entries: list[dict], protected: frozenset[str]) -> None:
        """Remove a pre-pulled image once the backend has stopped serving it for
        ``PRE_PULL_EVICT_UNLISTED_AFTER_SECONDS`` (0 keeps it until the disk floor needs the room).

        The clock starts at the first sweep that no longer lists it and resets if it comes back,
        so a template that drifts in and out at the edge of the top-N is not pulled and removed
        in turn. Only images this puller pulled are candidates; docker refuses to remove one a
        container still uses, and that image stays tracked for the next sweep."""
        after = settings.PRE_PULL_EVICT_UNLISTED_AFTER_SECONDS
        served = {
            f"{data['docker_image']}:{data['docker_image_tag']}"
            for data in entries
            if data.get("docker_image") and data.get("docker_image_tag")
        }
        now = time.time()
        for image_ref in list(self.state.images):
            record = self.state.images[image_ref]
            if image_ref in served or image_ref in protected or image_ref in self.protected:
                record.pop("unlisted_since", None)
                continue
            since = record.setdefault("unlisted_since", now)
            if after <= 0 or now - since < after:
                continue
            removed = await asyncio.to_thread(_remove_ref, self.client, image_ref)
            digest = record.get("digest")
            if digest:
                digest_ref = f"{image_ref.rpartition(':')[0]}@{digest}"
                removed = await asyncio.to_thread(_remove_ref, self.client, digest_ref) and removed
            if removed:
                self.state.forget(image_ref)
                logger.info(
                    f"pre_pull image={image_ref} digest={digest} outcome=evicted_unlisted "
                    f"detail=not served for {(now - since) / 3600:.1f}h"
                )

    async def _retire_superseded(self, image_ref: str, repo: str, digest: str) -> None:
        """When a tracked tag moved to a new digest, drop the old ``repo@<digest>`` reference:
        the re-tag leaves it behind as neither dangling nor a tag, so the disk guard could never
        reclaim its layers and the leak would grow by one image per refresh per pre-pulled tag."""
        old = (self.state.images.get(image_ref) or {}).get("digest")
        if old and old != digest:
            if await asyncio.to_thread(_remove_ref, self.client, f"{repo}@{old}"):
                logger.info(f"pre-pull: retired superseded {repo}@{old} for {image_ref}")

    async def _make_room(
        self, keep_ref: str, need_bytes: int, served: set[str]
    ) -> tuple[bool, str | None]:
        """Keep ``PRE_PULL_MIN_FREE_GB`` free after the pull, evicting LRU pre-pulled images first.

        Only images that left the backend's current top-N (``served``) are evictable: evicting a
        served one to fit another would re-pull the two alternately every sweep on a node whose
        free space fits either but not both. Such a node keeps what it has and skips the pull."""
        floor = settings.PRE_PULL_MIN_FREE_GB * GIB
        skip = served | {keep_ref}
        while True:
            free = psutil.disk_usage(DISK_PATH).free
            if free - need_bytes >= floor:
                return True, None
            # ``self.protected`` is re-read per victim: the loop may have refreshed it while
            # this sweep was running.
            victim = self.state.lru(skip | self.protected)
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
