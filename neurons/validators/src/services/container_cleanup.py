import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
import shlex
from typing import Awaitable, Callable, Optional

import asyncssh

from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from core.docker_utils import ALPINE_HELPER_IMAGE, DockerCommand, df_available_bytes
from core.utils import _m
from services.const import (
    DPHN_CACHE_LISTING_FLOOR_GB,
    CACHE_SWEEP_CONTAINER_NAME,
    EDIT_PARKED_SUFFIX,
    FILLER_CACHE_VOLUME_PREFIXES,
    FILLER_CONTAINER_GRACE_MINUTES,
    FILLER_CONTAINER_PREFIX,
    POD_CONTAINER_PREFIX,
    RENTAL_CONTAINER_PREFIXES,
)

logger = logging.getLogger(__name__)

# Anonymous docker volumes are named with a 64-char hex hash.
ANON_VOLUME_NAME_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# DAH-2475: a node whose free disk has fallen below the rental listing floor must give the filler
# cache back — it is worth ~37 GB and a delisted node earns nothing, so the cache is the first thing
# to sacrifice. This is the SAME floor the affordability gate keeps a node above, so they share one
# constant: split them and a node could be granted a cache it is immediately reclaimed for.
DPHN_CACHE_RECLAIM_FREE_FLOOR_GB = DPHN_CACHE_LISTING_FLOOR_GB

# Remove at most this many per pass; GC runs each cycle, so a backlog drains
# over a few passes without a huge `docker volume rm` command.
VOLUME_RM_MAX_PER_PASS = 200

# A rental's own volume is created as volume_<pod uuid>; nothing else on the host is named so.
POD_VOLUME_NAME_PATTERN = re.compile(r"^volume_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")
# Backend lookups go per candidate volume; a backlog drains over a few passes.
ORPHAN_VOLUME_LOOKUPS_PER_PASS = 50
# Wall-clock budget for those lookups, well inside the executor task timeout; the rest wait a pass.
ORPHAN_VOLUME_LOOKUP_BUDGET_SECONDS = 30


# DAH-2805: how long an abandoned download temporary may sit before the sweep removes it. The window
# has to clear the longest silence a HEALTHY download can have, because age is the only signal (see
# sweep_abandoned_download_temporaries): with hf_xet the destination file is written in 64 MiB
# bursts, which at the slowest per-file rate measured in prod (~50 KB/s) is ~22 min between writes.
# 4 hours clears that by 10x while bounding the standing garbage to ~70 GB at the observed leak rate.
# The residual case is accepted, not overlooked: a download whose process is alive but has written
# nothing for four hours loses its file. It loses nothing real — huggingface_hub never resumes a
# temporary anyway, so that attempt was already going to restart from byte zero.
DOWNLOAD_TEMPORARY_MAX_AGE_MINUTES = 240

# Depth from the volume root: `hub/models--*/blobs` (Dolphin) and
# `models/<repo>/.cache/huggingface/download` (ENGY); 8 reaches both with slack. The bound matters
# because these volumes also hold the multi-GB xet chunk cache and the worker runtimes.
DOWNLOAD_TEMPORARY_MAX_SEARCH_DEPTH = 8

# The whole helper run is bounded: the walk itself takes hundreds of ms on a real cache, so anything
# near a minute means a wedged docker daemon rather than work in progress.
DOWNLOAD_TEMPORARY_SWEEP_TIMEOUT_SECONDS = 60

# The sweep helper mounts every cache volume. `--rm` only fires once the container has run, so a
# `docker run` whose client is killed (timeout, dropped SSH) between create and start leaves it in
# Created for good, and dockerd then refuses `volume rm` on the cache it mounts ("volume is in use").
REMOVE_CACHE_SWEEP_CONTAINER = f"/usr/bin/docker rm -f {CACHE_SWEEP_CONTAINER_NAME} >/dev/null 2>&1"

# The images of the fillers whose caches the reclaims below give back (Engy's alone is ~32 GB). No
# `-f`: dockerd refuses an image any container still uses, running or stopped, so a live filler keeps it.
FILLER_IMAGE_REPOSITORIES = ("ghcr.io/hanlinai/engy-golden-serve", "daturaai/dolphin", "daturaai/dlph")
REMOVE_IDLE_FILLER_IMAGES = (
    "/usr/bin/docker image ls "
    + " ".join(f"--filter reference={repository}" for repository in FILLER_IMAGE_REPOSITORIES)
    + " --format '{{.Repository}}:{{.Tag}}' | xargs -r /usr/bin/docker image rm >/dev/null 2>&1 || true"
)


async def evict_cache_volumes_oldest_first(
    ssh_client: asyncssh.SSHClientConnection,
    volume_names: list[str],
    run_removal: Callable[[str], Awaitable[object]],
    has_room: Callable[[], Awaitable[bool]],
) -> None:
    """Remove filler cache volumes oldest first, stopping as soon as `has_room()` says the disk is enough.

    DAH-5014: an Engy checkpoint is up to ~1.6 TB and buys a fast restart, so a rental or the listing
    floor takes only what it needs. The caller has already found there is not enough room, so the
    idle filler images go first without a re-check. An order docker cannot report falls back to name order.
    """
    created_at: dict[str, str] = {}
    try:
        inspected = await ssh_client.run(
            "/usr/bin/docker volume inspect --format '{{.Name}} {{.CreatedAt}}' "
            + " ".join(shlex.quote(name) for name in volume_names)
        )
        for line in (inspected.stdout or "").splitlines():
            name, _, created = line.strip().partition(" ")
            created_at[name] = created
    except asyncio.CancelledError:
        raise
    except Exception:
        pass
    # the image first: a pull brings it back far faster than a checkpoint download
    await run_removal(REMOVE_IDLE_FILLER_IMAGES)
    if await has_room():
        return
    ordered: list[str] = sorted(volume_names, key=lambda name: (created_at.get(name, "~"), name))
    for index, name in enumerate(ordered):
        await run_removal(f"{REMOVE_CACHE_SWEEP_CONTAINER}; {DockerCommand.volume_remove(name)}")
        if index + 1 < len(ordered) and await has_room():
            return


RENTED_LIST_UNAVAILABLE = "rented_list_unavailable"
RENTED_LIST_EMPTY = "rented_list_empty"


def listed_container_names(rented_data: Optional[RentedExecutorsResponse]) -> set[str]:
    """Every container name the fleet snapshot lists, under any executor id.

    Pod names carry a UUID and filler names a run id, so a listed name is unique across the fleet
    and protecting it on every host costs nothing. Reading only this executor's entry deleted live
    pods whose node row the backend keyed on a twin executor id (one IP:port registered twice).
    """
    if not rented_data:
        return set()
    names: set[str] = set()
    for executor in rented_data.executors.values():
        # DAH-4315: a pod the backend has not named yet protects the pod_<pod_id> it is created as
        names.update(pod.docker_container_name for pod in executor.pods)
        # An edit parks the pod's current container under <name>__prev while the replacement
        # is created; it is the customer's only copy until then, whatever its age
        names.update(f"{pod.docker_container_name}{EDIT_PARKED_SUFFIX}" for pod in executor.pods)
    # Every filler is protected — a GPU-split node runs one per VRAM bundle, and reaping a
    # sibling kills a live worker mid-cycle. The legacy single map covers an older backend.
    for fillers in rented_data.all_filler_containers_by_executor.values():
        names.update(fillers)
    names.update(name for name in rented_data.filler_containers_by_executor.values() if name)
    return names


def rented_list_unknown_reason(rented_data: Optional[RentedExecutorsResponse]) -> Optional[str]:
    """Why the snapshot cannot be trusted to authorise a removal, or None.

    Only a snapshot that is missing or lists nothing anywhere in the fleet is untrusted: that is the
    one shape a backend hiccup takes. An executor absent from a non-empty snapshot is an idle node,
    and its orphans must still go, or they hold the rental ports and the node scores 0 every cycle.
    The scheduled paths skip the cycle when the fetch fails, so None is a defensive branch.
    """
    if rented_data is None:
        return RENTED_LIST_UNAVAILABLE
    if not listed_container_names(rented_data):
        return RENTED_LIST_EMPTY
    return None


# The machine scrape's two capability checks (machine_scrape.run_capability_check): named since
# DAH-5192; before that unnamed, so recognised by the exact check, the executor image running
# `nvidia-smi`. Only never-started (Created) ones are removed: a running one is a check in progress.
CAPABILITY_CHECK_NAME_PREFIX = "lium_capability_check_"
CAPABILITY_CHECK_IMAGE = "daturaai/compute-subnet-executor:latest"
LIST_CREATED_CONTAINERS = (
    "/usr/bin/docker ps -a --no-trunc --filter status=created --format '{{.ID}} {{.Names}} {{.Image}} {{.Command}}'"
)


def never_started_check_containers(listing: str) -> list[str]:
    ids: list[str] = []
    for fields in (line.split(None, 3) for line in listing.splitlines()):
        if len(fields) == 4 and (
            fields[1].startswith(CAPABILITY_CHECK_NAME_PREFIX)
            or (fields[2] == CAPABILITY_CHECK_IMAGE and fields[3].strip().strip('"') == "nvidia-smi")
        ):
            ids.append(fields[0])
    return ids


class ContainerCleanup:
    """Service for cleaning up stale containers on executor machines."""

    def __init__(self, stale_threshold_minutes: int = FILLER_CONTAINER_GRACE_MINUTES, dry_run: bool = False):
        self.stale_threshold_minutes = stale_threshold_minutes
        self.dry_run = dry_run
        # per executor: the last orphan-volume candidate examined, so a capped pass resumes after it
        self._orphan_volume_cursor: dict[str, str] = {}

    def _get_rented_containers(
        self, rented_data: Optional[RentedExecutorsResponse], executor_uuid: str
    ) -> set[str]:
        """Kept for callers outside this file (#1448's DinD volume sweep); the executor id is unused."""
        return listed_container_names(rented_data)

    async def cleanup(
        self,
        ssh_client,
        rented_data: Optional[RentedExecutorsResponse],
        executor_uuid: str,
        on_before_remove: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[int, list[str], list[str]]:
        """Remove containers that are not in rented data and are older than threshold.

        ``on_before_remove(container_name)`` is awaited right before each removal (DAH-3338: the
        check queues the reap for the backend first, so a crash between the removal and the report
        does not lose it). It must not raise; an error inside it is the caller's to swallow.

        Returns:
            Tuple of (number_removed, removed container names, orphaned containers that survived
            removal — they still hold their ports, so the port check names them).
        """
        removed_names = []
        unremovable_names: list[str] = []
        extra = {
            "executor_uuid": executor_uuid,
            "threshold_minutes": self.stale_threshold_minutes,
        }

        unknown_reason = rented_list_unknown_reason(rented_data)
        if unknown_reason is not None:
            logger.warning(
                _m(
                    "Skipping stale container removal: the fleet rented snapshot is empty or unknown",
                    extra={**extra, "skip_reason": unknown_reason},
                )
            )
            await self.prune_dangling_anonymous_volumes(ssh_client, executor_uuid)
            return 0, [], []

        try:
            # Get all containers with rental prefixes.
            all_containers = await self._get_all_rental_containers(ssh_client)
            extra["total_containers"] = len(all_containers)

            # Get currently rented containers for this executor
            rented_containers = listed_container_names(rented_data)
            extra["rented_container_count"] = len(rented_containers)

            # Check each container
            for container_name in all_containers:
                stripped_name = container_name.strip()
                if stripped_name in rented_containers:
                    continue

                age_minutes = await self._get_container_age_minutes(ssh_client, stripped_name)
                if age_minutes and age_minutes > self.stale_threshold_minutes:
                    if self.dry_run:
                        logger.info(
                            _m(
                                f"[DRY RUN] Would remove stale container {stripped_name}",
                                extra={
                                    **extra,
                                    "container_name": stripped_name,
                                    "age_minutes": round(age_minutes, 1),
                                    "dry_run": True,
                                }
                            )
                        )
                        continue

                    if on_before_remove is not None:
                        await on_before_remove(stripped_name)
                    if await self._remove_container(ssh_client, stripped_name):
                        removed_names.append(stripped_name)
                        logger.info(
                            _m(
                                f"Removed stale container {stripped_name}",
                                extra={
                                    **extra,
                                    "container_name": stripped_name,
                                    "age_minutes": round(age_minutes, 1),
                                }
                            )
                        )
                    else:
                        unremovable_names.append(stripped_name)

        except Exception as e:
            logger.warning(
                _m(
                    "Error during stale container cleanup",
                    extra={**extra, "error": str(e)}
                )
            )

        if removed_names:
            logger.info(
                _m(
                    f"Cleaned up {len(removed_names)} stale containers",
                    extra={
                        **extra,
                        "removed_count": len(removed_names),
                        "removed_containers": removed_names,
                    }
                )
            )

        # DAH-2375: reap anonymous volumes orphaned by historical `docker rm`
        # without -v. Best-effort — never raises, never changes this return.
        await self.prune_dangling_anonymous_volumes(ssh_client, executor_uuid)
        await self.remove_never_started_check_containers(ssh_client, executor_uuid)

        return len(removed_names), removed_names, unremovable_names

    async def remove_never_started_check_containers(self, ssh_client, executor_uuid: str) -> int:
        # best-effort like the volume prune above: never raises
        extra = {"executor_uuid": executor_uuid}
        try:
            listed = await ssh_client.run(LIST_CREATED_CONTAINERS)
            if getattr(listed, "exit_status", 0) != 0:
                return 0
            ids: list[str] = never_started_check_containers(listed.stdout or "")
            if not ids or self.dry_run:
                return 0
            await ssh_client.run(f"/usr/bin/docker rm -f {' '.join(ids)} >/dev/null 2>&1 || true")
            logger.info(_m(f"Removed {len(ids)} never-started capability check container(s)", extra=extra))
            return len(ids)
        except Exception as e:
            logger.warning(_m("Never-started check container removal failed", extra=extra | {"error": str(e)}))
            return 0

    async def prune_dangling_anonymous_volumes(
        self,
        ssh_client,
        executor_uuid: str,
    ) -> int:
        # reap orphaned anonymous volumes left behind by container removals without -v
        extra = {"executor_uuid": executor_uuid}
        try:
            dangling_anonymous_volumes: Optional[list[str]] = await self.get_dangling_anonymous_volumes(ssh_client)
            if dangling_anonymous_volumes is None:
                logger.warning(_m("Listing dangling volumes failed", extra=extra))
                return 0

            volumes_to_remove = dangling_anonymous_volumes[:VOLUME_RM_MAX_PER_PASS]
            if not volumes_to_remove:
                return 0

            if self.dry_run:
                logger.info(
                    _m(
                        f"[DRY RUN] Would remove {len(volumes_to_remove)} dangling anonymous volume(s)",
                        extra=extra | {"volumes": volumes_to_remove, "dry_run": True},
                    )
                )
                return 0

            await ssh_client.run(DockerCommand.volume_remove(*volumes_to_remove))
            logger.info(
                _m(
                    f"Removed {len(volumes_to_remove)} of {len(dangling_anonymous_volumes)} "
                    "dangling anonymous volume(s)",
                    extra=extra,
                )
            )
            return len(volumes_to_remove)

        except Exception as e:
            logger.warning(
                _m(
                    "Dangling anonymous volume prune failed",
                    extra=extra | {"error": str(e)},
                )
            )
            return 0

    async def reclaim_dphn_cache_when_disk_is_tight(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        executor_uuid: str,
    ) -> int:
        """Give the DPHN filler cache back when the node can no longer afford it.

        DAH-2475: the create-time sweep only ever removes STALE cache versions — it deliberately never
        reclaims, because a node's free disk is low precisely BECAUSE the cache is downloaded, so
        reclaiming there would make the backend grant it again next cycle and the node would
        re-download ~37 GB forever. This backstop is the one place that reclaims, and it decides on
        real free space rather than on a request: below the rental listing floor the node earns
        nothing, so the cache is the first thing to sacrifice.

        Gated on NO live filler container: a cache a sibling still mounts is in use (docker refuses to
        remove it anyway), and a node still filling bundles is not the node this is meant to rescue.
        Nor a live customer pod (Fish 26 Oct): the cache is kept across rentals and listed as used, so
        a renter who took the listed free disk leaves the node under the floor by design.
        Returns how many volumes were removed.
        """
        extra = {"executor_uuid": executor_uuid}
        try:
            # Cheapest and most selective guard first. Most executors hold no DPHN cache at all — PEARL
            # nodes, miner-assigned nodes, nodes that never ran Dolphin — and this check runs on every
            # node every cycle. Measuring the disk before knowing there is anything to reclaim made all
            # of them start a throwaway container just to learn there was nothing to do.
            # DAH-5014: the Engy checkpoint cache is reclaimed here too, oldest first.
            cache_volumes = await self._get_cache_volumes(ssh_client, FILLER_CACHE_VOLUME_PREFIXES)
            if not cache_volumes:
                return 0

            # RUNNING only: _get_all_rental_containers uses `docker ps -a`, so one long-exited
            # filler_* would make "no live filler" false forever and the reclaim would never fire.
            if await self._has_running_filler_or_pod(ssh_client):
                return 0

            free_gb = await self._get_free_disk_gb(ssh_client)
            if free_gb is None or free_gb >= DPHN_CACHE_RECLAIM_FREE_FLOOR_GB:
                return 0

            if self.dry_run:
                logger.info(
                    _m(
                        f"[DRY RUN] Would reclaim {len(cache_volumes)} DPHN cache volume(s)",
                        extra=extra | {"volumes": cache_volumes, "free_gb": free_gb, "dry_run": True},
                    )
                )
                return 0

            async def above_floor() -> bool:
                free_now: float | None = await self._get_free_disk_gb(ssh_client)
                return free_now is not None and free_now >= DPHN_CACHE_RECLAIM_FREE_FLOOR_GB

            await evict_cache_volumes_oldest_first(ssh_client, cache_volumes, ssh_client.run, above_floor)
            # `volume_remove` ends in `|| true`, so its exit status cannot tell a removal from a
            # refusal — and dockerd refuses any volume a container still references, in ANY state.
            # Re-list instead: reporting a rescue that did not happen leaves the node delisted while
            # the check event says it was saved.
            surviving: set[str] = set(await self._get_cache_volumes(ssh_client, FILLER_CACHE_VOLUME_PREFIXES))
            reclaimed: list[str] = [name for name in cache_volumes if name not in surviving]
            if not reclaimed:
                logger.warning(
                    _m(
                        "DPHN cache reclaim removed nothing; dockerd still holds the volumes",
                        extra=extra | {"volumes": cache_volumes, "free_gb": free_gb},
                    )
                )
                return 0
            logger.info(
                _m(
                    f"Reclaimed {len(reclaimed)} filler cache volume(s) to get the node back over "
                    "the rental listing floor",
                    extra=extra | {"volumes": reclaimed, "kept_volumes": sorted(surviving), "free_gb": free_gb},
                )
            )
            return len(reclaimed)
        except Exception as e:
            logger.warning(_m("DPHN cache reclaim failed", extra=extra | {"error": str(e)}))
            return 0

    async def remove_orphan_rental_volumes(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        rented_data: Optional[RentedExecutorsResponse],
        executor_uuid: str,
        is_orphan: Callable[[str, timedelta], Awaitable[bool]],
    ) -> int:
        """Remove `volume_<uuid>` volumes whose pod or filler run is over and whose container is gone.

        The orphan reaper removes a `volume_<id>` only together with its container, and the dangling
        prune only matches anonymous (64-hex) names, so a named volume whose container was already
        gone (a delete whose volume removal failed, a filler stopped without it) stayed for good.
        A candidate must be a `volume_<uuid>` no container (running or stopped) mounts, and not a pod
        or filler the fleet snapshot lists: the snapshot carries every pod status but BROKEN and
        DELETING, so a stopped or failed-reboot pod keeps its volume. ``is_orphan(uuid, age)`` then
        decides from the backend, ``age`` being the time since the volume was created. Returns how
        many volumes were removed.
        """
        extra = {"executor_uuid": executor_uuid}
        try:
            if rented_list_unknown_reason(rented_data) is not None:
                return 0
            dangling: Optional[list[str]] = await self._get_dangling_volumes(ssh_client)
            listed_names: set[str] = listed_container_names(rented_data)
            listed_pod_ids: set[str] = {
                pod.pod_id for executor in rented_data.executors.values() for pod in executor.pods
            }
            candidates: list[str] = sorted(
                name
                for name in dangling or []
                if (match := POD_VOLUME_NAME_PATTERN.match(name))
                and match[1] not in listed_pod_ids
                and f"{POD_CONTAINER_PREFIX}{match[1]}" not in listed_names
                and f"{FILLER_CONTAINER_PREFIX}{match[1]}" not in listed_names
            )
            # resume after the last pass's batch, wrapping around, so volumes that are kept or cannot
            # be dated never starve the ones behind them
            after: str = self._orphan_volume_cursor.get(executor_uuid, "")
            candidates = ([n for n in candidates if n > after] + [n for n in candidates if n <= after])[
                :ORPHAN_VOLUME_LOOKUPS_PER_PASS
            ]
            if not candidates:
                return 0
            self._orphan_volume_cursor[executor_uuid] = candidates[-1]

            ages: dict[str, timedelta] = await self._volume_ages(ssh_client, candidates)
            orphans: list[str] = []
            # a pass cut off by the budget resumes AT the in-flight candidate, so a slow one is not
            # skipped behind faster ones; only a first candidate that eats the whole budget is passed
            self._orphan_volume_cursor[executor_uuid] = candidates[0]
            try:
                async with asyncio.timeout(ORPHAN_VOLUME_LOOKUP_BUDGET_SECONDS):
                    for name in candidates:
                        if name in ages and await is_orphan(name.removeprefix("volume_"), ages[name]):
                            orphans.append(name)
                        self._orphan_volume_cursor[executor_uuid] = name
            except TimeoutError:
                logger.info(
                    _m("Orphan volume lookups hit the pass budget; resuming next pass", extra=extra)
                )
            if not orphans:
                return 0
            if self.dry_run:
                logger.info(
                    _m(
                        f"[DRY RUN] Would remove {len(orphans)} orphan rental volume(s)",
                        extra=extra | {"volumes": orphans, "dry_run": True},
                    )
                )
                return 0

            await ssh_client.run(DockerCommand.volume_remove(*orphans))
            # `volume_remove` ends in `|| true`: re-list to report only what is really gone
            surviving: set[str] = set(await self._get_cache_volumes(ssh_client, ("volume_",)))
            removed: list[str] = [name for name in orphans if name not in surviving]
            if removed:
                logger.info(
                    _m(f"Removed {len(removed)} orphan rental volume(s)", extra=extra | {"volumes": removed})
                )
            return len(removed)
        except Exception as e:
            logger.warning(_m("Orphan rental volume removal failed", extra=extra | {"error": str(e)}))
            return 0

    async def _volume_ages(self, ssh_client: asyncssh.SSHClientConnection, names: list[str]) -> dict[str, timedelta]:
        # time since each volume was created, by the host's clock; a volume it cannot date is left out
        quoted = " ".join(shlex.quote(name) for name in names)
        result = await ssh_client.run(
            f"date +%s; /usr/bin/docker volume inspect --format '{{{{.Name}}}} {{{{.CreatedAt}}}}' {quoted}"
        )
        lines = (result.stdout or "").splitlines()
        if not lines or not lines[0].strip().isdigit():
            return {}
        now = datetime.fromtimestamp(int(lines[0].strip()), UTC)
        ages: dict[str, timedelta] = {}
        for line in lines[1:]:
            name, _, created = line.strip().partition(" ")
            try:
                ages[name] = now - datetime.fromisoformat(created.strip())
            except ValueError:
                continue
        return ages

    async def _get_dangling_volumes(self, ssh_client) -> Optional[list[str]]:
        # every volume no container (in any state) references; None if listing failed
        result = await ssh_client.run(DockerCommand.volume_ls_dangling())
        if result.exit_status != 0:
            return None
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    async def _has_running_filler_or_pod(self, ssh_client: asyncssh.SSHClientConnection) -> bool:
        # A cache a live filler still mounts is in use; a live rental keeps it (two name filters OR).
        result = await ssh_client.run(
            f'/usr/bin/docker ps --filter "name={FILLER_CONTAINER_PREFIX}" --filter "name={POD_CONTAINER_PREFIX}" '
            '--format "{{.Names}}"'
        )
        if getattr(result, "exit_status", 0) != 0:
            return True  # cannot prove the node is idle -> do not reclaim
        return bool((result.stdout or "").strip())

    async def _get_free_disk_gb(self, ssh_client: asyncssh.SSHClientConnection) -> float | None:
        # The root is DISCOVERED, not assumed: a node with docker moved to a dedicated disk would
        # otherwise be judged by its root filesystem, and the reclaim would either never fire while
        # docker's disk is full or fire and delete a healthy cache the node had room for.
        info = await ssh_client.run("/usr/bin/docker info --format '{{.DockerRootDir}}'")
        if getattr(info, "exit_status", 0) != 0:
            return None
        docker_root: str = (info.stdout or "").strip() or "/var/lib/docker"
        try:
            return await df_available_bytes(ssh_client, docker_root) / (1024**3)
        except Exception:
            # A backstop that cannot measure declines rather than guesses; the caller treats None as
            # "leave the cache alone".
            return None

    async def _get_cache_volumes(
        self, ssh_client: asyncssh.SSHClientConnection, name_prefixes: tuple[str, ...]
    ) -> list[str]:
        result = await ssh_client.run('/usr/bin/docker volume ls --format "{{.Name}}"')
        if getattr(result, "exit_status", 0) != 0:
            return []
        return sorted(
            name
            for name in (line.strip() for line in (result.stdout or "").splitlines())
            if name.startswith(name_prefixes)
        )

    async def sweep_abandoned_download_temporaries(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        executor_uuid: str,
    ) -> int:
        """Delete `*.incomplete` files no writer has touched in DOWNLOAD_TEMPORARY_MAX_AGE_MINUTES.

        DAH-2805: a filler's weight download that is killed mid-flight leaves its temporary behind,
        and nothing ever reads it again (see the constant for the mechanism). One prod node reached
        741 GB of them and dropped out of the rental listing.

        It runs HERE, not inside the filler image, for three reasons: this loop visits a node on
        every cycle whatever image is on it, one prefix list covers DPHN and ENGY alike, and it
        cleans the garbage already on the fleet without waiting for an image rollout.

        Only files matching `*.incomplete`, only inside volumes whose name carries one of our own
        cache prefixes, and only when they are older than the window — a renter's volume is never a
        candidate. Returns how many files were removed.
        """
        extra = {"executor_uuid": executor_uuid}
        try:
            cache_volumes: list[str] = await self._get_cache_volumes(ssh_client, FILLER_CACHE_VOLUME_PREFIXES)
            if not cache_volumes:
                return 0
            if self.dry_run:
                logger.info(
                    _m(
                        "[DRY RUN] Would sweep abandoned download temporaries",
                        extra=extra | {"volumes": cache_volumes, "dry_run": True},
                    )
                )
                return 0

            removed_paths: list[str] = await self._remove_aged_download_temporaries(ssh_client, cache_volumes)
            if not removed_paths:
                return 0
            logger.info(
                _m(
                    f"Swept {len(removed_paths)} abandoned download temporaries from the filler cache",
                    extra=extra
                    | {
                        "volumes": cache_volumes,
                        "max_age_minutes": DOWNLOAD_TEMPORARY_MAX_AGE_MINUTES,
                        "sample_paths": removed_paths[:5],
                    },
                )
            )
            return len(removed_paths)
        except Exception as e:
            logger.warning(_m("Download-temporary sweep failed", extra=extra | {"error": str(e)}))
            return 0

    async def _remove_aged_download_temporaries(
        self, ssh_client: asyncssh.SSHClientConnection, volume_names: list[str]
    ) -> list[str]:
        """Run the find INSIDE a helper container with the volumes mounted by name.

        Same reason `df_available_bytes` measures disk through a helper: the validator's SSH session
        lands inside the executor container, where the volumes' host paths do not exist. Mounting by
        NAME also removes the mount-point lookup, and the helper cannot reach outside what we mount.

        `-name` comes before `-type`/`-mmin` so only a name match pays a stat(), and the removal is
        `-exec rm`, not `-delete`: `-delete` turns on `-depth`, which silently disables the prunes.
        Verified against busybox find (alpine 3.19): it removes the aged temporary, keeps a fresh one,
        keeps a finished blob, and never descends into the xet or runtimes trees.
        """
        mount_flags: str = " ".join(
            f"-v {shlex.quote(name)}:/cache{index}" for index, name in enumerate(volume_names)
        )
        search_roots: str = " ".join(f"/cache{index}" for index in range(len(volume_names)))
        # The `rm -f` after: a run killed before its container started leaves it Created, pinning the
        # volumes (see REMOVE_CACHE_SWEEP_CONTAINER). The `rm -f` first: `--rm` does not fire if dockerd is restarted mid-run, and a leftover
        # container would hold the name for good — the sweep would then fail silently on that node
        # forever. Worst case it interrupts a sweep another validator started, which simply runs again.
        # `timeout` bounds a wedged dockerd the same way machine_scrape does for `docker stats`: this
        # call sits in the per-executor pipeline ahead of the fatal port checks, so a hang here would
        # cost the node its whole cycle. A killed sweep simply removes nothing and runs again later.
        result = await ssh_client.run(
            f"{REMOVE_CACHE_SWEEP_CONTAINER}; "
            f"timeout {DOWNLOAD_TEMPORARY_SWEEP_TIMEOUT_SECONDS} "
            f"/usr/bin/docker run --rm --name {CACHE_SWEEP_CONTAINER_NAME} {mount_flags} {ALPINE_HELPER_IMAGE} "
            f"find {search_roots} -maxdepth {DOWNLOAD_TEMPORARY_MAX_SEARCH_DEPTH} "
            "-type d \\( -name xet -o -name runtimes \\) -prune -o "
            f"-name '*.incomplete' -type f -mmin +{DOWNLOAD_TEMPORARY_MAX_AGE_MINUTES} "
            "-print -exec rm -f {} + 2>/dev/null; "
            f"{REMOVE_CACHE_SWEEP_CONTAINER} || true"
        )
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    async def get_dangling_anonymous_volumes(self, ssh_client) -> Optional[list[str]]:
        # anonymous (64-hex) volumes referenced by no container; None if listing failed
        names = await self._get_dangling_volumes(ssh_client)
        if names is None:
            return None
        return [name for name in names if ANON_VOLUME_NAME_PATTERN.match(name)]

    async def force_remove_health_checks(
        self,
        ssh_client,
        executor_uuid: str,
    ) -> int:
        """Force-remove every `health_check_*` container on the executor.

        DAH-1991: backend-spawned `health_check_<epoch>` probes compete for
        the rental port range during the 20-60s gap inside
        DockerService.create_container. Cleaning them at the spawn site
        (RentalVerificationCheck) closes the orphan window from minutes
        (the 15-min stale sweep) to seconds. No age threshold is applied —
        the caller has just finished the API call that produced the probe,
        so any matching container is already done.

        Failures are swallowed; the caller treats this as best-effort and
        must not flip its own outcome based on cleanup result.
        """
        try:
            result = await ssh_client.run(
                "docker ps -q --filter 'name=^health_check_' | xargs -r docker rm -fv"
            )
            removed = [
                line for line in (result.stdout or "").strip().split("\n")
                if line.strip()
            ]
            count = len(removed)
            if count:
                logger.info(
                    _m(
                        f"Force-removed {count} health_check_ container(s)",
                        extra={
                            "executor_uuid": executor_uuid,
                            "removed_count": count,
                        },
                    )
                )
            return count
        except Exception as e:
            logger.warning(
                _m(
                    "health_check_ post-verification cleanup failed",
                    extra={"executor_uuid": executor_uuid, "error": str(e)},
                )
            )
            return 0

    async def _get_all_rental_containers(self, ssh_client) -> list[str]:
        """Get all containers with rental-related prefixes.

        Iterates RENTAL_CONTAINER_PREFIXES (services/const.py) so any short-lived
        container that competes for the rental port range is caught by cleanup.
        A stale running `health_check_*` from a crashed backend would otherwise
        hold ports 9100-9130 indefinitely.
        """
        try:
            patterns = [f"{prefix}*" for prefix in RENTAL_CONTAINER_PREFIXES]
            result = await ssh_client.run(DockerCommand.ps_filter(*patterns))
            if result.stdout and result.stdout.strip():
                return result.stdout.strip().split('\n')
        except Exception as e:
            logger.warning(
                _m(
                    f"Error getting rental containers",
                    extra={
                        "error": str(e)
                    }
                )
            )

        return []

    async def _get_container_age_minutes(self, ssh_client, container_name: str) -> Optional[float]:
        """Get container age in minutes, returns None if unable to determine."""
        try:
            # Get container creation timestamp
            created_result = await ssh_client.run(
                DockerCommand.inspect_created_timestamp(container_name)
            )
            if created_result.exit_status != 0:
                return None
            created_timestamp = int(created_result.stdout.strip())

            # Get current time on the machine
            current_result = await ssh_client.run("date +%s")
            if current_result.exit_status != 0:
                return None
            current_timestamp = int(current_result.stdout.strip())

            # Calculate age
            age_minutes = (current_timestamp - created_timestamp) / 60
            return age_minutes

        except Exception as e:
            logger.warning(
                _m(
                    f"Error checking container age for {container_name}",
                    extra={"container_name": container_name, "error": str(e)}
                )
            )
            return None

    async def _remove_container(self, ssh_client, container_name: str) -> bool:
        """Remove a container and its associated resources.

        True once the container itself is gone. The pod volume removal after it is best-effort:
        an error there is logged and does not turn a removed container into an unremovable one —
        the caller reads False as "still on the host" (the port check names it, DAH-3338 drops its
        reaped report), and both would be wrong about a container `docker rm` already took.
        """
        try:
            # Remove stale containers together with anonymous Docker volumes.
            result = await ssh_client.run(DockerCommand.remove_with_volumes(container_name))
            if result.exit_status != 0:
                # DAH-2991: dockerd could not kill the container (ticket-0287: "tried to kill
                # container, but did not receive an exit event", 4 backend deletes failed the same
                # way and the orphan held 8 of 10 rental ports for 5 h). Used to return False here in
                # silence, every cycle. Kill its init and shim directly and try once more.
                error = (result.stderr or result.stdout or "").strip()
                logger.warning(
                    _m(
                        f"docker rm -f failed for {container_name}; killing its processes directly",
                        extra={"container_name": container_name, "error": error},
                    )
                )
                killed = await ssh_client.run(DockerCommand.kill_container_processes(container_name))
                result = await ssh_client.run(DockerCommand.remove_with_volumes(container_name))
                if result.exit_status != 0:
                    logger.error(
                        _m(
                            f"Container {container_name} survives docker rm -f and a direct kill",
                            extra={
                                "container_name": container_name,
                                "error": (result.stderr or result.stdout or "").strip(),
                                "killed": (getattr(killed, "stdout", "") or "").strip(),
                            },
                        )
                    )
                    return False
        except Exception as e:
            logger.warning(
                _m(
                    f"Error removing container {container_name}",
                    extra={"container_name": container_name, "error": str(e)}
                )
            )
            return False

        # A create names both the container and its volume after payload.pod_id (the pod id or the
        # filler run id), so `volume_<id>` is this container's own volume and no one else's.
        for prefix in (POD_CONTAINER_PREFIX, FILLER_CONTAINER_PREFIX):
            if not container_name.startswith(prefix):
                continue
            volume_name = f"volume_{container_name.removeprefix(prefix)}"
            try:
                result = await ssh_client.run(DockerCommand.volume_remove_strict(volume_name))
                output = (result.stderr or result.stdout or "").strip()
                error = "" if result.exit_status == 0 else output or f"exit_status={result.exit_status}"
            except Exception as e:
                error = str(e) or type(e).__name__
            if error and "no such volume" not in error.lower():
                logger.warning(
                    _m(
                        f"Removed container {container_name} but not its volume",
                        extra={"container_name": container_name, "volume": volume_name, "error": error},
                    )
                )

        return True
