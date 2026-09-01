import logging
import re
import shlex
from typing import Optional

import asyncssh

from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from core.docker_utils import ALPINE_HELPER_IMAGE, DockerCommand, df_available_bytes
from core.utils import _m
from services.const import (
    DPHN_CACHE_LISTING_FLOOR_GB,
    DPHN_CACHE_VOLUME_PREFIX,
    CACHE_SWEEP_CONTAINER_NAME,
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


class ContainerCleanup:
    """Service for cleaning up stale containers on executor machines."""

    def __init__(self, stale_threshold_minutes: int = FILLER_CONTAINER_GRACE_MINUTES, dry_run: bool = False):
        self.stale_threshold_minutes = stale_threshold_minutes
        self.dry_run = dry_run

    async def cleanup(
        self,
        ssh_client,
        rented_data: Optional[RentedExecutorsResponse],
        executor_uuid: str,
    ) -> tuple[int, list[str]]:
        """Remove containers that are not in rented data and are older than threshold.

        Returns:
            Tuple of (number_removed, list_of_removed_container_names)
        """
        removed_names = []
        extra = {
            "executor_uuid": executor_uuid,
            "threshold_minutes": self.stale_threshold_minutes,
        }

        try:
            # Get all containers with rental prefixes.
            all_containers = await self._get_all_rental_containers(ssh_client)
            extra["total_containers"] = len(all_containers)

            # Get currently rented containers for this executor
            rented_containers = self._get_rented_containers(rented_data, executor_uuid)
            extra["rented_containers"] = str(rented_containers)

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

        return len(removed_names), removed_names

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
        Returns how many volumes were removed.
        """
        extra = {"executor_uuid": executor_uuid}
        try:
            # Cheapest and most selective guard first. Most executors hold no DPHN cache at all — PEARL
            # nodes, miner-assigned nodes, nodes that never ran Dolphin — and this check runs on every
            # node every cycle. Measuring the disk before knowing there is anything to reclaim made all
            # of them start a throwaway container just to learn there was nothing to do.
            cache_volumes = await self._get_cache_volumes(ssh_client, (DPHN_CACHE_VOLUME_PREFIX,))
            if not cache_volumes:
                return 0

            # RUNNING only: _get_all_rental_containers uses `docker ps -a`, so one long-exited
            # filler_* would make "no live filler" false forever and the reclaim would never fire.
            if await self._has_running_filler_container(ssh_client):
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

            await ssh_client.run(DockerCommand.volume_remove(*cache_volumes))
            # `volume_remove` ends in `|| true`, so its exit status cannot tell a removal from a
            # refusal — and dockerd refuses any volume a container still references, in ANY state.
            # Re-list instead: reporting a rescue that did not happen leaves the node delisted while
            # the check event says it was saved.
            surviving: set[str] = set(await self._get_cache_volumes(ssh_client, (DPHN_CACHE_VOLUME_PREFIX,)))
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
                    f"Reclaimed {len(reclaimed)} DPHN cache volume(s) to get the node back over "
                    "the rental listing floor",
                    extra=extra | {"volumes": reclaimed, "kept_volumes": sorted(surviving), "free_gb": free_gb},
                )
            )
            return len(reclaimed)
        except Exception as e:
            logger.warning(_m("DPHN cache reclaim failed", extra=extra | {"error": str(e)}))
            return 0

    async def _has_running_filler_container(self, ssh_client: asyncssh.SSHClientConnection) -> bool:
        # A cache a live filler still mounts is in use; docker refuses to remove it anyway.
        result = await ssh_client.run(
            f'/usr/bin/docker ps --filter "name={FILLER_CONTAINER_PREFIX}" --format "{{{{.Names}}}}"'
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
        # The `rm -f` first: `--rm` does not fire if dockerd is restarted mid-run, and a leftover
        # container would hold the name for good — the sweep would then fail silently on that node
        # forever. Worst case it interrupts a sweep another validator started, which simply runs again.
        # `timeout` bounds a wedged dockerd the same way machine_scrape does for `docker stats`: this
        # call sits in the per-executor pipeline ahead of the fatal port checks, so a hang here would
        # cost the node its whole cycle. A killed sweep simply removes nothing and runs again later.
        result = await ssh_client.run(
            f"/usr/bin/docker rm -f {CACHE_SWEEP_CONTAINER_NAME} >/dev/null 2>&1; "
            f"timeout {DOWNLOAD_TEMPORARY_SWEEP_TIMEOUT_SECONDS} "
            f"/usr/bin/docker run --rm --name {CACHE_SWEEP_CONTAINER_NAME} {mount_flags} {ALPINE_HELPER_IMAGE} "
            f"find {search_roots} -maxdepth {DOWNLOAD_TEMPORARY_MAX_SEARCH_DEPTH} "
            "-type d \\( -name xet -o -name runtimes \\) -prune -o "
            f"-name '*.incomplete' -type f -mmin +{DOWNLOAD_TEMPORARY_MAX_AGE_MINUTES} "
            "-print -exec rm -f {} + 2>/dev/null || true"
        )
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    async def get_dangling_anonymous_volumes(self, ssh_client) -> Optional[list[str]]:
        # anonymous (64-hex) volumes referenced by no container; None if listing failed
        result = await ssh_client.run(DockerCommand.volume_ls_dangling())
        if result.exit_status != 0:
            return None
        names = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
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

    def _get_rented_containers(
        self,
        rented_data: Optional[RentedExecutorsResponse],
        executor_uuid: str
    ) -> set[str]:
        """Get currently rented container names for this executor."""
        if not rented_data:
            return set()

        rented_containers = set()
        executor = rented_data.executors.get(executor_uuid)
        if executor:
            rented_containers.update(pod.container_name for pod in executor.pods)

        # Every filler container on the node is protected — a GPU-split node runs one per VRAM
        # bundle (DAH-2465), and reaping a sibling kills a live worker mid-cycle.
        rented_containers.update(rented_data.get_filler_containers(executor_uuid))

        return rented_containers

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
        """Remove a container and its associated resources."""
        try:
            # Remove stale containers together with anonymous Docker volumes.
            result = await ssh_client.run(DockerCommand.remove_with_volumes(container_name))
            if result.exit_status != 0:
                return False

            # Remove associated volume if it's a pod container
            if container_name.startswith(POD_CONTAINER_PREFIX):
                pod_id = container_name.removeprefix(POD_CONTAINER_PREFIX)
                await ssh_client.run(DockerCommand.volume_remove(f"volume_{pod_id}"))

            return True

        except Exception as e:
            logger.warning(
                _m(
                    f"Error removing container {container_name}",
                    extra={"container_name": container_name, "error": str(e)}
                )
            )
            return False
