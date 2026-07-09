import logging
import re
from typing import Optional

from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from core.docker_utils import DockerCommand
from core.utils import _m
from services.const import (
    FILLER_CONTAINER_GRACE_MINUTES,
    POD_CONTAINER_PREFIX,
    RENTAL_CONTAINER_PREFIXES,
)

logger = logging.getLogger(__name__)

# Anonymous docker volumes are named with a 64-char hex hash.
ANON_VOLUME_NAME_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# Remove at most this many per pass; GC runs each cycle, so a backlog drains
# over a few passes without a huge `docker volume rm` command.
VOLUME_RM_MAX_PER_PASS = 200


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
            volumes = await self.get_dangling_anonymous_volumes(ssh_client)
            if volumes is None:
                logger.warning(_m("Listing dangling volumes failed", extra=extra))
                return 0

            volumes = volumes[:VOLUME_RM_MAX_PER_PASS]
            if not volumes:
                return 0

            if self.dry_run:
                logger.info(
                    _m(
                        f"[DRY RUN] Would remove {len(volumes)} dangling anonymous volume(s)",
                        extra=extra | {"volumes": volumes, "dry_run": True},
                    )
                )
                return 0

            await ssh_client.run(DockerCommand.volume_remove(*volumes))
            logger.info(_m(f"Removed {len(volumes)} dangling anonymous volume(s)", extra=extra))
            return len(volumes)

        except Exception as e:
            logger.warning(
                _m(
                    "Dangling anonymous volume prune failed",
                    extra=extra | {"error": str(e)},
                )
            )
            return 0

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

        filler_container = rented_data.get_filler_container(executor_uuid)
        if filler_container:
            rented_containers.add(filler_container)

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
