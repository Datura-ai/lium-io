import logging
from typing import Optional

from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from core.docker_utils import DockerCommand
from core.utils import _m
from services.const import POD_CONTAINER_PREFIX

logger = logging.getLogger(__name__)


class ContainerCleanup:
    """Service for cleaning up stale containers on executor machines."""

    def __init__(self, stale_threshold_minutes: int = 15, dry_run: bool = True):
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
            # Get all containers with rental prefixes (both pod_ and container_)
            all_containers = await self._get_all_rental_containers(ssh_client)
            if not all_containers:
                return 0, []

            extra["total_containers"] = len(all_containers)

            # Get currently rented containers for this executor
            rented_containers = self._get_rented_containers(rented_data, executor_uuid)
            extra["rented_containers"] = sorted(rented_containers)

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

        return len(removed_names), removed_names

    async def _get_all_rental_containers(self, ssh_client) -> list[str]:
        """Get all containers with rental-related prefixes."""
        try:
            # Get both pod_ and container_ prefixed containers in one command
            result = await ssh_client.run(
                DockerCommand.ps_filter(f"{POD_CONTAINER_PREFIX}*", "container_*")
            )
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
        if not rented_data or executor_uuid not in rented_data.executors:
            return set()

        executor = rented_data.executors[executor_uuid]
        return {pod.container_name for pod in executor.pods}

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
            # Remove container
            result = await ssh_client.run(DockerCommand.remove(container_name))
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
