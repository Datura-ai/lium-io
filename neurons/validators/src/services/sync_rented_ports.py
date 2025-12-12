"""
Service for synchronizing rented ports state with backend.

When validator receives RentedMachineResponse from backend, this service
compares local port_mappings DB state with backend's list of rented pods
and releases ports for pods that are no longer rented on backend.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from protocol.vc_protocol.compute_requests import RentedMachineResponse

if TYPE_CHECKING:
    from daos.port_mapping_dao import PortMappingDao
    from services.redis_service import RedisService

logger = logging.getLogger(__name__)


@dataclass
class SyncRentedPortsResult:
    """Result of sync_rented_ports operation."""

    released_port_count: int
    stale_pod_ids: set[UUID]
    skipped_pod_ids: set[UUID]  # Pods skipped due to renting_in_progress
    unknown_pod_ids: set[UUID]


async def sync_rented_ports(
    response: RentedMachineResponse,
    port_mapping_dao: "PortMappingDao",
    redis_service: "RedisService",
    threshold_minutes: int = 30,
) -> SyncRentedPortsResult | None:
    """
    Synchronize local port rental state with backend.

    Compares pod_ids from backend response with local DB and:
    1. Releases ports for pods that exist locally but not on backend (stale)
    2. Logs warning for pods that exist on backend but not locally (unknown)
    """
    try:
        # 1. Extract pod_ids from backend response
        backend_pod_ids: set[UUID] = set()
        for machine in response.machines:
            for container in machine.containers:
                try:
                    backend_pod_ids.add(UUID(container.pod_id))
                except ValueError:
                    logger.warning(f"sync_rented_ports: invalid pod_id format: {container.pod_id}")

        # 2. Get pod_ids from local DB (older than threshold to avoid race conditions)
        local_pod_ids = await port_mapping_dao.get_rented_pod_ids_older_than(threshold_minutes)

        # 3. Find stale pods (in local DB but not on backend)
        stale_pod_ids = local_pod_ids - backend_pod_ids

        # 4. Find unknown pods (on backend but not in local DB)
        unknown_pod_ids = backend_pod_ids - local_pod_ids

        # 5. Release ports for stale pods (skip if renting in progress)
        released_port_count = 0
        skipped_pod_ids: set[UUID] = set()
        for pod_id in stale_pod_ids:
            # Check if renting is in progress for this pod
            executor_info = await port_mapping_dao.get_executor_info_for_pod(pod_id)
            if executor_info:
                miner_hotkey, executor_id = executor_info
                if await redis_service.renting_in_progress(miner_hotkey, executor_id, str(pod_id)):
                    logger.info(f"sync_rented_ports: skipping pod {pod_id} - renting in progress")
                    skipped_pod_ids.add(pod_id)
                    continue

            released = await port_mapping_dao.release_ports_for_pod(pod_id)
            released_port_count += released

        # Remove skipped pods from stale_pod_ids for accurate reporting
        stale_pod_ids = stale_pod_ids - skipped_pod_ids

        # 6. Log results
        if stale_pod_ids:
            logger.warning(
                f"sync_rented_ports: released {released_port_count} ports for "
                f"{len(stale_pod_ids)} stale pods: {sorted(str(p) for p in stale_pod_ids)}"
            )

        if unknown_pod_ids:
            logger.warning(
                f"sync_rented_ports: found {len(unknown_pod_ids)} unknown pods in backend "
                f"(not in local DB): {sorted(str(p) for p in unknown_pod_ids)}"
            )

        return SyncRentedPortsResult(
            released_port_count=released_port_count,
            stale_pod_ids=stale_pod_ids,
            skipped_pod_ids=skipped_pod_ids,
            unknown_pod_ids=unknown_pod_ids,
        )

    except Exception as e:
        logger.error(f"sync_rented_ports: failed to sync - {e}", exc_info=True)
        return None
