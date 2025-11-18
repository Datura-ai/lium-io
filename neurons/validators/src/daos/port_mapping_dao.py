import logging
import time
from asyncio import Semaphore
from uuid import UUID
from core.db import POOL_SIZE

from sqlalchemy import select, update

from daos.base import BaseDao
from models.port_mapping import PortMapping

logger = logging.getLogger(__name__)
upsert_semaphore = Semaphore(POOL_SIZE)


class PortMappingDao(BaseDao):
    """DAO for port mapping operations with per-operation sessions."""

    async def upsert_port_results(self, port_results: list[PortMapping]) -> None:
        """Batch upsert port verification results for single executor."""
        if not port_results:
            return

        async with upsert_semaphore:
            # All ports should be from same executor
            executor_id = port_results[0].executor_id
            async with self.get_session() as session:
                try:
                    # Process in chunks of 1000 for memory efficiency
                    chunk_size = 1000

                    for i in range(0, len(port_results), chunk_size):
                        chunk = port_results[i : i + chunk_size]
                        ports_dict = {p.external_port: p for p in chunk}
                        stmt = select(PortMapping.uuid, PortMapping.external_port).where(
                            PortMapping.executor_id == executor_id,
                            PortMapping.external_port.in_(list(ports_dict.keys())),
                        )
                        existing_result = await session.exec(stmt)
                        existing_ports = {port: uuid for uuid, port in existing_result.all()}

                        new_ports = []
                        updates = []
                        for port_num, new_port in ports_dict.items():
                            if port_num in existing_ports:
                                # Prepare bulk update
                                updates.append({
                                    'uuid': existing_ports[port_num],
                                    'verification_time': new_port.verification_time,
                                    'is_successful': new_port.is_successful,
                                    'miner_hotkey': new_port.miner_hotkey,
                                })
                            else:
                                # Add new
                                new_ports.append(new_port)

                        # Bulk update existing ports for this chunk
                        if updates:
                            stmt = update(PortMapping)
                            await session.execute(stmt, updates)

                        # Bulk insert new ports for this chunk
                        if new_ports:
                            session.add_all(new_ports)
                        await session.commit()

                except Exception as e:
                    logger.error(
                        f"Error upserting {len(port_results)} port results: {e}", exc_info=True
                    )
                    raise

    async def clean_ports(self, executor_id: UUID, period_minutes: int = 120) -> int:
        """delete ports older than period_minutes from DB"""
        async with self.get_session() as session:
            try:
                from sqlalchemy import delete, text

                # Bulk DELETE operation
                stmt = delete(PortMapping).where(
                    PortMapping.executor_id == executor_id,
                    PortMapping.rented_for_pod_id==None,
                    PortMapping.verification_time
                    < text(f"now() - interval '{period_minutes} minutes'"),
                )
                result = await session.exec(stmt)
                deleted_count = result.rowcount
                return deleted_count
            except Exception as e:
                logger.error(f"Error cleaning ports: {e}", exc_info=True)
                return 0

    async def get_successful_ports(self, executor_id: UUID, limit: int | None = None) -> dict[int, PortMapping]:
        """Get successful ports as dictionary {external_port: PortMapping} for fast lookup."""
        async with self.get_session() as session:
            try:
                stmt = (
                    select(PortMapping)
                    .where(PortMapping.executor_id == executor_id, PortMapping.is_successful)
                    .order_by(PortMapping.verification_time.desc())
                )
                if limit is not None:
                    stmt = stmt.limit(limit)
                result = await session.exec(stmt)
                ports = result.scalars().all()
                return {port.external_port: port for port in ports}
            except Exception as e:
                logger.error(f"Error getting successful ports as dict: {e}", exc_info=True)
                return {}

    async def get_successful_ports_count(self, executor_id: UUID | str) -> int:
        """Get count of successful ports for executor."""
        async with self.get_session() as session:
            try:
                from sqlalchemy import func

                stmt = select(func.count(PortMapping.uuid)).where(
                    PortMapping.executor_id == executor_id,
                    PortMapping.is_successful,
                    PortMapping.rented_for_pod_id.is_(None),
                )
                result = await session.exec(stmt)
                return result.scalar() or 0
            except Exception as e:
                logger.error(f"Error counting successful ports: {e}", exc_info=True)
                return 0

    async def get_ports_for_pod(self, pod_id: UUID) -> dict[int, PortMapping]:
        """
        Get all ports reserved for a specific pod as dictionary {docker_port: PortMapping}.

        For backward compatibility with old records, if docker_port is None, falls back
        to using external_port as the key.
        """
        async with self.get_session() as session:
            try:
                stmt = select(PortMapping).where(PortMapping.rented_for_pod_id == pod_id)
                result = await session.exec(stmt)
                ports = result.scalars().all()
                port_dict = {}
                for port in ports:
                    # Use docker_port as key if available, otherwise fallback to external_port
                    key = port.docker_port if port.docker_port is not None else port.external_port
                    port_dict[key] = port
                    if port.docker_port is None:
                        logger.debug(
                            f"Port {port.external_port} for pod {pod_id} has no docker_port, "
                            f"using external_port as fallback"
                        )

                if port_dict:
                    docker_ports = sorted([k for k in port_dict.keys()])
                    external_ports = sorted([p.external_port for p in port_dict.values()])
                    logger.info(
                        f"Found {len(port_dict)} ports for pod {pod_id}: "
                        f"docker_ports={docker_ports}, external={external_ports}"
                    )

                return port_dict
            except Exception as e:
                logger.error(f"Error getting ports for pod {pod_id}: {e}", exc_info=True)
                return {}

    async def reserve_ports_for_pod(
        self, executor_id: UUID, mappings: list[tuple[int, int, int]], pod_id: UUID
    ) -> None:
        """
        Reserve ports for a specific pod on an executor.
        - Clears pod_id and docker_port from all ports that were previously rented by this pod
          but are not in the new mappings
        - Sets pod_id and docker_port for all ports in the mappings

        :param executor_id: Executor UUID to operate on
        :param mappings: List of (docker_port, internal_port, external_port) tuples to reserve
        :param pod_id: Pod UUID to reserve ports for
        """
        async with self.get_session() as session:
            try:
                external_ports = [m[2] for m in mappings] if mappings else []

                # Step 1: Clear pod_id and docker_port from ports that were rented by this pod
                # but are not in external_ports
                release_stmt = (
                    update(PortMapping)
                    .where(
                        PortMapping.executor_id == executor_id,
                        PortMapping.rented_for_pod_id == pod_id,
                        PortMapping.external_port.notin_(external_ports) if external_ports else True,
                    )
                    .values(rented_for_pod_id=None, docker_port=None)
                )
                release_result = await session.exec(release_stmt)
                released_count = release_result.rowcount

                # Step 2: Set pod_id and docker_port for each mapping
                for docker_port, internal_port, external_port in mappings:
                    reserve_stmt = (
                        update(PortMapping)
                        .where(
                            PortMapping.executor_id == executor_id,
                            PortMapping.external_port == external_port,
                        )
                        .values(rented_for_pod_id=pod_id, docker_port=docker_port)
                    )
                    await session.exec(reserve_stmt)

                await session.commit()

                # Log detailed port mappings
                port_mappings_str = ", ".join([f"{m[0]}->{m[2]}" for m in mappings])
                logger.info(
                    f"Reserved {len(mappings)} ports for pod {pod_id} on executor {executor_id} "
                    f"(released {released_count} old ports): [{port_mappings_str}]"
                )
            except Exception as e:
                logger.error(
                    f"Error reserving ports for pod {pod_id} on executor {executor_id}: {e}",
                    exc_info=True,
                )
                raise

    async def release_ports_for_pod(self, pod_id: UUID) -> int:
        """Release all ports for a specific pod by clearing rented_for_pod_id and docker_port."""
        async with self.get_session() as session:
            try:
                stmt = (
                    update(PortMapping)
                    .where(PortMapping.rented_for_pod_id == pod_id)
                    .values(rented_for_pod_id=None, docker_port=None)
                )
                result = await session.exec(stmt)
                await session.commit()
                released_count = result.rowcount
                logger.info(f"Released {released_count} ports for pod {pod_id}")
                return released_count
            except Exception as e:
                logger.error(f"Error releasing ports for pod {pod_id}: {e}", exc_info=True)
                return 0

    async def get_available_ports_excluding_rented(
        self, executor_id: UUID, limit: int | None = None
    ) -> dict[int, PortMapping]:
        """Get successful ports that are not rented (rented_for_pod_id IS NULL) as dictionary."""
        async with self.get_session() as session:
            try:
                stmt = (
                    select(PortMapping)
                    .where(
                        PortMapping.executor_id == executor_id,
                        PortMapping.is_successful,
                        PortMapping.rented_for_pod_id.is_(None),
                    )
                    .order_by(PortMapping.verification_time.desc())
                )
                if limit is not None:
                    stmt = stmt.limit(limit)
                result = await session.exec(stmt)
                ports = result.scalars().all()
                return {port.external_port: port for port in ports}
            except Exception as e:
                logger.error(
                    f"Error getting available ports excluding rented: {e}", exc_info=True
                )
                return {}

    async def get_busy_external_ports(self) -> set[int]:
        """Get set of external ports that are currently rented (rented_for_pod_id IS NOT NULL)."""
        async with self.get_session() as session:
            try:
                # Get both ports and pod_ids for logging
                stmt = select(PortMapping.external_port, PortMapping.rented_for_pod_id).where(
                    PortMapping.rented_for_pod_id.isnot(None)
                )
                result = await session.exec(stmt)
                rows = result.all()
                
                if not rows:
                    return set()

                ports = set(row[0] for row in rows)
                pod_ids = set(str(row[1]) for row in rows)

                if ports:
                    logger.info(
                        f"Found {len(ports)} busy external ports for {len(pod_ids)} pods: {sorted(pod_ids)}"
                    )

                return ports
            except Exception as e:
                logger.error(f"Error getting busy external ports: {e}", exc_info=True)
                return set()
