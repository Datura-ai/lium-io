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
                    PortMapping.executor_id == executor_id, PortMapping.is_successful
                )
                result = await session.exec(stmt)
                return result.scalar() or 0
            except Exception as e:
                logger.error(f"Error counting successful ports: {e}", exc_info=True)
                return 0
