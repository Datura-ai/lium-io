import logging
from uuid import UUID

from daos.port_mapping_dao import PortMappingDao
from models.port_mapping import PortMapping
from services.executor_connectivity.models import PortVerificationResult

logger = logging.getLogger(__name__)


class PortResultPersister:
    """Persists port verification results."""

    def __init__(self, port_mapping_dao: PortMappingDao):
        self.dao = port_mapping_dao

    async def save(
        self,
        result: PortVerificationResult,
        executor_id: str,
        miner_hotkey: str,
    ) -> None:
        try:
            successful = list(result.successful_ports)
            failed = list(result.failed_ports)
            records = [
                PortMapping(
                    miner_hotkey=miner_hotkey,
                    executor_id=UUID(executor_id),
                    internal_port=p.internal,
                    external_port=p.external,
                    is_successful=True,
                )
                for p in successful
            ]
            records.extend([
                PortMapping(
                    miner_hotkey=miner_hotkey,
                    executor_id=UUID(executor_id),
                    internal_port=p.internal,
                    external_port=p.external,
                    is_successful=False,
                )
                for p in failed
            ])

            if records:
                await self.dao.upsert_port_results(records)
                cleaned = await self.dao.clean_ports(records[0].executor_id, 24 * 60)
                logger.info("saved %s ports to db, cleaned=%s", len(records), cleaned)
        except Exception as e:
            logger.error("save to db failed: %s", e, exc_info=True)
