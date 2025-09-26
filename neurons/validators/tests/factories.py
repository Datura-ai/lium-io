from datetime import datetime, timedelta
from uuid import UUID, uuid4

from models.port_mapping import PortMapping


def create_port_mapping(
    miner_hotkey: str = "test_miner_hotkey",
    executor_id: UUID | None = None,
    internal_port: int = 9000,
    external_port: int = 9000,
    is_successful: bool = True,
    verification_time: datetime | None = None,
) -> PortMapping:
    """Factory function to create a PortMapping instance with default values."""
    if executor_id is None:
        executor_id = uuid4()

    if verification_time is None:
        verification_time = datetime.utcnow()

    return PortMapping(
        miner_hotkey=miner_hotkey,
        executor_id=executor_id,
        internal_port=internal_port,
        external_port=external_port,
        is_successful=is_successful,
        verification_time=verification_time,
    )


def create_port_mappings_batch(
    count: int,
    miner_hotkey: str = "test_miner_hotkey",
    executor_id: UUID | None = None,
    is_successful: bool = True,
    base_port: int = 9000,
    time_offset_minutes: int = 0,
) -> list[PortMapping]:
    """Create a batch of PortMapping instances."""
    if executor_id is None:
        executor_id = uuid4()

    mappings = []
    for i in range(count):
        port = base_port + i
        verification_time = datetime.utcnow() - timedelta(minutes=time_offset_minutes + i)

        mappings.append(
            create_port_mapping(
                miner_hotkey=miner_hotkey,
                executor_id=executor_id,
                internal_port=port,
                external_port=port,
                is_successful=is_successful,
                verification_time=verification_time,
            )
        )

    return mappings
