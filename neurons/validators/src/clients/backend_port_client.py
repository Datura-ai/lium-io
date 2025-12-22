"""HTTP client for fetching rented ports from compute backend API."""

import logging
from uuid import UUID

from pydantic import BaseModel

from clients.backend_client import BackendClient
from core.utils import _m

logger = logging.getLogger(__name__)


class RentedPortsResponse(BaseModel):
    rented_external_ports: list[int]


class BackendPortClient:
    def __init__(self, backend_client: BackendClient):
        self.backend_client = backend_client

    async def get_rented_ports(
        self, executor_id: str | UUID, context: dict | None = None
    ) -> set[int]:
        executor_id_str = str(executor_id)
        path = f"/internal/executors/{executor_id_str}/rented-ports"

        response = await self.backend_client.get(path, RentedPortsResponse)
        if response is None:
            logger.warning(_m("Failed to fetch rented ports from backend", extra=context or {}))
            return set()

        ports = set(response.rented_external_ports)
        logger.info(_m(f"Fetched {len(ports)} rented ports", extra=context or {}))
        return ports
