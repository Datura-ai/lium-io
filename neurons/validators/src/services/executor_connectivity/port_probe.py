import logging

from core.utils import _m, get_extra_info
from services.executor_connectivity.models import PortPair, PortProbeResult
from services.executor_connectivity.port_verifiers import BatchVerifier, FallbackVerifier

logger = logging.getLogger(__name__)


class PortProbe:
    """Runs batch verification with fallback."""

    def __init__(
        self,
        batch_verifier: BatchVerifier,
        fallback_verifier: FallbackVerifier,
    ):
        self.batch_verifier = batch_verifier
        self.fallback_verifier = fallback_verifier

    async def probe(
        self,
        ports: list[PortPair],
        *,
        ssh_client,
        host: str,
        log_ctx: dict | None = None,
    ) -> PortProbeResult:
        log_ctx = log_ctx or {}
        successful, failed = await self.batch_verifier.verify(
            ports,
            ssh_client=ssh_client,
            host=host,
            log_ctx=log_ctx,
        )

        if not successful:
            logger.warning(
                _m("batch verification failed, trying fallback", extra=get_extra_info(log_ctx))
            )
            successful, failed = await self.fallback_verifier.verify(
                ports,
                ssh_client=ssh_client,
                host=host,
                max_ports=10,
                log_ctx=log_ctx,
            )

        return PortProbeResult(tuple(successful), tuple(failed))
