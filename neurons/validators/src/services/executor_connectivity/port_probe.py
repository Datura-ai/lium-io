import logging

from core.utils import _m, get_extra_info
from services.executor_connectivity.models import PortPair, PortProbeResult
from services.executor_connectivity.port_verifiers import BatchVerifier, FallbackVerifier, SemiBatchVerifier

logger = logging.getLogger(__name__)


class PortProbe:
    """Runs batch verification with fallback."""

    def __init__(
        self,
        batch_verifier: BatchVerifier,
        semi_batch_verifier: SemiBatchVerifier,
        fallback_verifier: FallbackVerifier,
    ):
        self.batch_verifier = batch_verifier
        self.semi_batch_verifier = semi_batch_verifier
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
        batch = await self.batch_verifier.verify(
            ports,
            ssh_client=ssh_client,
            host=host,
            log_ctx=log_ctx,
        )
        successful, failed = batch.successful, batch.failed

        if not successful:
            logger.warning(
                _m("batch verification failed, trying semi-batch", extra=get_extra_info(log_ctx))
            )
            successful, failed = await self.semi_batch_verifier.verify(
                ports,
                ssh_client=ssh_client,
                host=host,
                max_ports=50,
                log_ctx=log_ctx,
            )

        if not successful:
            logger.warning(
                _m("semi-batch verification failed, trying fallback", extra=get_extra_info(log_ctx))
            )
            successful, failed = await self.fallback_verifier.verify(
                ports,
                ssh_client=ssh_client,
                host=host,
                max_ports=10,
                log_ctx=log_ctx,
            )

        return PortProbeResult(tuple(successful), tuple(failed), batch_completed=batch.completed)

    async def probe_spread(
        self,
        ports: list[PortPair],
        *,
        ssh_client,
        host: str,
        log_ctx: dict | None = None,
    ) -> PortProbeResult:
        """The second pass: the batch tier only, one attempt, so it costs at most one container."""
        batch = await self.batch_verifier.verify(
            ports,
            ssh_client=ssh_client,
            host=host,
            log_ctx={**(log_ctx or {}), "port_pass": 2},
            max_attempts=1,
        )
        return PortProbeResult(tuple(batch.successful), tuple(batch.failed), batch_completed=batch.completed)
