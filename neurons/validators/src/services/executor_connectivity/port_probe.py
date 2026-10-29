import logging

from core.utils import _m, get_extra_info
from services.const import MIN_PORT_COUNT
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
        min_ports: int = MIN_PORT_COUNT,
    ) -> PortProbeResult:
        """`min_ports`: the verified ports PortCountCheck needs to list the node (more for a split node)."""
        log_ctx = log_ctx or {}
        batch = await self.batch_verifier.verify(
            ports,
            ssh_client=ssh_client,
            host=host,
            log_ctx=log_ctx,
            min_ports=min_ports,
        )
        if batch.external_blocked:
            # the other tiers probe from the validator too, so they would pass the same host
            return PortProbeResult((), tuple(ports), "external_blocked")
        successful, failed = batch.successful, batch.failed
        tier = "batch"

        if not successful:
            # DAH-3593: tier changes are DEBUG; the verdict line carries the outcome.
            logger.debug(
                _m("batch verification failed, trying semi-batch", extra=get_extra_info(log_ctx))
            )
            successful, failed = await self.semi_batch_verifier.verify(
                ports,
                ssh_client=ssh_client,
                host=host,
                max_ports=max(50, min_ports),
                log_ctx=log_ctx,
            )
            tier = "semi_batch"

        if not successful:
            logger.debug(
                _m("semi-batch verification failed, trying fallback", extra=get_extra_info(log_ctx))
            )
            successful, failed = await self.fallback_verifier.verify(
                ports,
                ssh_client=ssh_client,
                host=host,
                max_ports=max(10, min_ports),
                log_ctx=log_ctx,
            )
            tier = "fallback"

        return PortProbeResult(tuple(successful), tuple(failed), tier)
