import logging
import time

from core.config import settings
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
        if batch.external == "blocked" and batch.successful:
            # the other tiers probe from the validator too, so they would pass the same host. The validator's
            # ports are kept: PortConnectivityCheck decides whether this reading fails the node.
            return PortProbeResult(tuple(batch.successful), tuple(batch.failed), "batch", external="blocked")
        successful, failed = batch.successful, batch.failed
        tier = "batch"
        if batch.start_hung and settings.PORT_PROBE_SKIP_TIERS_ON_HUNG_START:
            # DAH-5682: their containers start on the same stuck Docker; ~90 s (semi-batch 60 s, fallback 30 s) for the
            # same verdict
            return PortProbeResult((), tuple(failed), tier, batch.external)

        later_tiers_started_at = time.monotonic()
        if not successful:
            # DAH-3593: tier changes are DEBUG; the verdict line carries the outcome.
            logger.debug(
                _m("batch verification failed, trying semi-batch", extra=get_extra_info(log_ctx))
            )
            successful, failed = await self.semi_batch_verifier.verify(
                ports,
                ssh_client=ssh_client,
                host=host,
                max_ports=50,
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
                max_ports=10,
                log_ctx=log_ctx,
            )
            tier = "fallback"

        if batch.start_hung:
            logger.info(
                _m(
                    "port check would skip later tiers",
                    extra=get_extra_info(
                        {
                            **log_ctx,
                            "later_tiers_found_ports": bool(successful),
                            "later_tiers_port_count": len(successful),
                            "later_tiers_seconds": round(time.monotonic() - later_tiers_started_at, 1),
                            "tier": tier,
                        }
                    ),
                )
            )
        return PortProbeResult(tuple(successful), tuple(failed), tier, batch.external)
