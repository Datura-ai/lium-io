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

        def short(ok) -> bool:
            # a split node (min_ports above the floor) keeps probing until it has its ports or every selected port
            # answered; the tier that verified most wins. Otherwise a later tier runs only when the earlier one
            # verified nothing.
            return not ok or (min_ports > MIN_PORT_COUNT and len(ok) < min(min_ports, len(ports)))

        if batch.external == "blocked" and not short(batch.successful):
            # the other tiers probe from the validator too, so they would pass the same host. The validator's
            # ports are kept: PortConnectivityCheck decides whether this reading fails the node. A short split
            # node still runs the later tiers; the result keeps the "blocked" reading.
            return PortProbeResult(tuple(batch.successful), tuple(batch.failed), "batch", external="blocked")
        successful, failed = list(batch.successful), list(batch.failed)
        tier, best = "batch", len(successful)
        if batch.start_hung:
            # DAH-5682: both batch container starts hung, so the later tiers' containers start on the same stuck
            # Docker; ~90 s (semi-batch 60 s, fallback 30 s) for the same verdict. The DinD probe still runs.
            return PortProbeResult((), tuple(failed), tier, batch.external)

        def merge(ok, bad, name) -> None:
            # a port any tier verified counts: tiers can verify different ports of the same node
            nonlocal successful, failed, tier, best
            successful = list(dict.fromkeys([*successful, *ok]))
            failed = [p for p in dict.fromkeys([*failed, *bad]) if p not in set(successful)]
            if len(ok) >= best:
                tier, best = name, len(ok)

        if short(successful):
            # DAH-3593: tier changes are DEBUG; the verdict line carries the outcome.
            logger.debug(
                _m("batch verification short, trying semi-batch", extra=get_extra_info(log_ctx))
            )
            semi_ok, semi_failed = await self.semi_batch_verifier.verify(
                ports,
                ssh_client=ssh_client,
                host=host,
                max_ports=max(50, min_ports),
                log_ctx=log_ctx,
            )
            merge(semi_ok, semi_failed, "semi_batch")

        if short(successful):
            logger.debug(
                _m("semi-batch verification short, trying fallback", extra=get_extra_info(log_ctx))
            )
            fb_ok, fb_failed = await self.fallback_verifier.verify(
                ports,
                ssh_client=ssh_client,
                host=host,
                max_ports=max(10, min_ports),
                log_ctx=log_ctx,
            )
            merge(fb_ok, fb_failed, "fallback")

        return PortProbeResult(tuple(successful), tuple(failed), tier, batch.external)
