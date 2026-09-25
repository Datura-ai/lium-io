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
    ) -> PortProbeResult:
        log_ctx = log_ctx or {}
        successful, failed = await self.batch_verifier.verify(
            ports,
            ssh_client=ssh_client,
            host=host,
            log_ctx=log_ctx,
        )
        tier = "batch"

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
            tier = "semi_batch"

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
            tier = "fallback"

        return PortProbeResult(tuple(successful), tuple(failed), tier)

    async def top_up(
        self,
        successful: list[PortPair],
        failed: list[PortPair],
        *,
        ssh_client,
        host: str,
        log_ctx: dict | None = None,
    ) -> PortProbeResult:
        """Re-probe `failed` through the published-port (-p) tiers until MIN_PORT_COUNT answer.

        A host-network batch can reach only a few listeners (a ufw INPUT policy drops them, or a
        probe lands before its nc has bound) while published ports, the path a renter's pod uses,
        would answer; below the floor that partial count hides the node, so the -p tiers get a turn.
        """
        log_ctx = log_ctx or {}
        logger.warning(
            _m(
                f"batch verified {len(successful)}/{len(successful) + len(failed)}, below the floor of "
                f"{MIN_PORT_COUNT}; re-probing the failed ports through published ports",
                extra=get_extra_info(log_ctx),
            )
        )
        tiers = ["batch"]
        for name, verifier, max_ports in (
            ("semi_batch", self.semi_batch_verifier, 50),
            ("fallback", self.fallback_verifier, 10),
        ):
            if len(successful) >= MIN_PORT_COUNT or not failed:
                break
            recovered, _ = await verifier.verify(
                failed,
                ssh_client=ssh_client,
                host=host,
                max_ports=max_ports,
                log_ctx=log_ctx,
            )
            if recovered:
                tiers.append(name)
                recovered_set = set(recovered)
                successful = successful + recovered
                failed = [p for p in failed if p not in recovered_set]
        return PortProbeResult(tuple(successful), tuple(failed), "+".join(tiers))
