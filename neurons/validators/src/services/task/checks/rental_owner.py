import logging

from core.utils import _m, get_extra_info
from protocol.vc_protocol.compute_requests import RentedExecutor

from ..pipeline import Context

logger = logging.getLogger(__name__)


def rental_of_validated_miner(
    ctx: Context, rented_executor: RentedExecutor | None, log_mismatch: bool = False
) -> RentedExecutor | None:
    """The rental listed under this executor's UUID, or None when it belongs to another miner on a
    different endpoint. On the same IP and port the renter's pod is on this box, so it stays rented
    and the VerifyX/matmul probes never run beside it."""
    if not rented_executor or rented_executor.miner_hotkey == ctx.miner_hotkey:
        return rented_executor
    # once per cycle, from TenantEnforcementCheck; every other check reads the same answer
    if log_mismatch:
        logger.warning(
            _m(
                "RENTED_OWNER_MISMATCH",
                extra=get_extra_info(
                    {
                        **ctx.default_extra,
                        "executor_uuid": ctx.executor.uuid,
                        "rental_miner_hotkey": rented_executor.miner_hotkey,
                    }
                ),
            )
        )
    same_endpoint = (rented_executor.executor_ip_address, str(rented_executor.executor_ip_port)) == (
        ctx.executor.address,
        str(ctx.executor.port),
    )
    return rented_executor if same_endpoint else None


def rental_of_this_node(ctx: Context, log_mismatch: bool = False) -> RentedExecutor | None:
    """This node's rental from the cycle snapshot, read through rental_of_validated_miner: the
    executor UUID is the miner's word, so a UUID match alone does not make the rental this node's."""
    rented_data = ctx.state.rented_data
    return rental_of_validated_miner(
        ctx, rented_data.executors.get(ctx.executor.uuid) if rented_data else None, log_mismatch
    )
