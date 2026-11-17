"""I-78: which SSH-key uploads carry a dated nonce, shared by the validator's own uploads and the backend health check's."""

import logging

from clients.subtensor_client import SubtensorClient
from datura.requests.validator_requests import fresh_upload_nonce

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

# bittensor packs a.b.c into axon_info.version as a*1000**2 + b*1000 + c. 10.5.0 is what
# miner-v1.003+ announces; the upload nonce field shipped in miner-v1.002 (fbfbde79, DAH-2338).
MIN_DATED_NONCE_MINER_AXON_VERSION = 10_005_000


def upload_nonce_for_miner(miner_hotkey: str) -> str | None:
    """A dated nonce for an SSH-key upload, so a captured signature goes stale (I-78).

    Only for a miner proven to relay the nonce field: the central miner, or an axon announcing
    bittensor 10.5.0+. Older miners drop the field and the upload fails, so they get none.
    """
    subtensor = SubtensorClient.get_instance()
    if miner_hotkey in subtensor.hotkeys_routed_via_central_miner:
        return fresh_upload_nonce()
    # in-memory metagraph snapshot only: no chain read on the upload path
    neuron = next((n for n in subtensor.miners if n.hotkey == miner_hotkey), None)
    axon_version = getattr(neuron.axon_info, "version", None) if neuron is not None else None
    if axon_version and axon_version >= MIN_DATED_NONCE_MINER_AXON_VERSION:
        return fresh_upload_nonce()
    logger.info(
        _m(
            "dated_upload_nonce_skipped",
            extra=get_extra_info({"miner_hotkey": miner_hotkey, "axon_version": axon_version}),
        )
    )
    return None
