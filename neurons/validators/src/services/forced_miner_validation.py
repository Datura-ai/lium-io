"""DAH-2090: validate one miner on demand, instead of waiting for the next block window.

A staging development tool. The scheduled cycle waits on every registered miner and most of
them never answer, each costing a 30 second timeout, so it takes minutes; one miner returns in
seconds.

This lives beside MinerService rather than inside it: the run needs a backend client and the
file encryptor, and widening the shared constructor would reach into the validator process and
every test that builds a MinerService, for a tool only staging uses.
"""

import asyncio
import logging

from pydantic import BaseModel

from clients.backend_client import BackendClient
from clients.subtensor_client import SubtensorClient
from core.config import settings
from core.utils import _m, get_extra_info
from payload_models.payloads import MinerJobEnryptedFiles, MinerJobRequestPayload
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.default_docker_image_digest_service import fetch_default_image_digests
from services.file_encrypt_service import FileEncryptService
from services.miner_service import MinerService
from services.task_service import JobResult

logger = logging.getLogger(__name__)

# ecrypt_miner_job_files wipes and rebuilds one fixed temp directory, so two runs at once would
# delete each other's binaries.
_one_run_at_a_time = asyncio.Lock()


class ValidationInputs(BaseModel):
    """Everything one run needs, gathered before the miner is contacted."""

    payload: MinerJobRequestPayload
    encrypted_files: MinerJobEnryptedFiles
    rented_executors: RentedExecutorsResponse
    default_docker_image_digests: dict[str, str]


async def validate_one_miner_now(
    miner_service: MinerService,
    backend_client: BackendClient,
    file_encrypt_service: FileEncryptService,
    miner_hotkey: str,
) -> None:
    """Validate every executor of one miner now, and publish the specs.

    The same work the scheduled cycle does for this miner: the same checks, the same scoring
    inputs, the same machine-spec publishing. Staging only -- the caller gates on the
    environment.
    """
    async with _one_run_at_a_time:
        inputs = await _gather_inputs(
            backend_client, file_encrypt_service, miner_hotkey
        )
        if inputs is None:
            return
        job = await miner_service.request_job_to_miner(
            payload=inputs.payload,
            encrypted_files=inputs.encrypted_files,
            rented_data=inputs.rented_executors,
            default_docker_image_digests=inputs.default_docker_image_digests,
        )

    results: list[JobResult] = job["results"]
    logger.info(
        _m(
            "Forced miner validation finished",
            extra=get_extra_info({
                "miner_hotkey": miner_hotkey,
                "job_batch_id": inputs.payload.job_batch_id,
                "results": len(results),
            }),
        ),
    )
    await miner_service.publish_machine_specs(
        results, miner_hotkey, inputs.payload.miner_coldkey
    )


async def _gather_inputs(
    backend_client: BackendClient,
    file_encrypt_service: FileEncryptService,
    miner_hotkey: str,
) -> ValidationInputs | None:
    """Collect what the run needs, or None when the run must not start.

    The four calls are independent. The block read and the PyInstaller build inside
    ecrypt_miner_job_files block the loop, and the connector process also serves container
    create and delete traffic, so those two go to threads.
    """
    subtensor_client = SubtensorClient.get_instance()
    try:
        miner = await subtensor_client.get_miner(miner_hotkey)
    except Exception as exc:
        logger.error(
            _m(
                "Forced miner validation stopped: miner is not in the metagraph",
                extra=get_extra_info({"miner_hotkey": miner_hotkey, "error": str(exc)}),
            ),
        )
        return None

    (
        job_batch_id,
        rented_executors,
        encrypted_files,
        default_docker_image_digests,
    ) = await asyncio.gather(
        _current_cycle_batch_id(subtensor_client),
        backend_client.get_all_rented_executors(),
        asyncio.to_thread(file_encrypt_service.ecrypt_miner_job_files),
        _default_docker_image_digests_or_empty(miner_hotkey),
    )

    # The scheduled cycle skips its whole iteration when this fetch fails (core/validator.py),
    # because an empty rental map reads every rented machine as free: the run would then
    # disturb a customer's box and score it by the wrong rule set.
    if rented_executors is None:
        logger.error(
            _m(
                "Forced miner validation stopped: could not fetch rented executors",
                extra=get_extra_info({"miner_hotkey": miner_hotkey}),
            ),
        )
        return None

    return ValidationInputs(
        payload=MinerJobRequestPayload(
            job_batch_id=job_batch_id,
            miner_hotkey=miner_hotkey,
            miner_coldkey=miner.coldkey,
            miner_address=miner.axon_info.ip,
            miner_port=miner.axon_info.port,
        ),
        encrypted_files=encrypted_files,
        rented_executors=rented_executors,
        default_docker_image_digests=default_docker_image_digests,
    )


async def _current_cycle_batch_id(subtensor_client: SubtensorClient) -> str:
    """The batch id of the cycle running now, in the shape the scheduled cycle sends.

    The backend parses this field as a "%Y-%m-%d %H:%M:%S" cycle timestamp, so a forced run
    cannot invent its own id -- it would fail the spec ingestion.
    """
    current_block: int = await asyncio.to_thread(subtensor_client.get_current_block)
    cycle_block: int = (current_block // settings.BLOCKS_FOR_JOB) * settings.BLOCKS_FOR_JOB
    return await subtensor_client.get_time_from_block(cycle_block)


async def _default_docker_image_digests_or_empty(miner_hotkey: str) -> dict[str, str]:
    """The digest snapshot the scheduled cycle validates against, empty when Docker Hub fails.

    Fail-open like the cycle: an empty snapshot skips the digest check. Without the fetch a
    forced run would score the miner by a weaker rule set than the cycle does.
    """
    try:
        return await fetch_default_image_digests()
    except Exception as exc:
        logger.error(
            _m(
                "Forced miner validation could not fetch image digests; digest checks skip",
                extra=get_extra_info({"miner_hotkey": miner_hotkey, "error": str(exc)}),
            ),
        )
        return {}
