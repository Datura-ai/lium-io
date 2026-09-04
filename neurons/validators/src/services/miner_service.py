import asyncio
import json
import logging
import os
import shlex
import time
from typing import Annotated
from uuid import UUID

import aiohttp
from asyncssh import SSHKey
import asyncssh
import bittensor
from clients.miner_client import MinerClient
from datura.requests.miner_requests import (
    AcceptJobRequest,
    AcceptSSHKeyRequest,
    DeclineJobRequest,
    ExecutorSSHInfo,
    FailedRequest,
    PodLogsResponse,
    RequestType,
    SSHKeyRemoved,
)
from datura.requests.validator_requests import (
    SSHPubKeyRemoveRequest,
    SSHPubKeySubmitRequest,
    GetPodLogsRequest,
    AuthenticationPayload,
    ssh_pubkey_signing_blob,
)
from fastapi import Depends
from clients.validator_portal_api import ValidatorPortalAPI
from payload_models.payloads import (
    BackupContainerRequest,
    CancelStorageOperationRequest,
    RestoreContainerRequest,
    ContainerBaseRequest,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    AddSshPublicKeyRequest,
    RemoveSshPublicKeysRequest,
    FailedContainerErrorCodes,
    FailedContainerErrorTypes,
    FailedContainerRequest,
    MinerJobEnryptedFiles,
    MinerJobRequestPayload,
    GetPodLogsRequestFromServer,
    PodLogsResponseToServer,
    FailedGetPodLogs,
    AddDebugSshKeyRequest,
    DebugSshKeyAdded,
    FailedAddDebugSshKey,
    InstallJupyterServerRequest,
    JupyterServerInstalled,
    JupyterInstallationFailed,
    WorkloadKind,
)
from tenacity import RetryError

from core.config import settings
from core.utils import _m, _StructuredMessage, get_extra_info
from protocol.vc_protocol.compute_requests import RentedExecutorsResponse
from services.attestation_service import AttestationService
from services.docker_service import DockerService, inflight_creates
from services.executor_image_policy import ExpectedImageSnapshot
from services.redis_service import MACHINE_SPEC_CHANNEL, RedisService
from services.roce_link_probe import measure_and_attach
from services.ssh_service import SSHService
from incentive.config import BASE_GPU_MAP
from services.task_service import TaskService, JobResult
from services.storage_operations import cancel_storage_operation, start_storage_operation

logger = logging.getLogger(__name__)


def _miner_job_script_path(script_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "miner_jobs", script_name)


def _nohup_command(argv: list[str], log_path: str) -> str:
    return f"{shlex.join(argv)} > {shlex.quote(log_path)} 2>&1 &"


def _get_error_details(error: Exception) -> str:
    """Extract exception details. For RetryError unwraps the underlying exception."""
    last_attempt = getattr(error, 'last_attempt', None)
    if last_attempt:
        last_exc = last_attempt.exception()
        if last_exc:
            return f"RetryError, {type(last_exc).__name__}: {str(last_exc)}"
    return f"{type(error).__name__}: {str(error)}"


def _storage_repository_spec(
    volume_info,
    password: str | None,
) -> dict[str, object]:
    return {
        "bucket": volume_info.name,
        "access_key_id": volume_info.iam_user_access_key,
        "secret_access_key": volume_info.iam_user_secret_key,
        "session_token": volume_info.session_token,
        "password": password,
    }


def _parse_miner_response(response_data: dict) -> AcceptSSHKeyRequest | FailedRequest | PodLogsResponse:
    """Parse miner REST API response based on message_type field.

    Args:
        response_data: JSON response data from miner

    Returns:
        Parsed response model (AcceptSSHKeyRequest, PodLogsResponse, or FailedRequest)

    Raises:
        ValueError: If message_type is missing or unknown
    """
    message_type_str = response_data.get("message_type")
    if not message_type_str:
        logger.error(
            f"Response missing message_type field. Raw payload: {json.dumps(response_data)}"
        )
        # Treat missing message_type as unexpected response
        return FailedRequest(
            message_type=RequestType.FailedRequest,
            details="Missing message_type in response"
        )

    try:
        message_type = RequestType(message_type_str)
    except ValueError:
        logger.error(
            f"Unknown message_type: {message_type_str}. Raw payload: {json.dumps(response_data)}"
        )
        # Treat unknown message_type as unexpected response
        return FailedRequest(
            message_type=RequestType.FailedRequest,
            details=f"Unknown message_type: {message_type_str}"
        )

    # Dispatch to appropriate model based on message_type
    if message_type == RequestType.AcceptSSHKeyRequest:
        return AcceptSSHKeyRequest.model_validate(response_data)
    elif message_type == RequestType.PodLogsResponse:
        return PodLogsResponse.model_validate(response_data)
    elif message_type in [RequestType.FailedRequest, RequestType.UnAuthorizedRequest, RequestType.SSHKeyRemoved]:
        return FailedRequest.model_validate(response_data)
    elif message_type == RequestType.SSHKeyRemoved:
        return SSHKeyRemoved.model_validate(response_data)
    else:
        # Handle any other unknown message types
        logger.error(
            f"Unexpected message_type: {message_type}. Raw payload: {json.dumps(response_data)}"
        )
        return FailedRequest(
            message_type=RequestType.FailedRequest,
            details=f"Unexpected message_type: {message_type}"
        )


def _bypasses_renting_in_progress(payload: ContainerBaseRequest) -> bool:
    if not isinstance(payload, ContainerDeleteRequest):
        return False
    if payload.workload_kind == WorkloadKind.FILLER:
        return True
    # DAH-2728: the pending-pod flag this guard reads is set by the very create this delete is
    # about to cancel, so declining here would leave that create to finish and orphan its
    # container — the exact failure the delete was sent to prevent.
    return inflight_creates.is_running(payload.pod_id)


JOB_LENGTH = 30

# HTTP timeout constants for REST API calls
REST_SSH_SUBMIT_TIMEOUT = 30  # Timeout for SSH key submission requests
REST_CONTAINER_OP_TIMEOUT = 30  # Timeout for container operations
REST_POD_LOGS_TIMEOUT = 30  # Timeout for pod logs requests
REST_SSH_REMOVE_TIMEOUT = 10  # Timeout for SSH key removal requests

# Emitted instead of a validation result for an executor under a special manual (bare-metal)
# rental. Distinct string so manual passes are greppable in Loki and can never be mistaken for a
# node that actually passed validation -- nothing about this node was verified.
MANUAL_RENTAL_FORCED_PASS_EVENT = "Executor force-passed as special manual rental (not validated)"


class MinerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        task_service: Annotated[TaskService, Depends(TaskService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
        attestation_service: Annotated[AttestationService, Depends(AttestationService)],
    ):
        self.ssh_service = ssh_service
        self.task_service = task_service
        self.redis_service = redis_service
        self.attestation_service = attestation_service

    @staticmethod
    def _normalize_public_key(public_key: bytes | str) -> str:
        return public_key.decode("utf-8") if isinstance(public_key, bytes) else public_key

    def _sign_validator_pubkey(
        self,
        keypair: bittensor.Keypair,
        public_key: bytes | str,
        nonce: str | None = None,
    ) -> str:
        """Sign the canonical SSH-pubkey blob (see datura.ssh_pubkey_signing_blob).

        Without a nonce this is the legacy signature over the bare public key.
        When an attestation nonce rides the request it MUST be passed here so it
        is covered by the signature — otherwise the executor rejects the request.
        """
        pubkey = self._normalize_public_key(public_key)
        return f"0x{keypair.sign(ssh_pubkey_signing_blob(pubkey, nonce)).hex()}"

    async def request_job_to_miner(
        self,
        payload: MinerJobRequestPayload,
        encrypted_files: MinerJobEnryptedFiles,
        rented_data: RentedExecutorsResponse,
        default_docker_image_digests: dict[str, str],
        executor_image_snapshot: ExpectedImageSnapshot | None = None,
    ):
        """Request job to miner - uses REST API if configured, otherwise WebSocket."""
        if settings.USE_REST_API:
            logger.info(
                _m(
                    "Routing request_job_to_miner to REST API",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                    }),
                ),
            )
            return await self._request_job_to_miner(
                payload,
                encrypted_files,
                rented_data,
                default_docker_image_digests,
                executor_image_snapshot,
            )
        else:
            logger.info(
                _m(
                    "Routing request_job_to_miner to WebSocket",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                    }),
                ),
            )
        
        loop = asyncio.get_event_loop()
        # DAH-2667: what the RoCE probe subtracts from the cycle's own wait, so a job that already
        # spent most of it takes a shorter sweep instead of overrunning and scoring the miner zero.
        job_started_at: float = time.monotonic()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "job_batch_id": payload.job_batch_id,
            "miner_hotkey": payload.miner_hotkey,
            "miner_address": payload.miner_address,
            "miner_port": payload.miner_port,
        }

        try:
            logger.info(_m("Requesting job to miner", extra=get_extra_info(default_extra)))
            miner_client = MinerClient(
                loop=loop,
                miner_address=payload.miner_address,
                miner_port=payload.miner_port,
                miner_hotkey=payload.miner_hotkey,
                my_hotkey=my_key.ss58_address,
                keypair=my_key,
                miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/websocket/{my_key.ss58_address}"
            )

            async with miner_client:
                # generate ssh key and send it to miner
                private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)

                # G3 — attestation event: when due, mint a challenge that executors
                # must echo in TDX report_data[32:64] / GPU evidence. The nonce is
                # covered by validator_signature so it cannot be stripped or swapped.
                attestation_nonce = await self.attestation_service.maybe_issue_nonce(
                    payload.miner_hotkey
                )
                nonce_hex = attestation_nonce.value_hex if attestation_nonce else None

                await miner_client.send_model(
                    SSHPubKeySubmitRequest(
                        public_key=public_key,
                        validator_signature=self._sign_validator_pubkey(my_key, public_key, nonce=nonce_hex),
                        miner_hotkey=payload.miner_hotkey, # include miner's hotkey in the request
                        nonce=nonce_hex,
                    )
                )

                try:
                    msg = await asyncio.wait_for(
                        miner_client.job_state.miner_accepted_ssh_key_or_failed_future, JOB_LENGTH
                    )
                except TimeoutError:
                    logger.error(
                        _m(
                            "Waiting accepted ssh key or failed request from miner resulted in TimeoutError",
                            extra=get_extra_info(default_extra),
                        ),
                    )
                    msg = None
                except Exception:
                    logger.error(
                        _m(
                            "Waiting accepted ssh key or failed request from miner resulted in an exception",
                            extra=get_extra_info(default_extra),
                        ),
                    )
                    msg = None

                if msg is None:
                    # Deliberately NOT force-passed, unlike the zero-executors case below. There we
                    # know the miner is alive and chose to drop the executor because it could not
                    # install our key -- exactly the manual-rental shape. Here we have no signal
                    # from the miner at all, so we cannot distinguish "renter holds the box" from
                    # "this miner is gone"; force-passing would pay emissions on strictly less
                    # evidence than the case it is meant to cover. This assumes the miner neuron is
                    # shared infrastructure independent of the rented host (CENTRAL_MODE), so that
                    # an unreachable miner is a real fault rather than an expected consequence of
                    # handing the box over -- CENTRAL_MODE defaults to False and confirming the
                    # target deployment is still an open question on the plan.
                    # Accepted for beta -- see the plan's V3 "Known limitation".
                    return self._build_failed_job_result(
                        payload,
                        "Miner did not respond after SSH key submission",
                    )

                if isinstance(msg, AcceptSSHKeyRequest):
                    logger.info(
                        _m(
                            "Received AcceptSSHKeyRequest for miner. Running tasks for executors",
                            extra=get_extra_info(
                                {**default_extra, "executors": len(msg.executors)}
                            ),
                        ),
                    )
                    if len(msg.executors) == 0 and not self._has_manual_rental_executors(
                        payload, rented_data
                    ):
                        # Zero executors is normally a miner failure. It is the *expected* shape when
                        # every executor this miner has is under a manual rental, though -- the miner
                        # drops each one because it can no longer install our key. Only fail when
                        # there is genuinely nothing to score; otherwise fall through to synthesis.
                        return self._build_failed_job_result(
                            payload,
                            "Miner returned zero executors in AcceptSSHKeyRequest",
                        )
                    tasks = [
                        asyncio.create_task(
                            asyncio.wait_for(
                                self.task_service.create_task(
                                    miner_info=payload,
                                    executor_info=executor_info,
                                    keypair=my_key,
                                    private_key=private_key.decode("utf-8"),
                                    public_key=public_key.decode("utf-8"),
                                    encrypted_files=encrypted_files,
                                    rented_data=rented_data,
                                    default_docker_image_digests=default_docker_image_digests,
                                    executor_image_snapshot=executor_image_snapshot,
                                    attestation_nonce=attestation_nonce,
                                ),
                                timeout=settings.JOB_TIME_OUT - 120
                            )
                        )
                        for executor_info in msg.executors
                    ]

                    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
                    results = self._filter_task_results(msg.executors, raw_results, default_extra)
                    results.extend(
                        self._build_manual_rental_results(payload, rented_data, existing=results)
                    )

                    # DAH-2667: while the miner's key is still installed and its idle hosts are
                    # free, measure every RoCE pair. Best effort — an unmeasured pair is simply not
                    # sold as a cluster.
                    await measure_and_attach(
                        results,
                        self.ssh_service.decrypt_payload(
                            my_key.ss58_address, private_key.decode("utf-8")
                        ),
                        default_extra,
                        job_started_at,
                    )

                    logger.info(
                        _m(
                            "Finished running tasks for executors",
                            extra=get_extra_info({**default_extra, "executors": len(results)}),
                        ),
                    )

                    try:
                        await miner_client.send_model(SSHPubKeyRemoveRequest(
                            public_key=public_key,
                            validator_signature=self._sign_validator_pubkey(my_key, public_key),
                            miner_hotkey=payload.miner_hotkey
                        ))
                    except Exception as e:
                        logger.warning(
                            _m(
                                "Failed to send SSHPubKeyRemoveRequest (non-critical)",
                                extra=get_extra_info({
                                    **default_extra,
                                    "error": _get_error_details(e),
                                }),
                            ),
                        )

                    return {
                        "miner_hotkey": payload.miner_hotkey,
                        "miner_coldkey": payload.miner_coldkey,
                        "results": results,
                    }
                elif isinstance(msg, FailedRequest):
                    logger.warning(
                        _m(
                            "Requesting job failed for miner",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )
                    return self._build_failed_job_result(
                        payload,
                        f"Miner returned FailedRequest: {msg.details or 'unknown reason'}",
                    )
                elif isinstance(msg, DeclineJobRequest):
                    logger.warning(
                        _m(
                            "Requesting job declined for miner",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )
                    return self._build_failed_job_result(
                        payload,
                        "Miner declined the job request",
                    )
                else:
                    logger.error(
                        _m(
                            "Unexpected msg",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )
                    return self._build_failed_job_result(
                        payload,
                        f"Unexpected response from miner: {msg}",
                    )
        except asyncio.CancelledError:
            logger.error(
                _m("Requesting job to miner was cancelled", extra=get_extra_info(default_extra)),
            )
            return self._build_failed_job_result(
                payload,
                "Validator cancelled the request before completion",
            )
        except asyncio.TimeoutError:
            logger.error(
                _m("Requesting job to miner was timed out", extra=get_extra_info(default_extra)),
            )
            return self._build_failed_job_result(
                payload,
                "Request to miner timed out",
            )
        except RetryError as e:
            last_attempt = getattr(e, "last_attempt", None)
            root_exc = last_attempt.exception() if last_attempt else None
            if isinstance(root_exc, AttributeError):
                friendly_reason = "Failed to send SSH key because miner websocket never connected"
            else:
                friendly_reason = str(root_exc or e)

            logger.error(
                _m(
                    "Requesting job to miner resulted in a retry exhaustion error",
                    extra=get_extra_info({**default_extra, "error": friendly_reason}),
                ),
            )
            return self._build_failed_job_result(
                payload,
                friendly_reason,
            )
        except Exception as e:
            logger.error(
                _m(
                    "Requesting job to miner resulted in an exception",
                    extra=get_extra_info({
                        **default_extra,
                        "error": _get_error_details(e),
                    }),
                ),
            )
            return self._build_failed_job_result(
                payload,
                str(e),
            )

    def _filter_task_results(
        self,
        executors: list[ExecutorSSHInfo],
        raw_results: list,
        default_extra: dict,
    ) -> list[JobResult]:
        # keep successful JobResults; log every executor whose task ended in an
        # exception/timeout/empty result instead of silently dropping it (DAH-2365)
        results: list[JobResult] = []
        for executor_info, result in zip(executors, raw_results):
            if result and not isinstance(result, BaseException):
                results.append(result)
                continue
            if result is None:
                error_type = None
                error = "empty result"
            else:
                error_type = type(result).__name__
                # str(TimeoutError()) is "", so fall back to the type name to avoid an empty error
                error = str(result) or error_type
            logger.warning(
                _m(
                    "Executor task dropped without result",
                    extra=get_extra_info(
                        {
                            **default_extra,
                            "executor_uuid": str(executor_info.uuid),
                            "executor_ip_address": executor_info.address,
                            "error_type": error_type,
                            "error": error,
                        }
                    ),
                ),
            )
        return results

    def _iter_manual_rental_candidates(
        self,
        payload: MinerJobRequestPayload,
        rented_data: RentedExecutorsResponse | None,
    ):
        """Yield (executor_uuid, info, rented_executor) for this miner's force-passable rentals.

        Shared by the cheap `_has_manual_rental_executors` probe and the actual synthesis so the
        two can never disagree about which executors qualify.
        """
        if not rented_data or not rented_data.manual_rental_executors:
            return

        for executor_uuid, info in rented_data.manual_rental_executors.items():
            executor_uuid = str(executor_uuid)

            rented_executor = rented_data.executors.get(executor_uuid)
            if not rented_executor:
                # Flagged by the backend but not listed as rented (e.g. the pod is DELETING, which
                # get_rented_pods_with_ports excludes). Without an address there is nothing to build.
                continue
            if rented_executor.miner_hotkey != payload.miner_hotkey:
                # Belongs to a different miner's job request.
                continue
            if info.gpu_model not in BASE_GPU_MAP:
                # A real result can never reach the incentive layer with an unknown model --
                # GpuModelValidCheck is fatal and halts the pipeline first. A synthetic one skips
                # the whole pipeline, so it has to make that check itself: RentalPriceIncentive's
                # get_base_model_for_gpu does a raising BASE_GPU_MAP[...] subscript, and it is
                # called from calculate_mining_scores, which has no per-result try/except. An
                # unknown model (e.g. a SKU retired from the map since the rental was created)
                # would therefore abort weight-setting for EVERY miner on the subnet this cycle.
                # Dropping the entry costs this one node its emission; raising costs everyone's.
                logger.warning(
                    _m(
                        "Manual rental executor has a GPU model that is not in BASE_GPU_MAP; "
                        "skipping its forced pass to protect this cycle's weight calculation",
                        extra=get_extra_info({
                            "job_batch_id": payload.job_batch_id,
                            "miner_hotkey": payload.miner_hotkey,
                            "executor_uuid": executor_uuid,
                            "gpu_model": info.gpu_model,
                        }),
                    ),
                )
                continue

            yield executor_uuid, info, rented_executor

    def _has_manual_rental_executors(
        self,
        payload: MinerJobRequestPayload,
        rented_data: RentedExecutorsResponse | None,
    ) -> bool:
        """Whether this miner has any force-passable manual rental, without building anything.

        Used to decide whether a zero-executor AcceptSSHKeyRequest is a genuine miner failure or
        the expected shape when every executor has been handed to a renter.
        """
        return any(self._iter_manual_rental_candidates(payload, rented_data))

    def _build_manual_rental_results(
        self,
        payload: MinerJobRequestPayload,
        rented_data: RentedExecutorsResponse | None,
        existing: list[JobResult],
    ) -> list[JobResult]:
        """Synthesise forced-pass results for this miner's special manual (bare-metal) rentals.

        A manually-rented node is handed to the renter at root level and the platform stops
        provisioning it, so the miner can no longer install our SSH key on it. The miner silently
        drops such an executor from AcceptSSHKeyRequest (it filters on a successful pubkey upload),
        which means TaskService.create_task never runs for it and it would score 0. The forced pass
        therefore has to be synthesised out here, from the backend's list, rather than short-circuited
        inside per-executor validation.

        The node is never contacted: score, gpu_model and gpu_count come entirely from the backend.

        `spec` is deliberately left None. The backend skips its whole executor upsert on a null spec,
        which is what we want -- a synthesised spec that validated but disagreed with the stored row
        would flip the executor inactive and take billing down with it.
        """
        already_scored = {str(result.executor_info.uuid) for result in existing}
        results: list[JobResult] = []

        for executor_uuid, info, rented_executor in self._iter_manual_rental_candidates(
            payload, rented_data
        ):
            if executor_uuid in already_scored:
                # The node answered after all. A real result always beats a synthesised one.
                continue

            try:
                executor_port = int(rented_executor.executor_ip_port)
            except (TypeError, ValueError):
                executor_port = 0

            executor_info = ExecutorSSHInfo(
                uuid=executor_uuid,
                address=rented_executor.executor_ip_address,
                port=executor_port,
                # No SSH is attempted for a manual rental; these exist only to satisfy the model.
                ssh_username="",
                ssh_port=0,
                python_path="",
                root_dir="",
            )

            # Read the incentive-layer gates off rented_data exactly as ResultHandler does, rather
            # than hardcoding them: forcing score=1.0 buys a place in the mining pool, it does not
            # exempt the node from spot-tier or Discord exclusions.
            is_spot = executor_uuid in (rented_data.spot_executor_ids or [])
            discord_connected_ids = rented_data.provider_discord_connected_executor_ids
            provider_discord_connected = (
                True if discord_connected_ids is None else executor_uuid in discord_connected_ids
            )

            log_text = _m(
                MANUAL_RENTAL_FORCED_PASS_EVENT,
                extra=get_extra_info({
                    "job_batch_id": payload.job_batch_id,
                    "miner_hotkey": payload.miner_hotkey,
                    "executor_uuid": executor_uuid,
                    "executor_ip_address": rented_executor.executor_ip_address,
                    "gpu_model": info.gpu_model,
                    "gpu_count": info.gpu_count,
                    "reason": "special_manual_rental",
                }),
            ).to_full_string()

            results.append(
                JobResult(
                    spec=None,
                    executor_info=executor_info,
                    score=1.0,
                    job_score=1.0,
                    job_batch_id=payload.job_batch_id,
                    log_status="success",
                    log_text=log_text,
                    gpu_model=info.gpu_model,
                    gpu_count=info.gpu_count,
                    # Rented exempts the minimum-driver gate, which we cannot measure here.
                    is_rented=True,
                    # Not measurable without the node; a false reading would silently cut the score.
                    sysbox_runtime=True,
                    is_spot=is_spot,
                    provider_discord_connected=provider_discord_connected,
                )
            )

        if results:
            logger.info(
                _m(
                    "Forced pass for special manual rentals",
                    extra=get_extra_info({
                        "job_batch_id": payload.job_batch_id,
                        "miner_hotkey": payload.miner_hotkey,
                        "executors": len(results),
                        "executor_uuids": [str(r.executor_info.uuid) for r in results],
                    }),
                ),
            )

        return results

    def _build_failed_job_result(self, payload: MinerJobRequestPayload, reason: str):
        executor_info = ExecutorSSHInfo(
            # Special uuid for failed miners
            uuid="11111111-1111-1111-1111-111111111111",
            address=payload.miner_address,
            port=payload.miner_port,
            ssh_username="unknown",
            ssh_port=0,
            python_path="",
            root_dir="",
        )

        job_result = JobResult(
            spec=None,
            executor_info=executor_info,
            score=0,
            job_score=0,
            collateral_deposited=False,
            job_batch_id=payload.job_batch_id,
            log_status="error",
            log_text=reason,
            gpu_model=None,
            gpu_count=0,
            sysbox_runtime=False,
        )

        return {
            "miner_hotkey": payload.miner_hotkey,
            "miner_coldkey": payload.miner_coldkey,
            "results": [job_result],
        }

    async def request_validation_cycle_now(self) -> None:
        """Ask the validator loop to start its cycle now, not at the next block window.

        The loop runs in another process, so the request crosses over Redis. Staging only --
        the caller gates on the environment.
        """
        await self.redis_service.request_forced_validation_cycle()
        logger.info(_m("Forced validation cycle requested", extra=get_extra_info({})))

    async def publish_machine_specs(
        self, results: list[JobResult], miner_hotkey: str, miner_coldkey: str
    ):
        """Publish machine specs to compute app connector process"""
        default_extra = {
            "miner_hotkey": miner_hotkey,
        }
        if not results:
            return

        if settings.DRY_RUN:
            logger.info(
                _m(
                    "DRY_RUN: Skipping publish_machine_specs to compute app",
                    extra=get_extra_info({**default_extra, "job_batch_id": results[0].job_batch_id, "results": len(results)}),
                ),
            )
            return

        logger.info(
            _m(
                "Publishing machine specs to compute app connector process",
                extra=get_extra_info({**default_extra, "job_batch_id": results[0].job_batch_id, "results": len(results)}),
            ),
        )
        batch_total = len(results)
        for result in results:
            try:
                await self.redis_service.publish(
                    MACHINE_SPEC_CHANNEL,
                    {
                        "specs": result.spec,
                        "miner_hotkey": miner_hotkey,
                        "miner_coldkey": miner_coldkey,
                        "executor_uuid": result.executor_info.uuid,
                        "executor_ip": result.executor_info.address,
                        "executor_port": result.executor_info.port,
                        "executor_ssh_port": result.executor_info.ssh_port,
                        "price_per_gpu": result.executor_info.price_per_gpu,
                        "score": result.score,
                        "synthetic_job_score": result.job_score,
                        "job_batch_id": result.job_batch_id,
                        "netuid": settings.BITTENSOR_NETUID,
                        "scored_at": result.scored_at.isoformat() if result.scored_at else None,
                        "incentive": result.incentive if result.incentive is not None else 0.0,
                        "incentive_rented": result.incentive_rented,
                        "incentive_idle": result.incentive_idle,
                        "incentive_source": result.incentive_source,
                        "node_state_at_cycle": result.node_state_at_cycle,
                        "incentive_formula_version": result.incentive_formula_version,
                        "incentive_formula_inputs": result.incentive_formula_inputs,
                        "log_status": result.log_status,
                        "log_text": result.full_log_text,
                        # [] means this cycle reported no catalogued zero-incentive reason.
                        "incentive_reasons": [
                            reason.model_dump(mode="json") for reason in result.zero_incentive_reasons
                        ],
                        "collateral_deposited": result.collateral_deposited,
                        "ssh_pub_keys": result.ssh_pub_keys,
                        "attestation_digest": result.attestation_digest,
                        "tee_type": result.tee_type,
                        "tdx_attestation_passed": result.tdx_attestation_passed,
                        "gpu_attestation_passed": result.gpu_attestation_passed,
                        "executor_image": result.executor_image_report,
                        "sent_at": time.time(),
                        "batch_total": batch_total,
                        "availability_errors": result.availability_errors,
                    },
                )
            except Exception as e:
                logger.error(
                    _m(
                        f"Error publishing machine specs of {miner_hotkey} to compute app connector process",
                        extra=get_extra_info({**default_extra, "error": str(e)}),
                    ),
                    exc_info=True,
                )

    def _handle_container_error(
        self,
        payload: ContainerBaseRequest,
        msg: str | _StructuredMessage,
        error_code: FailedContainerErrorCodes,
    ):
        # DAH-2475: two texts with two audiences. `msg` is the HEADLINE — it feeds renter-visible
        # events on the backend, so it must never carry host details. `detail` is the full structured
        # text (headline + the `extra` dict with the actual exception, executor host, failure step);
        # it exists because a bare "Resulted in an exception" cost a manual investigation per failure —
        # the backend stores it in filler_run.failure_reason and logs, never in customer events.
        # Logging keeps the structured object so `extra` still lands in Loki as fields.
        logger.error(msg)
        headline: str = str(msg)
        detail: str | None = msg.to_full_string() if isinstance(msg, _StructuredMessage) else None

        if isinstance(payload, ContainerCreateRequest):
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=headline,
                detail=detail,
                error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                error_code=error_code,
            )

        elif isinstance(payload, ContainerDeleteRequest):
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=headline,
                detail=detail,
                error_type=FailedContainerErrorTypes.ContainerDeletionFailed,
                error_code=error_code,
            )
        elif isinstance(payload, AddSshPublicKeyRequest):
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=headline,
                detail=detail,
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=error_code,
            )
        elif isinstance(payload, InstallJupyterServerRequest):
            return JupyterInstallationFailed(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=detail or headline,
            )
        else:
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=headline,
                detail=detail,
                error_type=FailedContainerErrorTypes.UnknownRequest,
                error_code=error_code,
            )

    async def handle_container(self, payload: ContainerBaseRequest):
        """Handle container request, tracking the create so its own delete can cancel it."""
        # DAH-2728: register the create BEFORE the miner handshake — a delete whose handshake is
        # quicker would otherwise reach the executor first and see a pod that does not exist yet.
        if not isinstance(payload, ContainerCreateRequest):
            return await self._route_container(payload)
        with inflight_creates.track(payload.pod_id):
            return await self._route_container(payload)

    async def _route_container(self, payload: ContainerBaseRequest):
        """Route a container request - uses REST API if configured, otherwise WebSocket."""
        if settings.USE_REST_API:
            logger.info(
                _m(
                    "Routing handle_container to REST API",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                        "request_type": str(payload.message_type),
                    }),
                ),
            )
            return await self._handle_container(payload)
        else:
            logger.info(
                _m(
                    "Routing handle_container to WebSocket",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                        "request_type": str(payload.message_type),
                    }),
                ),
            )
        
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_id": payload.executor_id,
            "pod_id": payload.pod_id,
            "workload_kind": payload.workload_kind.value,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
            "container_request_type": str(payload.message_type),
        }

        docker_service = DockerService(
            ssh_service=self.ssh_service,
            redis_service=self.redis_service,
            attestation_service=self.attestation_service,
        )

        try:
            miner_client = MinerClient(
                loop=loop,
                miner_address=payload.miner_address,
                miner_port=payload.miner_port,
                miner_hotkey=payload.miner_hotkey,
                my_hotkey=my_key.ss58_address,
                keypair=my_key,
                miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/websocket/{my_key.ss58_address}",
            )

            async with miner_client:
                # generate ssh key and send it to miner
                private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)

                await miner_client.send_model(
                    SSHPubKeySubmitRequest(
                        public_key=public_key,
                        validator_signature=self._sign_validator_pubkey(my_key, public_key),
                        executor_id=payload.executor_id,
                        is_rental_request=isinstance(payload, ContainerCreateRequest),
                        miner_hotkey=payload.miner_hotkey
                    )
                )

                logger.info(
                    _m("Sent SSH key to miner.", extra=get_extra_info(default_extra)),
                )

                msg = await asyncio.wait_for(
                    miner_client.job_state.miner_accepted_ssh_key_or_failed_future,
                    timeout=JOB_LENGTH,
                )

                if isinstance(msg, AcceptSSHKeyRequest):
                    logger.info(
                        _m(
                            "Received AcceptSSHKeyRequest",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )

                    try:
                        executor = msg.executors[0]
                    except Exception as e:
                        executor = None

                    if executor is None or executor.uuid != payload.executor_id:
                        log_text = _m("Error: Invalid executor id", extra=get_extra_info(default_extra))

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key, 
                                validator_signature=self._sign_validator_pubkey(my_key, public_key),
                                executor_id=payload.executor_id, 
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        if executor:
                            logger.info(
                                _m(
                                    "Remove rented machine from redis",
                                    extra=get_extra_info(default_extra),
                                ),
                            )
                            await self.redis_service.remove_rented_machine(executor)

                        return self._handle_container_error(
                            payload=payload,
                            msg=log_text,
                            error_code=FailedContainerErrorCodes.InvalidExecutorId
                        )

                    renting_in_progress = await self.redis_service.renting_in_progress(payload.miner_hotkey, payload.executor_id, payload.pod_id)
                    if renting_in_progress and not _bypasses_renting_in_progress(payload):
                        log_text = _m(
                            "Decline renting pod request. Renting is still in progress",
                            extra=get_extra_info(default_extra),
                        )

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key, 
                                validator_signature=self._sign_validator_pubkey(my_key, public_key),
                                executor_id=payload.executor_id, 
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return self._handle_container_error(
                            payload=payload,
                            msg=log_text,
                            error_code=FailedContainerErrorCodes.RentingInProgress,
                        )

                    # get private key for ssh connection - asyncssh
                    ssh_pkey = asyncssh.import_private_key(
                        self.ssh_service.decrypt_payload(
                            my_key.ss58_address, private_key.decode("utf-8")
                        )
                    )


                    if isinstance(payload, ContainerCreateRequest):
                        logger.info(
                            _m(
                                "Creating container",
                                extra=get_extra_info(
                                    {**default_extra, "payload": str(payload)}
                                ),
                            ),
                        )
                        result = await docker_service.create_container(
                            payload,
                            executor,
                            my_key,
                            private_key.decode("utf-8"),
                        )

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key,
                                validator_signature=self._sign_validator_pubkey(my_key, public_key),
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return result

                    elif isinstance(payload, ContainerDeleteRequest):
                        logger.info(
                            _m(
                                "Deleting container",
                                extra=get_extra_info(
                                    {**default_extra, "payload": str(payload)}
                                ),
                            ),
                        )
                        result = await docker_service.delete_container(
                            payload,
                            executor,
                            my_key,
                            private_key.decode("utf-8"),
                        )

                        logger.info(
                            _m(
                                "Deleted Container",
                                extra=get_extra_info(
                                    {**default_extra, "payload": str(payload)}
                                ),
                            ),
                        )
                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key,
                                validator_signature=self._sign_validator_pubkey(my_key, public_key),
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return result
                    elif isinstance(payload, AddSshPublicKeyRequest):
                        logger.info(
                            _m(
                                "adding ssh key to container",
                                extra=get_extra_info(
                                    {**default_extra, "payload": str(payload)}
                                ),
                            ),
                        )
                        result = await docker_service.add_ssh_key(
                            payload,
                            executor,
                            my_key,
                            private_key.decode("utf-8"),
                        )

                        logger.info(
                            _m(
                                "Added ssh to the container",
                                extra=get_extra_info(
                                    {**default_extra, "payload": str(payload)}
                                ),
                            ),
                        )

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key,
                                validator_signature=self._sign_validator_pubkey(my_key, public_key),
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return result
                    elif isinstance(payload, RemoveSshPublicKeysRequest):
                        result = await docker_service.remove_ssh_keys(payload, executor, my_key, private_key.decode("utf-8"))

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key,
                                validator_signature=self._sign_validator_pubkey(my_key, public_key),
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return result
                    elif isinstance(payload, InstallJupyterServerRequest):
                        result = await docker_service.install_jupyter_server(payload, executor, my_key, private_key.decode("utf-8"))

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key,
                                validator_signature=self._sign_validator_pubkey(my_key, public_key),
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return result
                    elif isinstance(payload, BackupContainerRequest):
                        return await self.handle_backup_container_req(executor, payload, ssh_pkey)
                    elif isinstance(payload, RestoreContainerRequest):
                        return await self.handle_restore_container_req(executor, payload, ssh_pkey)
                    elif isinstance(payload, CancelStorageOperationRequest):
                        return await self.handle_cancel_storage_operation_req(executor, payload, ssh_pkey)
                    else:
                        log_text = _m(
                            "Unexpected request",
                            extra=get_extra_info(
                                {**default_extra, "payload": str(payload)}
                            ),
                        )

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=public_key,
                                validator_signature=self._sign_validator_pubkey(my_key, public_key),
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return self._handle_container_error(
                            payload=payload,
                            msg=log_text,
                            error_code=FailedContainerErrorCodes.UnknownError,
                        )

                elif isinstance(msg, FailedRequest):
                    log_text = _m(
                        "Error: Miner failed job",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    )

                    return self._handle_container_error(
                        payload=payload,
                        msg=log_text,
                        error_code=FailedContainerErrorCodes.FailedMsgFromMiner,
                    )
                else:
                    log_text = _m(
                        "Error: Unexpected msg",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    )

                    return self._handle_container_error(
                        payload=payload,
                        msg=log_text,
                        error_code=FailedContainerErrorCodes.UnknownError,
                    )
        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            return self._handle_container_error(
                payload=payload,
                msg=log_text,
                error_code=FailedContainerErrorCodes.ExceptionError,
            )

    async def get_pod_logs(self, payload: GetPodLogsRequestFromServer) -> PodLogsResponseToServer:
        """Get pod logs - uses REST API if configured, otherwise WebSocket."""
        if settings.USE_REST_API:
            logger.info(
                _m(
                    "Routing get_pod_logs to REST API",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                    }),
                ),
            )
            return await self._get_pod_logs(payload)
        else:
            logger.info(
                _m(
                    "Routing get_pod_logs to WebSocket",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                    }),
                ),
            )
        
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "pod_id": payload.pod_id,
            "executor_id": payload.executor_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
            "container_name": payload.container_name,
        }

        try:
            miner_client = MinerClient(
                loop=loop,
                miner_address=payload.miner_address,
                miner_port=payload.miner_port,
                miner_hotkey=payload.miner_hotkey,
                my_hotkey=my_key.ss58_address,
                keypair=my_key,
                miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/websocket/{my_key.ss58_address}",
            )

            async with miner_client:
                # generate ssh key and send it to miner
                await miner_client.send_model(
                    GetPodLogsRequest(
                        container_name=payload.container_name,
                        pod_id=payload.pod_id,
                        executor_id=payload.executor_id, 
                        miner_hotkey=payload.miner_hotkey,
                    )
                )

                logger.info(
                    _m("Getting logs from executor", extra=get_extra_info(default_extra)),
                )

                msg = await asyncio.wait_for(
                    miner_client.job_state.miner_accepted_ssh_key_or_failed_future,
                    timeout=JOB_LENGTH,
                )

                if isinstance(msg, PodLogsResponse):
                    logger.info(
                        _m(
                            "Pod Log result",
                            extra=get_extra_info({**default_extra, "logs": len(msg.logs)}),
                        )
                    )
                    return PodLogsResponseToServer(
                        miner_hotkey=payload.miner_hotkey,
                        pod_id=payload.pod_id,
                        executor_id=payload.executor_id,
                        container_name=payload.container_name,
                        logs=msg.logs
                    )

                elif isinstance(msg, FailedRequest):
                    log_text = _m(
                        "Error: FailedRequest",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    )
                    logger.error(log_text)

                    return FailedGetPodLogs(
                        miner_hotkey=payload.miner_hotkey,
                        pod_id=payload.pod_id,
                        executor_id=payload.executor_id,
                        container_name=payload.container_name,
                        msg=log_text.to_full_string(),
                    )

                else:
                    log_text = _m(
                        "Error: Unexpected msg",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    )
                    logger.error(log_text)

                    return FailedGetPodLogs(
                        miner_hotkey=payload.miner_hotkey,
                        pod_id=payload.pod_id,
                        executor_id=payload.executor_id,
                        container_name=payload.container_name,
                        msg=log_text.to_full_string(),
                    )

        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text)

            return FailedGetPodLogs(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                container_name=payload.container_name,
                msg=log_text.to_full_string(),
            )

    async def add_debug_ssh_key(self, payload: AddDebugSshKeyRequest) -> DebugSshKeyAdded:
        """Add debug SSH key - uses REST API if configured, otherwise WebSocket."""
        if settings.USE_REST_API:
            logger.info(
                _m(
                    "Routing add_debug_ssh_key to REST API",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                    }),
                ),
            )
            return await self._add_debug_ssh_key(payload)
        else:
            logger.info(
                _m(
                    "Routing add_debug_ssh_key to WebSocket",
                    extra=get_extra_info({
                        "use_rest_api": settings.USE_REST_API,
                        "miner_hotkey": payload.miner_hotkey,
                        "executor_id": payload.executor_id,
                    }),
                ),
            )
        
        loop = asyncio.get_event_loop()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_id": payload.executor_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
        }

        try:
            miner_client = MinerClient(
                loop=loop,
                miner_address=payload.miner_address,
                miner_port=payload.miner_port,
                miner_hotkey=payload.miner_hotkey,
                my_hotkey=my_key.ss58_address,
                keypair=my_key,
                miner_url=f"ws://{payload.miner_address}:{payload.miner_port}/websocket/{my_key.ss58_address}",
            )

            async with miner_client:

                await miner_client.send_model(
                    SSHPubKeySubmitRequest(
                        public_key=payload.public_key,
                        validator_signature=self._sign_validator_pubkey(my_key, payload.public_key),
                        executor_id=payload.executor_id,
                        is_rental_request=False,
                        miner_hotkey=payload.miner_hotkey,
                    )
                )

                logger.info(
                    _m("Sent SSH key to miner.", extra=get_extra_info(default_extra)),
                )

                msg = await asyncio.wait_for(
                    miner_client.job_state.miner_accepted_ssh_key_or_failed_future,
                    timeout=JOB_LENGTH,
                )

                if isinstance(msg, AcceptSSHKeyRequest):
                    logger.info(
                        _m(
                            "Received AcceptSSHKeyRequest",
                            extra=get_extra_info({**default_extra, "msg": str(msg)}),
                        ),
                    )

                    try:
                        executor = msg.executors[0]
                    except Exception as e:
                        executor = None

                    if executor is None or executor.uuid != payload.executor_id:
                        log_text = _m("Error: Invalid executor id", extra=get_extra_info(default_extra))
                        logger.error(log_text)

                        await miner_client.send_model(
                            SSHPubKeyRemoveRequest(
                                public_key=payload.public_key, 
                                validator_signature=self._sign_validator_pubkey(my_key, payload.public_key),
                                executor_id=payload.executor_id,
                                miner_hotkey=payload.miner_hotkey
                            )
                        )

                        return FailedAddDebugSshKey(
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            msg=log_text.to_full_string(),
                        )

                    logger.info(
                        _m(
                            "Added debug public key",
                            extra=get_extra_info(default_extra),
                        ),
                    )

                    return DebugSshKeyAdded(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        address=executor.address,
                        port=executor.port,
                        ssh_username=executor.ssh_username,
                        ssh_port=executor.ssh_port,
                    )

                else:
                    log_text = _m(
                        "Error: Failed to add debug public key",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    )
                    logger.error(log_text)

                    return FailedAddDebugSshKey(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=log_text.to_full_string(),
                    )

        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return FailedAddDebugSshKey(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                msg=log_text.to_full_string(),
            )

    async def handle_backup_container_req(self, executor_info: ExecutorSSHInfo, payload: BackupContainerRequest, pkey: SSHKey):
        """Handle backup container request."""
        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=None,
        ) as ssh_client:

            if payload.backup_engine == "restic":
                operation_id = UUID(payload.backup_log_id)
                await start_storage_operation(
                    ssh_client,
                    executor_info.python_path,
                    operation_id,
                    self._restic_backup_operation_spec(payload),
                    retain_terminal_artifacts=False,
                )
                return

            remote_script_path = "/root/app/backup_storage.py"
            remote_helper_path = "/root/app/workspace_mount.py"
            local_script_path = _miner_job_script_path("backup_storage.py")
            local_helper_path = _miner_job_script_path("workspace_mount.py")

            logger.info(
                _m(
                    "Uploading backup_storage.py script to the remote server", 
                    extra=get_extra_info({ "remote_script_path": remote_script_path, "local_script_path": local_script_path })
                ),
            )

            async with ssh_client.start_sftp_client() as sftp:
                await sftp.put(local_script_path, remote_script_path)
                await sftp.put(local_helper_path, remote_helper_path)

            argv = [
                executor_info.python_path,
                "/root/app/backup_storage.py",
                "--api-url", settings.COMPUTE_REST_API_URL_EXTERNAL,
                "--source-volume", payload.source_volume,
                "--backup-path", payload.backup_path,
                "--auth-token", payload.auth_token,
                "--backup-log-id", payload.backup_log_id,
                "--backup-volume-name", payload.backup_volume_info.name,
                "--backup-volume-iam_user_access_key", payload.backup_volume_info.iam_user_access_key,
                "--backup-volume-iam_user_secret_key", payload.backup_volume_info.iam_user_secret_key,
                "--source-volume-path", payload.source_volume_path,
                "--backup-target-path", payload.backup_target_path,
            ]
            await ssh_client.run(
                _nohup_command(["nohup", *argv], "/root/app/backup_storage.log"),
                timeout=50,
                check=True,
            )

    async def handle_restore_container_req(self, executor_info: ExecutorSSHInfo, payload: RestoreContainerRequest, pkey: SSHKey):
        """Handle restore container request."""
        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=None,
        ) as ssh_client:

            if payload.backup_engine == "restic":
                operation_id = UUID(payload.restore_log_id)
                await start_storage_operation(
                    ssh_client,
                    executor_info.python_path,
                    operation_id,
                    self._restic_restore_operation_spec(payload),
                    retain_terminal_artifacts=False,
                )
                return

            remote_script_path = "/root/app/restore_storage.py"
            remote_helper_path = "/root/app/workspace_mount.py"
            local_script_path = _miner_job_script_path("restore_storage.py")
            local_helper_path = _miner_job_script_path("workspace_mount.py")

            logger.info(
                _m(
                    "Uploading restore_storage.py script to the remote server for restore operation", 
                    extra=get_extra_info({ "remote_script_path": remote_script_path, "local_script_path": local_script_path })
                ),
            )

            async with ssh_client.start_sftp_client() as sftp:
                await sftp.put(local_script_path, remote_script_path)
                await sftp.put(local_helper_path, remote_helper_path)

            argv = [
                executor_info.python_path,
                "/root/app/restore_storage.py",
                "--api-url", settings.COMPUTE_REST_API_URL_EXTERNAL,
                "--target-volume", payload.target_volume,
                "--restore-path", payload.restore_path,
                "--backup-source-path", payload.backup_source_path,
                "--auth-token", payload.auth_token,
                "--restore-log-id", payload.restore_log_id,
                "--backup-volume-name", payload.backup_volume_info.name,
                "--backup-volume-iam_user_access_key", payload.backup_volume_info.iam_user_access_key,
                "--backup-volume-iam_user_secret_key", payload.backup_volume_info.iam_user_secret_key,
                "--target-volume-path", payload.target_volume_path,
            ]
            await ssh_client.run(
                _nohup_command(["nohup", *argv], "/root/app/restore_storage.log"),
                timeout=50,
                check=True,
            )

    async def handle_cancel_storage_operation_req(
        self,
        executor_info: ExecutorSSHInfo,
        payload: CancelStorageOperationRequest,
        pkey: SSHKey,
    ) -> None:
        async with asyncssh.connect(
            host=executor_info.address,
            port=executor_info.ssh_port,
            username=executor_info.ssh_username,
            client_keys=[pkey],
            known_hosts=None,
        ) as ssh_client:
            await cancel_storage_operation(ssh_client, UUID(payload.operation_id))

    @staticmethod
    def _restic_backup_operation_spec(payload: BackupContainerRequest) -> dict[str, object]:
        return {
            "operation_id": payload.backup_log_id,
            "pod_id": payload.pod_id,
            "repository_pod_id": payload.repository_pod_id or payload.pod_id,
            "action": "backup",
            "engine": "restic",
            "repository": _storage_repository_spec(
                payload.backup_volume_info,
                payload.repository_password,
            ),
            "workspace": {
                "mode": "encrypted_running" if payload.volume_encrypted else "plain_volume",
                "volume_name": payload.source_volume,
                "volume_path": payload.source_volume_path,
                "requested_path": payload.backup_path or payload.source_volume_path,
                "container_name": payload.container_name,
            },
            "reporter": {
                "api_url": settings.COMPUTE_REST_API_URL_EXTERNAL,
                "auth_token": payload.auth_token,
                "resource": "backup",
                "failure_timeout_seconds": payload.failure_timeout_seconds,
            },
        }

    @staticmethod
    def _restic_restore_operation_spec(payload: RestoreContainerRequest) -> dict[str, object]:
        return {
            "operation_id": payload.restore_log_id,
            "pod_id": payload.pod_id,
            "repository_pod_id": payload.repository_pod_id or payload.pod_id,
            "action": "restore",
            "engine": "restic",
            "snapshot_id": payload.snapshot_id,
            "repository": _storage_repository_spec(
                payload.backup_volume_info,
                payload.repository_password,
            ),
            "workspace": {
                "mode": "encrypted_running" if payload.volume_encrypted else "plain_volume",
                "volume_name": payload.target_volume,
                "volume_path": payload.target_volume_path,
                "requested_path": payload.restore_path or payload.target_volume_path,
                "container_name": payload.container_name,
            },
            "reporter": {
                "api_url": settings.COMPUTE_REST_API_URL_EXTERNAL,
                "auth_token": payload.auth_token,
                "resource": "restore",
                "failure_timeout_seconds": payload.failure_timeout_seconds,
            },
        }

    def _generate_auth_headers(self, my_key: bittensor.Keypair, miner_hotkey: str) -> dict:
        """Generate authentication headers for REST API requests.

        IMPORTANT: Uses AuthenticationPayload.blob_for_signing() for canonical serialization.
        This contract is defined in datura/datura/requests/validator_requests.py
        """
        payload = AuthenticationPayload(
            validator_hotkey=my_key.ss58_address,
            miner_hotkey=miner_hotkey,
            timestamp=int(time.time()),
        )
        signature = f"0x{my_key.sign(payload.blob_for_signing()).hex()}"

        return {
            "X-Validator-Hotkey": my_key.ss58_address,
            "X-Miner-Hotkey": miner_hotkey,
            "X-Timestamp": str(payload.timestamp),
            "X-Signature": signature,
        }

    async def _make_rest_request(
        self,
        method: str,
        url: str,
        json_data: dict,
        headers: dict,
        timeout: int,
        log_extra: dict,
        operation_name: str,
    ) -> tuple[int, dict | None]:
        """Make a REST API request to miner with proper error handling and logging.

        Args:
            method: HTTP method (e.g., 'POST', 'GET')
            url: Full URL to request
            json_data: JSON payload to send
            headers: HTTP headers
            timeout: Request timeout in seconds
            log_extra: Additional logging context
            operation_name: Name of operation for logging (e.g., 'SSH key submit')

        Returns:
            Tuple of (status_code, response_json). response_json is None if request failed
            or response is not valid JSON.

        Raises:
            asyncio.TimeoutError: If request times out
            aiohttp.ClientError: For other HTTP client errors
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    response.raise_for_status
                    response_data = await response.json()
                    return response.status, response_data
        except asyncio.TimeoutError:
            logger.error(
                _m(
                    f"REST API {operation_name} timed out after {timeout}s",
                    extra=get_extra_info({
                        **log_extra,
                        "timeout": timeout,
                        "url": url,
                    }),
                ),
            )
            raise
        except aiohttp.ClientError as e:
            logger.error(
                _m(
                    f"REST API {operation_name} client error",
                    extra=get_extra_info({
                        **log_extra,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "url": url,
                    }),
                ),
            )
            raise
        except Exception as e:
            logger.error(
                _m(
                    f"REST API {operation_name} unexpected error",
                    extra=get_extra_info({
                        **log_extra,
                        "error": _get_error_details(e),
                        "url": url,
                    }),
                ),
                exc_info=True,
            )
            raise

    async def _remove_ssh_key_via_rest(
        self,
        base_url: str,
        my_key: bittensor.Keypair,
        public_key: bytes,
        miner_hotkey: str,
        executor_id: str | None,
        log_extra: dict,
    ) -> bool:
        """Remove SSH key from miner via REST API.

        Args:
            base_url: Base URL of miner (e.g., 'http://192.168.1.1:8000')
            my_key: Validator's keypair for authentication
            public_key: SSH public key to remove
            miner_hotkey: Miner's hotkey
            executor_id: Optional executor ID
            log_extra: Additional logging context

        Returns:
            True if removal was successful (status 200), False otherwise.
            Logs warnings for failures but does not raise exceptions.
        """
        try:
            remove_request = SSHPubKeyRemoveRequest(
                public_key=public_key,
                validator_signature=self._sign_validator_pubkey(my_key, public_key),
                executor_id=executor_id,
                miner_hotkey=miner_hotkey,
            )

            status, _ = await self._make_rest_request(
                method="POST",
                url=f"{base_url}/api/validator/ssh-pubkey-remove",
                json_data=self._serialize_request(remove_request),
                headers=self._generate_auth_headers(my_key, miner_hotkey),
                timeout=REST_SSH_REMOVE_TIMEOUT,
                log_extra=log_extra,
                operation_name="SSH key removal",
            )

            if status != 200:
                logger.warning(
                    _m(
                        "Failed to remove SSH key via REST API. Validator key may still be present on miner",
                        extra=get_extra_info({
                            **log_extra,
                            "status": status,
                            "miner_hotkey": miner_hotkey,
                            "executor_id": executor_id,
                        }),
                    ),
                )
                return False

            return True

        except Exception as e:
            logger.warning(
                _m(
                    "Failed to remove SSH key via REST API. Validator key may still be present on miner",
                    extra=get_extra_info({
                        **log_extra,
                        "error": _get_error_details(e),
                        "miner_hotkey": miner_hotkey,
                        "executor_id": executor_id,
                    }),
                ),
            )
            return False

    def _serialize_request(self, request) -> dict:
        """Serialize a Pydantic request model to dict for JSON serialization.
        
        Handles bytes fields by ensuring they're properly encoded.
        """
        # Use model_dump_json and parse back to ensure proper serialization
        # This handles bytes fields correctly (base64 encoding)
        return json.loads(request.model_dump_json())

    async def _request_job_to_miner(
        self,
        payload: MinerJobRequestPayload,
        encrypted_files: MinerJobEnryptedFiles,
        rented_data: RentedExecutorsResponse,
        default_docker_image_digests: dict[str, str],
        executor_image_snapshot: ExpectedImageSnapshot | None = None,
    ):
        """REST API version of request_job_to_miner."""
        # DAH-2667: see the WebSocket path — the RoCE probe measures the cycle's remaining time
        # from here.
        job_started_at: float = time.monotonic()
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "job_batch_id": payload.job_batch_id,
            "miner_hotkey": payload.miner_hotkey,
            "miner_address": payload.miner_address,
            "miner_port": payload.miner_port,
        }

        try:
            logger.info(_m("Requesting job to miner via REST API", extra=get_extra_info(default_extra)))
            
            # Generate SSH key
            private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)

            # G3 — attestation event (REST path); see the WebSocket path for details.
            attestation_nonce = await self.attestation_service.maybe_issue_nonce(
                payload.miner_hotkey
            )
            nonce_hex = attestation_nonce.value_hex if attestation_nonce else None

            # Prepare request
            ssh_request = SSHPubKeySubmitRequest(
                public_key=public_key,
                validator_signature=self._sign_validator_pubkey(my_key, public_key, nonce=nonce_hex),
                miner_hotkey=payload.miner_hotkey,
                nonce=nonce_hex,
            )
            
            # Make REST API call
            base_url = f"http://{payload.miner_address}:{payload.miner_port}"
            headers = self._generate_auth_headers(my_key, payload.miner_hotkey)
            headers["Content-Type"] = "application/json"
            
            status, response_data = await self._make_rest_request(
                method="POST",
                url=f"{base_url}/api/validator/ssh-pubkey-submit",
                json_data=self._serialize_request(ssh_request),
                headers=headers,
                timeout=REST_SSH_SUBMIT_TIMEOUT,
                log_extra=default_extra,
                operation_name="SSH key submit",
            )

            if status != 200 or response_data is None:
                return self._build_failed_job_result(
                    payload,
                    "Failed to submit SSH key to miner via REST API",
                )

            msg = _parse_miner_response(response_data)
            
            # Track whether SSH key was successfully accepted
            ssh_key_accepted = False
            
            if isinstance(msg, AcceptSSHKeyRequest):
                ssh_key_accepted = True
                logger.info(
                    _m(
                        "Received AcceptSSHKeyRequest for miner via REST API. Running tasks for executors",
                        extra=get_extra_info(
                            {**default_extra, "executors": len(msg.executors)}
                        ),
                    ),
                )
                if len(msg.executors) == 0 and not self._has_manual_rental_executors(
                    payload, rented_data
                ):
                    # See the WebSocket path: zero executors is the expected shape when every
                    # executor is under a manual rental, so only fail when nothing can be scored.
                    return self._build_failed_job_result(
                        payload,
                        "Miner returned zero executors in AcceptSSHKeyRequest",
                    )
                tasks = [
                    asyncio.create_task(
                        asyncio.wait_for(
                            self.task_service.create_task(
                                miner_info=payload,
                                executor_info=executor_info,
                                keypair=my_key,
                                private_key=private_key.decode("utf-8"),
                                public_key=public_key.decode("utf-8"),
                                encrypted_files=encrypted_files,
                                rented_data=rented_data,
                                default_docker_image_digests=default_docker_image_digests,
                                executor_image_snapshot=executor_image_snapshot,
                                attestation_nonce=attestation_nonce,
                            ),
                            timeout=settings.JOB_TIME_OUT - 120
                        )
                    )
                    for executor_info in msg.executors
                ]

                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
                results = self._filter_task_results(msg.executors, raw_results, default_extra)
                results.extend(
                    self._build_manual_rental_results(payload, rented_data, existing=results)
                )

                # DAH-2667: while the miner's key is still installed and its idle hosts are free,
                # measure every RoCE pair. Best effort — an unmeasured pair is simply not sold as a
                # cluster.
                await measure_and_attach(
                    results,
                    self.ssh_service.decrypt_payload(
                        my_key.ss58_address, private_key.decode("utf-8")
                    ),
                    default_extra,
                    job_started_at,
                )

                logger.info(
                    _m(
                        "Finished running tasks for executors",
                        extra=get_extra_info({**default_extra, "executors": len(results)}),
                    ),
                )

                # Remove SSH key only if it was successfully accepted
                if ssh_key_accepted:
                    await self._remove_ssh_key_via_rest(
                        base_url=base_url,
                        my_key=my_key,
                        public_key=public_key,
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=None,
                        log_extra=default_extra,
                    )

                return {
                    "miner_hotkey": payload.miner_hotkey,
                    "miner_coldkey": payload.miner_coldkey,
                    "results": results,
                }
            elif isinstance(msg, FailedRequest):
                logger.warning(
                    _m(
                        "Requesting job failed for miner via REST API",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    ),
                )
                return self._build_failed_job_result(
                    payload,
                    f"Miner returned FailedRequest: {msg.details or 'unknown reason'}",
                )
            else:
                logger.error(
                    _m(
                        "Unexpected response from miner via REST API",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    ),
                )
                return self._build_failed_job_result(
                    payload,
                    f"Unexpected response from miner: {msg}",
                )
        except asyncio.CancelledError:
            logger.error(
                _m("Requesting job to miner via REST API was cancelled", extra=get_extra_info(default_extra)),
            )
            return self._build_failed_job_result(
                payload,
                "Requesting job to miner via REST API was cancelled",
            )
        except asyncio.TimeoutError:
            logger.error(
                _m("Requesting job to miner via REST API was timed out", extra=get_extra_info(default_extra)),
            )
            return self._build_failed_job_result(
                payload,
                "Requesting job to miner via REST API was timed out",
            )
        except Exception as e:
            logger.error(
                _m(
                    "Requesting job to miner via REST API resulted in an exception",
                    extra=get_extra_info({
                        **default_extra,
                        "error": _get_error_details(e),
                    }),
                ),
            )
            return self._build_failed_job_result(
                payload,
                "Requesting job to miner via REST API resulted in an exception",
            )

    async def _handle_container(self, payload: ContainerBaseRequest):
        """REST API version of handle_container."""
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_id": payload.executor_id,
            "pod_id": payload.pod_id,
            "workload_kind": payload.workload_kind.value,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
            "container_request_type": str(payload.message_type),
        }

        docker_service = DockerService(
            ssh_service=self.ssh_service,
            redis_service=self.redis_service,
            attestation_service=self.attestation_service,
        )

        try:
            base_url = f"http://{payload.miner_address}:{payload.miner_port}"
            headers = self._generate_auth_headers(my_key, payload.miner_hotkey)
            headers["Content-Type"] = "application/json"
            
            # Generate SSH key and send it to miner
            private_key, public_key = self.ssh_service.generate_ssh_key(my_key.ss58_address)

            ssh_request = SSHPubKeySubmitRequest(
                public_key=public_key,
                validator_signature=self._sign_validator_pubkey(my_key, public_key),
                executor_id=payload.executor_id,
                is_rental_request=isinstance(payload, ContainerCreateRequest),
                miner_hotkey=payload.miner_hotkey
            )

            logger.info(
                _m("Sent SSH key to miner via REST API.", extra=get_extra_info(default_extra)),
            )

            status, response_data = await self._make_rest_request(
                method="POST",
                url=f"{base_url}/api/validator/ssh-pubkey-submit",
                json_data=self._serialize_request(ssh_request),
                headers=headers,
                timeout=REST_CONTAINER_OP_TIMEOUT,
                log_extra=default_extra,
                operation_name="SSH key submit",
            )

            if status != 200 or response_data is None:
                error_msg = "Failed to submit SSH key"
                if response_data:
                    error_msg = f"{error_msg}: {response_data}"
                return self._handle_container_error(
                    payload=payload,
                    msg=error_msg,
                    error_code=FailedContainerErrorCodes.FailedMsgFromMiner,
                )

            msg = _parse_miner_response(response_data)

            # Track whether SSH key was successfully accepted
            ssh_key_accepted = False

            if isinstance(msg, AcceptSSHKeyRequest):
                ssh_key_accepted = True
                logger.info(
                    _m(
                        "Received AcceptSSHKeyRequest via REST API",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    ),
                )

                try:
                    executor = msg.executors[0]
                except Exception as e:
                    executor = None

                if executor is None or executor.uuid != payload.executor_id:
                    log_text = _m("Error: Invalid executor id", extra=get_extra_info(default_extra))

                    # Remove SSH key only if it was accepted
                    if ssh_key_accepted:
                        await self._remove_ssh_key_via_rest(
                            base_url=base_url,
                            my_key=my_key,
                            public_key=public_key,
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            log_extra=default_extra,
                        )

                    if executor:
                        logger.info(
                            _m(
                                "Remove rented machine from redis",
                                extra=get_extra_info(default_extra),
                            ),
                        )
                        await self.redis_service.remove_rented_machine(executor)

                    return self._handle_container_error(
                        payload=payload,
                        msg=log_text,
                        error_code=FailedContainerErrorCodes.InvalidExecutorId
                    )

                renting_in_progress = await self.redis_service.renting_in_progress(payload.miner_hotkey, payload.executor_id, payload.pod_id)
                if renting_in_progress and not _bypasses_renting_in_progress(payload):
                    log_text = _m(
                        "Decline renting pod request. Renting is still in progress",
                        extra=get_extra_info(default_extra),
                    )

                    # Remove SSH key only if it was accepted
                    if ssh_key_accepted:
                        await self._remove_ssh_key_via_rest(
                            base_url=base_url,
                            my_key=my_key,
                            public_key=public_key,
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            log_extra=default_extra,
                        )

                    return self._handle_container_error(
                        payload=payload,
                        msg=log_text,
                        error_code=FailedContainerErrorCodes.RentingInProgress,
                    )

                # Get private key for ssh connection - asyncssh
                ssh_pkey = asyncssh.import_private_key(
                    self.ssh_service.decrypt_payload(
                        my_key.ss58_address, private_key.decode("utf-8")
                    )
                )

                # Handle different container request types
                result = None
                if isinstance(payload, ContainerCreateRequest):
                    # DAH-2272: no pre-flag port-check removal here. The probe
                    # force-remove lives inside create_container (right before
                    # `docker run`, AFTER add_pending_pod sets the pending-pod
                    # flag), so a probe killed by the rental is covered by
                    # PortConnectivityCheck's renting_in_progress tolerate.
                    # Removing a probe here would run while renting_in_progress
                    # is still False and wrongly ding the miner's sysbox score.
                    logger.info(
                        _m(
                            "Creating container",
                            extra=get_extra_info(
                                {**default_extra, "payload": str(payload)}
                            ),
                        ),
                    )
                    result = await docker_service.create_container(
                        payload,
                        executor,
                        my_key,
                        private_key.decode("utf-8"),
                    )
                elif isinstance(payload, ContainerDeleteRequest):
                    logger.info(
                        _m(
                            "Deleting container",
                            extra=get_extra_info(
                                {**default_extra, "payload": str(payload)}
                            ),
                        ),
                    )
                    result = await docker_service.delete_container(
                        payload,
                        executor,
                        my_key,
                        private_key.decode("utf-8"),
                    )
                elif isinstance(payload, AddSshPublicKeyRequest):
                    logger.info(
                        _m(
                            "adding ssh key to container",
                            extra=get_extra_info(
                                {**default_extra, "payload": str(payload)}
                            ),
                        ),
                    )
                    result = await docker_service.add_ssh_key(
                        payload,
                        executor,
                        my_key,
                        private_key.decode("utf-8"),
                    )
                elif isinstance(payload, RemoveSshPublicKeysRequest):
                    result = await docker_service.remove_ssh_keys(payload, executor, my_key, private_key.decode("utf-8"))
                elif isinstance(payload, InstallJupyterServerRequest):
                    result = await docker_service.install_jupyter_server(payload, executor, my_key, private_key.decode("utf-8"))
                elif isinstance(payload, BackupContainerRequest):
                    result = await self.handle_backup_container_req(executor, payload, ssh_pkey)
                elif isinstance(payload, RestoreContainerRequest):
                    result = await self.handle_restore_container_req(executor, payload, ssh_pkey)
                elif isinstance(payload, CancelStorageOperationRequest):
                    result = await self.handle_cancel_storage_operation_req(executor, payload, ssh_pkey)
                else:
                    log_text = _m(
                        "Unexpected request",
                        extra=get_extra_info(
                            {**default_extra, "payload": str(payload)}
                        ),
                    )
                    result = self._handle_container_error(
                        payload=payload,
                        msg=log_text,
                        error_code=FailedContainerErrorCodes.UnknownError,
                    )

                # Remove SSH key after operation only if it was accepted
                if ssh_key_accepted:
                    await self._remove_ssh_key_via_rest(
                        base_url=base_url,
                        my_key=my_key,
                        public_key=public_key,
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        log_extra=default_extra,
                    )

                return result

            elif isinstance(msg, FailedRequest):
                log_text = _m(
                    "Error: Miner failed job",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )

                return self._handle_container_error(
                    payload=payload,
                    msg=log_text,
                    error_code=FailedContainerErrorCodes.FailedMsgFromMiner,
                )
            else:
                log_text = _m(
                    "Error: Unexpected msg",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )

                return self._handle_container_error(
                    payload=payload,
                    msg=log_text,
                    error_code=FailedContainerErrorCodes.UnknownError,
                )
        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            return self._handle_container_error(
                payload=payload,
                msg=log_text,
                error_code=FailedContainerErrorCodes.ExceptionError,
            )

    async def _get_pod_logs(self, payload: GetPodLogsRequestFromServer) -> PodLogsResponseToServer:
        """REST API version of get_pod_logs."""
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "pod_id": payload.pod_id,
            "executor_id": payload.executor_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
            "container_name": payload.container_name,
        }

        try:
            base_url = f"http://{payload.miner_address}:{payload.miner_port}"
            headers = self._generate_auth_headers(my_key, payload.miner_hotkey)
            headers["Content-Type"] = "application/json"
            
            logs_request = GetPodLogsRequest(
                container_name=payload.container_name,
                pod_id=payload.pod_id,
                executor_id=payload.executor_id, 
                miner_hotkey=payload.miner_hotkey,
            )

            logger.info(
                _m("Getting logs from executor via REST API", extra=get_extra_info(default_extra)),
            )

            status, response_data = await self._make_rest_request(
                method="POST",
                url=f"{base_url}/api/validator/pod-logs",
                json_data=self._serialize_request(logs_request),
                headers=headers,
                timeout=REST_POD_LOGS_TIMEOUT,
                log_extra=default_extra,
                operation_name="pod logs",
            )

            if status != 200 or response_data is None:
                error_msg = "Failed to get pod logs"
                if response_data:
                    error_msg = f"{error_msg}: {response_data}"
                log_text = _m(
                    "Error: FailedRequest",
                    extra=get_extra_info({**default_extra, "error": error_msg}),
                )
                logger.error(log_text)
                return FailedGetPodLogs(
                    miner_hotkey=payload.miner_hotkey,
                    pod_id=payload.pod_id,
                    executor_id=payload.executor_id,
                    container_name=payload.container_name,
                    msg=log_text.to_full_string(),
                )

            msg = _parse_miner_response(response_data)

            if isinstance(msg, PodLogsResponse):
                logger.info(
                    _m(
                        "Pod Log result via REST API",
                        extra=get_extra_info({**default_extra, "logs": len(msg.logs)}),
                    )
                )
                return PodLogsResponseToServer(
                    miner_hotkey=payload.miner_hotkey,
                    pod_id=payload.pod_id,
                    executor_id=payload.executor_id,
                    container_name=payload.container_name,
                    logs=msg.logs
                )

            elif isinstance(msg, FailedRequest):
                log_text = _m(
                    "Error: FailedRequest",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )
                logger.error(log_text)

                return FailedGetPodLogs(
                    miner_hotkey=payload.miner_hotkey,
                    pod_id=payload.pod_id,
                    executor_id=payload.executor_id,
                    container_name=payload.container_name,
                    msg=log_text.to_full_string(),
                )

            else:
                log_text = _m(
                    "Error: Unexpected msg",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )
                logger.error(log_text)

                return FailedGetPodLogs(
                    miner_hotkey=payload.miner_hotkey,
                    pod_id=payload.pod_id,
                    executor_id=payload.executor_id,
                    container_name=payload.container_name,
                    msg=log_text.to_full_string(),
                )

        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text)

            return FailedGetPodLogs(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                container_name=payload.container_name,
                msg=log_text.to_full_string(),
            )

    async def _add_debug_ssh_key(self, payload: AddDebugSshKeyRequest) -> DebugSshKeyAdded:
        """REST API version of add_debug_ssh_key."""
        my_key: bittensor.Keypair = settings.get_bittensor_wallet().get_hotkey()
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_id": payload.executor_id,
            "executor_ip": payload.miner_address,
            "executor_port": payload.miner_port,
        }

        try:
            base_url = f"http://{payload.miner_address}:{payload.miner_port}"
            headers = self._generate_auth_headers(my_key, payload.miner_hotkey)
            headers["Content-Type"] = "application/json"
            
            ssh_request = SSHPubKeySubmitRequest(
                public_key=payload.public_key,
                validator_signature=self._sign_validator_pubkey(my_key, payload.public_key),
                executor_id=payload.executor_id,
                is_rental_request=False,
                miner_hotkey=payload.miner_hotkey,
            )

            logger.info(
                _m("Sent SSH key to miner via REST API.", extra=get_extra_info(default_extra)),
            )

            status, response_data = await self._make_rest_request(
                method="POST",
                url=f"{base_url}/api/validator/ssh-pubkey-submit",
                json_data=self._serialize_request(ssh_request),
                headers=headers,
                timeout=REST_SSH_SUBMIT_TIMEOUT,
                log_extra=default_extra,
                operation_name="SSH key submit (debug)",
            )

            if status != 200 or response_data is None:
                error_msg = "Failed to add debug public key"
                if response_data:
                    error_msg = f"{error_msg}: {response_data}"
                log_text = _m(
                    "Error: Failed to add debug public key",
                    extra=get_extra_info({**default_extra, "error": error_msg}),
                )
                logger.error(log_text)
                return FailedAddDebugSshKey(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    msg=log_text.to_full_string(),
                )

            msg = _parse_miner_response(response_data)

            # Track whether SSH key was successfully accepted
            ssh_key_accepted = False

            if isinstance(msg, AcceptSSHKeyRequest):
                ssh_key_accepted = True
                logger.info(
                    _m(
                        "Received AcceptSSHKeyRequest via REST API",
                        extra=get_extra_info({**default_extra, "msg": str(msg)}),
                    ),
                )

                try:
                    executor = msg.executors[0]
                except Exception as e:
                    executor = None

                if executor is None or executor.uuid != payload.executor_id:
                    log_text = _m("Error: Invalid executor id", extra=get_extra_info(default_extra))
                    logger.error(log_text)

                    # Remove SSH key only if it was accepted
                    if ssh_key_accepted:
                        await self._remove_ssh_key_via_rest(
                            base_url=base_url,
                            my_key=my_key,
                            public_key=payload.public_key,
                            miner_hotkey=payload.miner_hotkey,
                            executor_id=payload.executor_id,
                            log_extra=default_extra,
                        )

                    return FailedAddDebugSshKey(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        msg=log_text.to_full_string(),
                    )

                logger.info(
                    _m(
                        "Added debug public key",
                        extra=get_extra_info(default_extra),
                    ),
                )

                return DebugSshKeyAdded(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    address=executor.address,
                    port=executor.port,
                    ssh_username=executor.ssh_username,
                    ssh_port=executor.ssh_port,
                )

            else:
                log_text = _m(
                    "Error: Failed to add debug public key",
                    extra=get_extra_info({**default_extra, "msg": str(msg)}),
                )
                logger.error(log_text)

                return FailedAddDebugSshKey(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    msg=log_text.to_full_string(),
                )

        except Exception as e:
            log_text = _m(
                "Resulted in an exception",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return FailedAddDebugSshKey(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                msg=log_text.to_full_string(),
            )

MinerServiceDep = Annotated[MinerService, Depends(MinerService)]
