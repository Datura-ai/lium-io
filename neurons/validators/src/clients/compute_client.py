import asyncio
import json
import logging
import time
from contextlib import suppress
from contextvars import ContextVar
from typing import NoReturn
from uuid import uuid4

import aiohttp
import bittensor
import pydantic
import tenacity
import websockets
from payload_models.payloads import (
    BackupContainerRequest,
    DeliveryStamps,
    CancelStorageOperationRequest,
    RestoreContainerRequest,
    BaseServerRequest,
    BaseValidatorResponse,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    ContainerStartRequest,
    ContainerStopRequest,
    AddSshPublicKeyRequest,
    RemoveSshPublicKeysRequest,
    ContainerCreated,
    ContainerStarted,
    ContainerStopped,
    SshPubKeyAdded,
    SshPubKeyRemoved,
    ContainerDeleted,
    DuplicateExecutorsResponse,
    FailedContainerErrorCodes,
    FailedContainerErrorTypes,
    FailedContainerRequest,
    ExecutorRentFinishedRequest,
    ForcedValidationCycleRequest,
    GetEstimateRequest,
    GetPodLogsRequestFromServer,
    PodLogsResponseToServer,
    FailedGetPodLogs,
    AddDebugSshKeyRequest,
    DebugSshKeyAdded,
    FailedAddDebugSshKey,
    InstallJupyterServerRequest,
    PodGpuResetResult,
    ResetPodGpuRequest,
    JupyterServerInstalled,
    JupyterInstallationFailed,
)
from protocol.vc_protocol.compute_requests import (
    Error,
    ExecutorUptimeResponse,
    RentedMachineResponse,
    Response,
    RevenuePerGpuTypeResponse,
)
from protocol.vc_protocol.validator_requests import (
    AuthenticateRequest,
    DuplicateExecutorsRequest,
    EstimateResponse,
    ExecutorSpecRequest,
    GpuEstimatesRequest,
    InspectorEventRequest,
    LogStreamRequest,
    PodStatesReport,
    RentedMachineRequest,
    ResetVerifiedJobEvidence,
    ResetVerifiedJobRequest,
    NormalizedScoreRequest,
    RevenuePerGpuTypeRequest,
    ScorePortionPerGpuTypeRequest,
)
from websockets.asyncio.client import ClientConnection

from core.config import settings
from core.utils import _m, get_extra_info
from clients.subtensor_client import SubtensorClient
from services.docker_service import create_steps_after_reply, inflight_creates
from services.gpu_reset import renter_gpu_reset_in_progress, renter_gpu_reset_recent, reset_caused
from services.miner_service import MinerService
from incentive.rental_price import ExecutorEstimateParams, RentalPriceSnapshot, estimate_executor
from services.redis_service import (
    BackendCommand,
    DUPLICATED_MACHINE_SET,
    EXECUTORS_UPTIME_PREFIX,
    GPU_ESTIMATES_CHANNEL,
    INSPECTOR_EVENT_CHANNEL,
    RENTAL_SUCCEED_MACHINE_SET,
    MACHINE_SPEC_CHANNEL,
    POD_STATES_CHANNEL,
    RENTED_MACHINE_PREFIX,
    RESET_VERIFIED_JOB_CHANNEL,
    STREAMING_LOG_CHANNEL,
    NORMALIZED_SCORE_CHANNEL,
)
from clients.handlers.backup_handler import BackupHandler

logger = logging.getLogger(__name__)

# DAH-2792: the keepalive pong queues behind a scoring-cycle burst and missed the 20 s default;
# mirrors ws_ping_* in compute-app apps/server/src/core/uvicorn_worker.py, which holds the burst.
WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 40
# not an executor step, so the backend never counts it against the host
CREATE_CUT_OFF_BY_RESTART_STEP = "validator_restarted"


command_responses: ContextVar[list[str] | None] = ContextVar("command_responses", default=None)


class OutgoingMessages(list[DeliveryStamps]):
    # every append wakes the send loop, so a reply leaves at once instead of on the next 1 s poll
    def __init__(self) -> None:
        super().__init__()
        self.appended = asyncio.Event()

    def append(self, message: DeliveryStamps) -> None:
        if (responses := command_responses.get()) is not None:
            responses.append(message.model_dump_json())
        super().append(message)
        self.appended.set()


class AuthenticationError(Exception):
    def __init__(self, reason: str, errors: list[Error]):
        self.reason = reason
        self.errors = errors


class ComputeClient:
    HEARTBEAT_PERIOD = 60

    def __init__(
        self, keypair: bittensor.Keypair, compute_app_uri: str, miner_service: MinerService
    ):
        self.keypair = keypair
        self.ws: ClientConnection | None = None
        self.compute_app_uri = compute_app_uri
        self.compute_app_rest_api_uri = compute_app_uri.replace("wss", "https").replace("ws", "http")
        self.miner_drivers = asyncio.Queue()
        self.miner_driver_awaiter_task = asyncio.create_task(self.miner_driver_awaiter())
        # self.heartbeat_task = asyncio.create_task(self.heartbeat())
        self.miner_service = miner_service
        self.message_queue = OutgoingMessages()
        self.lock = asyncio.Lock()

        self.logging_extra = {
            "validator_hotkey": self.my_hotkey(),
            "compute_app_uri": compute_app_uri,
        }

        # initiate handlers
        self.backup_handler = BackupHandler(self)

    def accepted_request_type(self) -> type[BaseServerRequest]:
        return BaseServerRequest

    def connect(self):
        """Create an awaitable/async-iterable websockets.connect() object"""
        logger.info(
            _m(
                "Connecting to backend app",
                extra=get_extra_info(self.logging_extra)
            )
        )
        return websockets.connect(
            self.compute_app_uri,
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=WS_PING_TIMEOUT,
        )

    async def miner_driver_awaiter(self):
        """avoid memory leak by awaiting miner driver tasks"""
        while True:
            task = await self.miner_drivers.get()
            if task is None:
                return

            try:
                await task
            except Exception as exc:
                logger.error(
                    _m(
                        "Error occurred during driving a miner client",
                        extra=get_extra_info({
                            **self.logging_extra,
                            "error": str(exc)
                        }),
                    )
                )

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.miner_drivers.put(None)
        await self.miner_driver_awaiter_task
        # the drivers' creates replied before their key removal and session close; finish those too
        await create_steps_after_reply.wait_until_all_done()

        # Cleanup subtensor client
        if hasattr(self, 'subtensor_client'):
            await SubtensorClient.shutdown()

    def my_hotkey(self) -> str:
        return self.keypair.ss58_address

    async def run_forever(self) -> NoReturn:
        self.subtensor_client = await SubtensorClient.initialize(chain_reads_in_thread=True)

        try:
            await self.fail_creates_cut_off_by_restart()
        except Exception as exc:  # the sweep is best-effort; it must never keep the connector from starting
            logger.error(
                _m(
                    "Failed to report creates cut off by a connector restart",
                    extra=get_extra_info({**self.logging_extra, "error": str(exc)}),
                )
            )
        try:
            await self.miner_service.redis_service.clear_running_backend_commands()
        except Exception:
            logger.exception("Failed to clear running backend commands on connector start")
        asyncio.create_task(self.handle_send_messages())
        asyncio.create_task(self.subscribe_mesages_from_redis())
        asyncio.create_task(self.poll_rented_machines())
        asyncio.create_task(self.poll_executors_uptime())
        asyncio.create_task(self.poll_revenue_per_gpu_type())

        max_delay = 60

        while True:
            reconnect_delay = 1
            try:
                async for ws in self.connect():
                    try:
                        logger.info(
                            _m(
                                "Connected to backend app",
                                extra=get_extra_info(self.logging_extra),
                            )
                        )
                        await self.handle_connection(ws)
                    except websockets.ConnectionClosed as exc:
                        self.ws = None
                        logger.warning(
                            _m(
                                f"validator connection to backend app closed, reconnecting in {reconnect_delay}s...",
                                extra=get_extra_info(
                                    {
                                        **self.logging_extra,
                                        "retry_in": reconnect_delay,
                                        "error": str(exc),
                                    }
                                ),
                            )
                        )
                    except asyncio.CancelledError:
                        self.ws = None
                        logger.warning(
                            _m(
                                "Facilitator client received cancel, stopping",
                                extra=get_extra_info(self.logging_extra),
                            )
                        )
                        raise
                    except Exception as exc:
                        self.ws = None
                        logger.error(
                            _m(
                                "Error connecting to compute app, retrying...",
                                extra=get_extra_info(
                                    {
                                        **self.logging_extra,
                                        "error": str(exc),
                                        "retry_in": reconnect_delay,
                                    }
                                ),
                            ),
                            exc_info=True,
                        )
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, max_delay)
            except Exception:
                logger.error(
                    _m(
                        "Error connecting to compute app, retrying...",
                        extra=get_extra_info(self.logging_extra),
                    ),
                    exc_info=True,
                )

                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay)

    async def handle_connection(self, ws: ClientConnection):
        """handle a single websocket connection"""
        await ws.send(AuthenticateRequest.from_keypair(self.keypair).model_dump_json())

        raw_msg = await ws.recv()
        try:
            response = Response.model_validate_json(raw_msg)
        except pydantic.ValidationError as exc:
            raise AuthenticationError(
                "did not receive Response for AuthenticationRequest", []
            ) from exc
        if response.status != "success":
            raise AuthenticationError("auth request received failed response", response.errors)

        self.ws = ws

        async for raw_msg in ws:
            await self.handle_message(raw_msg)

    async def subscribe_mesages_from_redis(self):
        validator_hotkey = self.my_hotkey()

        while True:
            pubsub = None
            try:
                pubsub = await self.miner_service.redis_service.subscribe(
                    MACHINE_SPEC_CHANNEL,
                    STREAMING_LOG_CHANNEL,
                    INSPECTOR_EVENT_CHANNEL,
                    RESET_VERIFIED_JOB_CHANNEL,
                    NORMALIZED_SCORE_CHANNEL,
                    GPU_ESTIMATES_CHANNEL,
                    POD_STATES_CHANNEL,
                )
                async for message in pubsub.listen():
                    try:
                        channel = message['channel'].decode('utf-8')
                        data = json.loads(message['data'])
                    except Exception:
                        continue

                    logger.info(
                        _m(
                            'Received message from redis',
                            extra=get_extra_info({
                                **self.logging_extra,
                                "channel": channel,
                            }),
                        )
                    )

                    if channel == MACHINE_SPEC_CHANNEL:
                        logger.info(
                            _m(
                                "Received machine specs from redis",
                                extra=get_extra_info({
                                    **self.logging_extra,
                                    "job_batch_id": data['job_batch_id'],
                                    "miner_hotkey": data["miner_hotkey"],
                                    "executor_uuid": data["executor_uuid"],
                                    "actual_score": data["score"],
                                    "synthetic_job_score": data["synthetic_job_score"],
                                }),
                            )
                        )
                        specs = ExecutorSpecRequest(
                            specs=data["specs"],
                            score=data["score"],
                            synthetic_job_score=data["synthetic_job_score"],
                            log_status=data["log_status"],
                            job_batch_id=data['job_batch_id'],
                            netuid=data.get("netuid"),
                            scored_at=data.get("scored_at"),
                            incentive=data.get("incentive"),
                            incentive_rented=data.get("incentive_rented"),
                            incentive_idle=data.get("incentive_idle"),
                            incentive_source=data.get("incentive_source"),
                            node_state_at_cycle=data.get("node_state_at_cycle"),
                            incentive_formula_version=data.get("incentive_formula_version"),
                            incentive_formula_inputs=data.get("incentive_formula_inputs"),
                            log_text=data["log_text"],
                            validation_event=data.get("validation_event"),
                            incentive_reasons=data.get("incentive_reasons"),
                            miner_hotkey=data["miner_hotkey"],
                            miner_coldkey=data["miner_coldkey"],
                            validator_hotkey=validator_hotkey,
                            executor_uuid=data["executor_uuid"],
                            executor_ip=data["executor_ip"],
                            executor_port=data["executor_port"],
                            executor_ssh_port=data["executor_ssh_port"],
                            price_per_gpu=data["price_per_gpu"],
                            collateral_deposited=data["collateral_deposited"],
                            ssh_pub_keys=data["ssh_pub_keys"],
                            tee_type=data.get("tee_type"),
                            attestation_digest=data.get("attestation_digest"),
                            tdx_attestation_passed=data.get("tdx_attestation_passed"),
                            gpu_attestation_passed=data.get("gpu_attestation_passed"),
                            executor_image=data.get("executor_image"),
                            sent_at=data.get("sent_at"),
                            batch_total=data.get("batch_total"),
                            availability_errors=data.get("availability_errors"),
                            pod_states=data.get("pod_states"),
                            pod_ssh=data.get("pod_ssh"),
                            ssh_host_key=data.get("ssh_host_key"),
                        )

                        async with self.lock:
                            self.message_queue.append(specs)
                    elif channel == POD_STATES_CHANNEL:
                        # DAH-3338: one chunk of the cycle's container states, published after the spec
                        report = PodStatesReport(
                            validator_hotkey=validator_hotkey,
                            miner_hotkey=data["miner_hotkey"],
                            executor_uuid=data["executor_uuid"],
                            job_batch_id=data["job_batch_id"],
                            chunk_index=data["chunk_index"],
                            chunk_total=data["chunk_total"],
                            pod_states=data["pod_states"],
                            sent_at=data.get("sent_at"),
                        )

                        async with self.lock:
                            self.message_queue.append(report)
                    elif channel == STREAMING_LOG_CHANNEL:
                        log_stream = LogStreamRequest(
                            logs=data["logs"],
                            miner_hotkey=data["miner_hotkey"],
                            validator_hotkey=validator_hotkey,
                            executor_uuid=data["executor_uuid"],
                            pod_id=data["pod_id"],
                        )

                        async with self.lock:
                            self.message_queue.append(log_stream)
                    elif channel == INSPECTOR_EVENT_CHANNEL:
                        inspector_event = InspectorEventRequest(
                            validator_hotkey=validator_hotkey,
                            miner_hotkey=data["miner_hotkey"],
                            executor_id=data["executor_id"],
                            job_batch_id=data["job_batch_id"],
                            pod_ids=data["pod_ids"],
                            outcome=data["outcome"],
                            reason_code=data["reason_code"],
                            report=data.get("report"),
                            error=data.get("error"),
                            context=data.get("context") or {},
                            pipeline_id=data.get("pipeline_id"),
                            trace_id=data.get("trace_id"),
                            when=data["when"],
                        )

                        async with self.lock:
                            self.message_queue.append(inspector_event)
                    elif channel == RESET_VERIFIED_JOB_CHANNEL:
                        # DAH-4658: the renter is resetting the node's GPUs, or did a moment ago; what a reset can
                        # cause is the renter's doing, never the provider's fault (an unreadable mark sends it)
                        evidence = ResetVerifiedJobEvidence.model_validate(data.get("evidence") or {})
                        reset_marks = (
                            self.miner_service.redis_service,
                            data["miner_hotkey"],
                            data["executor_uuid"],
                            evidence.pod_id,
                        )
                        if reset_caused(data.get("check_id"), data.get("reason_code"), evidence) and (
                            await renter_gpu_reset_in_progress(*reset_marks)
                            or await renter_gpu_reset_recent(*reset_marks)
                        ):
                            logger.info(
                                _m(
                                    "Verified-job reset not sent: renter GPU reset in progress",
                                    extra=get_extra_info({
                                        **self.logging_extra,
                                        "miner_hotkey": data["miner_hotkey"],
                                        "executor_uuid": data["executor_uuid"],
                                        "check_id": data.get("check_id"),
                                    }),
                                )
                            )
                            continue
                        reset_request = ResetVerifiedJobRequest(
                            miner_hotkey=data["miner_hotkey"],
                            validator_hotkey=validator_hotkey,
                            executor_uuid=data["executor_uuid"],
                            reason=data["reason"],
                            reason_code=data.get("reason_code"),
                            check_id=data.get("check_id"),
                            evidence=data.get("evidence"),
                        )

                        async with self.lock:
                            self.message_queue.append(reset_request)
                    elif channel == NORMALIZED_SCORE_CHANNEL:
                        normalized_score = NormalizedScoreRequest(
                            normalized_scores=data["normalized_scores"],
                        )

                        async with self.lock:
                            self.message_queue.append(normalized_score)
                    elif channel == GPU_ESTIMATES_CHANNEL:
                        redis_service = self.miner_service.redis_service
                        estimates = await redis_service.get_gpu_estimates()
                        if estimates:
                            async with self.lock:
                                self.message_queue.append(GpuEstimatesRequest(estimates=estimates))

            except Exception as exc:
                logger.error(
                    _m(
                        "Error: unknown",
                        extra=get_extra_info({
                            **self.logging_extra,
                            "error": str(exc),
                        }),
                    )
                )
            finally:
                # The connection goes back to the (bounded) pubsub pool only on aclose(); without
                # this every reconnect round leaked one, and the pool would drain within the hour.
                if pubsub is not None:
                    try:
                        await pubsub.aclose()
                    except Exception:
                        pass

            await asyncio.sleep(1)

    async def heartbeat(self):
        pass
        # while True:
        #     if self.ws is not None:
        #         try:
        #             await self.send_model(Heartbeat())
        #         except Exception as exc:
        #             msg = f"Error occurred while sending heartbeat: {exc}"
        #             logger.warning(msg)
        #     await asyncio.sleep(self.HEARTBEAT_PERIOD)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, exp_base=2, min=1, max=10),
        retry=tenacity.retry_if_exception_type(websockets.ConnectionClosed),
    )
    async def send_model(self, msg: DeliveryStamps):
        # stamped inside the retry so a message resent after a reconnect reports its own delay
        msg.forwarded_at = time.time()
        msg.queue_depth = len(self.message_queue)
        await self.ws.send(msg.model_dump_json())

    async def handle_send_messages(self):
        while True:
            if self.ws is None:
                await asyncio.sleep(1)
                continue

            if len(self.message_queue) > 0:
                async with self.lock:
                    log_to_send = self.message_queue.pop(0)

                if log_to_send:
                    try:
                        await self.send_model(log_to_send)
                        if getattr(log_to_send, "failure_step", None) == CREATE_CUT_OFF_BY_RESTART_STEP:
                            # kept until now so a second restart before the send reports it again
                            await self._untrack_backend_create_for_restart_report(log_to_send.pod_id)
                    except Exception as exc:
                        async with self.lock:
                            self.message_queue.insert(0, log_to_send)
                        logger.error(
                            _m(
                                "Error: message sent error",
                                extra=get_extra_info({
                                    **self.logging_extra,
                                    "error": str(exc),
                                }),
                            )
                        )
            else:
                self.message_queue.appended.clear()
                await self.message_queue.appended.wait()

    async def poll_rented_machines(self):
        while True:
            async with self.lock:
                logger.info(
                    _m(
                        "Request rented machines",
                        extra=get_extra_info(self.logging_extra),
                    )
                )
                self.message_queue.append(RentedMachineRequest())

                logger.info(
                    _m(
                        "Request duplicated machines",
                        extra=get_extra_info(self.logging_extra),
                    )
                )
                self.message_queue.append(DuplicateExecutorsRequest())

            await asyncio.sleep(10 * 60)

    async def poll_executors_uptime(self):
        while True:
            try:
                await self.get_executors_uptime()
            except Exception as exc:
                logger.error(
                    _m("Exception during get executors uptime from compute app", extra={**self.logging_extra, "error": str(exc)}),
                    exc_info=True
                )
            finally:
                await asyncio.sleep(20 * 60)

    async def poll_revenue_per_gpu_type(self):
        while True:
            async with self.lock:
                logger.info(
                    _m(
                        "Request revenue per gpu type",
                        extra=get_extra_info(self.logging_extra),
                    )
                )
                self.message_queue.append(RevenuePerGpuTypeRequest())

            await asyncio.sleep(10 * 60)  # poll every 10 mins

    async def get_executors_uptime(self):
        """Get executors uptime from compute app."""
        default_log_info = get_extra_info(self.logging_extra)
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.compute_app_rest_api_uri}/executors"
                headers = {
                    'X-Validator-Signature': f"0x{self.keypair.sign(self.my_hotkey()).hex()}"
                }
                async with session.post(url, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        logger.error(
                            _m("Failed to get executors uptime from compute app", extra={**default_log_info, "status": response.status, "response": error_msg}),
                        )
                        return

                    executors_json = await response.json()
                    executors = [ExecutorUptimeResponse(**executor) for executor in executors_json]
                    logger.info(
                        _m("Successfully got response from compute app", extra={**default_log_info, "status": len(executors), "time_cost": time.time() - start_time}),
                    )
                    # clean up old redis data
                    redis_service = self.miner_service.redis_service
                    await redis_service.delete(EXECUTORS_UPTIME_PREFIX)
                    # need to store them into redis db.
                    for executor in executors:
                        await redis_service.add_executor_uptime(executor)

                    logger.info(
                        _m("Successfully stored executors uptime into redis", extra={**default_log_info, "time_cost": time.time() - start_time}),
                    )
            except Exception as exc:
                logger.error(
                    _m("Exception during get executors uptime from compute app", extra={**default_log_info, "error": str(exc)}),
                    exc_info=True
                )

    async def handle_message(self, raw_msg: str | bytes):
        """handle message received from facilitator"""
        try:
            response = Response.model_validate_json(raw_msg)
        except pydantic.ValidationError:
            pass
        else:
            if response.status != "success":
                logger.error(
                    _m(
                        "received error response from compute app",
                        extra=get_extra_info({
                            **self.logging_extra,
                            "response": str(response)
                        }),
                    )
                )
            return

        try:
            response: RentedMachineResponse = pydantic.TypeAdapter(RentedMachineResponse).validate_json(raw_msg)
        except pydantic.ValidationError:
            pass
        else:
            logger.info(
                _m(
                    "Rented machines",
                    extra=get_extra_info({
                        **self.logging_extra,
                        "machines": len(response.machines),
                        "banned_guids": len(response.banned_guids)
                    }),
                )
            )
            redis_service = self.miner_service.redis_service
            await redis_service.delete(RENTED_MACHINE_PREFIX)
            for machine in response.machines:
                await redis_service.add_rented_machine(machine)
            await redis_service.set_banned_guids(response.banned_guids)
            return

        try:
            response: DuplicateExecutorsResponse = pydantic.TypeAdapter(DuplicateExecutorsResponse).validate_json(raw_msg)
        except pydantic.ValidationError:
            pass
        else:
            logger.info(
                _m(
                    "Duplicated executors",
                    extra=get_extra_info({
                        **self.logging_extra,
                        "executors": len(response.executors),
                        "rental_succeed_executors": len(response.rental_succeed_executors) if response.rental_succeed_executors else 0
                    }),
                )
            )

            # Reset duplicated machines
            redis_service = self.miner_service.redis_service
            await redis_service.delete(DUPLICATED_MACHINE_SET)

            for _, details_list in response.executors.items():
                for detail in details_list:
                    executor_id = detail.get("executor_id")
                    miner_hotkey = detail.get("miner_hotkey")
                    await redis_service.sadd(
                        DUPLICATED_MACHINE_SET, f"{miner_hotkey}:{executor_id}"
                    )

            # Reset rental failed machines
            await redis_service.delete(RENTAL_SUCCEED_MACHINE_SET)
            if response.rental_succeed_executors:
                for executor_uuid in response.rental_succeed_executors:
                    await redis_service.sadd(
                        RENTAL_SUCCEED_MACHINE_SET, executor_uuid
                    )

            return

        try:
            response: RevenuePerGpuTypeResponse = pydantic.TypeAdapter(RevenuePerGpuTypeResponse).validate_json(raw_msg)
        except pydantic.ValidationError:
            pass
        else:
            logger.info(
                _m(
                    "RevenuePerGpuTypeResponse",
                    extra=get_extra_info({
                        **self.logging_extra,
                        "revenues": response.revenues
                    }),
                )
            )

            portions = {}
            redis_service = self.miner_service.redis_service
            for gpu_type, revenue in response.revenues.items():
                prev_portion = await redis_service.get_portion_per_gpu_type(gpu_type)
                portion = prev_portion + settings.TIME_DELTA_FOR_EMISSION * (revenue - prev_portion)
                portions[gpu_type] = portion
                await redis_service.set_portion_per_gpu_type(gpu_type, portion)

            async with self.lock:
                self.message_queue.append(ScorePortionPerGpuTypeRequest(portions=portions))
            return

        try:
            req: GetEstimateRequest = pydantic.TypeAdapter(GetEstimateRequest).validate_json(raw_msg)
        except pydantic.ValidationError:
            pass
        else:
            await self._handle_get_estimate(req)
            return

        try:
            pydantic.TypeAdapter(ForcedValidationCycleRequest).validate_json(raw_msg)
        except pydantic.ValidationError:
            pass
        else:
            await self.handle_forced_validation_cycle()
            return

        try:
            job_request = self.accepted_request_type().parse(raw_msg)
        except Exception as ex:
            error_msg = "Invalid message received from backend"
            logger.error(
                _m(
                    error_msg,
                    extra=get_extra_info(
                        {
                            **self.logging_extra,
                            "error_type": type(ex).__name__,
                            "raw_message_length": len(raw_msg),
                        }
                    ),
                )
            )
            pass
        else:
            task = asyncio.create_task(self.miner_driver(job_request))
            await self.miner_drivers.put(task)
            return

    async def _handle_get_estimate(self, req: GetEstimateRequest):
        started_at = time.perf_counter()
        redis_service = self.miner_service.redis_service
        snapshot_started_at = time.perf_counter()
        snapshot_data = await redis_service.get_incentive_snapshot()
        if snapshot_data is None:
            logger.warning(
                _m(
                    "No incentive snapshot available for estimate request",
                    extra=get_extra_info({**self.logging_extra, "request_id": req.request_id}),
                )
            )
            return
        try:
            snapshot = RentalPriceSnapshot.model_validate(snapshot_data)
        except Exception as exc:
            logger.error(
                _m(
                    "Failed to parse incentive snapshot for estimate request",
                    extra=get_extra_info({**self.logging_extra, "error": str(exc)}),
                )
            )
            return
        snapshot_extraction_ms = round((time.perf_counter() - snapshot_started_at) * 1000, 2)
        result = await estimate_executor(
            config=settings.incentive,
            redis_service=redis_service,
            snapshot=snapshot,
            params=ExecutorEstimateParams(
                gpu_model=req.gpu_model,
                gpu_count=req.gpu_count,
                is_rented=req.is_rented,
                gpu_splitting=req.gpu_splitting,
                gpu_splitting_min_count=req.gpu_splitting_min_count,
                sysbox_runtime=req.sysbox_runtime,
                collateral_deposited=req.collateral_deposited,
            ),
        )
        async with self.lock:
            self.message_queue.append(
                EstimateResponse(
                    request_id=req.request_id,
                    estimate=result.model_dump(),
                )
            )
        logger.info(
            _m(
                "Processed estimate request",
                extra=get_extra_info(
                    {
                        **self.logging_extra,
                        "request_id": req.request_id,
                        "gpu_model": req.gpu_model,
                        "snapshot_extraction_ms": snapshot_extraction_ms,
                        "total_processing_ms": round((time.perf_counter() - started_at) * 1000, 2),
                        "is_rented": req.is_rented,
                    }
                ),
            )
        )

    async def handle_forced_validation_cycle(self) -> None:
        """Start a validation cycle now instead of waiting for the next block window.

        Staging only. The backend gates the request too; this is the second gate, so a message
        that reaches a production validator does nothing.
        """
        if settings.DEPLOY_ENV == "PROD":
            logger.warning(
                _m(
                    "Forced validation cycle is disabled in production",
                    extra=get_extra_info(self.logging_extra),
                ),
            )
            return

        await self.miner_service.request_validation_cycle_now()

    async def get_miner_axon_info(self, hotkey: str) -> bittensor.AxonInfo:
        miner = await self.subtensor_client.get_miner(hotkey)
        return miner.axon_info

    async def miner_driver(self, job_request: BaseServerRequest):
        if not job_request.command_id:
            return await self._run_miner_job(job_request)
        command_id = job_request.command_id
        redis_service = self.miner_service.redis_service
        claim = BackendCommand(state="running", owner=uuid4().hex).model_dump_json()
        try:
            existing = await redis_service.claim_backend_command(command_id, claim)
        except Exception:
            logger.exception(_m(
                "Backend command dedup failed open",
                extra=get_extra_info({**self.logging_extra, "command_id": command_id}),
            ))
            return await self._run_miner_job(job_request)
        if existing is not None:
            return await self._handle_backend_command_duplicate(job_request, existing)

        responses: list[str] = []
        token = command_responses.set(responses)
        heartbeat = asyncio.create_task(self._refresh_backend_command(command_id, claim))
        succeeded = False
        try:
            result = await self._run_miner_job(job_request)
            succeeded = True
            return result
        finally:
            command_responses.reset(token)
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            # A failure without any answer releases the claim so a retry can run.
            value = (
                BackendCommand(state="finished", responses=responses).model_dump_json()
                if succeeded or responses
                else ""
            )
            try:
                await redis_service.update_backend_command(
                    command_id, claim, value, settings.BACKEND_COMMAND_ANSWER_TTL_SECONDS
                )
            except Exception:
                logger.exception("Failed to finish backend command: %s", command_id)

    async def _handle_backend_command_duplicate(
        self, job_request: BaseServerRequest, existing: BackendCommand
    ) -> None:
        command_id = job_request.command_id
        if existing.state == "finished":
            try:
                replies = [BaseValidatorResponse.parse(raw) for raw in existing.responses]
            except Exception:
                logger.exception(_m(
                    "Backend command reply replay failed open",
                    extra=get_extra_info({**self.logging_extra, "command_id": command_id}),
                ))
                await self._run_miner_job(job_request)
                return
            async with self.lock:
                for reply in replies:
                    self.message_queue.append(reply)
        else:
            logger.info(
                _m(
                    "Ignoring running backend command duplicate",
                    extra=get_extra_info({
                        **self.logging_extra,
                        "command_id": command_id,
                        "request_type": job_request.message_type.value,
                    }),
                )
            )

    async def _refresh_backend_command(self, command_id: str, claim: str) -> None:
        ttl = settings.BACKEND_COMMAND_CLAIM_TTL_SECONDS
        interval = min(settings.BACKEND_COMMAND_HEARTBEAT_SECONDS, ttl / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                await self.miner_service.redis_service.update_backend_command(
                    command_id, claim, claim, ttl
                )
            except Exception:
                logger.exception("Failed to refresh backend command claim: %s", command_id)

    async def _run_miner_job(
        self,
        job_request: ContainerCreateRequest
        | ContainerDeleteRequest
        | ContainerStopRequest
        | ContainerStartRequest
        | AddSshPublicKeyRequest
        | RemoveSshPublicKeysRequest
        | ExecutorRentFinishedRequest
        | GetPodLogsRequestFromServer
        | AddDebugSshKeyRequest
        | BackupContainerRequest
        | RestoreContainerRequest
        | InstallJupyterServerRequest
        | ResetPodGpuRequest,
    ):
        # DAH-2728: track the create from the moment this task picks it up. The axon lookup below
        # fetches the whole miner set on a cold cache, so a delete arriving seconds later can sail
        # past it and find nothing to cancel — and the create it raced then orphans its container.
        if not isinstance(job_request, ContainerCreateRequest):
            return await self._drive_miner_job(job_request)
        with inflight_creates.track(job_request.pod_id):
            await self._track_backend_create_for_restart_report(job_request)
            try:
                return await self._drive_miner_job(job_request)
            finally:
                await self._untrack_backend_create_for_restart_report(job_request.pod_id)

    async def _track_backend_create_for_restart_report(self, job_request: ContainerCreateRequest) -> None:
        # best-effort: a Redis error loses only the restart report, never the create itself
        try:
            await self.miner_service.redis_service.add_inflight_backend_create(
                job_request.pod_id, job_request.miner_hotkey, job_request.executor_id, job_request.workload_kind
            )
        except Exception as exc:
            self._log_backend_create_tracking_error(job_request.pod_id, exc)

    async def _untrack_backend_create_for_restart_report(self, pod_id: str) -> None:
        try:
            await self.miner_service.redis_service.remove_inflight_backend_create(pod_id)
        except Exception as exc:
            self._log_backend_create_tracking_error(pod_id, exc)

    def _log_backend_create_tracking_error(self, pod_id: str, exc: Exception) -> None:
        logger.error(
            _m(
                "Failed to track a backend create for the restart report",
                extra=get_extra_info({**self.logging_extra, "pod_id": pod_id, "error": str(exc)}),
            )
        )

    async def fail_creates_cut_off_by_restart(self) -> None:
        """DAH-4698: a create the previous connector was running when it died (every lium-io deploy recreates it) never
        replied, so the backend kept the pod PENDING / the filler STARTING, and its pending-pod flag refused the
        delete for 30 min. Each one is reported failed now, which frees the node, and its flag is cleared."""
        redis_service = self.miner_service.redis_service
        creates = await redis_service.inflight_backend_creates()
        for pod_id, create in creates.items():
            # the validator wipes these flags on its own start; this covers a connector that restarted alone
            await redis_service.remove_pending_pod(create.miner_hotkey, create.executor_id, pod_id)
            self.message_queue.append(
                FailedContainerRequest(
                    miner_hotkey=create.miner_hotkey,
                    executor_id=create.executor_id,
                    pod_id=pod_id,
                    workload_kind=create.workload_kind,
                    error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                    error_code=FailedContainerErrorCodes.UnknownError,
                    msg="The create was interrupted by a validator restart. Please try again.",
                    detail=f"validator connector restarted during the create (started {create.started_at_unix_seconds})",
                    failure_step=CREATE_CUT_OFF_BY_RESTART_STEP,
                )
            )
            logger.warning(
                _m(
                    "Create cut off by a connector restart, reported failed",
                    extra=get_extra_info({**self.logging_extra, "pod_id": pod_id, **create.model_dump(mode="json")}),
                )
            )

    async def _drive_miner_job(
        self,
        job_request: ContainerCreateRequest
        | ContainerDeleteRequest
        | ContainerStopRequest
        | ContainerStartRequest
        | AddSshPublicKeyRequest
        | RemoveSshPublicKeysRequest
        | ExecutorRentFinishedRequest
        | GetPodLogsRequestFromServer
        | AddDebugSshKeyRequest
        | BackupContainerRequest
        | RestoreContainerRequest
        | InstallJupyterServerRequest
        | ResetPodGpuRequest
    ):
        """drive a miner client from job start to completion, then close miner connection"""
        logger.info(
            _m(
                f"Getting miner axon info for {job_request.miner_hotkey}",
                extra=get_extra_info(self.logging_extra),
            )
        )
        miner_axon_info = await self.get_miner_axon_info(job_request.miner_hotkey)
        logging_extra = {
            **self.logging_extra,
            "miner_hotkey": job_request.miner_hotkey,
            "miner_ip": miner_axon_info.ip,
            "miner_port": miner_axon_info.port,
            "request_type": job_request.message_type.value,
            "executor_id": str(job_request.executor_id),
        }
        logger.info(
            _m(
                "Miner driver to miner",
                extra=get_extra_info(logging_extra),
            )
        )

        job_request.miner_address = miner_axon_info.ip
        job_request.miner_port = miner_axon_info.port

        if isinstance(job_request, ContainerCreateRequest):
            logger.info(
                _m(
                    "Creating container for executor.",
                    extra=get_extra_info({**logging_extra, "job_request": str(job_request)}),
                )
            )
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                ContainerCreated | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)

            logger.info(
                _m(
                    "Sending back created container info to compute app",
                    extra=get_extra_info({**logging_extra, "response": str(response)}),
                )
            )

            async with self.lock:
                self.message_queue.append(response)
        elif isinstance(job_request, ContainerDeleteRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                ContainerDeleted | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)

            logger.info(
                _m(
                    "Sending back deleted container info to compute app",
                    extra=get_extra_info({**logging_extra, "response": str(response)}),
                )
            )

            async with self.lock:
                self.message_queue.append(response)
        elif isinstance(job_request, ContainerStopRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                ContainerStopped | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)

            logger.info(
                _m(
                    "Sending back stopped container info to compute app",
                    extra=get_extra_info({**logging_extra, "response": str(response)}),
                )
            )

            async with self.lock:
                self.message_queue.append(response)
        elif isinstance(job_request, ContainerStartRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                ContainerStarted | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)

            logger.info(
                _m(
                    "Sending back started container info to compute app",
                    extra=get_extra_info({**logging_extra, "response": str(response)}),
                )
            )

            async with self.lock:
                self.message_queue.append(response)
        elif isinstance(job_request, AddSshPublicKeyRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                SshPubKeyAdded | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)
            # the backend matches the answer to its send by this id (a delayed answer must not confirm a later send)
            response.command_id = job_request.command_id

            logger.info(
                _m(
                    "Sending back ssh key add result to compute app",
                    extra=get_extra_info({**logging_extra, "response": str(response)}),
                )
            )

            async with self.lock:
                self.message_queue.append(response)
        elif isinstance(job_request, RemoveSshPublicKeysRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                SshPubKeyRemoved | FailedContainerRequest
            ) = await self.miner_service.handle_container(job_request)

            logger.info(
                _m(
                    "Sending back ssh key remove result to compute app",
                    extra=get_extra_info({**logging_extra, "response": str(response)}),
                )
            )

            async with self.lock:
                self.message_queue.append(response)
        elif isinstance(job_request, ExecutorRentFinishedRequest):
            logger.info(
                _m(
                    "Rent finished. Clear Pending flag",
                    extra=get_extra_info(logging_extra),
                )
            )

            await self.miner_service.redis_service.remove_pending_pod(job_request.miner_hotkey, job_request.executor_id, job_request.pod_id)
        elif isinstance(job_request, GetPodLogsRequestFromServer):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                PodLogsResponseToServer | FailedGetPodLogs
            ) = await self.miner_service.get_pod_logs(job_request)

            async with self.lock:
                self.message_queue.append(response)
        elif isinstance(job_request, AddDebugSshKeyRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                DebugSshKeyAdded | FailedAddDebugSshKey
            ) = await self.miner_service.add_debug_ssh_key(job_request)

            async with self.lock:
                self.message_queue.append(response)
        elif isinstance(job_request, InstallJupyterServerRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: (
                JupyterServerInstalled | JupyterInstallationFailed
            ) = await self.miner_service.handle_container(job_request)

            async with self.lock:
                self.message_queue.append(response)
        elif isinstance(job_request, ResetPodGpuRequest):
            job_request.miner_address = miner_axon_info.ip
            job_request.miner_port = miner_axon_info.port
            response: PodGpuResetResult = await self.miner_service.handle_container(job_request)

            async with self.lock:
                self.message_queue.append(response)
        elif isinstance(job_request, BackupContainerRequest):
            await self.backup_handler.handle_backup_container_req(job_request)
        elif isinstance(job_request, RestoreContainerRequest):
            await self.backup_handler.handle_restore_container_req(job_request)
        elif isinstance(job_request, CancelStorageOperationRequest):
            await self.backup_handler.handle_cancel_storage_operation_req(job_request)
