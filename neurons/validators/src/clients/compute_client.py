import asyncio
import json
import logging
import time
from typing import NoReturn

import aiohttp
import bittensor
import pydantic
import tenacity
import websockets
from datura.requests.base import BaseRequest
from payload_models.payloads import (
    ContainerBaseRequest,
    ContainerCreateRequest,
    ContainerDeleteRequest,
    ContainerStartRequest,
    ContainerStopRequest,
    AddSshPublicKeyRequest,
    ContainerCreated,
    ContainerStarted,
    ContainerStopped,
    SshPubKeyAdded,
    ContainerDeleted,
    DuplicateExecutorsResponse,
    FailedContainerRequest,
    ExecutorRentFinishedRequest,
    GetPodLogsRequestFromServer,
    PodLogsResponseToServer,
    FailedGetPodLogs,
    AddDebugSshKeyRequest,
    DebugSshKeyAdded,
    FailedAddDebugSshKey,
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
    ExecutorSpecRequest,
    LogStreamRequest,
    RentedMachineRequest,
    ResetVerifiedJobRequest,
    NormalizedScoreRequest,
    RevenuePerGpuTypeRequest,
    ScorePortionPerGpuTypeRequest,
)
from pydantic import BaseModel
from websockets.asyncio.client import ClientConnection

from core.config import settings
from core.utils import _m, get_extra_info
from clients.subtensor_client import SubtensorClient
from services.miner_service import MinerService
from services.redis_service import (
    DUPLICATED_MACHINE_SET,
    EXECUTORS_UPTIME_PREFIX,
    RENTAL_SUCCEED_MACHINE_SET,
    MACHINE_SPEC_CHANNEL,
    RENTED_MACHINE_PREFIX,
    RESET_VERIFIED_JOB_CHANNEL,
    STREAMING_LOG_CHANNEL,
    NORMALIZED_SCORE_CHANNEL,
)

logger = logging.getLogger(__name__)


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
        self.message_queue = []
        self.lock = asyncio.Lock()

        self.logging_extra = {
            "validator_hotkey": self.my_hotkey(),
            "compute_app_uri": compute_app_uri,
        }

    def accepted_request_type(self) -> type[BaseRequest]:
        return ContainerBaseRequest

    def connect(self):
        """Create an awaitable/async-iterable websockets.connect() object"""
        logger.info(
            _m(
                "Connecting to backend app",
                extra=get_extra_info(self.logging_extra)
            )
        )
        return websockets.connect(self.compute_app_uri)

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

        # Cleanup subtensor client
        if hasattr(self, 'subtensor_client'):
            await SubtensorClient.shutdown()

    def my_hotkey(self) -> str:
        return self.keypair.ss58_address

    async def run_forever(self) -> NoReturn:
        self.subtensor_client = await SubtensorClient.initialize()

        asyncio.create_task(self.handle_send_messages())
        asyncio.create_task(self.subscribe_mesages_from_redis())
        asyncio.create_task(self.poll_rented_machines())
        asyncio.create_task(self.poll_executors_uptime())
        asyncio.create_task(self.poll_revenue_per_gpu_type())

        try:
            while True:
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
                                f"validator connection to backend app closed with code {exc.code} and reason {exc.reason}, reconnecting...",
                                extra=get_extra_info(self.logging_extra),
                            )
                        )
                    except asyncio.exceptions.CancelledError:
                        self.ws = None
                        logger.warning(
                            _m(
                                "Facilitator client received cancel, stopping",
                                extra=get_extra_info(self.logging_extra),
                            )
                        )
                    except Exception as e:
                        self.ws = None
                        logger.error(
                            _m(
                                "Error in connecting to compute app",
                                extra=get_extra_info({
                                    **self.logging_extra,
                                    "error": str(e),
                                }),
                            )
                        )

        except Exception as exc:
            self.ws = None
            logger.error(
                _m(
                    "Connecting to compute app failed",
                    extra=get_extra_info({**self.logging_extra, "error": str(exc)}),
                ),
                exc_info=True,
            )

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
            try:
                pubsub = await self.miner_service.redis_service.subscribe(
                    MACHINE_SPEC_CHANNEL,
                    STREAMING_LOG_CHANNEL,
                    RESET_VERIFIED_JOB_CHANNEL,
                    NORMALIZED_SCORE_CHANNEL,
                )
                async for message in pubsub.listen():
                    try:
                        channel = message['channel'].decode('utf-8')
                        data = json.loads(message['data'])
                    except Exception as exc:
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
                        specs = ExecutorSpecRequest(
                            specs=data["specs"],
                            score=data["score"],
                            synthetic_job_score=data["synthetic_job_score"],
                            log_status=data["log_status"],
                            job_batch_id=data['job_batch_id'],
                            log_text=data["log_text"],
                            miner_hotkey=data["miner_hotkey"],
                            miner_coldkey=data["miner_coldkey"],
                            validator_hotkey=validator_hotkey,
                            executor_uuid=data["executor_uuid"],
                            executor_ip=data["executor_ip"],
                            executor_port=data["executor_port"],
                            executor_price=data["executor_price"],
                            collateral_deposited=data["collateral_deposited"],
                            ssh_pub_keys=data["ssh_pub_keys"],
                        )

                        async with self.lock:
                            self.message_queue.append(specs)
                    elif channel == STREAMING_LOG_CHANNEL:
                        log_stream = LogStreamRequest(
                            logs=data["logs"],
                            miner_hotkey=data["miner_hotkey"],
                            validator_hotkey=validator_hotkey,
                            executor_uuid=data["executor_uuid"],
                        )

                        async with self.lock:
                            self.message_queue.append(log_stream)
                    elif channel == RESET_VERIFIED_JOB_CHANNEL:
                        reset_request = ResetVerifiedJobRequest(
                            miner_hotkey=data["miner_hotkey"],
                            validator_hotkey=validator_hotkey,
                            executor_uuid=data["executor_uuid"],
                            reason=data["reason"]
                        )

                        async with self.lock:
                            self.message_queue.append(reset_request)
                    elif channel == NORMALIZED_SCORE_CHANNEL:
                        normalized_score = NormalizedScoreRequest(
                            normalized_scores=data["normalized_scores"],
                        )

                        async with self.lock:
                            self.message_queue.append(normalized_score)

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
    async def send_model(self, msg: BaseModel):
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
                await asyncio.sleep(1)

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
            response = pydantic.TypeAdapter(DuplicateExecutorsResponse).validate_json(raw_msg)
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
            response = pydantic.TypeAdapter(RevenuePerGpuTypeResponse).validate_json(raw_msg)
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
            job_request = self.accepted_request_type().parse(raw_msg)
        except Exception as ex:
            error_msg = f"Invalid message received from backend: {str(ex)}"
            logger.error(
                _m(
                    error_msg,
                    extra=get_extra_info({**self.logging_extra, "error": str(ex), "raw_msg": raw_msg}),
                )
            )
            pass
        else:
            task = asyncio.create_task(self.miner_driver(job_request))
            await self.miner_drivers.put(task)
            return

    async def get_miner_axon_info(self, hotkey: str) -> bittensor.AxonInfo:
        return self.subtensor_client.get_miner(hotkey).axon_info

    async def miner_driver(
        self,
        job_request: ContainerCreateRequest
        | ContainerDeleteRequest
        | ContainerStopRequest
        | ContainerStartRequest
        | AddSshPublicKeyRequest
        | ExecutorRentFinishedRequest
        | GetPodLogsRequestFromServer
        | AddDebugSshKeyRequest
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
            "job_request": str(job_request),
            "executor_id": str(job_request.executor_id),
        }
        logger.info(
            _m(
                "Miner driver to miner",
                extra=get_extra_info(logging_extra),
            )
        )

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

            logger.info(
                _m(
                    "Sending back ssh key add result to compute app",
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

            await self.miner_service.redis_service.remove_pending_pod(job_request.miner_hotkey, job_request.executor_id)
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
