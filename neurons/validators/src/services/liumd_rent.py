"""liumd: a customer rent carried out by an agent on the node (DAH-3980) — STAGING PROTOTYPE, NOT THE PRODUCT.

The agent keeps a socket open to compute-app; the connector builds the rent from the request it already has,
sends it on its own compute-app socket as `LiumdRentRequest`, and compute-app relays the frames both ways
(untracked/epics/fast-rent/design-liumd-rent-path.md §2–§3). No miner key exchange and no SSH session sit on the
rent's path; the inspector start runs after the reply (MinerService._start_inspector_after_agent_rent).

A `rent` frame carries the wrapped volume passphrase and the renter's keys: no frame is ever logged.
"""

import asyncio
import base64
import contextlib
import logging
import re
import secrets
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from uuid import UUID, uuid4

from docker.utils import parse_bytes

from core.config import settings
from core.utils import _m, get_extra_info
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    CustomOptions,
    FailedContainerErrorCodes,
    FailedContainerErrorTypes,
    FailedContainerRequest,
    LiumdAgentFrame,
    LiumdRentRequest,
    ProfilerStep,
    ProfilerStepName,
    VolumeEncryptionStatus,
    WorkloadKind,
    now_ms,
)
from services.docker_service import (
    _LOOPBACK_PLUGIN_ALIAS,
    _VOLUME_SETUP_EXEC_FAILURES,
    _VOLUME_SETUP_TMPFS,
    DockerService,
    _build_gocryptfs_setup_and_mount_script,
    _build_volume_setup_exec_script,
    _opaque_shell_name,
    _xor_wrap_passphrase,
    create_steps_after_reply,
    inflight_creates,
)
from services.gpu_power_limit import MIN_POWER_LIMIT_RATIO, read_gpu_power_restore_records
from services.redis_service import STREAMING_LOG_CHANNEL, RedisService
from services.rental_docker_sdk import GpuDockerConfig, build_environment_exec_spec
from services.volume_keys import VolumeKeyDeriver

logger = logging.getLogger(__name__)

# staging: the bench node only
LIUMD_EXECUTOR_IDS = frozenset({"df044c30-b8b4-4f4f-8860-9d451c16090c"})
# the agent's own budget for a rent, from the moment it reads the frame
LIUMD_RENT_DEADLINE_MS = 30_000
# the connector's wait past the agent's deadline before it cancels the rent and fails it
LIUMD_REPLY_GRACE_SECONDS = 5.0
# ponytail: inflight_creates has a flag and no event, so a waiting rent looks at it this often; an event
# on _InflightCreate would let the wait wake on the delete itself
LIUMD_DELETE_POLL_SECONDS = 0.1
LOG_TAG = "container_creation"
# the agents refuse a rent whose `validator_hotkey` is not this shape (liumd-rent agent.rs, check_rent_frame)
SS58_ADDRESS_PATTERN = re.compile(r"[1-9A-HJ-NP-Za-km-z]{47,48}")
# the agent's step names (design §2.3), each a `liumd <step>` profiler row; any other step a reply names is dropped
LIUMD_STEP_NAMES = frozenset(
    step.value.removeprefix("liumd ")
    for step in ProfilerStepName
    if step.value.startswith("liumd ") and step != ProfilerStepName.LIUMD_ROUND_TRIP
)
# a failed exec reports the step its exit code maps to (the rent's `exit_codes`)
LIUMD_FAILED_STEP_NAMES = LIUMD_STEP_NAMES | {step for step, _ in _VOLUME_SETUP_EXEC_FAILURES.values()}
# the `error` codes of design §2.3; any other code is reported as `unknown`
LIUMD_ERROR_CODES = frozenset(
    {"not_connected", "send_failed", "reply_lost", "protocol", "ineligible", "busy", "step_failed", "deadline", "cancelled"}
)
# compute-app's codes for a rent it may have handed to the agent: they cannot know the host's state, whatever
# their flags say
LIUMD_HOST_UNKNOWN_CODES = frozenset({"send_failed", "reply_lost"})
# the connector's word on an attempt, kept as `<state>|<executor_id>|<pod_id>` for the agent's
# `attempt_state unacked` and for a `result` that comes after its rent gave up (design §3)
ATTEMPT_ACCEPTING = "accepting"
ATTEMPT_ACKED = "acked"
ATTEMPT_FAILED = "failed"
ATTEMPT_MARK_TTL_SECONDS = 3600
ATTEMPT_ACKED_MARK_TRIES = 3
# the protocol's agent frames and attempt states (design §2.1, §2.3): only these are logged as they came
LIUMD_AGENT_FRAME_TYPES = frozenset({"hello", "ping", "attempt_state", "step", "result", "error"})
LIUMD_ATTEMPT_STATES = frozenset({"rolling_back", "rolled_back", "unacked", "rollback_failed"})


def liumd_ineligibility_reason(payload: ContainerCreateRequest) -> str | None:
    # why a rent of a listed executor keeps today's path (design §6), None when the agent may take it
    custom_options = CustomOptions.sanitize(payload.custom_options)
    failed_checks = {
        "not_customer_rental": payload.workload_kind != WorkloadKind.CUSTOMER_RENTAL,
        "no_volume_encryption": not (payload.enable_volume_encryption and settings.ENABLE_VOLUME_ENCRYPTION),
        "not_sysbox": not payload.is_sysbox,
        # the image starts sshd itself only when its own CMD and ENTRYPOINT run
        "image_does_not_manage_services": not payload.ships_sshd
        or bool((custom_options.startup_commands or "").strip())
        or bool((custom_options.entrypoint or "").strip()),
        # the Jupyter URL names the executor's address, which only the miner's key exchange gives
        "jupyter": bool(payload.enable_jupyter),
        "edit_of_existing_pod": bool(payload.local_volume),
        "no_volume_limit": not payload.volume_limit_gb,
        "whole_node_gpus": not payload.gpu_uuids,
        # the agent removes the port-check containers `container_<miner_hotkey>_*` by this hotkey
        "miner_hotkey_not_ss58": not SS58_ADDRESS_PATTERN.fullmatch(payload.miner_hotkey),
        "no_public_keys": not payload.user_public_keys,
        "external_volume": payload.external_volume_info is not None,
        "bootstrap_restore": payload.bootstrap_restore is not None,
        "restore_path": bool(payload.restore_path),
        "backup_log_id": bool(payload.backup_log_id),
        "cluster_membership": payload.cluster_membership is not None,
        "custom_dockerfile": bool(payload.dockerfile_content),
        "registry_credentials": bool(payload.docker_username),
        "cache_volumes": bool(payload.cache_volumes),
        "gpu_power_limits": bool(payload.gpu_power_limits),
        # the inspector start after an agent rent takes the REST key exchange (staging's mode)
        "miner_websocket_mode": not settings.USE_REST_API,
    }
    return next((reason for reason, failed in failed_checks.items() if failed), None)


class LiumdAgentFrames:
    """The connector's side of compute-app's relay: frames out, agent frames back to the waiting rent."""

    def __init__(self) -> None:
        # set by ComputeClient; False when the compute-app socket is down and nothing can be sent
        self.send_to_backend: Callable[[LiumdRentRequest], bool] | None = None
        # set by LiumdRentService: where the connector's word on each attempt is kept (design §3)
        self.redis_service: RedisService | None = None
        self._waiters_by_attempt: dict[str, _ReplyWaiter] = {}
        self._settle_tasks: set[asyncio.Task] = set()

    def send(self, request: LiumdRentRequest) -> bool:
        return self.send_to_backend is not None and self.send_to_backend(request)

    def is_rent_without_waiter(self, request: LiumdRentRequest) -> bool:
        # a `rent` still queued after its waiter gave up (timeout, cancel) must never reach the agent
        return request.frame.get("type") == "rent" and request.frame.get("attempt") not in self._waiters_by_attempt

    @contextlib.contextmanager
    def waiting_for(self, executor_id: str, pod_id: str, attempt: str) -> Iterator[asyncio.Queue[dict]]:
        waiter = _ReplyWaiter(executor_id=executor_id, pod_id=pod_id, replies=asyncio.Queue())
        self._waiters_by_attempt[attempt] = waiter
        try:
            yield waiter.replies
        finally:
            del self._waiters_by_attempt[attempt]

    def deliver(self, message: LiumdAgentFrame) -> None:
        # called from the compute-app receive loop: a bad frame is dropped here and never raises into it
        try:
            self._deliver(message)
        except Exception as exc:
            logger.warning(
                _m("liumd frame dropped", extra=get_extra_info({"executor_id": message.executor_id, "error": repr(exc)}))
            )

    def _deliver(self, message: LiumdAgentFrame) -> None:
        frame = message.frame
        rent_id, attempt = frame.get("rent_id"), frame.get("attempt")
        log_fields = {
            "executor_id": message.executor_id,
            "frame_type": _known_or_unknown(frame.get("type"), LIUMD_AGENT_FRAME_TYPES),
            "rent_id": rent_id if isinstance(rent_id, str) else None,
            "attempt": attempt if isinstance(attempt, str) else None,
        }
        if not (isinstance(rent_id, str) and isinstance(attempt, str)):
            logger.info(_m("liumd frame dropped: rent_id or attempt is not a string", extra=get_extra_info(log_fields)))
            return
        if frame.get("type") == "attempt_state":
            state = _known_or_unknown(frame.get("state"), LIUMD_ATTEMPT_STATES)
            logger.info(_m("liumd attempt_state", extra=get_extra_info({**log_fields, "state": state})))
            # a waiting rent settles its own attempt
            if state == "unacked" and attempt not in self._waiters_by_attempt:
                self._start_settle(message.executor_id, rent_id, attempt, late_result=False)
            return
        waiter = self._waiters_by_attempt.get(attempt)
        if waiter is None and frame.get("type") == "result":
            # its rent gave up (timeout, cancel): the agent keeps the pod until it hears `settle`
            self._start_settle(message.executor_id, rent_id, attempt, late_result=True)
            return
        if waiter is None or (waiter.executor_id, waiter.pod_id, waiter.pod_id) != (message.executor_id, message.pod_id, rent_id):
            logger.info(_m("liumd frame for no waiting rent", extra=get_extra_info(log_fields)))
            return
        waiter.replies.put_nowait(frame)

    def _start_settle(self, executor_id: str, rent_id: str, attempt: str, *, late_result: bool) -> None:
        task = asyncio.ensure_future(self._settle(executor_id, rent_id, attempt, late_result=late_result))
        self._settle_tasks.add(task)
        task.add_done_callback(self._settle_tasks.discard)

    async def _settle(self, executor_id: str, rent_id: str, attempt: str, *, late_result: bool) -> None:
        # an attempt the agent holds with no rent waiting on it: answered by the connector's mark of that attempt
        log_extra = {"executor_id": executor_id, "rent_id": rent_id, "attempt": attempt, "late_result": late_result}
        try:
            mark = await self.redis_service.get(attempt_mark_key(attempt))
        except Exception as exc:
            # unanswered, the agent keeps the attempt and reports it again on its next hello
            logger.warning(_m("liumd settle skipped: attempt mark unreadable", extra=get_extra_info({**log_extra, "error": repr(exc)})))
            return
        mark_fields = mark.decode().split("|") if mark else []
        if mark_fields and mark_fields[1:] != [executor_id, rent_id]:
            logger.warning(_m("liumd settle skipped: attempt marked for another executor or pod", extra=get_extra_info(log_extra)))
            return
        state = mark_fields[0] if mark_fields else None
        if state in (None, ATTEMPT_FAILED):
            action = "rollback"
        elif state == ATTEMPT_ACKED and not late_result:
            action = "ack"
        else:
            # `accepting`: the rent is still handing the pod on, the agent asks again at its next hello;
            # `acked` for a repeated result: the pod is already handed on
            action = None
        logger.info(_m("liumd settle", extra=get_extra_info({**log_extra, "mark": state, "action": action})))
        if action is None:
            return
        self.send(
            LiumdRentRequest(
                executor_id=executor_id,
                pod_id=rent_id,
                frame=_settle_frame(rent_id, attempt, action),
            )
        )


@dataclass
class _ReplyWaiter:
    # the one source a rent's replies are accepted from
    executor_id: str
    pod_id: str
    replies: asyncio.Queue[dict]


@dataclass
class _RentProgress:
    # how far a rent got, for what its exit must undo
    log_extra: dict
    attempt: str | None = None
    pending_marked: bool = False
    sent: bool = False


# In-process like inflight_creates: the rent and the compute-app socket live in the one connector loop.
liumd_agent_frames = LiumdAgentFrames()


class LiumdRentService:
    def __init__(
        self,
        redis_service: RedisService,
        docker_service: DockerService,
        agent_frames: LiumdAgentFrames = liumd_agent_frames,
    ) -> None:
        self.redis_service = redis_service
        self.docker_service = docker_service
        self.agent_frames = agent_frames
        agent_frames.redis_service = redis_service

    async def rent_through_agent(
        self, payload: ContainerCreateRequest
    ) -> ContainerCreated | FailedContainerRequest | None:
        # the rent's reply from the node's agent; None when today's path must take the rent (design §3)
        if payload.executor_id not in LIUMD_EXECUTOR_IDS:
            return None
        progress = _RentProgress(
            log_extra={"miner_hotkey": payload.miner_hotkey, "executor_id": payload.executor_id, "pod_id": payload.pod_id}
        )
        outcome: ContainerCreated | FailedContainerRequest | None = None
        try:
            outcome = await self._rent(payload, progress)
        except asyncio.CancelledError:
            if progress.sent:
                self._abandon_attempt(payload, progress.attempt)
            raise
        except Exception as exc:
            if not progress.sent:
                # the agent never heard of this rent
                _log_rent_result(progress.log_extra, attempt=progress.attempt, path="today", reason=f"connector_error_{type(exc).__name__}")
                return None
            self._abandon_attempt(payload, progress.attempt)
            _log_rent_result(progress.log_extra, attempt=progress.attempt, path="failed", reason=f"connector_error_{type(exc).__name__}")
            outcome = _failed(payload, "The node agent's rent failed", "liumd_connector_error")
        finally:
            if not isinstance(outcome, ContainerCreated):
                await self._release_failed_rent(payload, progress)
        return outcome

    async def _rent(
        self, payload: ContainerCreateRequest, progress: _RentProgress
    ) -> ContainerCreated | FailedContainerRequest | None:
        # rent_through_agent's work; `progress` tells it what an exit must undo
        started_ms = now_ms()
        reason = liumd_ineligibility_reason(payload)
        if reason is None and await self.redis_service.renting_in_progress(
            payload.miner_hotkey, payload.executor_id, payload.pod_id
        ):
            # today's path declines it with RentingInProgress
            reason = "renting_in_progress"
        if reason is not None:
            _log_rent_result(progress.log_extra, attempt=None, path="today", reason=reason)
            return None

        custom_options = CustomOptions.sanitize(payload.custom_options)
        port_maps, _ = await self.docker_service.generate_portMappings(
            payload.miner_hotkey,
            payload.executor_id,
            UUID(payload.pod_id),
            custom_options.internal_ports,
            custom_options.initial_port_count,
            payload.enable_jupyter,
            payload.available_ports,
            payload.pod_mapping,
            payload.workload_kind,
        )
        if not port_maps:
            _log_rent_result(progress.log_extra, attempt=None, path="today", reason="no_port_maps")
            return None
        port_maps_generated_ms = now_ms()

        attempt = progress.attempt = secrets.token_hex(8)
        log_extra = progress.log_extra = {**progress.log_extra, "attempt": attempt}
        rent_frame = await self._build_rent_frame(payload, custom_options, attempt, port_maps, log_extra)
        # the same exclusion today's create holds: the rental probe stays off the node, a second create waits
        async with self.redis_service.executor_create_exclusion(payload.executor_id):
            if inflight_creates.is_cancelled(payload.pod_id):
                _log_rent_result(log_extra, attempt=attempt, path="failed", reason="cancelled_by_delete")
                return _failed(payload, "Create cancelled by an in-flight delete", "cancelled_by_delete")
            # PortConnectivityCheck tolerates the port-check containers the agent removes while this is set
            progress.pending_marked = True
            await self.redis_service.add_pending_pod(payload.miner_hotkey, payload.executor_id, payload.pod_id)
            with self.agent_frames.waiting_for(payload.executor_id, payload.pod_id, attempt) as replies:
                sent_ms = now_ms()
                if not self._send_frame(payload, rent_frame):
                    _log_rent_result(log_extra, attempt=attempt, path="today", reason="compute_app_socket_down")
                    return None
                progress.sent = True
                ssh_port = next(external for docker, _, external in port_maps if docker == 22)
                await self._stream_logs(payload, [(f"Port mappings ready: 22->{ssh_port}", "success")])
                reply, steps_seen, cancel_sent = await self._wait_for_reply(payload, attempt, replies)
                replied_ms = now_ms()
                # a delete that landed while the reply was on its way wins over the reply
                cancelled_by_delete = cancel_sent or inflight_creates.is_cancelled(payload.pod_id)
                result_is_valid = reply is not None and reply["type"] == "result" and _is_result_of(reply, rent_frame)
                # stored while the waiter stands: once it is gone, an `unacked` or a repeated `result` of this
                # attempt reads the mark, and `accepting` keeps the agent's pod
                accepting = (
                    result_is_valid
                    and not cancelled_by_delete
                    and await self._mark_attempt(payload, attempt, ATTEMPT_ACCEPTING)
                )

        reply_type = reply["type"] if reply else "timeout"
        steps = _known_steps((reply or {}).get("steps")) or steps_seen
        if reply_type == "result" and not accepting:
            # the agent removes this attempt's container and volume by its label; an ack would leave them to nobody
            self._send_frame(payload, _settle_frame(payload.pod_id, attempt, "rollback"))
        if cancelled_by_delete:
            _log_rent_result(log_extra, attempt=attempt, path="failed", reason="cancelled_by_delete", steps=steps)
            return _failed(payload, "Create cancelled by an in-flight delete", "cancelled_by_delete")
        if reply_type == "timeout":
            self._send_frame(payload, _cancel_frame(payload, attempt, "timeout"))
            _log_rent_result(log_extra, attempt=attempt, path="failed", reason="timeout", steps=steps)
            return _failed(payload, "The node agent did not answer the rent in time", "liumd_timeout")
        if reply_type == "result":
            if not result_is_valid:
                _log_rent_result(log_extra, attempt=attempt, path="failed", reason="bad_result", steps=steps)
                return _failed(payload, "The node agent's rent failed", "liumd_bad_result")
            if not accepting:
                _log_rent_result(log_extra, attempt=attempt, path="failed", reason="mark_failed", steps=steps)
                return _failed(payload, "The node agent's rent failed", "liumd_mark_failed")
            await self._stream_logs(payload, [("Created Docker Container", "success")])
            created = self._container_created(
                payload, custom_options, steps, port_maps, started_ms, port_maps_generated_ms, sent_ms, replied_ms
            )
            # the hand-over: no await from this check to the return, so a delete either lands before it and rolls
            # the pod back, or after it and finds the pod handed on
            if inflight_creates.is_cancelled(payload.pod_id):
                self._send_frame(payload, _settle_frame(payload.pod_id, attempt, "rollback"))
                _log_rent_result(log_extra, attempt=attempt, path="failed", reason="cancelled_by_delete", steps=steps)
                return _failed(payload, "Create cancelled by an in-flight delete", "cancelled_by_delete")
            _log_rent_result(log_extra, attempt=attempt, path="agent", reason=None, steps=steps, round_trip_ms=replied_ms - sent_ms)
            # a task runs on a later loop turn, after the caller queued ContainerCreated for compute-app
            # (ComputeClient holds its lock across no await), so the ack follows it in the same queue
            create_steps_after_reply.start(payload.pod_id, self._acknowledge_result(payload, attempt))
            return created

        code = reply.get("code") if reply.get("code") in LIUMD_ERROR_CODES else "unknown"
        if code == "busy":
            _log_rent_result(log_extra, attempt=attempt, path="failed", reason=code, steps=steps)
            return _failed(
                payload,
                "Decline renting pod request. Renting is still in progress",
                None,
                error_code=FailedContainerErrorCodes.RentingInProgress,
            )
        if _host_confirmed_clean(reply):
            # nothing of this pod is left on the host: today's by-name cleanup touches nothing (LIUM-79 F1)
            _log_rent_result(log_extra, attempt=attempt, path="today", reason=f"agent_{code}", steps=steps)
            return None
        # the host may hold this attempt's objects: the agent rolls back on `cancel`; today's path would race it
        self._send_frame(payload, _cancel_frame(payload, attempt, "timeout"))
        failed_step = reply.get("failed_step") if reply.get("failed_step") in LIUMD_FAILED_STEP_NAMES else None
        detail = reply.get("detail") if isinstance(reply.get("detail"), dict) else {}
        exit_code = detail.get("exit_code") if _is_int(detail.get("exit_code")) else None
        failure_step = f"liumd_{failed_step or code}"
        _log_rent_result(log_extra, attempt=attempt, path="failed", reason=failure_step, steps=steps)
        return _failed(
            payload,
            "The node agent's rent failed",
            failure_step,
            detail=_m(
                "liumd rent failed and its objects could not be confirmed removed",
                extra=get_extra_info({**log_extra, "code": code, "failed_step": failed_step, "exit_code": exit_code}),
            ).to_full_string(),
            volume_encryption_status=VolumeEncryptionStatus.FAILED if failed_step == "exec_volume_setup" else None,
        )

    async def _acknowledge_result(self, payload: ContainerCreateRequest, attempt: str) -> None:
        # the agent may forget the attempt now; the mark answers its `unacked` if the ack is lost
        for _ in range(ATTEMPT_ACKED_MARK_TRIES):
            if await self._mark_attempt(payload, attempt, ATTEMPT_ACKED):
                break
        else:
            # the pod is reported RUNNING: the ack still goes, and one that arrives ends the agent's record;
            # a lost one leaves `accepting`, which the agent's `unacked` gets no answer to
            logger.error(_m("liumd acked mark not stored, ack sent anyway", extra=get_extra_info({"pod_id": payload.pod_id, "attempt": attempt})))
        self._send_frame(payload, {"type": "ack", "rent_id": payload.pod_id, "attempt": attempt})

    async def _release_failed_rent(self, payload: ContainerCreateRequest, progress: _RentProgress) -> None:
        # every exit but ContainerCreated, task cancellation included; never raises over the rent's outcome
        if progress.pending_marked:
            try:
                await self.redis_service.remove_pending_pod(payload.miner_hotkey, payload.executor_id, payload.pod_id)
            except Exception as exc:
                logger.warning(_m("liumd pending pod mark not removed", extra=get_extra_info({**progress.log_extra, "error": repr(exc)})))
        if progress.sent:
            await self._mark_attempt(payload, progress.attempt, ATTEMPT_FAILED)

    async def _mark_attempt(self, payload: ContainerCreateRequest, attempt: str, state: str) -> bool:
        # False when the mark is not stored
        try:
            await self.redis_service.set(
                attempt_mark_key(attempt), f"{state}|{payload.executor_id}|{payload.pod_id}", ex=ATTEMPT_MARK_TTL_SECONDS
            )
        except Exception as exc:
            logger.warning(_m("liumd attempt mark not stored", extra=get_extra_info({"attempt": attempt, "mark": state, "error": repr(exc)})))
            return False
        return True

    def _abandon_attempt(self, payload: ContainerCreateRequest, attempt: str) -> None:
        # the reply may or may not have come: `cancel` stops a running rent, `settle rollback` removes a finished
        # one; the agent ignores whichever does not fit the attempt's state
        self._send_frame(payload, _cancel_frame(payload, attempt, "timeout"))
        self._send_frame(payload, _settle_frame(payload.pod_id, attempt, "rollback"))

    def _send_frame(self, payload: ContainerCreateRequest, frame: dict) -> bool:
        return self.agent_frames.send(
            LiumdRentRequest(executor_id=payload.executor_id, pod_id=payload.pod_id, frame=frame)
        )

    async def _wait_for_reply(
        self, payload: ContainerCreateRequest, attempt: str, replies: asyncio.Queue[dict]
    ) -> tuple[dict | None, dict[str, int], bool]:
        # the agent's `result` or `error` (None after the deadline), the steps it reported, and whether a
        # delete cancelled the rent (its `cancel` is sent)
        deadline = time.monotonic() + LIUMD_RENT_DEADLINE_MS / 1000 + LIUMD_REPLY_GRACE_SECONDS
        steps_seen: dict[str, int] = {}
        cancelled_by_delete = False
        while (remaining := deadline - time.monotonic()) > 0:
            if not cancelled_by_delete and inflight_creates.is_cancelled(payload.pod_id):
                cancelled_by_delete = True
                self._send_frame(payload, _cancel_frame(payload, attempt, "pod_deleted"))
            try:
                frame = await asyncio.wait_for(replies.get(), min(remaining, LIUMD_DELETE_POLL_SECONDS))
            except asyncio.TimeoutError:
                continue
            if frame.get("type") == "step":
                steps_seen.update(_known_steps({frame.get("step"): frame.get("ms")}))
            elif frame.get("type") in ("result", "error"):
                return frame, steps_seen, cancelled_by_delete
        return None, steps_seen, cancelled_by_delete

    async def _build_rent_frame(
        self,
        payload: ContainerCreateRequest,
        custom_options: CustomOptions,
        attempt: str,
        port_maps: list[tuple[int, int, int]],
        log_extra: dict,
    ) -> dict:
        # the `rent` frame of design §2.2, from the builders today's create uses
        container_name = self.docker_service.get_container_name(payload)
        volume_name = f"volume_{payload.pod_id}"
        run_spec = self.docker_service._build_rental_container_run_spec(
            payload=payload,
            container_name=container_name,
            custom_options=custom_options,
            port_maps=port_maps,
            local_volume=volume_name,
            local_volume_path=_plaintext_path(custom_options),
            encrypted_local_volume=True,
            external_volume_name=None,
            # the agent maps gpu_uuids to its own device nodes
            gpu_devices=GpuDockerConfig(),
            effective_storage_limit_gb=payload.storage_limit_gb,
            cpu_count=payload.cpu_count,
        )
        restore_records = await read_gpu_power_restore_records(self.redis_service, payload.gpu_uuids, log_extra)
        restore_watts_by_uuid = {record.gpu_uuid: record.watts for record in restore_records.records}
        execs = [_volume_setup_exec(payload, custom_options)]
        environment_exec = build_environment_exec_spec(
            container_name=container_name, environment=custom_options.environment
        )
        if environment_exec is not None:
            execs.append({
                "name": "environment",
                "argv": list(environment_exec.argv),
                "stdin_b64": base64.b64encode(environment_exec.stdin.encode()).decode(),
                "exit_codes": {},
            })
        return {
            "type": "rent",
            "rent_id": payload.pod_id,
            "attempt": attempt,
            "executor_id": payload.executor_id,
            "deadline_ms": LIUMD_RENT_DEADLINE_MS,
            # the name the agents parse; the port-check containers are named by the miner's hotkey
            # (executor_connectivity/orchestrator.py), as today's rent removes them
            "validator_hotkey": payload.miner_hotkey,
            # null: the connector holds no registry digest before the host is asked (design §10)
            "image": {"ref": payload.docker_image, "expected_digest": None},
            "container": {
                "name": container_name,
                "command": list(run_spec.command) or None,
                "env": run_spec.environment,
                "ports": [
                    {"container_port": port.container_port, "host_port": port.host_port, "protocol": port.protocol}
                    for port in run_spec.ports
                ],
                "gpu_uuids": payload.gpu_uuids,
                "cpu_count": run_spec.cpu_count,
                "memory_gb": run_spec.memory_gb,
                "storage_gb": run_spec.storage_limit_gb,
                "shm_size_bytes": parse_bytes(run_spec.shm_size) if run_spec.shm_size else None,
            },
            "volume": {
                "name": volume_name,
                "driver": _LOOPBACK_PLUGIN_ALIAS,
                "size_gb": payload.volume_limit_gb,
                "sparse": payload.disk_share is not None and payload.disk_share >= 1.0,
                "create_timeout_s": DockerService._get_local_volume_create_timeout(payload.volume_limit_gb, 10),
            },
            "preempt": {"fillers": payload.workload_kind == WorkloadKind.CUSTOMER_RENTAL},
            "gpu_power": [
                {"gpu_uuid": gpu_uuid, "restore_watts": restore_watts_by_uuid.get(gpu_uuid)}
                for gpu_uuid in payload.gpu_uuids
            ],
            "gpu_power_floor_ratio": MIN_POWER_LIMIT_RATIO,
            "execs": execs,
        }

    def _container_created(
        self,
        payload: ContainerCreateRequest,
        custom_options: CustomOptions,
        steps: dict[str, int],
        port_maps: list[tuple[int, int, int]],
        started_ms: int,
        port_maps_generated_ms: int,
        sent_ms: int,
        replied_ms: int,
    ) -> ContainerCreated:
        # today's profile frame around the agent's steps, so the backend's spans and the bench's table still line up
        profilers = [
            step for step in map(ProfilerStep.from_wire, payload.pre_dispatch_profilers or []) if step is not None
        ]
        if payload.timestamp:
            profilers.append(ProfilerStep(name=ProfilerStepName.REQUESTED_FROM_BACKEND, timestamp=payload.timestamp))
        profilers += [
            ProfilerStep(name=ProfilerStepName.STARTED_IN_SUBNET, duration=started_ms - (payload.timestamp or started_ms)),
            ProfilerStep(name=ProfilerStepName.PORT_MAPPINGS_GENERATED, duration=port_maps_generated_ms - started_ms),
        ]
        profilers += [
            ProfilerStep(name=ProfilerStepName(f"liumd {step_name}"), duration=duration_ms)
            for step_name, duration_ms in steps.items()
        ]
        finished_ms = now_ms()
        profilers += [
            ProfilerStep(name=ProfilerStepName.LIUMD_ROUND_TRIP, duration=replied_ms - sent_ms),
            ProfilerStep(name=ProfilerStepName.INSPECTOR_START_AFTER_REPLY, skipped=not settings.ENABLE_INSPECTOR),
            ProfilerStep(name=ProfilerStepName.FINISHED_IN_SUBNET, duration=finished_ms - replied_ms, timestamp=finished_ms),
        ]
        return ContainerCreated(
            miner_hotkey=payload.miner_hotkey,
            executor_id=payload.executor_id,
            pod_id=payload.pod_id,
            workload_kind=payload.workload_kind,
            container_name=self.docker_service.get_container_name(payload),
            volume_name=f"volume_{payload.pod_id}",
            port_maps=[(docker_port, external_port) for docker_port, _, external_port in port_maps],
            profilers=profilers,
            warnings=[],
            storage_limit_gb=payload.storage_limit_gb,
            volume_limit_gb=payload.volume_limit_gb,
            local_volume_path=_plaintext_path(custom_options),
            volume_encryption_status=VolumeEncryptionStatus.ENABLED,
        )

    async def _stream_logs(self, payload: ContainerCreateRequest, lines: list[tuple[str, str]]) -> None:
        # the renter's create log, as DockerService.handle_stream_logs publishes it
        try:
            await self.redis_service.publish(
                STREAMING_LOG_CHANNEL,
                {
                    "logs": [{"log_text": text, "log_status": status, "log_tag": LOG_TAG} for text, status in lines],
                    "miner_hotkey": payload.miner_hotkey,
                    "executor_uuid": payload.executor_id,
                    "pod_id": payload.pod_id,
                },
            )
        except Exception as exc:
            logger.warning(_m("liumd create log publish failed", extra=get_extra_info({"pod_id": payload.pod_id, "error": str(exc)})))


def _plaintext_path(custom_options: CustomOptions) -> str:
    # where the renter sees the volume, as create_container reads it
    return custom_options.volumes[0].split(":")[-1] if custom_options.volumes else "/root"


def _volume_setup_exec(payload: ContainerCreateRequest, custom_options: CustomOptions) -> dict:
    # the encrypted volume's setup + the renter's keys, composed as setup_encrypted_local_volume composes them
    passphrase = VolumeKeyDeriver.from_settings(settings).material(payload.pod_id).passphrase
    pad_hex, wrapped_hex = _xor_wrap_passphrase(passphrase)
    pad_var = _opaque_shell_name()
    wrapped_var = _opaque_shell_name()
    while wrapped_var == pad_var:
        wrapped_var = _opaque_shell_name()
    setup_script_path = f"{_VOLUME_SETUP_TMPFS}/.x{uuid4().hex[:8]}"
    passfile_path = f"{_VOLUME_SETUP_TMPFS}/.x{uuid4().hex[:8]}"
    plaintext_path = _plaintext_path(custom_options)
    setup_script = _build_gocryptfs_setup_and_mount_script(
        plaintext_path,
        pad_hex=pad_hex,
        wrapped_hex=wrapped_hex,
        pad_var=pad_var,
        wrapped_var=wrapped_var,
        passfile_path=passfile_path,
    )
    program = _build_volume_setup_exec_script(
        plaintext_path,
        setup_script_path=setup_script_path,
        passfile_path=passfile_path,
        setup_script_size=len(setup_script.encode("utf-8")),
        with_authorized_keys=True,
    )
    key_data = "".join(f"{public_key}\n" for public_key in payload.user_public_keys)
    return {
        "name": "volume_setup",
        "argv": ["sh", "-c", program],
        "stdin_b64": base64.b64encode((setup_script + key_data).encode("utf-8")).decode(),
        "exit_codes": {str(code): step for code, (step, _) in _VOLUME_SETUP_EXEC_FAILURES.items()},
    }


def _failed(
    payload: ContainerCreateRequest,
    msg: str,
    failure_step: str | None,
    *,
    error_code: FailedContainerErrorCodes = FailedContainerErrorCodes.UnknownError,
    detail: str | None = None,
    volume_encryption_status: VolumeEncryptionStatus | None = None,
) -> FailedContainerRequest:
    return FailedContainerRequest(
        miner_hotkey=payload.miner_hotkey,
        executor_id=payload.executor_id,
        pod_id=payload.pod_id,
        workload_kind=payload.workload_kind,
        msg=msg,
        detail=detail,
        error_type=FailedContainerErrorTypes.ContainerCreationFailed,
        error_code=error_code,
        failure_step=failure_step,
        volume_encryption_status=volume_encryption_status,
    )


def _log_rent_result(
    log_extra: dict,
    *,
    attempt: str | None,
    path: str,
    reason: str | None,
    steps: dict | None = None,
    round_trip_ms: int | None = None,
) -> None:
    # one line per rent of a listed executor: path `agent`, `today` (fallback) or `failed`; never the frame
    logger.info(
        _m(
            "LIUMD_RENT_RESULT",
            extra=get_extra_info({
                **log_extra,
                "rent_id": log_extra["pod_id"],
                "attempt": attempt,
                "path": path,
                "reason": reason,
                "steps": steps or {},
                "round_trip_ms": round_trip_ms,
            }),
        )
    )


def attempt_mark_key(attempt: str) -> str:
    return f"liumd:attempt:{attempt}"


def _cancel_frame(payload: ContainerCreateRequest, attempt: str, reason: str) -> dict:
    return {"type": "cancel", "rent_id": payload.pod_id, "attempt": attempt, "reason": reason}


def _settle_frame(rent_id: str, attempt: str, action: str) -> dict:
    return {"type": "settle", "rent_id": rent_id, "attempt": attempt, "action": action}


def _known_or_unknown(value: object, known: frozenset[str]) -> str:
    # an agent's free text never reaches the logs
    return value if isinstance(value, str) and value in known else "unknown"


def _known_steps(steps: object) -> dict[str, int]:
    # only the agent's own step names with int durations reach the logs and the profile
    if not isinstance(steps, dict):
        return {}
    return {name: ms for name, ms in steps.items() if name in LIUMD_STEP_NAMES and _is_int(ms)}


def _is_int(value: object) -> bool:
    # JSON true / false arrive as bool, which isinstance counts as int
    return isinstance(value, int) and not isinstance(value, bool)


def _is_result_of(result: dict, rent_frame: dict) -> bool:
    # the container and volume the rent asked for; the connector reports these names to compute-app
    return (
        result.get("container_name") == rent_frame["container"]["name"]
        and result.get("volume_name") == rent_frame["volume"]["name"]
    )


def _host_confirmed_clean(error: dict) -> bool:
    # the agent's explicit word that nothing of the rent is on the host; a missing or non-bool flag is unknown
    if error.get("code") in LIUMD_HOST_UNKNOWN_CODES:
        return False
    host_touched = error.get("host_touched")
    return host_touched is False or (host_touched is True and error.get("cleaned") is True)
