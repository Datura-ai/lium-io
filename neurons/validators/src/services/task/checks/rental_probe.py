from __future__ import annotations

import asyncio
import logging
import shlex
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import asyncssh
from core.docker_utils import DockerCommand
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    ContainerDeleted,
    ContainerDeleteRequest,
    FailedContainerRequest,
    PayloadPortMapping,
)

from core.config import settings
from core.utils import _m, get_extra_info

from ...const import MIN_PORT_COUNT, POD_CONTAINER_PREFIX
from ..messages import RentalProbeMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# The steps a renter's first minute on a node goes through, in the order the probe runs them. The
# step that fails is what the portal shows the provider (`failed_step` in what_we_saw).
STEP_CONTAINER_START = "container_start"
STEP_SSHD_LISTEN = "sshd_listen"
STEP_SSH_LOGIN = "ssh_login"
STEP_GPU_COUNT = "gpu_count"
STEP_TEARDOWN = "teardown"

# create_container's failure_step values that fail before the validator has talked to the host over
# SSH. A failure there is the validator's (its Redis, its key, the attestation verifier), so the probe
# reaches no verdict about the node instead of penalising it.
_CREATE_STEPS_BEFORE_THE_HOST = frozenset(
    {
        "start",
        "prepare_request",
        "port_mapping",
        "validate_request",
        "pending_pod",
        "ssh_key_import",
        "attestation",
        "docker_sdk_ssh_host_key",
        "build_input_empty",
        "cancelled_by_delete",
    }
)

# What the provider is told for each failed step, in plain words. The mapped port is filled in.
_REMEDIATION_BY_STEP: dict[str, str] = {
    STEP_CONTAINER_START: (
        "The validator could not start a renter container on this node (create step: {create_step}). "
        "Check that Docker and the NVIDIA container runtime start a GPU container by hand, that the "
        "default renter image is present, and that the executor's port range is free."
    ),
    STEP_SSHD_LISTEN: (
        "The container started but sshd never listened on port {port} within {deadline} s. "
        "Check the host firewall and Docker port publishing: the container's port 22 must be "
        "reachable from the internet on the mapped port."
    ),
    STEP_SSH_LOGIN: (
        "sshd answered on port {port} but the SSH login with the injected key was refused. "
        "Check that the container's /root/.ssh/authorized_keys is written and that sshd inside the "
        "container allows public-key login as root."
    ),
    STEP_GPU_COUNT: (
        "The renter container started and SSH worked, but `nvidia-smi -L` inside it did not list "
        "the {expected} GPU(s) this node advertises. Check the NVIDIA container runtime and "
        "`docker run --gpus all nvidia-smi -L` by hand."
    ),
}

_SSHD_POLL_SECONDS = 2.0
_SSH_LOGIN_TIMEOUT_SECONDS = 30
_NVIDIA_SMI_TIMEOUT_SECONDS = 30
# create_container has its own bounded retries (a 90 s port-retry budget, SSH timeouts) but no overall
# deadline; the image is already on the host, so a create still running after this is a stuck host, and
# the run's outer timeout (JOB_TIME_OUT) must not be what ends it.
_CREATE_DEADLINE_SECONDS = 300
# create_step values for a create that never returned: its container may be half-made on the host
_CREATE_CUT_SHORT = frozenset({"deadline", "cancelled"})
_TEARDOWN_DEADLINE_SECONDS = 120
# one shell command over the validation connection (the image check, the by-name removals)
_SHELL_COMMAND_TIMEOUT_SECONDS = 60
# every string copied out of the host into the event is bounded (PR_PROCESS §5)
_TAIL_CHARS = 600
_REDIS_LAST_OK_PREFIX = "rental_probe_ok"


@dataclass
class _Step:
    name: str
    seconds: float
    ok: bool
    detail: str | None = None

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "step": self.name,
            "seconds": round(self.seconds, 2),
            "ok": self.ok,
        }
        if self.detail:
            record["detail"] = self.detail[-_TAIL_CHARS:]
        return record


@dataclass
class _ProbeOutcome:
    steps: list[_Step] = field(default_factory=list)
    failed_step: str | None = None
    # the create step create_container named when the container never started
    create_step: str | None = None
    ssh_port: int | None = None
    inconclusive_reason: str | None = None


class RentalProbeCheck:
    """Rent the idle node from the validator and prove a renter could use it (DAH-3436).

    Node 939b3e38 broke six renter pods in a row on 11 Sep (ports mapped, sshd never listened, the
    executor then reported offline) while every 15-minute check scored it 1.0: the checks prove the
    GPUs, the bandwidth and the filler, not the one thing a renter needs. This check starts the
    default renter image through `DockerService.create_container`, the path a renter's pod takes,
    with a probe-owned SSH key and the ports PortConnectivityCheck verified; waits for sshd on the
    mapped port; logs in; runs `nvidia-smi -L`; and tears the container down through
    `delete_container`. Each step is timed and lands in `what_we_saw["probe_steps"]`.

    A failed step zeroes the score and clears the verified job with reason code RENTAL_PROBE_FAILED
    and the step's name, the same mechanics as the GPU runtime quarantine in rental_verification.py.
    The penalty is the cycle's: the failure is not carried forward, so the next cycle verifies the
    node again unless the probe runs and fails again. It runs again when the node is still idle and
    the image is still pulled (the interval stamp is written on success only); a cycle that skips
    the node (filler running, image gone) or reads inconclusively restores it without a probe.
    Otherwise the probe runs at most once per RENTAL_PROBE_INTERVAL_HOURS per executor, or at once
    when the backend lists the executor in `rental_probe_requested_executor_ids`. It never runs on a rented node or beside a filler: the
    start path sweeps `pod_*`/`filler_*` containers it was not told about and lifts GPU power caps.
    The node's idleness is re-read from the backend and this validator's pending-pod marks right
    before the container starts, not taken from the cycle-start snapshot. Anything the probe cannot
    read (backend, Redis) makes it skip or reach no verdict; it never penalises on missing data.
    RENTAL_PROBE_ENABLED is off by default.
    """

    check_id = "executor.validate.rental_probe"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.RENTAL_PROBE_ENABLED:
            return CheckResult(
                passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            )

        skip_reason, skip_what = await _skip_reason(ctx)
        if skip_reason:
            return self._skipped(ctx, skip_reason, skip_what)

        missing = _missing_inputs(ctx)
        if missing:
            return self._inconclusive(
                ctx, f"validator has no {missing} for this executor", steps=[]
            )

        image_ref = await _default_renter_image(ctx)
        if image_ref is None:
            return self._inconclusive(
                ctx, "no default renter image for this GPU and driver", steps=[]
            )
        image_present = await _image_present_on_host(ctx, image_ref)
        if image_present is None:
            return self._inconclusive(
                ctx, "could not read the node's image list", steps=[], what={"image": image_ref}
            )
        if not image_present:
            # a pull would put minutes of Docker Hub traffic inside the validation cycle; the
            # cached-template check already reports a node that has not pre-pulled the image
            return self._skipped(
                ctx, "default renter image is not pulled on the node", {"image": image_ref}
            )

        # ctx.state.rented_data is the snapshot the cycle started from, minutes ago for the last
        # check in the pipeline. create_container sweeps every pod_* container it is not told
        # about, so the node must be idle NOW: a fresh backend read plus this validator's own
        # pending-pod mark (a rent it is creating this moment). Unknown counts as busy.
        busy_now = await _busy_now(ctx)
        if busy_now is None:
            return self._inconclusive(ctx, "could not confirm the node is idle", steps=[])
        if busy_now:
            return self._skipped(ctx, busy_now)

        outcome = await _probe(ctx, image_ref)
        # `steps` is the pipeline's own per-check duration summary on the run's last event
        # (summarize_steps); the probe's records need their own key to survive it.
        what = {
            "executor_uuid": ctx.executor.uuid,
            "image": image_ref,
            "ssh_port": outcome.ssh_port,
            "probe_steps": [step.as_record() for step in outcome.steps],
            "probe_seconds_total": round(sum(step.seconds for step in outcome.steps), 2),
        }

        if outcome.inconclusive_reason:
            return self._inconclusive(
                ctx, outcome.inconclusive_reason, steps=outcome.steps, what=what
            )

        if outcome.failed_step is None:
            await _stamp_last_ok(ctx)
            return CheckResult(
                passed=True,
                event=render_message(Msg.PROBE_OK, ctx=ctx, check_id=self.check_id, what=what),
            )

        # A renter may have taken the node while the probe ran: their create sweeps our container
        # away and a real rental is proof the node works. Never punish that race, and never
        # punish when the race cannot be ruled out.
        rented_meanwhile = await _rented_meanwhile(ctx)
        if rented_meanwhile is None:
            return self._inconclusive(
                ctx, "could not rule out a rental during the probe", steps=outcome.steps, what=what
            )
        if rented_meanwhile:
            return self._inconclusive(
                ctx, "node was rented during the probe", steps=outcome.steps, what=what
            )

        # a pass stamped earlier in the interval must not shield this failure: without this a node
        # the backend forced a probe on would skip as "within interval" next cycle and be relisted
        await _clear_last_ok(ctx)
        what["failed_step"] = outcome.failed_step
        if outcome.create_step:
            what["create_step"] = outcome.create_step
        event = render_message(
            Msg.PROBE_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            what=what,
            remediation=_remediation(outcome, expected_gpus=_expected_gpu_count(ctx)),
        )
        return CheckResult(
            passed=False,
            event=event,
            updates={
                "score": 0.0,
                "job_score": 0.0,
                "score_warning": f"Rental probe failed at {outcome.failed_step}",
                "clear_verified_job_info": True,
                "clear_verified_job_evidence": {
                    "reason_code": event.reason_code,
                    "check_id": self.check_id,
                    "failed_step": outcome.failed_step,
                    "create_step": outcome.create_step,
                    "ssh_port": outcome.ssh_port,
                },
            },
        )

    def _skipped(
        self, ctx: Context, reason: str, what: dict[str, Any] | None = None
    ) -> CheckResult:
        event = render_message(
            Msg.SKIPPED,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid, "reason": reason, **(what or {})},
        )
        return CheckResult(passed=True, event=event)

    def _inconclusive(
        self, ctx: Context, reason: str, *, steps: list[_Step], what: dict[str, Any] | None = None
    ) -> CheckResult:
        event = render_message(
            Msg.INCONCLUSIVE,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "executor_uuid": ctx.executor.uuid,
                "reason": reason,
                "probe_steps": [step.as_record() for step in steps],
                **(what or {}),
            },
        )
        return CheckResult(passed=True, event=event)


async def _skip_reason(ctx: Context) -> tuple[str | None, dict[str, Any]]:
    rented_data = ctx.state.rented_data
    rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
    if rented_executor and rented_executor.pods:
        return "rented", {"pods": [pod.pod_id for pod in rented_executor.pods]}
    filler_containers = rented_data.get_filler_containers(ctx.executor.uuid) if rented_data else []
    if filler_containers:
        return "filler running", {"filler_containers": filler_containers}

    requested = bool(
        rented_data and ctx.executor.uuid in rented_data.rental_probe_requested_executor_ids
    )
    if requested:
        return None, {}
    try:
        last_ok = await _last_ok_at(ctx)
    except Exception:
        # an unreadable stamp must not turn into a container per node per cycle
        _read_failed(ctx, "its interval stamp in Redis")
        return "interval state unreadable", {}
    interval_seconds = settings.RENTAL_PROBE_INTERVAL_HOURS * 3600
    if last_ok is not None and time.time() - last_ok < interval_seconds:
        return "within interval", {"last_ok_seconds_ago": int(time.time() - last_ok)}
    return None, {}


def _read_failed(ctx: Context, what: str) -> None:
    # every unreadable input ends as a skip or no verdict (never a penalty); the log line says which read
    logger.warning(
        _m(
            f"Rental probe could not read {what}; no verdict this cycle",
            extra=get_extra_info(ctx.default_extra),
        ),
        exc_info=True,
    )


async def _busy_now(ctx: Context) -> str | None:
    """Why the node is not idle right now, "" when it is, None when that could not be read."""
    try:
        rented = await ctx.services.backend.get_all_rented_executors()
    except Exception:
        _read_failed(ctx, "the backend's rented executors")
        return None
    if rented is None:
        return None
    rented_executor = rented.executors.get(ctx.executor.uuid)
    if rented_executor and rented_executor.pods:
        return "rented since the cycle started"
    if rented.get_filler_containers(ctx.executor.uuid):
        return "filler started since the cycle started"
    try:
        if await ctx.services.redis.renting_in_progress(ctx.miner_hotkey, ctx.executor.uuid):
            return "a rent is being created on the node"
    except Exception:
        _read_failed(ctx, "its pending-pod marks in Redis")
        return None
    return ""


def _missing_inputs(ctx: Context) -> str | None:
    # generate_portMappings refuses fewer than MIN_PORT_COUNT free ports, so a create with fewer would
    # fail at port_mapping every cycle
    if len(ctx.state.verified_port_pairs) < MIN_PORT_COUNT:
        return f"{MIN_PORT_COUNT} verified ports"
    if not _gpu_uuids(ctx):
        return "GPU UUIDs"
    if not ctx.executor_ssh_private_key_encrypted or ctx.config.validator_keypair is None:
        return "executor SSH key"
    return None


def _gpu_uuids(ctx: Context) -> list[str]:
    details = ctx.state.gpu_details or (ctx.state.specs or {}).get("gpu", {}).get("details", [])
    return [d["uuid"] for d in details if isinstance(d, dict) and d.get("uuid")]


def _expected_gpu_count(ctx: Context) -> int:
    return ctx.state.gpu_count or len(_gpu_uuids(ctx))


async def _default_renter_image(ctx: Context) -> str | None:
    gpu_model = ctx.state.gpu_model
    driver_version = str((ctx.state.specs or {}).get("gpu", {}).get("driver") or "")
    if not gpu_model or not driver_version:
        return None
    try:
        images = await ctx.services.backend.get_default_docker_image(gpu_model, driver_version)
    except Exception:
        _read_failed(ctx, "the default renter image from the backend")
        return None
    if not images:
        return None
    return images[0].image_ref


async def _image_present_on_host(ctx: Context, image_ref: str) -> bool | None:
    """Whether the image is on the node; None when the validation shell could not answer."""
    try:
        result = await asyncio.wait_for(
            ctx.ssh.run(
                f"/usr/bin/docker image inspect --format '{{{{.Id}}}}' {shlex.quote(image_ref)}",
                check=False,
            ),
            timeout=_SHELL_COMMAND_TIMEOUT_SECONDS,
        )
    except (TimeoutError, asyncssh.Error, OSError):
        return None
    return result.exit_status == 0 and bool((result.stdout or "").strip())


async def _last_ok_at(ctx: Context) -> float | None:
    """Epoch of the last passed probe, None when there is none; raises when Redis cannot be read."""
    raw = await ctx.services.redis.get(f"{_REDIS_LAST_OK_PREFIX}:{ctx.executor.uuid}")
    if raw is None:
        return None
    try:
        return float(raw.decode() if isinstance(raw, bytes) else raw)
    except (TypeError, ValueError):
        return None


async def _stamp_last_ok(ctx: Context) -> None:
    try:
        await ctx.services.redis.set(
            f"{_REDIS_LAST_OK_PREFIX}:{ctx.executor.uuid}", str(time.time())
        )
    except Exception:
        logger.warning(
            _m(
                "Rental probe passed but its interval stamp could not be written",
                extra=get_extra_info(ctx.default_extra),
            ),
            exc_info=True,
        )


async def _clear_last_ok(ctx: Context) -> None:
    try:
        await ctx.services.redis.delete(f"{_REDIS_LAST_OK_PREFIX}:{ctx.executor.uuid}")
    except Exception:
        logger.warning(
            _m(
                "Rental probe failed but its interval stamp could not be cleared; the node may skip the next probe",
                extra=get_extra_info(ctx.default_extra),
            ),
            exc_info=True,
        )


async def _rented_meanwhile(ctx: Context) -> bool | None:
    """Whether a renter took the node while the probe ran: the backend lists a pod on it now, or this
    validator holds a pending-pod mark for it (a renter's create it is running at this moment; the
    probe's own mark is already cleared when this runs). None when either could not be read."""
    try:
        rented = await ctx.services.backend.get_all_rented_executors()
    except Exception:
        _read_failed(ctx, "the backend's rented executors after the probe")
        return None
    if rented is None:
        return None
    rented_executor = rented.executors.get(ctx.executor.uuid)
    if rented_executor and rented_executor.pods:
        return True
    try:
        return bool(
            await ctx.services.redis.renting_in_progress(ctx.miner_hotkey, ctx.executor.uuid)
        )
    except Exception:
        _read_failed(ctx, "its pending-pod marks in Redis after the probe")
        return None


def _probe_payload(
    ctx: Context, *, pod_id: str, image_ref: str, public_key: str
) -> ContainerCreateRequest:
    return ContainerCreateRequest(
        miner_hotkey=ctx.miner_hotkey,
        miner_address=ctx.miner_address,
        miner_port=ctx.miner_port,
        executor_id=ctx.executor.uuid,
        pod_id=pod_id,
        docker_image=image_ref,
        user_public_keys=[public_key],
        gpu_uuids=_gpu_uuids(ctx),
        is_sysbox=ctx.state.sysbox_runtime,
        # the default renter image runs its own sshd; the same flag the backend sets for it
        ships_sshd=True,
        enable_jupyter=False,
        enable_volume_encryption=False,
        available_ports=[
            PayloadPortMapping(internal_port=internal, external_port=external)
            for internal, external in ctx.state.verified_port_pairs
        ],
        pod_mapping=[],
        # nothing to protect: the probe runs only when the backend lists no pod and no filler here
        active_container_names=[],
        active_volume_names=[],
    )


async def _probe(ctx: Context, image_ref: str) -> _ProbeOutcome:
    """Run the five steps; whatever happens, the node and this validator's Redis are left as found.

    The container's name and volume follow from the pod id (`pod_<id>`, `volume_<id>`), so the
    cleanup in `finally` can remove them by name over the validation shell even when
    create_container was cut short: by its own deadline here, or by the run's outer timeout
    (JOB_TIME_OUT), which arrives as CancelledError and passes every `except Exception`. The
    cleanup is shielded so a cancelled task still runs it; the cancellation itself propagates.
    """
    outcome = _ProbeOutcome()
    pod_id = str(uuid.uuid4())
    private_key, public_key = ctx.services.ssh.generate_keypair()
    docker = ctx.services.pod_recovery
    log_extra = {**ctx.default_extra, "probe_pod_id": pod_id, "image": image_ref}
    created: ContainerCreated | None = None

    try:
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                docker.create_container(
                    _probe_payload(ctx, pod_id=pod_id, image_ref=image_ref, public_key=public_key),
                    ctx.executor,
                    ctx.config.validator_keypair,
                    ctx.executor_ssh_private_key_encrypted,
                ),
                timeout=_CREATE_DEADLINE_SECONDS,
            )
        except TimeoutError:
            outcome.steps.append(
                _Step(
                    STEP_CONTAINER_START,
                    time.perf_counter() - started,
                    False,
                    f"create_container did not finish in {_CREATE_DEADLINE_SECONDS} s",
                )
            )
            outcome.failed_step = STEP_CONTAINER_START
            outcome.create_step = "deadline"
            return outcome
        except asyncio.CancelledError:
            # the run's outer timeout: create_container may have left `pod_<id>` half-made; _settle removes it by name
            outcome.create_step = "cancelled"
            raise
        except (
            Exception
        ) as exc:  # create_container reports its failures; a raise is the validator's own
            outcome.steps.append(
                _Step(STEP_CONTAINER_START, time.perf_counter() - started, False, repr(exc))
            )
            outcome.inconclusive_reason = "create_container raised"
            return outcome

        if not isinstance(result, ContainerCreated):
            detail, create_step = _failure_detail(result)
            outcome.steps.append(
                _Step(STEP_CONTAINER_START, time.perf_counter() - started, False, detail)
            )
            outcome.create_step = create_step
            if create_step is None:
                outcome.inconclusive_reason = "create returned a result without a failure step"
            elif create_step in _CREATE_STEPS_BEFORE_THE_HOST:
                outcome.inconclusive_reason = (
                    f"create failed on the validator's side at {create_step}"
                )
            else:
                outcome.failed_step = STEP_CONTAINER_START
            return outcome
        created = result
        outcome.steps.append(_Step(STEP_CONTAINER_START, time.perf_counter() - started, True))

        ssh_port = _mapped_ssh_port(created)
        outcome.ssh_port = ssh_port
        if ssh_port is None:
            # generate_portMappings always maps 22 first; a create without it is the validator's mapping
            outcome.steps.append(
                _Step(STEP_SSHD_LISTEN, 0.0, False, "no port mapping for container port 22")
            )
            outcome.inconclusive_reason = "create returned no mapping for container port 22"
            return outcome

        deadline = settings.RENTAL_PROBE_SSH_DEADLINE_SECONDS
        started = time.perf_counter()
        listen_error = await _wait_for_sshd(ctx.executor.address, ssh_port, deadline)
        outcome.steps.append(
            _Step(
                STEP_SSHD_LISTEN, time.perf_counter() - started, listen_error is None, listen_error
            )
        )
        if listen_error is not None:
            outcome.failed_step = STEP_SSHD_LISTEN
            return outcome

        login = await _login_and_list_gpus(ctx.executor.address, ssh_port, private_key)
        outcome.steps.append(
            _Step(STEP_SSH_LOGIN, login.login_seconds, login.login_error is None, login.login_error)
        )
        if login.login_error is not None:
            outcome.failed_step = STEP_SSH_LOGIN
            return outcome

        expected = _expected_gpu_count(ctx)
        if login.command_error is not None:
            gpu_ok = False
            detail = f"nvidia-smi -L did not run over the SSH session: {login.command_error}"
        else:
            smi = login.result
            seen = _count_gpus(smi.stdout or "")
            gpu_ok = smi.exit_status == 0 and seen == expected
            detail = (
                None
                if gpu_ok
                else f"expected {expected} GPU(s), nvidia-smi -L listed {seen} (exit {smi.exit_status}): "
                f"{(smi.stderr or smi.stdout or '')[-_TAIL_CHARS:]}"
            )
        outcome.steps.append(_Step(STEP_GPU_COUNT, login.command_seconds, gpu_ok, detail))
        if not gpu_ok:
            outcome.failed_step = STEP_GPU_COUNT
        return outcome
    finally:
        # the settle runs as its own task and is awaited to the end: a cancel that lands while it runs
        # (the first one, or a second) interrupts only this wait, never the settle, and is re-raised
        # once the settle is done so the run still ends as cancelled
        settle = asyncio.ensure_future(
            _settle(ctx, outcome, created=created, pod_id=pod_id, log_extra=log_extra)
        )
        cancelled = False
        while not settle.done():
            try:
                await asyncio.shield(settle)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError()


async def _settle(
    ctx: Context,
    outcome: _ProbeOutcome,
    *,
    created: ContainerCreated | None,
    pod_id: str,
    log_extra: dict[str, Any],
) -> None:
    """Leave the node and Redis as found.

    A created container goes through delete_container, the renter's unrent path. When that fails, or
    when the create was cut short by its deadline or by the run's cancellation (it may have left `pod_<id>` behind), the container
    and its volume are removed by name over the validation shell. The pending-pod mark is cleared in
    every case. A teardown that needed the fallback is recorded as a failed step: the node served the
    renter but its unrent path did not finish, so a clean run is INCONCLUSIVE (no stamp, probed again
    next cycle) rather than a pass.
    """
    try:
        if created is None and outcome.create_step not in _CREATE_CUT_SHORT:
            return  # nothing reached the host
        started = time.perf_counter()
        teardown_error = (
            await _teardown(ctx, created, pod_id) if created is not None else "create cut short"
        )
        leftover: str | None = None
        if teardown_error is not None:
            # the names the create used when it returned them; the names it would have used otherwise
            container_name = (
                created.container_name if created is not None else f"{POD_CONTAINER_PREFIX}{pod_id}"
            )
            volume_name = created.volume_name if created is not None else f"volume_{pod_id}"
            leftover = await _remove_over_shell(
                ctx, container_name=container_name, volume_name=volume_name
            )
            detail = (
                f"{teardown_error}; removed by name over the validation shell"
                if leftover is None
                else f"{teardown_error}; shell removal failed too: {leftover}"
            )
        else:
            detail = None
        outcome.steps.append(
            _Step(STEP_TEARDOWN, time.perf_counter() - started, teardown_error is None, detail)
        )
        if leftover is not None:
            logger.warning(
                _m(
                    "Rental probe container could not be removed; the stale-container sweep reaps it",
                    extra=get_extra_info({**log_extra, "error": detail}),
                ),
            )
        if (
            teardown_error is not None
            and outcome.failed_step is None
            and outcome.inconclusive_reason is None
        ):
            outcome.inconclusive_reason = "teardown did not finish"
    finally:
        await _remove_pending_pod(ctx, pod_id)


async def _remove_over_shell(ctx: Context, *, container_name: str, volume_name: str) -> str | None:
    """`docker rm -fv <container>` + `docker volume rm <volume>` over the validation shell, and the
    rented-pod cache entry dropped; None when nothing is left on the node."""
    try:
        removed = await asyncio.wait_for(
            ctx.ssh.run(
                DockerCommand.remove_with_volumes(shlex.quote(container_name)), check=False
            ),
            timeout=_SHELL_COMMAND_TIMEOUT_SECONDS,
        )
        await asyncio.wait_for(
            ctx.ssh.run(DockerCommand.volume_remove(volume_name), check=False),
            timeout=_SHELL_COMMAND_TIMEOUT_SECONDS,
        )
    except (TimeoutError, asyncssh.Error, OSError) as exc:
        return repr(exc)
    try:
        await ctx.services.redis.remove_rented_machine(ctx.executor, container_name)
    except Exception:
        logger.warning(
            _m(
                "Rental probe could not drop its rented-pod cache entry",
                extra=get_extra_info({**ctx.default_extra, "container_name": container_name}),
            ),
            exc_info=True,
        )
    # `docker rm -f` of a name that does not exist exits 1 with "No such container": nothing left behind.
    # Any other non-zero exit (dockerd down also exits 1) means the container's state is unknown.
    if removed.exit_status == 0 or "No such container" in (removed.stderr or ""):
        return None
    return f"docker rm exited {removed.exit_status}: {(removed.stderr or '')[-_TAIL_CHARS:]}"


def _failure_detail(result: Any) -> tuple[str, str | None]:
    if isinstance(result, FailedContainerRequest):
        return (result.detail or result.msg or "create failed")[-_TAIL_CHARS:], result.failure_step
    return f"unexpected create result {type(result).__name__}", None


def _mapped_ssh_port(created: ContainerCreated) -> int | None:
    for docker_port, external_port in created.port_maps:
        if docker_port == 22:
            return external_port
    return None


async def _wait_for_sshd(host: str, port: int, deadline_seconds: float) -> str | None:
    """None once a TCP connection to the mapped port opens, else the last error at the deadline."""
    last_error = "never tried"
    deadline = time.monotonic() + deadline_seconds
    while True:
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=5)
            writer.close()
            return None
        except (TimeoutError, OSError) as exc:
            last_error = repr(exc)
        if time.monotonic() >= deadline:
            return (
                f"port {port} did not accept a connection within {deadline_seconds} s: {last_error}"
            )
        await asyncio.sleep(_SSHD_POLL_SECONDS)


@dataclass
class _Login:
    login_error: str | None = None
    command_error: str | None = None
    result: Any = None
    login_seconds: float = 0.0
    command_seconds: float = 0.0


async def _login_and_list_gpus(host: str, port: int, private_key: str) -> _Login:
    """Log in with the probe key the way a renter does and run `nvidia-smi -L` inside the container.

    The two phases are reported apart: a refused or timed-out connection is the login step's failure,
    a session that opened but could not run the command is the GPU step's.
    """
    login = _Login()
    started = time.perf_counter()
    try:
        pkey = asyncssh.import_private_key(private_key)
        conn = await asyncssh.connect(
            host=host,
            port=port,
            username="root",
            client_keys=[pkey],
            known_hosts=None,
            connect_timeout=_SSH_LOGIN_TIMEOUT_SECONDS,
        )
    except (TimeoutError, asyncssh.Error, OSError) as exc:
        login.login_error = repr(exc)
        login.login_seconds = time.perf_counter() - started
        return login
    login.login_seconds = time.perf_counter() - started
    command_started = time.perf_counter()
    try:
        async with conn:
            login.result = await conn.run(
                "nvidia-smi -L", check=False, timeout=_NVIDIA_SMI_TIMEOUT_SECONDS
            )
    except (TimeoutError, asyncssh.Error, OSError) as exc:
        login.command_error = repr(exc)
    login.command_seconds = time.perf_counter() - command_started
    return login


def _count_gpus(nvidia_smi_output: str) -> int:
    return sum(1 for line in nvidia_smi_output.splitlines() if line.startswith("GPU "))


async def _teardown(ctx: Context, created: ContainerCreated, pod_id: str) -> str | None:
    docker = ctx.services.pod_recovery
    try:
        deleted = await asyncio.wait_for(
            docker.delete_container(
                ContainerDeleteRequest(
                    miner_hotkey=ctx.miner_hotkey,
                    miner_address=ctx.miner_address,
                    miner_port=ctx.miner_port,
                    executor_id=ctx.executor.uuid,
                    pod_id=pod_id,
                    container_name=created.container_name,
                    local_volume=created.volume_name,
                ),
                ctx.executor,
                ctx.config.validator_keypair,
                ctx.executor_ssh_private_key_encrypted,
            ),
            timeout=_TEARDOWN_DEADLINE_SECONDS,
        )
    except TimeoutError:
        return f"delete_container did not finish in {_TEARDOWN_DEADLINE_SECONDS} s"
    except Exception as exc:
        return repr(exc)
    if isinstance(deleted, ContainerDeleted):
        return None
    detail, _ = _failure_detail(deleted)
    return detail


async def _remove_pending_pod(ctx: Context, pod_id: str) -> None:
    # create_container marks the pod pending (miner_service declines a second create of the SAME pod id
    # while it stands); the backend's rent-finished request clears it for a renter, so the probe clears
    # its own
    try:
        await ctx.services.redis.remove_pending_pod(ctx.miner_hotkey, ctx.executor.uuid, pod_id)
    except Exception:
        logger.warning(
            _m(
                "Rental probe could not clear its pending pod",
                extra=get_extra_info({**ctx.default_extra, "probe_pod_id": pod_id}),
            ),
            exc_info=True,
        )


def _remediation(outcome: _ProbeOutcome, *, expected_gpus: int) -> str:
    template = _REMEDIATION_BY_STEP.get(outcome.failed_step or "", "")
    return template.format(
        create_step=outcome.create_step or "unknown",
        port=outcome.ssh_port if outcome.ssh_port is not None else "?",
        deadline=settings.RENTAL_PROBE_SSH_DEADLINE_SECONDS,
        expected=expected_gpus,
    )
