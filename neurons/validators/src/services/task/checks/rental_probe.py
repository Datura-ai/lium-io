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
# reaches no verdict about the node instead of penalising it. `cancelled_by_delete` is not positional:
# a delete_container for the same pod id cut the create short and removed the container itself.
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
        "The container started but sshd did not answer on port {port} within {deadline} s. "
        "Check that sshd starts inside the container (`docker logs <container>`), then the host "
        "firewall and Docker port publishing: the container's port 22 must be reachable from the "
        "internet on the mapped port."
    ),
    STEP_SSH_LOGIN: (
        "sshd answered on port {port} but the SSH login with the injected key was refused, or the "
        "connection kept dropping until the {deadline} s deadline. Check that the container's "
        "/root/.ssh/authorized_keys is written and that sshd inside the container allows public-key "
        "login as root."
    ),
    STEP_GPU_COUNT: (
        "The renter container started and SSH worked, but `nvidia-smi -L` inside it did not list "
        "the {expected} GPU(s) this node advertises. Check the NVIDIA container runtime and "
        "`docker run --gpus all nvidia-smi -L` by hand."
    ),
}

_SSHD_POLL_SECONDS = 2.0
# the identification string sshd sends first on every connection (RFC 4253 §4.2)
_SSH_BANNER_PREFIX = b"SSH-2.0"
_SSHD_CONNECT_TIMEOUT_SECONDS = 5
_SSHD_BANNER_TIMEOUT_SECONDS = 5
_SSH_LOGIN_TIMEOUT_SECONDS = 30
# waits between login attempts; the last value repeats until the deadline
_SSH_LOGIN_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)
_NVIDIA_SMI_TIMEOUT_SECONDS = 30
# create_container has its own bounded retries (a 90 s port-retry budget, SSH timeouts) but no overall
# deadline; the image is already on the host, so a create still running after this is a stuck host, and
# the run's outer timeout (JOB_TIME_OUT) must not be what ends it.
_CREATE_DEADLINE_SECONDS = 300
# the idleness re-read runs under the per-executor create lock; the lock's TTL (redis_service.py) covers
# this budget plus the create deadline, so a re-read that overruns it gives the lock back instead
_IDLENESS_REREAD_BUDGET_SECONDS = 30
# the outer cap on delete_container (DAH-3467, review): a stop with its 30 s grace, a `rm -f` that
# outlives the Docker SDK's 60 s read timeout, the 60 s inspect window plus one 15 s inspect past
# it, and six "volume is in use" retries 5 s apart come to about 190 s on the inspect-confirmed
# path; a stop that itself runs into the SDK's read timeout adds to that and stays under this
# cap, which is what ends a delete that is still slower (the probe then records "did not finish",
# and the by-name removals in _step_teardown run)
_TEARDOWN_DEADLINE_SECONDS = 300
# one shell command over the validation connection (the image check, the by-name removals)
_SHELL_COMMAND_TIMEOUT_SECONDS = 60
# every string copied out of the host into the event is bounded (PR_PROCESS §5)
_TAIL_CHARS = 600
_REDIS_LAST_OK_PREFIX = "rental_probe_ok"
# the failed step of the last probe, standing until a probe passes (review: a skipped or inconclusive
# next cycle must not relist the node without a clean probe). The key has NO lifetime: only a passed
# probe deletes it (review: with a lifetime, a node skipped or inconclusive for longer than it was
# relisted without a clean probe). The key is one short string per executor that failed and never
# passed again, deregistered ones included.
_REDIS_FAILED_PREFIX = "rental_probe_failed"
# the OK stamp expires: a deregistered executor's stamp ages out, and an expired stamp only makes the
# probe run again on the next idle cycle
_REDIS_STAMP_TTL_SECONDS = 30 * 24 * 3600


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


@dataclass
class _ProbeGate:
    """What `RentalProbeCheck._may_probe` hands the probe once every may-we-probe condition holds."""

    image_ref: str
    # held; `run` releases it after the probe (the probe itself releases it as soon as its create returns)
    create_lock: _CreateLock
    # the last probe's failed step, standing until a probe passes; None when there is none
    standing: _Failure | None


class RentalProbeCheck:
    """Rent the idle node from the validator and prove a renter could use it (DAH-3436).

    Node 939b3e38 broke six renter pods in a row on 11 Sep (ports mapped, sshd never listened, the
    executor then reported offline) while every 15-minute check scored it 1.0: the checks prove the
    GPUs, the bandwidth and the filler, not the one thing a renter needs. This check starts the
    default renter image through `DockerService.create_container`, the path a renter's pod takes,
    with a probe-owned SSH key and the ports PortConnectivityCheck verified; waits for sshd's banner
    on the mapped port; logs in, retrying until the same RENTAL_PROBE_SSH_DEADLINE_SECONDS deadline;
    runs `nvidia-smi -L`; and tears the container down through `delete_container`. Each step is
    timed and lands in `what_we_saw["probe_steps"]`.

    A failed step zeroes the score and clears the verified job with reason code RENTAL_PROBE_FAILED
    and the step's name, the same mechanics as the GPU runtime quarantine in rental_verification.py.
    The failure stands until a probe passes: the failed step is kept in Redis, and every later cycle
    that reaches no verdict of its own (the node skipped for a filler or a missing image, an
    inconclusive probe, an unreadable input) fails the node again with that step, so a node is not
    relisted without a clean probe while Redis is readable (an unreadable Redis is logged and the cycle
    decides as if no failure stood, as every other unreadable input does). Only a rented node is left
    alone, whichever read says so (the cycle's snapshot, the re-read before the create, a renter's
    create holding the lock, a rental during the probe): the probe cannot run there and the renter's
    pod is the backend's to judge; the standing failure applies again once the node is idle. It runs
    again when the node is idle and the image is pulled (the interval stamp is written
    on success only). Otherwise the probe runs at most once per RENTAL_PROBE_INTERVAL_HOURS per
    executor. It never runs on a rented node or beside a filler: the start path sweeps
    `pod_*`/`filler_*` containers it was not told about and lifts GPU power caps. Before the
    container starts the probe takes the per-executor create lock a renter's create holds around
    `create_container` (RedisService.executor_create_exclusion) without waiting, skips the node when a
    renter holds it, and re-reads the node's idleness from the backend and this validator's pending-pod
    marks under the lock, not from the cycle-start snapshot; the lock is released as soon as the
    create returns. Anything the probe cannot read (backend, Redis) makes it skip or reach no verdict;
    it never penalises on missing data alone. RENTAL_PROBE_ENABLED is off by default.
    """

    check_id = "executor.validate.rental_probe"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        gate = await self._may_probe(ctx)
        if isinstance(gate, CheckResult):
            return gate
        standing = gate.standing
        image_ref = gate.image_ref
        try:
            outcome = await _probe(ctx, image_ref, gate.create_lock)
        finally:
            await gate.create_lock.release(ctx)
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
                ctx, outcome.inconclusive_reason, steps=outcome.steps, what=what, standing=standing
            )

        if outcome.failed_step is None:
            await _stamp_last_ok(ctx)
            await _clear_failure(ctx)
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
                ctx,
                "could not rule out a rental during the probe",
                steps=outcome.steps,
                what=what,
                standing=standing,
            )
        if rented_meanwhile:
            # rented: the standing failure is neither applied nor cleared (class docstring); a filler
            # that started during the probe is not a renter and carries it, as _busy_now's filler does
            return self._inconclusive(
                ctx,
                f"node was {rented_meanwhile} during the probe",
                steps=outcome.steps,
                what=what,
                standing=standing if rented_meanwhile == "given a filler" else None,
            )

        # a pass stamped earlier in the interval must not shield this failure, and the failure stands
        # until a probe passes (see _standing_failure)
        await _clear_last_ok(ctx)
        failure = _Failure(outcome.failed_step, outcome.create_step)
        await _stamp_failure(ctx, failure)
        what["failed_step"] = outcome.failed_step
        if outcome.create_step:
            what["create_step"] = outcome.create_step
        return self._failed(ctx, failure, what=what, ssh_port=outcome.ssh_port)

    async def _may_probe(self, ctx: Context) -> _ProbeGate | CheckResult:
        """May the probe run on this node now? The gate, or the verdict-less result that stops the cycle.

        In order: the feature flag, the standing failure, the cycle's skip reasons (rented, filler,
        interval), the validator's inputs, the default renter image and its presence on the node, the
        per-executor create lock, and the node's idleness re-read under that lock. A gate is returned
        with the lock held; every refusal after the lock was taken releases it first.
        """
        if not settings.RENTAL_PROBE_ENABLED:
            return CheckResult(
                passed=True, event=render_message(Msg.DISABLED, ctx=ctx, check_id=self.check_id)
            )

        # the last probe's failed step, standing until a probe passes; None when there is none or
        # Redis could not be read (then this cycle behaves as if there were none, and logs why)
        standing = await _standing_failure(ctx)

        skip_reason, skip_what = await _skip_reason(ctx)
        if skip_reason == "rented":
            return self._skipped(ctx, skip_reason, skip_what)
        if skip_reason:
            return self._skipped(ctx, skip_reason, skip_what, standing=standing)

        missing = _missing_inputs(ctx)
        if missing:
            return self._inconclusive(
                ctx, f"validator has no {missing} for this executor", steps=[], standing=standing
            )

        image_ref = await _default_renter_image(ctx)
        if image_ref is None:
            return self._inconclusive(
                ctx, "no default renter image for this GPU and driver", steps=[], standing=standing
            )
        image_present = await _image_present_on_host(ctx, image_ref)
        if image_present is None:
            return self._inconclusive(
                ctx,
                "could not read the node's image list",
                steps=[],
                what={"image": image_ref},
                standing=standing,
            )
        if not image_present:
            # a pull would put minutes of Docker Hub traffic inside the validation cycle; the
            # cached-template check already reports a node that has not pre-pulled the image
            return self._skipped(
                ctx,
                "default renter image is not pulled on the node",
                {"image": image_ref},
                standing=standing,
            )

        # Review: a renter's create holds the per-executor create lock around create_container. The
        # probe takes it without waiting (a renter never waits on the probe) and holds it until its own
        # create returns, so its sweep of pod_* containers cannot run beside a renter's create.
        create_lock = await _take_create_lock(ctx)
        if create_lock is None:
            return self._inconclusive(
                ctx, "could not take the node's create lock", steps=[], standing=standing
            )
        if not create_lock.held:
            # a renter's create: rented, as far as the standing failure is concerned
            return self._skipped(ctx, _RENT_IN_PROGRESS)

        try:
            refusal = await self._idle_under_lock(ctx, standing)
        except BaseException:
            await create_lock.release(ctx)
            raise
        if refusal is not None:
            await create_lock.release(ctx)
            return refusal
        return _ProbeGate(image_ref=image_ref, create_lock=create_lock, standing=standing)

    async def _idle_under_lock(self, ctx: Context, standing: _Failure | None) -> CheckResult | None:
        """Re-read the node's idleness under the create lock; the result that stops the cycle, or None.

        ctx.state.rented_data is the snapshot the cycle started from, minutes ago for the last check
        in the pipeline. create_container sweeps every pod_* container it is not told about, so the
        node must be idle NOW: a fresh backend read plus this validator's own pending-pod mark (a rent
        it is creating this moment). Unknown counts as busy.
        """
        reread_started = time.monotonic()
        busy_now = await _busy_now(ctx)
        if busy_now is None:
            return self._inconclusive(
                ctx, "could not confirm the node is idle", steps=[], standing=standing
            )
        if busy_now:
            return self._skipped(
                ctx, busy_now, standing=None if busy_now in _RENTED_NOW else standing
            )
        if time.monotonic() - reread_started > _IDLENESS_REREAD_BUDGET_SECONDS:
            # the lock's TTL would run out under the create; give it back rather than sweep late
            return self._inconclusive(
                ctx, "the idleness re-read overran its budget", steps=[], standing=standing
            )
        return None

    def _failed(
        self, ctx: Context, failure: _Failure, *, what: dict[str, Any], ssh_port: int | None
    ) -> CheckResult:
        event = render_message(
            Msg.PROBE_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            what=what,
            remediation=_remediation(
                failure, ssh_port=ssh_port, expected_gpus=_expected_gpu_count(ctx)
            ),
        )
        return CheckResult(
            passed=False,
            event=event,
            updates={
                "score": 0.0,
                "job_score": 0.0,
                "score_warning": f"Rental probe failed at {failure.step}",
                "clear_verified_job_info": True,
                "clear_verified_job_evidence": {
                    "reason_code": event.reason_code,
                    "check_id": self.check_id,
                    "failed_step": failure.step,
                    "create_step": failure.create_step,
                    "ssh_port": ssh_port,
                },
            },
        )

    def _carried(self, ctx: Context, standing: _Failure, *, no_verdict: str) -> CheckResult:
        """This cycle reached no verdict of its own and the last probe failed: fail the node again with
        that step (review). `no_verdict` is what this cycle would otherwise have reported."""
        what: dict[str, Any] = {
            "executor_uuid": ctx.executor.uuid,
            "failed_step": standing.step,
            "standing_failure": True,
            "no_verdict_this_cycle": no_verdict,
        }
        if standing.create_step:
            what["create_step"] = standing.create_step
        return self._failed(ctx, standing, what=what, ssh_port=None)

    def _skipped(
        self,
        ctx: Context,
        reason: str,
        what: dict[str, Any] | None = None,
        *,
        standing: _Failure | None = None,
    ) -> CheckResult:
        if standing is not None:
            return self._carried(ctx, standing, no_verdict=f"skipped: {reason}")
        event = render_message(
            Msg.SKIPPED,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid, "reason": reason, **(what or {})},
        )
        return CheckResult(passed=True, event=event)

    def _inconclusive(
        self,
        ctx: Context,
        reason: str,
        *,
        steps: list[_Step],
        what: dict[str, Any] | None = None,
        standing: _Failure | None = None,
    ) -> CheckResult:
        if standing is not None:
            return self._carried(ctx, standing, no_verdict=f"inconclusive: {reason}")
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


_RENTED_SINCE_CYCLE_START = "rented since the cycle started"
_RENT_IN_PROGRESS = "a rent is being created on the node"
# the busy reasons that mean a renter has the node: a standing failure is not applied on these
_RENTED_NOW = frozenset({_RENTED_SINCE_CYCLE_START, _RENT_IN_PROGRESS})


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
        return _RENTED_SINCE_CYCLE_START
    if rented.get_filler_containers(ctx.executor.uuid):
        return "filler started since the cycle started"
    try:
        if await ctx.services.redis.renting_in_progress(ctx.miner_hotkey, ctx.executor.uuid):
            return _RENT_IN_PROGRESS
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
            f"{_REDIS_LAST_OK_PREFIX}:{ctx.executor.uuid}",
            str(time.time()),
            ex=_REDIS_STAMP_TTL_SECONDS,
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


@dataclass(frozen=True)
class _Failure:
    step: str
    create_step: str | None = None


async def _standing_failure(ctx: Context) -> _Failure | None:
    """The last probe's failed step, kept until a probe passes; None when there is none or Redis could
    not be read (logged; this cycle then decides as if none stood, which is what today's code does)."""
    try:
        raw = await ctx.services.redis.get(f"{_REDIS_FAILED_PREFIX}:{ctx.executor.uuid}")
    except Exception:
        _read_failed(ctx, "its standing failure in Redis")
        return None
    if raw is None:
        return None
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    step, _, create_step = text.partition(":")
    return _Failure(step, create_step or None) if step else None


async def _stamp_failure(ctx: Context, failure: _Failure) -> None:
    # no lifetime: the failure stands until _clear_failure, however long the node is skipped or
    # inconclusive in between (see _REDIS_FAILED_PREFIX)
    try:
        await ctx.services.redis.set(
            f"{_REDIS_FAILED_PREFIX}:{ctx.executor.uuid}",
            f"{failure.step}:{failure.create_step or ''}",
        )
    except Exception:
        logger.warning(
            _m(
                "Rental probe failed but the failure could not be recorded; a cycle without a verdict may relist the node",
                extra=get_extra_info(ctx.default_extra),
            ),
            exc_info=True,
        )


async def _clear_failure(ctx: Context) -> None:
    try:
        await ctx.services.redis.delete(f"{_REDIS_FAILED_PREFIX}:{ctx.executor.uuid}")
    except Exception:
        logger.warning(
            _m(
                "Rental probe passed but its standing failure could not be cleared; the node fails until it is",
                extra=get_extra_info(ctx.default_extra),
            ),
            exc_info=True,
        )


@dataclass
class _CreateLock:
    """The per-executor create lock the probe holds from its idleness re-read until its create returns."""

    lock: Any
    held: bool

    async def release(self, ctx: Context) -> None:
        if not self.held:
            return
        self.held = False
        try:
            await self.lock.release()
        except Exception:
            # a release past the TTL (LockNotOwnedError) or with Redis gone: the lock is not held anyway
            logger.warning(
                _m(
                    "Rental probe could not release the node's create lock",
                    extra=get_extra_info(ctx.default_extra),
                ),
                exc_info=True,
            )


async def _take_create_lock(ctx: Context) -> _CreateLock | None:
    """The create lock, held or not (a renter's create holds it); None when Redis could not answer."""
    lock = ctx.services.redis.executor_create_lock(ctx.executor.uuid)
    try:
        held = bool(await lock.acquire(blocking=False))
    except Exception:
        _read_failed(ctx, "the node's create lock in Redis")
        return None
    return _CreateLock(lock, held)


async def _rented_meanwhile(ctx: Context) -> str | None:
    """How the node was taken while the probe ran, "" when it was not, None when that could not be read.

    "rented": the backend lists a pod on it now, or this validator holds a pending-pod mark for it (a
    renter's create it is running at this moment; the probe's own mark is already cleared when this
    runs). "given a filler": the backend lists a filler on it now; a filler's create goes through the
    same create_container and sweeps `pod_<probe>` the same way, and its pending mark is gone by the
    time this runs.
    """
    try:
        rented = await ctx.services.backend.get_all_rented_executors()
    except Exception:
        _read_failed(ctx, "the backend's rented executors after the probe")
        return None
    if rented is None:
        return None
    rented_executor = rented.executors.get(ctx.executor.uuid)
    if rented_executor and rented_executor.pods:
        return "rented"
    if rented.get_filler_containers(ctx.executor.uuid):
        return "given a filler"
    try:
        if await ctx.services.redis.renting_in_progress(ctx.miner_hotkey, ctx.executor.uuid):
            return "rented"
    except Exception:
        _read_failed(ctx, "its pending-pod marks in Redis after the probe")
        return None
    return ""


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


async def _probe(ctx: Context, image_ref: str, create_lock: _CreateLock) -> _ProbeOutcome:
    """Run the five steps in order; whatever happens, the node and this validator's Redis are left as found.

    1. `_step_container_start`: create the container the way a renter's create does.
    2. `_step_sshd_listen`: wait for sshd's banner on the mapped port.
    3. `_step_ssh_login`: log in with the probe's key, retrying until the deadline, and run `nvidia-smi -L`.
    4. `_step_gpu_count`: compare the listed GPUs with what the node advertises.
    5. `_step_teardown`: remove the container and the pending-pod mark, whichever step ended the probe.

    A step that ends the probe returns None and says why in `outcome` (a failed step, or an inconclusive
    reason). The teardown runs in `finally`, so it also runs after a raise or a cancel.
    """
    outcome = _ProbeOutcome()
    pod_id = str(uuid.uuid4())
    private_key, public_key = ctx.services.ssh.generate_keypair()
    log_extra = {**ctx.default_extra, "probe_pod_id": pod_id, "image": image_ref}
    created: ContainerCreated | None = None

    try:
        created = await _step_container_start(
            ctx,
            outcome,
            pod_id=pod_id,
            image_ref=image_ref,
            public_key=public_key,
            create_lock=create_lock,
        )
        if created is None:
            return outcome
        listening = await _step_sshd_listen(ctx, outcome, created)
        if listening is None:
            return outcome
        ssh_port, deadline_at = listening
        login = await _step_ssh_login(ctx, outcome, private_key, ssh_port, deadline_at=deadline_at)
        if login is None:
            return outcome
        _step_gpu_count(ctx, outcome, login)
        return outcome
    finally:
        await _step_teardown(ctx, outcome, created=created, pod_id=pod_id, log_extra=log_extra)


async def _step_container_start(
    ctx: Context,
    outcome: _ProbeOutcome,
    *,
    pod_id: str,
    image_ref: str,
    public_key: str,
    create_lock: _CreateLock,
) -> ContainerCreated | None:
    """Step 1: start the container through create_container, the path a renter's pod takes.

    The container's name and volume follow from the pod id (`pod_<id>`, `volume_<id>`), so the
    teardown can remove them by name over the validation shell even when create_container returned
    no ContainerCreated: it failed on the host after `docker_run`, was cut short by its own deadline
    here, raised, or was cancelled by the run's outer timeout (JOB_TIME_OUT), which arrives as
    CancelledError and passes every `except Exception`. The create lock is released as soon as
    create_container returns, raises or is cut short: a renter must wait behind the probe's sweep of
    `pod_*` containers, not behind its sshd wait or teardown. Returns the created container, or None
    when the probe ends here.
    """
    docker = ctx.services.pod_recovery
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
        return None
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
        return None
    finally:
        await asyncio.shield(create_lock.release(ctx))

    if not isinstance(result, ContainerCreated):
        detail, create_step = _failure_detail(result)
        outcome.steps.append(
            _Step(STEP_CONTAINER_START, time.perf_counter() - started, False, detail)
        )
        outcome.create_step = create_step
        if create_step is None:
            outcome.inconclusive_reason = "create returned a result without a failure step"
        elif create_step in _CREATE_STEPS_BEFORE_THE_HOST:
            outcome.inconclusive_reason = f"create failed on the validator's side at {create_step}"
        else:
            outcome.failed_step = STEP_CONTAINER_START
        return None
    outcome.steps.append(_Step(STEP_CONTAINER_START, time.perf_counter() - started, True))
    return result


async def _step_sshd_listen(
    ctx: Context, outcome: _ProbeOutcome, created: ContainerCreated
) -> tuple[int, float] | None:
    """Step 2: wait for sshd's banner on the mapped port.

    Returns the mapped port and the deadline (a `time.monotonic()` instant) the login retries share
    with this wait: one deadline for the renter's first minute. None when the probe ends here.
    """
    ssh_port = _mapped_ssh_port(created)
    outcome.ssh_port = ssh_port
    if ssh_port is None:
        # generate_portMappings always maps 22 first; a create without it is the validator's mapping
        outcome.steps.append(
            _Step(STEP_SSHD_LISTEN, 0.0, False, "no port mapping for container port 22")
        )
        outcome.inconclusive_reason = "create returned no mapping for container port 22"
        return None

    deadline_seconds = settings.RENTAL_PROBE_SSH_DEADLINE_SECONDS
    deadline_at = time.monotonic() + deadline_seconds
    started = time.perf_counter()
    listen_error = await _wait_for_sshd(
        ctx.executor.address,
        ssh_port,
        deadline_at=deadline_at,
        deadline_seconds=deadline_seconds,
    )
    outcome.steps.append(
        _Step(STEP_SSHD_LISTEN, time.perf_counter() - started, listen_error is None, listen_error)
    )
    if listen_error is not None:
        outcome.failed_step = STEP_SSHD_LISTEN
        return None
    return ssh_port, deadline_at


async def _step_ssh_login(
    ctx: Context, outcome: _ProbeOutcome, private_key: str, ssh_port: int, *, deadline_at: float
) -> _Login | None:
    """Step 3: log in with the probe's key, retrying until the deadline, and run `nvidia-smi -L`.

    Returns the login (with the command's result) for the GPU count, or None when the probe ends here.
    """
    login = await _login_and_list_gpus(
        ctx.executor.address, ssh_port, private_key, deadline_at=deadline_at
    )
    outcome.steps.append(
        _Step(STEP_SSH_LOGIN, login.login_seconds, login.login_error is None, login.login_detail())
    )
    if login.login_error is not None:
        outcome.failed_step = STEP_SSH_LOGIN
        return None
    return login


def _step_gpu_count(ctx: Context, outcome: _ProbeOutcome, login: _Login) -> None:
    """Step 4: `nvidia-smi -L` inside the container must list the GPUs this node advertises."""
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


async def _step_teardown(
    ctx: Context,
    outcome: _ProbeOutcome,
    *,
    created: ContainerCreated | None,
    pod_id: str,
    log_extra: dict[str, Any],
) -> None:
    """Step 5: `_settle`, run to the end even when the probe's task is cancelled.

    The settle runs as its own task and is awaited to the end: a cancel that lands while it runs
    (the first one, or a second) interrupts only this wait, never the settle, and is re-raised once
    the settle is done so the run still ends as cancelled.
    """
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

    A created container goes through delete_container, the renter's unrent path. In every other case
    where the create reached the host (it failed after `ssh_connect`, ran into its deadline, was
    cancelled, or raised), and when the unrent path fails, the container and its volume are removed by
    name over the validation shell. create_container does run `docker rm -fv` itself when a step after
    `docker_run` fails (`cleanup_failed_container_creation`), but that cleanup is best effort: its own
    failure is logged and swallowed, and a cancel or the deadline here skips it. A `pod_<id>` left
    behind holds the GPUs and the verified ports until the stale sweep, so the by-name removal runs
    as the backstop in every path; a name that is already gone costs one shell command. Only a create
    that failed before the host (`_CREATE_STEPS_BEFORE_THE_HOST`) removes nothing. The pending-pod
    mark is cleared in every case. An unrent path that needed the fallback is recorded as a failed
    step: the node served the renter but its unrent path did not finish, so a clean run is
    INCONCLUSIVE (no stamp, probed again next cycle) rather than a pass.
    """
    try:
        if created is None and outcome.create_step in _CREATE_STEPS_BEFORE_THE_HOST:
            return  # the create failed before the validator's SSH session to the host
        started = time.perf_counter()
        teardown_error: str | None = None
        leftover: str | None = None
        if created is not None:
            teardown_error = await _teardown(ctx, created, pod_id)
        if created is None or teardown_error is not None:
            # the names the create used when it returned them; the names it would have used otherwise
            container_name = (
                created.container_name if created is not None else f"{POD_CONTAINER_PREFIX}{pod_id}"
            )
            volume_name = created.volume_name if created is not None else f"volume_{pod_id}"
            leftover = await _remove_over_shell(
                ctx, container_name=container_name, volume_name=volume_name
            )
        if created is None:
            why = f"create ended at {outcome.create_step or 'an unknown step'}"
        else:
            why = teardown_error
        if why is None:
            detail = None
        elif leftover is None:
            # `docker rm -fv` by name exits 0 for a container it removed and 1 "No such container" for
            # one that was already gone; both leave nothing on the node
            detail = f"{why}; docker rm by name over the validation shell left nothing"
        else:
            detail = f"{why}; shell removal failed too: {leftover}"
        outcome.steps.append(
            _Step(
                STEP_TEARDOWN,
                time.perf_counter() - started,
                teardown_error is None and leftover is None,
                detail,
            )
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
        # review: DockerCommand.volume_remove masks its exit with `|| true`; the probe needs the answer
        volume_removed = await asyncio.wait_for(
            ctx.ssh.run(DockerCommand.volume_remove_strict(volume_name), check=False),
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
    if removed.exit_status != 0 and "No such container" not in (removed.stderr or ""):
        return f"docker rm exited {removed.exit_status}: {(removed.stderr or '')[-_TAIL_CHARS:]}"
    # the same for the named volume: "no such volume" (either spelling the CLI has used) is nothing
    # left behind, anything else is unknown
    if (
        volume_removed.exit_status != 0
        and "no such volume" not in (volume_removed.stderr or "").lower()
    ):
        return (
            f"docker volume rm exited {volume_removed.exit_status}: "
            f"{(volume_removed.stderr or '')[-_TAIL_CHARS:]}"
        )
    return None


def _failure_detail(result: Any) -> tuple[str, str | None]:
    if isinstance(result, FailedContainerRequest):
        return (result.detail or result.msg or "create failed")[-_TAIL_CHARS:], result.failure_step
    return f"unexpected create result {type(result).__name__}", None


def _mapped_ssh_port(created: ContainerCreated) -> int | None:
    for docker_port, external_port in created.port_maps:
        if docker_port == 22:
            return external_port
    return None


async def _wait_for_sshd(
    host: str, port: int, *, deadline_at: float, deadline_seconds: float
) -> str | None:
    """None once the mapped port answers with sshd's `SSH-2.0` banner, else the last error at the deadline.

    A TCP accept alone proves nothing: docker-proxy accepts on a published port as soon as the
    container exists, before sshd inside it listens (the default image's start.sh runs `service ssh
    start` after its own setup), and closes the connection at once. The banner is sshd's first
    write on every connection, so it is what a renter's client waits for too.
    """
    last_error = "never tried"
    while True:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=_SSHD_CONNECT_TIMEOUT_SECONDS
            )
            try:
                banner = await asyncio.wait_for(
                    reader.readline(), timeout=_SSHD_BANNER_TIMEOUT_SECONDS
                )
            finally:
                writer.close()
            if banner.startswith(_SSH_BANNER_PREFIX):
                return None
            last_error = (
                f"port {port} accepted the connection but sent no SSH banner "
                f"({banner[:40]!r}; docker-proxy answers before sshd listens)"
            )
        except (TimeoutError, OSError, ValueError) as exc:
            # ValueError: readline's line limit, a server that talks but not SSH
            last_error = repr(exc)
        if time.monotonic() >= deadline_at:
            return f"sshd did not answer on port {port} within {deadline_seconds} s: {last_error}"
        await asyncio.sleep(_SSHD_POLL_SECONDS)


@dataclass
class _Login:
    login_error: str | None = None
    login_attempts: int = 0
    command_error: str | None = None
    result: Any = None
    login_seconds: float = 0.0
    command_seconds: float = 0.0

    def login_detail(self) -> str | None:
        if self.login_error is not None:
            return f"{self.login_error} (attempt {self.login_attempts})"
        if self.login_attempts > 1:
            return f"connected on attempt {self.login_attempts}"
        return None


async def _login_and_list_gpus(
    host: str, port: int, private_key: str, *, deadline_at: float
) -> _Login:
    """Log in with the probe key the way a renter does and run `nvidia-smi -L` inside the container.

    The connect is retried with backoff until `deadline_at` (the sshd step's deadline, shared): a
    freshly started sshd drops the first connections while it is still forking or under MaxStartups,
    and a renter's client retries too. A refused key (PermissionDenied) is final: the create wrote
    authorized_keys before sshd started, so it will not change. The two phases are reported apart: a
    connection that never opened is the login step's failure, a session that opened but could not run
    the command is the GPU step's.
    """
    login = _Login()
    started = time.perf_counter()
    try:
        pkey = asyncssh.import_private_key(private_key)
    except (ValueError, asyncssh.Error) as exc:  # KeyImportError is a ValueError
        login.login_error = repr(exc)
        login.login_attempts = 1
        login.login_seconds = time.perf_counter() - started
        return login
    while True:
        login.login_attempts += 1
        try:
            conn = await asyncssh.connect(
                host=host,
                port=port,
                username="root",
                client_keys=[pkey],
                known_hosts=None,
                connect_timeout=_SSH_LOGIN_TIMEOUT_SECONDS,
            )
            break
        except asyncssh.PermissionDenied as exc:
            login.login_error = repr(exc)
            login.login_seconds = time.perf_counter() - started
            return login
        except (TimeoutError, asyncssh.Error, OSError) as exc:
            login.login_error = repr(exc)
        backoff = _SSH_LOGIN_BACKOFF_SECONDS[
            min(login.login_attempts, len(_SSH_LOGIN_BACKOFF_SECONDS)) - 1
        ]
        if time.monotonic() + backoff >= deadline_at:
            login.login_seconds = time.perf_counter() - started
            return login
        await asyncio.sleep(backoff)
    login.login_error = None
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


def _remediation(failure: _Failure, *, ssh_port: int | None, expected_gpus: int) -> str:
    template = _REMEDIATION_BY_STEP.get(failure.step, "")
    return template.format(
        create_step=failure.create_step or "unknown",
        port=ssh_port if ssh_port is not None else "?",
        deadline=settings.RENTAL_PROBE_SSH_DEADLINE_SECONDS,
        expected=expected_gpus,
    )
