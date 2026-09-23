"""DAH-3436: the synthetic rental probe.

Every test names the regression that fails it. The probe rents a real container on a provider's node,
so most regressions here are of the "rents when it must not" or "penalises when it must not" kind.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest
from neurons.validators.src.services.task import pipeline_factory
from neurons.validators.src.services.task.checks import rental_probe as module
from neurons.validators.src.services.task.checks.rental_probe import (
    STEP_CONTAINER_START,
    STEP_EGRESS,
    STEP_GPU_COUNT,
    STEP_SSH_LOGIN,
    STEP_SSHD_LISTEN,
    STEP_TEARDOWN,
    RentalProbeCheck,
)
from neurons.validators.src.services.task.messages import RentalProbeMessages as Msg
from neurons.validators.src.services.task.pipeline import (
    Pipeline,
    updates_with_clear_verified_job_evidence,
)
from payload_models.payloads import (
    ContainerCreated,
    ContainerCreateRequest,
    ContainerDeleted,
    ContainerDeleteRequest,
    FailedContainerErrorCodes,
    FailedContainerErrorTypes,
    FailedContainerRequest,
)
from protocol.vc_protocol.compute_requests import (
    DefaultDockerImage,
    RentedExecutor,
    RentedExecutorsResponse,
    RentedPod,
)

from services.ssh_service import SSHService
from tests.helpers import (
    build_context_config,
    build_services,
    build_state,
    default_executor,
    make_context,
)

EXECUTOR = default_executor()
MINER = "miner-hotkey"
IMAGE = DefaultDockerImage(docker_image="daturaai/pytorch", docker_image_tag="2.7.0-cuda12.8")
GPU_DETAILS = [{"uuid": "GPU-aaaa", "name": "RTX 5090"}, {"uuid": "GPU-bbbb", "name": "RTX 5090"}]
# a NAT'd node: the container binds 20001 on the host, renters reach it on 30001
PORT_PAIRS = [(20001, 30001), (20002, 30002), (20003, 30003)]
SMI_TWO_GPUS = (
    "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-aaaa)\n"
    "GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-bbbb)\n"
)


@contextmanager
def probe_settings(
    *,
    enabled: bool = True,
    interval_hours: float = 6.0,
    deadline: int = 1,
    egress_check: bool = False,
    egress_enforced: bool = False,
):
    with patch("neurons.validators.src.services.task.checks.rental_probe.settings") as s:
        s.RENTAL_PROBE_ENABLED = enabled
        s.RENTAL_PROBE_INTERVAL_HOURS = interval_hours
        s.RENTAL_PROBE_SSH_DEADLINE_SECONDS = deadline
        s.NO_OUTBOUND_INTERNET_CHECK_ENABLED = egress_check
        s.NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED = egress_enforced
        yield s


def created(pod_id: str = "pod") -> ContainerCreated:
    return ContainerCreated(
        miner_hotkey=MINER,
        executor_id=EXECUTOR.uuid,
        pod_id=pod_id,
        container_name=f"pod_{pod_id}",
        volume_name=f"volume_{pod_id}",
        # docker port 22 lands on the node's second verified pair, as generate_portMappings may pick
        port_maps=[(22, 30002)],
    )


def create_failed(failure_step: str) -> FailedContainerRequest:
    return FailedContainerRequest(
        miner_hotkey=MINER,
        executor_id=EXECUTOR.uuid,
        pod_id="pod",
        msg="Failed create_container",
        detail="Failed create_container: docker: Error response from daemon: OCI runtime create failed",
        error_type=FailedContainerErrorTypes.ContainerCreationFailed,
        error_code=FailedContainerErrorCodes.UnknownError,
        failure_step=failure_step,
    )


class FakeDocker:
    """The two DockerService entry points the probe reuses, answering with canned results."""

    def __init__(self, *, create_result=None, delete_result=None):
        self.create_result = create_result if create_result is not None else created()
        self.delete_result = (
            delete_result
            if delete_result is not None
            else ContainerDeleted(miner_hotkey=MINER, executor_id=EXECUTOR.uuid, pod_id="pod")
        )
        self.create_calls: list[tuple] = []
        self.delete_calls: list[tuple] = []

    async def create_container(self, payload, executor_info, keypair, private_key):
        self.create_calls.append((payload, executor_info, keypair, private_key))
        if isinstance(self.create_result, Exception):
            raise self.create_result
        return self.create_result

    async def delete_container(self, payload, executor_info, keypair, private_key):
        self.delete_calls.append((payload, executor_info, keypair, private_key))
        return self.delete_result

    async def recover_pod_after_stale_vloopback_mount(self, **kwargs):  # PodRecoverer
        return False


class FakeCreateLock:
    """`RedisService.executor_create_lock`'s Lock as the probe uses it: `acquire(blocking=False)`, `release()`."""

    def __init__(self, redis: FakeRedis):
        self.redis = redis

    async def acquire(self, blocking=True):
        if self.redis.create_lock_broken:
            raise ConnectionError("redis down")
        if self.redis.create_lock_held_by_renter:
            return False
        self.redis.create_lock_events.append("acquired")
        return True

    async def release(self):
        self.redis.create_lock_events.append("released")


class FakeRedis:
    def __init__(
        self,
        last_ok: float | None = None,
        *,
        renting=False,
        broken: bool = False,
        failed_step: str | None = None,
        create_lock_held_by_renter: bool = False,
        create_lock_broken: bool = False,
    ):
        self.store: dict[str, str] = {}
        if last_ok is not None:
            self.store[f"rental_probe_ok:{EXECUTOR.uuid}"] = str(last_ok)
        if failed_step is not None:
            self.store[f"rental_probe_failed:{EXECUTOR.uuid}"] = f"{failed_step}:"
        self.removed_pending: list[tuple[str, str, str]] = []
        self.removed_rented: list[str] = []
        # a bool, or a list with one answer per renting_in_progress call (before the probe, after a failure)
        self.renting = renting
        self.broken = broken
        self.create_lock_held_by_renter = create_lock_held_by_renter
        self.create_lock_broken = create_lock_broken
        # the probe's create-lock acquires and releases, in order
        self.create_lock_events: list[str] = []
        self.expiries: dict[str, int | None] = {}

    def executor_create_lock(self, executor_id):
        assert executor_id == EXECUTOR.uuid
        return FakeCreateLock(self)

    async def get(self, key):
        if self.broken:
            raise ConnectionError("redis down")
        value = self.store.get(key)
        return value.encode() if value is not None else None

    async def renting_in_progress(self, miner_hotkey, executor_id, pod_id=None):
        if isinstance(self.renting, list):
            return self.renting.pop(0)
        return self.renting

    async def remove_rented_machine(self, executor, container_name=None):
        self.removed_rented.append(container_name)

    async def set(self, key: str, value: str, ex: int | None = None):
        # the real RedisService.set signature; the OK stamp passes `ex` (30-day lifetime), the failure none
        self.store[key] = value
        self.expiries[key] = ex

    async def delete(self, key):
        self.store.pop(key, None)
        self.expiries.pop(key, None)

    def age_past_every_lifetime(self) -> None:
        # what Redis does once every `ex` has run out: a key written with a lifetime is gone
        for key, ex in list(self.expiries.items()):
            if ex is not None:
                self.store.pop(key, None)
                self.expiries.pop(key, None)

    async def remove_pending_pod(self, miner_hotkey, executor_id, pod_id):
        self.removed_pending.append((miner_hotkey, executor_id, pod_id))

    def stamped(self) -> bool:
        return f"rental_probe_ok:{EXECUTOR.uuid}" in self.store

    def standing_failure(self) -> str | None:
        raw = self.store.get(f"rental_probe_failed:{EXECUTOR.uuid}")
        return raw.split(":")[0] if raw else None


def rented_data(*, pods: int = 0, fillers: int = 0) -> RentedExecutorsResponse:
    executors = {}
    if pods:
        executors[EXECUTOR.uuid] = RentedExecutor(
            miner_hotkey=MINER,
            executor_ip_address=EXECUTOR.address,
            executor_ip_port=str(EXECUTOR.port),
            pods=[RentedPod(pod_id=f"p{i}", container_name=f"pod_p{i}") for i in range(pods)],
        )
    return RentedExecutorsResponse(
        executors=executors,
        all_filler_containers_by_executor={EXECUTOR.uuid: [f"filler_{i}" for i in range(fillers)]}
        if fillers
        else {},
    )


def make_probe_context(
    *,
    docker: FakeDocker | None = None,
    redis: FakeRedis | None = None,
    rented: RentedExecutorsResponse | None = None,
    port_pairs=PORT_PAIRS,
    image_present: bool = True,
    images=(IMAGE,),
    rented_now=None,
):
    docker = docker or FakeDocker()
    redis = redis or FakeRedis()
    backend = AsyncMock()
    backend.get_default_docker_image.return_value = list(images) if images else None
    # the fresh backend reads: before the probe (is the node idle now?) and, on a failed verdict, after
    # it (was it rented meanwhile?). A list gives one answer per call.
    if isinstance(rented_now, list):
        backend.get_all_rented_executors.side_effect = rented_now
    else:
        backend.get_all_rented_executors.return_value = (
            rented_now if rented_now is not None else rented_data()
        )
    services = build_services(pod_recovery=docker, redis=redis, backend=backend, ssh=SSHService())
    ssh = AsyncMock()
    ssh.run.return_value = MagicMock(
        exit_status=0 if image_present else 1, stdout="sha256:abc" if image_present else ""
    )
    state = build_state(
        specs={
            "verified_ports": [ext for _, ext in port_pairs],
            "gpu": {"driver": "580.65.06", "details": GPU_DETAILS},
        },
        gpu_model="RTX 5090",
        gpu_count=2,
        gpu_details=GPU_DETAILS,
        sysbox_runtime=True,
        verified_port_pairs=list(port_pairs),
        rented_data=rented if rented is not None else rented_data(),
    )
    ctx = make_context(
        services=services,
        state=state,
        config=build_context_config(validator_keypair=object()),
        ssh=ssh,
        executor_ssh_private_key_encrypted="gAAAAA-encrypted-executor-key",
        miner_hotkey=MINER,
    )
    return ctx, docker, redis


class FakeSSHConnection:
    """What `await asyncssh.connect(...)` gives back: an async context manager with `run`."""

    def __init__(
        self,
        stdout: str,
        exit_status: int,
        command_error: Exception | None,
        egress_stdout: str | None = None,
    ):
        self.stdout = stdout
        self.exit_status = exit_status
        self.command_error = command_error
        self.egress_stdout = egress_stdout
        self.commands: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, command, *, check=False, timeout=None):
        self.commands.append(command)
        if self.command_error is not None:
            raise self.command_error
        if command.startswith("sh -c ") and self.egress_stdout is not None:
            return MagicMock(stdout=self.egress_stdout, stderr="", exit_status=0)
        return MagicMock(stdout=self.stdout, stderr="", exit_status=self.exit_status)


SSH_BANNER = b"SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\r\n"


class FakeReader:
    def __init__(self, first_line: bytes):
        self.first_line = first_line

    async def readline(self):
        return self.first_line


@contextmanager
def renter_path(
    *,
    sshd_listens: bool = True,
    sshd_up_after: int | None = 0,
    login_error: Exception | list | None = None,
    command_error: Exception | None = None,
    smi_stdout: str = SMI_TWO_GPUS,
    smi_exit: int = 0,
    egress_stdout: str | None = None,
):
    """What the probe sees from the validator's side of the network: the TCP port and the login.

    `sshd_listens=False` refuses the TCP connection (no docker-proxy, no container port published).
    `sshd_up_after=N` is a published port whose container has not started sshd yet: docker-proxy
    accepts the first N connections and closes them with no banner, and an SSH login in that window is
    dropped; from connection N+1 sshd answers with its banner. None: sshd never comes up.
    `login_error` is one exception for every attempt, or a list with one entry per attempt (None = success).
    """
    seen: dict = {"tcp_connects": 0, "ssh_attempts": 0}
    login_errors = list(login_error) if isinstance(login_error, list) else None

    def sshd_up() -> bool:
        return sshd_up_after is not None and seen["tcp_connects"] > sshd_up_after

    async def open_connection(host, port):
        seen["tcp"] = (host, port)
        if not sshd_listens:
            raise ConnectionRefusedError(111, "Connection refused")
        seen["tcp_connects"] += 1
        return FakeReader(SSH_BANNER if sshd_up() else b""), MagicMock()

    async def connect(**kwargs):
        seen["ssh"] = kwargs
        seen["ssh_attempts"] += 1
        if not sshd_up():
            raise asyncssh.ConnectionLost("Connection lost")
        if login_errors is not None:
            error = login_errors.pop(0) if login_errors else None
        else:
            error = login_error
        if error is not None:
            raise error
        seen["conn"] = FakeSSHConnection(smi_stdout, smi_exit, command_error, egress_stdout)
        return seen["conn"]

    with (
        patch.object(module.asyncio, "open_connection", open_connection),
        patch.object(module.asyncssh, "connect", connect),
        patch.object(module, "_SSHD_POLL_SECONDS", 0.0),
        patch.object(module, "_SSH_LOGIN_BACKOFF_SECONDS", (0.01,)),
    ):
        yield seen


@pytest.mark.asyncio
async def test_off_by_default_rents_nothing():
    """Regression: a validator that did not opt in rents a container on every idle node each cycle, either
    because the check ignores the flag or because the setting's default was flipped to True."""
    assert type(module.settings).model_fields["RENTAL_PROBE_ENABLED"].default is False
    ctx, docker, _ = make_probe_context()
    with probe_settings(enabled=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed and result.event.reason_code == Msg.DISABLED.reason
    assert docker.create_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rented,expected_reason",
    [
        (rented_data(pods=1), "rented"),
        (rented_data(fillers=1), "filler running"),
    ],
)
async def test_never_rents_beside_a_renter_or_a_filler(rented, expected_reason):
    """Regression: create_container sweeps every pod_/filler_ container it was not told about and lifts GPU
    power caps, so a probe beside a renter or a PEARL filler destroys their container or uncaps their card."""
    ctx, docker, _ = make_probe_context(rented=rented)
    with probe_settings():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed and result.event.reason_code == Msg.SKIPPED.reason
    assert result.event.what_we_saw["reason"] == expected_reason
    assert docker.create_calls == []


@pytest.mark.asyncio
async def test_interval_skips_a_recently_passed_node():
    """Regression: the interval is ignored (a container per node per cycle)."""
    recent = FakeRedis(last_ok=time.time() - 3600)
    ctx, docker, _ = make_probe_context(redis=recent)
    with probe_settings(interval_hours=6):
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.event.what_we_saw["reason"] == "within interval"
    assert docker.create_calls == []

    ctx, docker, _ = make_probe_context(redis=FakeRedis(last_ok=time.time() - 7 * 3600))
    with probe_settings(interval_hours=6), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.PROBE_OK.reason
    assert len(docker.create_calls) == 1


@pytest.mark.asyncio
async def test_a_failed_probe_clears_an_earlier_pass_stamp_and_records_the_failed_step():
    """Regression: a pass stamped earlier in the interval shields the failure, so the next cycle skips the
    node as "within interval" and relists it; or the failure is not recorded, so a cycle without a verdict
    relists it (review)."""
    redis = FakeRedis(last_ok=time.time() - 7 * 3600)
    ctx, docker, _ = make_probe_context(redis=redis)
    with probe_settings(interval_hours=6, deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is False and result.event.reason_code == Msg.PROBE_FAILED.reason
    assert not redis.stamped()
    assert redis.standing_failure() == STEP_SSHD_LISTEN
    # review: the failure has no lifetime; only a passed probe clears it
    assert redis.expiries[f"rental_probe_failed:{EXECUTOR.uuid}"] is None

    # next cycle: the node is probed again, not skipped
    ctx, docker, _ = make_probe_context(redis=redis)
    with probe_settings(interval_hours=6, deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.PROBE_FAILED.reason
    assert len(docker.create_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_ctx, no_verdict",
    [
        (
            lambda redis: make_probe_context(redis=redis, rented=rented_data(fillers=1)),
            "skipped: filler running",
        ),
        (
            lambda redis: make_probe_context(redis=redis, image_present=False),
            "skipped: default renter image is not pulled on the node",
        ),
        (
            lambda redis: make_probe_context(redis=redis, rented_now=[None]),
            "inconclusive: could not confirm the node is idle",
        ),
        (
            lambda redis: make_probe_context(
                redis=redis, docker=FakeDocker(create_result=create_failed("attestation"))
            ),
            "inconclusive: create failed on the validator's side at attestation",
        ),
    ],
)
async def test_a_standing_failure_holds_through_a_cycle_without_a_verdict(make_ctx, no_verdict):
    """Regression (review): a failed probe only clears verification for one cycle, so a skipped or
    inconclusive next cycle relists the node without a clean probe. The failed step stands until a probe
    passes, and a cycle that reaches no verdict of its own fails the node with it again."""
    redis = FakeRedis(failed_step=STEP_SSHD_LISTEN)
    ctx, docker, _ = make_ctx(redis)
    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is False and result.event.reason_code == Msg.PROBE_FAILED.reason
    assert result.event.what_we_saw["failed_step"] == STEP_SSHD_LISTEN
    assert result.event.what_we_saw["standing_failure"] is True
    assert result.event.what_we_saw["no_verdict_this_cycle"] == no_verdict
    assert result.updates["clear_verified_job_info"] is True
    assert result.updates["clear_verified_job_evidence"]["failed_step"] == STEP_SSHD_LISTEN
    assert redis.standing_failure() == STEP_SSHD_LISTEN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_ctx, reason_code, reason",
    [
        (
            lambda redis: make_probe_context(redis=redis, rented=rented_data(pods=1)),
            Msg.SKIPPED.reason,
            "rented",
        ),
        (
            lambda redis: make_probe_context(redis=redis, rented_now=[rented_data(pods=1)]),
            Msg.SKIPPED.reason,
            "rented since the cycle started",
        ),
        (
            lambda redis: make_probe_context(redis=redis, rented_now=[rented_data()]),
            Msg.SKIPPED.reason,
            "a rent is being created on the node",
        ),
        (
            lambda redis: make_probe_context(
                redis=redis, rented_now=[rented_data(), rented_data(pods=1)]
            ),
            Msg.INCONCLUSIVE.reason,
            "node was rented during the probe",
        ),
    ],
)
async def test_a_standing_failure_leaves_a_rented_node_alone_whichever_read_says_so(
    make_ctx, reason_code, reason
):
    """Regression (self-review): the standing failure was applied when the rent showed up after the cycle's
    snapshot (the re-read, a renter's create holding the lock, a rental during the probe) and only the
    snapshot's "rented" was exempt: a node rented with a failed step in Redis lost its score and verified job
    with the renter's pod on it. The probe cannot run on a rented node and the renter's pod is the backend's
    to judge, so the failure is neither applied nor cleared there."""
    redis = FakeRedis(failed_step=STEP_GPU_COUNT)
    if reason == "a rent is being created on the node":
        redis.renting = True
    ctx, docker, _ = make_ctx(redis)
    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == reason_code
    assert result.event.what_we_saw["reason"] == reason and result.updates == {}
    assert redis.standing_failure() == STEP_GPU_COUNT


@pytest.mark.asyncio
async def test_a_standing_failure_ends_with_a_passed_probe():
    redis = FakeRedis(failed_step=STEP_GPU_COUNT)
    ctx, docker, _ = make_probe_context(redis=redis)
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.PROBE_OK.reason
    assert redis.standing_failure() is None and redis.stamped()


@pytest.mark.asyncio
async def test_a_standing_failure_outlives_every_stamp_lifetime_until_a_probe_passes():
    """Regression (review): the failure was written with the OK stamp's 30-day lifetime, so a node skipped
    or inconclusive for longer than that (image never pulled, a filler always on it) was relisted without a
    clean probe once the key expired. The failure now has no lifetime; only a passed probe clears it."""
    redis = FakeRedis()
    ctx, docker, _ = make_probe_context(redis=redis)
    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        assert (await RentalProbeCheck().run(ctx)).passed is False
    assert redis.standing_failure() == STEP_SSHD_LISTEN

    # 31 days of cycles that reach no verdict: Redis drops every key that had a lifetime, the failure stays
    redis.age_past_every_lifetime()
    for _ in range(3):
        ctx, docker, _ = make_probe_context(redis=redis, image_present=False)
        with probe_settings(deadline=1):
            result = await RentalProbeCheck().run(ctx)
        assert result.passed is False and result.event.reason_code == Msg.PROBE_FAILED.reason
        assert result.event.what_we_saw["standing_failure"] is True
        assert redis.standing_failure() == STEP_SSHD_LISTEN

    # the first probe that passes clears it
    ctx, docker, _ = make_probe_context(redis=redis)
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.PROBE_OK.reason
    assert redis.standing_failure() is None


@pytest.mark.asyncio
async def test_the_probe_holds_the_create_lock_from_the_idleness_re_read_until_its_create_returns():
    """Regression (review): a renter's create can start after the idleness re-read, and the probe's create
    then sweeps the renter's `pod_*` container (its active_container_names is empty). The probe takes the
    per-executor create lock a renter's create holds around create_container, re-reads idleness under it,
    and releases it as soon as its own create returns, before the sshd wait and the teardown."""
    events: list[str] = []

    class RecordingDocker(FakeDocker):
        async def create_container(self, payload, executor_info, keypair, private_key):
            events.append("create")
            return await super().create_container(payload, executor_info, keypair, private_key)

        async def delete_container(self, payload, executor_info, keypair, private_key):
            events.append("delete")
            return await super().delete_container(payload, executor_info, keypair, private_key)

    class RecordingRedis(FakeRedis):
        async def renting_in_progress(self, miner_hotkey, executor_id, pod_id=None):
            events.append("idleness re-read")
            return await super().renting_in_progress(miner_hotkey, executor_id, pod_id)

    redis = RecordingRedis()
    redis.create_lock_events = events
    ctx, docker, _ = make_probe_context(docker=RecordingDocker(), redis=redis)
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.PROBE_OK.reason
    assert events == ["acquired", "idleness re-read", "create", "released", "delete"]


@pytest.mark.asyncio
async def test_a_renters_create_holding_the_lock_skips_the_probe_without_waiting():
    """Regression (review): the probe waits for a renter's create, or runs beside it."""
    ctx, docker, redis = make_probe_context(
        redis=FakeRedis(create_lock_held_by_renter=True, failed_step=STEP_SSHD_LISTEN)
    )
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.SKIPPED.reason
    assert result.event.what_we_saw["reason"] == "a rent is being created on the node"
    assert docker.create_calls == [] and redis.create_lock_events == []
    # a renter has the node: the standing failure is not applied (and not cleared)
    assert result.updates == {} and redis.standing_failure() == STEP_SSHD_LISTEN


@pytest.mark.asyncio
async def test_an_idleness_re_read_that_overruns_its_budget_gives_the_lock_back_without_a_create():
    """Regression (self-review): the lock's TTL covers the re-read budget plus the create deadline; a slow
    backend read under the lock would let the TTL lapse under the create, and the probe's sweep run after
    a renter acquired the lock."""
    ctx, docker, redis = make_probe_context()
    with (
        probe_settings(),
        renter_path(),
        patch.object(module, "_IDLENESS_REREAD_BUDGET_SECONDS", -1.0),  # any re-read overruns it
    ):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "the idleness re-read overran its budget"
    assert docker.create_calls == [] and redis.create_lock_events == ["acquired", "released"]


@pytest.mark.asyncio
async def test_a_create_lock_that_cannot_be_taken_is_no_verdict():
    ctx, docker, _ = make_probe_context(redis=FakeRedis(create_lock_broken=True))
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "could not take the node's create lock"
    assert docker.create_calls == []


@pytest.mark.asyncio
async def test_the_create_lock_is_released_when_the_create_is_cut_short():
    """Regression: the create's deadline, a raise or the cycle's cancellation leaves the lock held until
    its TTL, and a renter's create waits that long."""

    class RaisingDocker(FakeDocker):
        async def create_container(self, payload, executor_info, keypair, private_key):
            raise RuntimeError("redis down")

    ctx, _, redis = make_probe_context(docker=RaisingDocker())
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert redis.create_lock_events == ["acquired", "released"]

    class HangingDocker(FakeDocker):
        async def create_container(self, payload, executor_info, keypair, private_key):
            await asyncio.sleep(30)

    ctx, _, redis = make_probe_context(docker=HangingDocker())
    with probe_settings(), renter_path():
        task = asyncio.ensure_future(RentalProbeCheck().run(ctx))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert redis.create_lock_events == ["acquired", "released"]


@pytest.mark.asyncio
async def test_success_records_every_step_tears_down_and_stamps_the_interval():
    """Regression: a passing probe leaves its container on the node or its pending-pod mark in Redis (the
    probe then skips the node as busy for up to 30 min), or forgets the stamp and rents again next cycle."""
    ctx, docker, redis = make_probe_context()
    with probe_settings(), renter_path() as seen:
        result = await RentalProbeCheck().run(ctx)

    assert result.passed and result.event.reason_code == Msg.PROBE_OK.reason
    what = result.event.what_we_saw
    assert [step["step"] for step in what["probe_steps"]] == [
        STEP_CONTAINER_START,
        STEP_SSHD_LISTEN,
        STEP_SSH_LOGIN,
        STEP_GPU_COUNT,
        STEP_TEARDOWN,
    ]
    assert all(step["ok"] for step in what["probe_steps"])
    assert what["ssh_port"] == 30002
    # the renter's view: the public address and the mapped port, root with the probe key only
    assert seen["tcp"] == (EXECUTOR.address, 30002)
    assert seen["ssh"]["port"] == 30002 and seen["ssh"]["username"] == "root"
    assert len(docker.delete_calls) == 1
    delete_payload: ContainerDeleteRequest = docker.delete_calls[0][0]
    assert (
        delete_payload.container_name == "pod_pod" and delete_payload.local_volume == "volume_pod"
    )
    assert redis.stamped()
    assert redis.expiries[f"rental_probe_ok:{EXECUTOR.uuid}"] == module._REDIS_STAMP_TTL_SECONDS
    probe_pod_id = docker.create_calls[0][0].pod_id
    assert redis.removed_pending == [(MINER, EXECUTOR.uuid, probe_pod_id)]
    assert result.updates == {}


@pytest.mark.asyncio
async def test_create_payload_is_what_a_renter_gets():
    """Regression: the probe hands create_container the external ports as internal ones (a NAT'd node then
    binds the wrong host port), omits the probe key, or forgets ships_sshd so the validator bootstraps a
    second sshd into an image that runs its own."""
    ctx, docker, _ = make_probe_context()
    with probe_settings(), renter_path() as seen:
        await RentalProbeCheck().run(ctx)

    payload, executor_info, keypair, private_key = docker.create_calls[0]
    assert isinstance(payload, ContainerCreateRequest)
    assert executor_info is ctx.executor and keypair is ctx.config.validator_keypair
    assert private_key == ctx.executor_ssh_private_key_encrypted
    assert payload.docker_image == IMAGE.image_ref
    assert payload.gpu_uuids == ["GPU-aaaa", "GPU-bbbb"]
    assert (
        payload.ships_sshd is True and payload.is_sysbox is True and payload.enable_jupyter is False
    )
    assert [(p.internal_port, p.external_port) for p in payload.available_ports] == PORT_PAIRS
    assert payload.pod_mapping == [] and payload.active_container_names == []
    assert len(payload.user_public_keys) == 1 and payload.user_public_keys[0].startswith(
        "ssh-ed25519 "
    )
    # the key injected into the container is the one the login step then authenticates with
    login_key = seen["ssh"]["client_keys"][0]
    assert (
        login_key.export_public_key().decode().split()[:2]
        == payload.user_public_keys[0].split()[:2]
    )


@pytest.mark.asyncio
async def test_container_start_failure_on_the_host_zeroes_the_score_and_clears_verified():
    """Regression: the node that broke 6 renter pods (DAH-3432) keeps score 1.0 and stays listed because
    the probe's verdict never reaches the score or the verified job."""
    docker = FakeDocker(create_result=create_failed("docker_run"))
    ctx, docker, redis = make_probe_context(docker=docker)
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)

    assert result.passed is False
    assert result.event.reason_code == Msg.PROBE_FAILED.reason
    assert result.event.what_we_saw["failed_step"] == STEP_CONTAINER_START
    assert result.event.what_we_saw["create_step"] == "docker_run"
    assert "create step: docker_run" in result.event.remediation
    assert result.updates["score"] == 0.0 and result.updates["job_score"] == 0.0
    assert result.updates["clear_verified_job_info"] is True
    evidence = updates_with_clear_verified_job_evidence(result, RentalProbeCheck.check_id)[
        "clear_verified_job_evidence"
    ]
    assert evidence["reason_code"] == "RENTAL_PROBE_FAILED"
    assert evidence["check_id"] == "executor.validate.rental_probe"
    assert evidence["failed_step"] == STEP_CONTAINER_START
    # nothing was created, so nothing to delete; the pending mark create_container set is cleared
    assert docker.delete_calls == []
    assert len(redis.removed_pending) == 1
    assert not redis.stamped()


@pytest.mark.asyncio
async def test_a_create_that_fails_after_docker_run_still_removes_the_container_by_name():
    """Regression: create_container fails at add_public_keys, set_environment or finalize; its own
    `cleanup_failed_container_creation` is best effort (a failed `docker rm` is logged and swallowed), and
    the settle sees no ContainerCreated and returns early, so a `pod_<id>` that cleanup missed keeps running
    on the node with the GPUs and the verified ports until the stale sweep."""
    for step in ("add_public_keys", "set_environment", "finalize"):
        ctx, docker, redis = make_probe_context(
            docker=FakeDocker(create_result=create_failed(step))
        )
        with probe_settings(), renter_path():
            result = await RentalProbeCheck().run(ctx)

        assert result.passed is False, step
        assert result.event.what_we_saw["failed_step"] == STEP_CONTAINER_START
        assert result.event.what_we_saw["create_step"] == step
        probe_pod_id = docker.create_calls[0][0].pod_id
        shell_commands = [call.args[0] for call in ctx.ssh.run.call_args_list]
        assert any(f"docker rm -fv pod_{probe_pod_id}" in cmd for cmd in shell_commands), step
        assert any(f"volume_{probe_pod_id}" in cmd for cmd in shell_commands), step
        assert docker.delete_calls == []
        assert redis.removed_rented == [f"pod_{probe_pod_id}"]
        teardown = next(
            s for s in result.event.what_we_saw["probe_steps"] if s["step"] == STEP_TEARDOWN
        )
        assert teardown["ok"] is True
        assert (
            teardown["detail"]
            == f"create ended at {step}; docker rm by name over the validation shell left nothing"
        )


@pytest.mark.asyncio
async def test_a_create_that_raises_still_removes_the_container_by_name():
    """Regression: create_container ends in a raise instead of a FailedContainerRequest (a BaseException that
    is not a cancel, a bug in its handler) after it may have started `pod_<id>`; the probe reaches no verdict
    and nothing removes the container."""
    ctx, docker, redis = make_probe_context(
        docker=FakeDocker(create_result=RuntimeError("unexpected"))
    )
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "create_container raised"
    probe_pod_id = docker.create_calls[0][0].pod_id
    shell_commands = [call.args[0] for call in ctx.ssh.run.call_args_list]
    assert any(f"docker rm -fv pod_{probe_pod_id}" in cmd for cmd in shell_commands)
    assert any(f"volume_{probe_pod_id}" in cmd for cmd in shell_commands)
    assert redis.removed_rented == [f"pod_{probe_pod_id}"]
    assert redis.removed_pending == [(MINER, EXECUTOR.uuid, probe_pod_id)]


@pytest.mark.asyncio
async def test_create_failure_before_the_host_is_not_the_nodes_fault():
    """Regression: a validator-side failure (its Redis lock, its key, the attestation verifier) zeroes an
    honest node's score; or the settle runs `docker rm` on a node the create never reached and reports a
    teardown step for it."""
    ctx, docker, redis = make_probe_context(
        docker=FakeDocker(create_result=create_failed("attestation"))
    )
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.updates == {}
    assert not redis.stamped()
    # the only shell command was the image check before the create; nothing was removed by name
    shell_commands = [call.args[0] for call in ctx.ssh.run.call_args_list]
    assert not any("docker rm" in cmd for cmd in shell_commands)
    assert [s["step"] for s in result.event.what_we_saw["probe_steps"]] == [STEP_CONTAINER_START]

    ctx, docker, redis = make_probe_context(
        docker=FakeDocker(create_result=RuntimeError("redis down"))
    )
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.INCONCLUSIVE.reason


@pytest.mark.asyncio
async def test_sshd_never_listening_fails_at_sshd_listen_and_names_the_port():
    """Regression (node 939b3e38): ports mapped, container RUNNING, sshd never reachable, and the provider
    is told nothing about which port to check."""
    ctx, docker, redis = make_probe_context()
    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)

    assert result.passed is False and result.event.reason_code == Msg.PROBE_FAILED.reason
    what = result.event.what_we_saw
    assert what["failed_step"] == STEP_SSHD_LISTEN
    listen = next(step for step in what["probe_steps"] if step["step"] == STEP_SSHD_LISTEN)
    assert listen["ok"] is False and "30002" in listen["detail"] and listen["seconds"] >= 1.0
    assert "port 30002" in result.event.remediation and "within 1 s" in result.event.remediation
    assert result.updates["clear_verified_job_info"] is True
    # the container is torn down even though the probe failed
    assert len(docker.delete_calls) == 1
    assert not redis.stamped()


@pytest.mark.asyncio
async def test_login_refused_fails_at_ssh_login():
    """Regression: sshd answers but the injected key is not honoured, and the probe reports the node healthy
    because it only checked the TCP port. Also: a refused key is retried until the deadline, which adds
    the whole deadline to every cycle on a node that cannot pass (authorized_keys is written before sshd
    starts, so a PermissionDenied does not clear itself)."""
    ctx, docker, _ = make_probe_context()
    with (
        probe_settings(deadline=5),
        renter_path(login_error=asyncssh.PermissionDenied("Permission denied")) as seen,
    ):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is False
    assert result.event.what_we_saw["failed_step"] == STEP_SSH_LOGIN
    assert (
        "port 30002" in result.event.remediation and "authorized_keys" in result.event.remediation
    )
    assert seen["ssh_attempts"] == 1
    assert len(docker.delete_calls) == 1


@pytest.mark.asyncio
async def test_a_slow_starting_sshd_behind_docker_proxy_passes():
    """Regression: docker-proxy accepts on the published port as soon as the container exists, seconds
    before the image's start.sh runs `service ssh start`. A bare TCP accept ends the sshd_listen wait at
    once, the single login is dropped, and a healthy node scores 0 with failed_step=ssh_login."""
    ctx, docker, redis = make_probe_context()
    with probe_settings(deadline=5), renter_path(sshd_up_after=3) as seen:
        result = await RentalProbeCheck().run(ctx)

    assert result.passed is True and result.event.reason_code == Msg.PROBE_OK.reason
    steps = {step["step"]: step for step in result.event.what_we_saw["probe_steps"]}
    assert steps[STEP_SSHD_LISTEN]["ok"] is True and steps[STEP_SSH_LOGIN]["ok"] is True
    # the banner was waited for: three accepted-then-closed connections, then the fourth answered
    assert seen["tcp_connects"] == 4
    assert seen["ssh_attempts"] == 1
    assert redis.stamped() and len(docker.delete_calls) == 1


@pytest.mark.asyncio
async def test_a_port_that_accepts_but_never_sends_a_banner_fails_at_sshd_listen():
    """Regression: the container runs but sshd inside it never starts. docker-proxy keeps accepting, so the
    probe blames ssh_login and the provider is told to check authorized_keys instead of sshd."""
    ctx, docker, _ = make_probe_context()
    with probe_settings(deadline=1), renter_path(sshd_up_after=None) as seen:
        result = await RentalProbeCheck().run(ctx)

    assert result.passed is False and result.event.reason_code == Msg.PROBE_FAILED.reason
    what = result.event.what_we_saw
    assert what["failed_step"] == STEP_SSHD_LISTEN
    listen = next(step for step in what["probe_steps"] if step["step"] == STEP_SSHD_LISTEN)
    assert listen["ok"] is False and "no SSH banner" in listen["detail"]
    assert listen["seconds"] >= 1.0 and seen["tcp_connects"] > 1
    assert seen["ssh_attempts"] == 0
    assert "sshd did not answer on port 30002" in result.event.remediation
    assert len(docker.delete_calls) == 1


@pytest.mark.asyncio
async def test_a_dropped_first_login_is_retried_until_the_deadline():
    """Regression: sshd's banner is up but the first connection is dropped (sshd still forking, MaxStartups)
    and a single connect attempt zeroes the node; or the retries never stop and the probe outlives the
    deadline it told the provider about."""
    ctx, docker, _ = make_probe_context()
    dropped = asyncssh.ConnectionLost("Connection lost")
    with probe_settings(deadline=5), renter_path(login_error=[dropped, dropped, None]) as seen:
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.PROBE_OK.reason
    assert seen["ssh_attempts"] == 3
    login = next(
        step for step in result.event.what_we_saw["probe_steps"] if step["step"] == STEP_SSH_LOGIN
    )
    assert login["ok"] is True and login["detail"] == "connected on attempt 3"

    ctx, docker, _ = make_probe_context()
    with probe_settings(deadline=1), renter_path(login_error=dropped) as seen:
        started = time.perf_counter()
        result = await RentalProbeCheck().run(ctx)
        elapsed = time.perf_counter() - started
    assert result.passed is False and result.event.what_we_saw["failed_step"] == STEP_SSH_LOGIN
    assert seen["ssh_attempts"] > 1 and elapsed < 3
    login = next(
        step for step in result.event.what_we_saw["probe_steps"] if step["step"] == STEP_SSH_LOGIN
    )
    assert login["ok"] is False and f"(attempt {seen['ssh_attempts']})" in login["detail"]
    assert len(docker.delete_calls) == 1


@pytest.mark.asyncio
async def test_gpu_count_mismatch_fails_at_gpu_count():
    """Regression: a node whose container runtime hands the renter one card of two passes because the probe
    only checks that nvidia-smi ran."""
    ctx, docker, _ = make_probe_context()
    one_gpu = "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-aaaa)\n"
    with probe_settings(), renter_path(smi_stdout=one_gpu):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is False
    assert result.event.what_we_saw["failed_step"] == STEP_GPU_COUNT
    gpu = next(
        step for step in result.event.what_we_saw["probe_steps"] if step["step"] == STEP_GPU_COUNT
    )
    assert "expected 2 GPU(s), nvidia-smi -L listed 1" in gpu["detail"]
    assert "the 2 GPU(s)" in result.event.remediation

    ctx, docker, _ = make_probe_context()
    with (
        probe_settings(),
        renter_path(smi_stdout="Failed to initialize NVML: Unknown Error\n", smi_exit=255),
    ):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is False and result.event.what_we_saw["failed_step"] == STEP_GPU_COUNT

    # the session opened but the command could not run: the GPU step's failure, not a refused login
    ctx, docker, _ = make_probe_context()
    with probe_settings(), renter_path(command_error=asyncssh.ChannelOpenError(2, "open failed")):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is False and result.event.what_we_saw["failed_step"] == STEP_GPU_COUNT
    login = next(
        step for step in result.event.what_we_saw["probe_steps"] if step["step"] == STEP_SSH_LOGIN
    )
    assert login["ok"] is True
    assert "nvidia-smi -L did not run" in result.event.what_we_saw["probe_steps"][-2]["detail"]


@pytest.mark.asyncio
async def test_a_node_rented_during_the_probe_is_not_penalised():
    """Regression: a renter's create sweeps the probe container away mid-run (their pod is proof the node
    works) and the probe zeroes the node for its own missing container; or the backend cannot be read
    afterwards and the race is assumed away."""
    ctx, docker, _ = make_probe_context(rented_now=[rented_data(), rented_data(pods=1)])
    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "node was rented during the probe"
    assert result.updates == {}
    assert len(docker.create_calls) == 1

    # self-review: a filler's create goes through the same create_container and sweeps pod_<probe> too;
    # its pending mark is gone by the time the probe looks, and a false failure would then stand for the
    # filler's whole run ("skipped: filler running")
    ctx, docker, redis = make_probe_context(rented_now=[rented_data(), rented_data(fillers=1)])
    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason and result.updates == {}
    assert result.event.what_we_saw["reason"] == "node was given a filler during the probe"
    assert redis.standing_failure() is None
    # ... but a filler is not a renter: a standing failure is carried through it, as at the re-read
    ctx, docker, redis = make_probe_context(
        redis=FakeRedis(failed_step=STEP_GPU_COUNT),
        rented_now=[rented_data(), rented_data(fillers=1)],
    )
    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is False and result.event.what_we_saw["standing_failure"] is True

    ctx, docker, _ = make_probe_context(rented_now=[rented_data(), None])
    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "could not rule out a rental during the probe"

    # the pending-pod read failing after the probe is the same unknown
    class RedisFailingAfterTheProbe(FakeRedis):
        async def renting_in_progress(self, miner_hotkey, executor_id, pod_id=None):
            if self.removed_pending:  # the probe has settled: this is the post-failure read
                raise ConnectionError("redis down")
            return False

    ctx, docker, _ = make_probe_context(redis=RedisFailingAfterTheProbe())
    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "could not rule out a rental during the probe"


@pytest.mark.asyncio
async def test_a_renters_create_running_beside_the_probe_is_not_penalised():
    """Regression: miner_service declines a second create only for the SAME pod id, so a renter's create
    dispatched during the probe runs beside it and sweeps `pod_<probe>` away; the backend may not list
    that pod yet, and only this validator's pending-pod mark shows a rent is being created."""
    ctx, docker, _ = make_probe_context(redis=FakeRedis(renting=[False, True]))
    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "node was rented during the probe"
    assert len(docker.create_calls) == 1 and result.updates == {}


@pytest.mark.asyncio
async def test_the_cycles_cancellation_mid_create_still_removes_the_container_by_name():
    """Regression: JOB_TIME_OUT cancels the run while create_container is on the host; the CancelledError
    passes every `except Exception`, so nothing removes `pod_<id>` and its volume or clears the pending
    mark, and the container binds the verified ports into the next cycle."""

    class HangingDocker(FakeDocker):
        async def create_container(self, payload, executor_info, keypair, private_key):
            self.create_calls.append((payload, executor_info, keypair, private_key))
            await asyncio.sleep(30)
            return created()

    ctx, docker, redis = make_probe_context(docker=HangingDocker())
    with probe_settings(), renter_path():
        task = asyncio.ensure_future(RentalProbeCheck().run(ctx))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    probe_pod_id = docker.create_calls[0][0].pod_id
    shell_commands = [call.args[0] for call in ctx.ssh.run.call_args_list]
    assert any(f"docker rm -fv pod_{probe_pod_id}" in cmd for cmd in shell_commands)
    assert any(f"volume_{probe_pod_id}" in cmd for cmd in shell_commands)
    assert redis.removed_pending == [(MINER, EXECUTOR.uuid, probe_pod_id)]
    assert redis.removed_rented == [f"pod_{probe_pod_id}"]


@pytest.mark.asyncio
async def test_a_cancel_during_the_teardown_finishes_it_and_still_ends_the_run_cancelled():
    """Regression: a cancel landing while delete_container runs is swallowed, the check returns a verdict
    built from a half-settled outcome (a clean run stamped OK before its teardown failed) and the run ends
    as a normal result past its deadline."""

    class SlowDeleteDocker(FakeDocker):
        delete_finished = False

        async def delete_container(self, payload, executor_info, keypair, private_key):
            self.delete_calls.append((payload, executor_info, keypair, private_key))
            await asyncio.sleep(0.2)
            self.delete_finished = True
            return self.delete_result

    ctx, docker, redis = make_probe_context(docker=SlowDeleteDocker())
    with probe_settings(), renter_path():
        task = asyncio.ensure_future(RentalProbeCheck().run(ctx))
        # let the probe reach the teardown, then cancel while delete_container is running
        while not docker.delete_calls:
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # the teardown ran to its end and Redis is clean; no verdict was stamped from a half-settled outcome
    assert docker.delete_finished is True
    assert len(docker.delete_calls) == 1
    assert len(redis.removed_pending) == 1
    assert not redis.stamped()


def test_docker_rm_exit_one_means_gone_only_with_no_such_container():
    """Regression: `docker rm -f` exit 1 is read as "already gone" while dockerd is down (also exit 1), so a
    leftover container is reported removed."""

    async def run_case(exit_status, stderr):
        ctx, _, redis = make_probe_context()
        ctx.ssh.run.side_effect = [
            MagicMock(exit_status=exit_status, stdout="", stderr=stderr),
            MagicMock(exit_status=0, stdout="volume_x\n", stderr=""),
        ]
        return await module._remove_over_shell(
            ctx, container_name="pod_x", volume_name="volume_x"
        ), redis

    error, redis = asyncio.run(run_case(1, "Error response from daemon: No such container: pod_x"))
    assert error is None and redis.removed_rented == ["pod_x"]
    error, _ = asyncio.run(
        run_case(1, "Cannot connect to the Docker daemon at unix:///var/run/docker.sock")
    )
    assert error is not None and "docker rm exited 1" in error
    error, _ = asyncio.run(run_case(0, ""))
    assert error is None


def test_a_volume_rm_that_fails_is_a_leftover_not_a_clean_teardown():
    """Regression (review): the volume command's result was ignored, and `DockerCommand.volume_remove` masks
    every failure with `|| true`, so a teardown reported clean while `volume_<id>` stayed on the node."""

    async def run_case(exit_status, stderr):
        ctx, _, _ = make_probe_context()
        ctx.ssh.run.side_effect = [
            MagicMock(exit_status=0, stdout="pod_x\n", stderr=""),
            MagicMock(exit_status=exit_status, stdout="", stderr=stderr),
        ]
        error = await module._remove_over_shell(ctx, container_name="pod_x", volume_name="volume_x")
        return error, [call.args[0] for call in ctx.ssh.run.call_args_list][1]

    error, command = asyncio.run(
        run_case(1, "Error response from daemon: remove volume_x: volume is in use")
    )
    assert error is not None and "docker volume rm exited 1" in error and "in use" in error
    assert command == "/usr/bin/docker volume rm volume_x" and "|| true" not in command
    error, _ = asyncio.run(
        run_case(
            1,
            "Error response from daemon: get volume_x: no such volume\nError: No such volume: volume_x",
        )
    )
    assert error is None
    error, _ = asyncio.run(run_case(0, ""))
    assert error is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rented_now,redis,expected_reason",
    [
        ([rented_data(pods=1)], FakeRedis(), "rented since the cycle started"),
        ([rented_data(fillers=1)], FakeRedis(), "filler started since the cycle started"),
        ([rented_data()], FakeRedis(renting=True), "a rent is being created on the node"),
    ],
)
async def test_the_node_must_be_idle_now_not_at_cycle_start(rented_now, redis, expected_reason):
    """Regression: `ctx.state.rented_data` is the snapshot the cycle started from, minutes before this
    last check runs; a pod rented in between is force-removed by create_container's container sweep."""
    ctx, docker, _ = make_probe_context(rented_now=rented_now, redis=redis)
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.SKIPPED.reason
    assert result.event.what_we_saw["reason"] == expected_reason
    assert docker.create_calls == []


@pytest.mark.asyncio
async def test_unreadable_state_rents_nothing():
    """Regression: a backend or Redis outage is read as "idle, never probed" and the probe rents a
    container on every node every cycle while the outage lasts."""
    ctx, docker, _ = make_probe_context(rented_now=[None])
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "could not confirm the node is idle"
    assert docker.create_calls == []

    ctx, docker, _ = make_probe_context(redis=FakeRedis(broken=True))
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.event.what_we_saw["reason"] == "interval state unreadable"
    assert docker.create_calls == []


@pytest.mark.asyncio
async def test_a_create_that_hangs_is_cut_off_and_its_container_removed_by_name():
    """Regression: a stuck create_container runs until the cycle's outer timeout cancels the task, and the
    cancellation skips every `except Exception`: the pending-pod mark and a half-made `pod_<id>` stay on
    the node."""

    class HangingDocker(FakeDocker):
        async def create_container(self, payload, executor_info, keypair, private_key):
            self.create_calls.append((payload, executor_info, keypair, private_key))
            await asyncio.sleep(30)
            return created()

    ctx, docker, redis = make_probe_context(docker=HangingDocker())
    with probe_settings(), renter_path(), patch.object(module, "_CREATE_DEADLINE_SECONDS", 0.05):
        result = await RentalProbeCheck().run(ctx)

    assert result.passed is False and result.event.reason_code == Msg.PROBE_FAILED.reason
    assert result.event.what_we_saw["failed_step"] == STEP_CONTAINER_START
    assert result.event.what_we_saw["create_step"] == "deadline"
    probe_pod_id = docker.create_calls[0][0].pod_id
    shell_commands = [call.args[0] for call in ctx.ssh.run.call_args_list]
    assert any(f"docker rm -fv pod_{probe_pod_id}" in cmd for cmd in shell_commands)
    assert any(f"volume_{probe_pod_id}" in cmd for cmd in shell_commands)
    assert docker.delete_calls == []
    assert redis.removed_pending == [(MINER, EXECUTOR.uuid, probe_pod_id)]
    assert redis.removed_rented == [f"pod_{probe_pod_id}"]


@pytest.mark.asyncio
async def test_teardown_failure_after_a_clean_run_neither_penalises_nor_stamps():
    """Regression: the container is left on the node binding the verified ports (PortCount then fails the
    next cycle) and the probe stamps success, so nothing looks at the node again for six hours."""
    failed_delete = FailedContainerRequest(
        miner_hotkey=MINER,
        executor_id=EXECUTOR.uuid,
        pod_id="pod",
        msg="Unknown Error delete_container",
        error_type=FailedContainerErrorTypes.ContainerDeletionFailed,
        error_code=FailedContainerErrorCodes.UnknownError,
    )
    ctx, docker, redis = make_probe_context(docker=FakeDocker(delete_result=failed_delete))
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "teardown did not finish"
    teardown = next(
        step for step in result.event.what_we_saw["probe_steps"] if step["step"] == STEP_TEARDOWN
    )
    assert teardown["ok"] is False and "delete_container" in teardown["detail"]
    # the leftover is removed by name over the validation shell, so no port stays bound until the sweep
    assert "docker rm by name over the validation shell left nothing" in teardown["detail"]
    shell_commands = [call.args[0] for call in ctx.ssh.run.call_args_list]
    assert any("docker rm -fv pod_pod" in cmd for cmd in shell_commands)
    assert redis.removed_rented == ["pod_pod"]
    assert not redis.stamped()
    assert len(redis.removed_pending) == 1


@pytest.mark.asyncio
async def test_missing_image_on_the_node_skips_instead_of_pulling():
    """Regression: the probe pulls a multi-GB image inside the validation cycle on every node that has not
    pre-pulled it."""
    ctx, docker, _ = make_probe_context(image_present=False)
    with probe_settings():
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.event.what_we_saw["reason"] == "default renter image is not pulled on the node"
    assert docker.create_calls == []

    ctx, docker, _ = make_probe_context(images=())
    with probe_settings():
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert docker.create_calls == []


@pytest.mark.asyncio
async def test_an_unreadable_image_list_or_a_create_without_port_22_reaches_no_verdict():
    """Regression: an SSH error on `docker image inspect` is read as "image not pulled" (a skip that hides a
    dead validation shell), or a ContainerCreated with no mapping for port 22 (the validator's own port
    mapping, which always maps 22 first) is blamed on the node as sshd_listen."""
    ctx, docker, _ = make_probe_context()
    ctx.ssh.run.side_effect = asyncssh.ConnectionLost("lost")
    with probe_settings():
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "could not read the node's image list"
    assert docker.create_calls == []

    no_ssh_mapping = created()
    no_ssh_mapping.port_maps = [(8888, 30003)]
    ctx, docker, redis = make_probe_context(docker=FakeDocker(create_result=no_ssh_mapping))
    with probe_settings(), renter_path():
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is True and result.event.reason_code == Msg.INCONCLUSIVE.reason
    assert result.event.what_we_saw["reason"] == "create returned no mapping for container port 22"
    assert result.updates == {}
    # the container was still created, so it is still torn down
    assert len(docker.delete_calls) == 1 and not redis.stamped()


@pytest.mark.asyncio
async def test_no_verified_ports_or_no_key_reaches_no_verdict():
    """Regression: create_container is called with no ports (it fails at port_mapping and the node is blamed)
    or with no executor key (it raises)."""
    # fewer than MIN_PORT_COUNT pairs: generate_portMappings would refuse the create at port_mapping
    ctx, docker, _ = make_probe_context(port_pairs=PORT_PAIRS[:2])
    with probe_settings():
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason and docker.create_calls == []
    assert (
        result.event.what_we_saw["reason"] == "validator has no 3 verified ports for this executor"
    )

    ctx, docker, _ = make_probe_context()
    ctx = ctx.model_copy(update={"executor_ssh_private_key_encrypted": None})
    with probe_settings():
        result = await RentalProbeCheck().run(ctx)
    assert result.event.reason_code == Msg.INCONCLUSIVE.reason and docker.create_calls == []


def test_every_failed_step_has_a_remediation_that_renders():
    """Regression: a new step is added without provider text, or a template's placeholder goes stale and
    `.format` raises inside the check, turning a verdict into a crashed cycle."""
    for step in (STEP_CONTAINER_START, STEP_SSHD_LISTEN, STEP_SSH_LOGIN, STEP_GPU_COUNT):
        failure = module._Failure(step, "docker_run")
        with probe_settings(deadline=90):
            text = module._remediation(failure, ssh_port=30002, expected_gpus=8)
        assert text and "{" not in text, (step, text)
    with probe_settings(deadline=90):
        assert "port 30002" in module._remediation(
            module._Failure(STEP_SSHD_LISTEN), ssh_port=30002, expected_gpus=1
        )
        assert "the 8 GPU(s)" in module._remediation(
            module._Failure(STEP_GPU_COUNT), ssh_port=None, expected_gpus=8
        )


def test_count_gpus_reads_real_nvidia_smi_output():
    """Regression: the count trips on the UUID lines or on a MIG listing and every node fails gpu_count."""
    mig = (
        "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-1111)\n"
        "  MIG 1g.10gb     Device  0: (UUID: MIG-2222)\n"
        "GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-3333)\n"
    )
    assert module._count_gpus(mig) == 2
    assert module._count_gpus("Failed to initialize NVML: Unknown Error") == 0


def test_gpu_count_step_fails_a_listing_with_the_right_count_but_a_non_zero_exit():
    """Regression: nvidia-smi lists every advertised GPU and still exits non-zero (a driver error after
    the listing) and the gpu_count step counts it as a pass."""
    ctx, _, _ = make_probe_context()
    two_gpus = "GPU 0: NVIDIA RTX (UUID: GPU-aaaa)\nGPU 1: NVIDIA RTX (UUID: GPU-bbbb)\n"
    smi = MagicMock(stdout=two_gpus, stderr="", exit_status=1)
    outcome = module._ProbeOutcome()
    module._step_gpu_count(ctx, outcome, module._Login(result=smi, command_seconds=0.5))
    assert outcome.failed_step == STEP_GPU_COUNT
    assert outcome.steps[-1].ok is False and "listed 2 (exit 1)" in outcome.steps[-1].detail


def test_the_probe_runs_last_before_scoring_and_never_in_dry_run():
    """Regression: the probe is placed before the port or rented checks it depends on, or the dry-run
    pipeline (staging, DRY_RUN=true) starts creating containers on providers' nodes."""
    ids = [type(check).__name__ for check in pipeline_factory.PipelineFactory.build_checks()]
    assert ids.index("RentalProbeCheck") == ids.index("RentalVerificationCheck") + 1
    assert ids.index("RentalProbeCheck") == ids.index("ScoreCheck") - 1
    assert ids.index("RentalProbeCheck") > ids.index("PortConnectivityCheck")
    assert ids.index("RentalProbeCheck") > ids.index("TenantEnforcementCheck")
    dry = [
        type(check).__name__ for check in pipeline_factory.PipelineFactory.build_dry_run_checks()
    ]
    assert "RentalProbeCheck" not in dry


@pytest.mark.asyncio
async def test_pipeline_carries_the_failed_step_to_the_reset_evidence():
    """Regression: the reset the backend receives says DEFAULT with no check name, so the penalty row and the
    portal cannot show which step of the rental failed (the DAH-3386 contract)."""
    ctx, _, _ = make_probe_context()

    class Sink:
        async def emit(self, event):  # pragma: no cover
            pass

    with probe_settings(deadline=1), renter_path(sshd_listens=False):
        ok, events, last_ctx = await Pipeline([RentalProbeCheck()], sink=Sink()).run(ctx)
    assert ok is False
    assert last_ctx.clear_verified_job_info is True
    assert last_ctx.clear_verified_job_evidence["reason_code"] == "RENTAL_PROBE_FAILED"
    assert last_ctx.clear_verified_job_evidence["failed_step"] == STEP_SSHD_LISTEN
    assert last_ctx.score == 0.0 and last_ctx.job_score == 0.0
    # the pipeline writes its own per-check `steps` summary onto the last event; the probe's records must
    # survive it under their own key, since this event is what the portal shows the provider
    assert events[-1].what_we_saw["steps_failed"] == RentalProbeCheck.check_id
    assert set(events[-1].what_we_saw["steps"]) == {RentalProbeCheck.check_id}
    assert [step["step"] for step in events[-1].what_we_saw["probe_steps"]] == [
        STEP_CONTAINER_START,
        STEP_SSHD_LISTEN,
        STEP_TEARDOWN,
    ]


@pytest.mark.asyncio
async def test_the_real_redis_service_accepts_the_stamps_lifetime(monkeypatch):
    """Regression (self-review): FakeRedis.set took an `ex` the real `RedisService.set` did not, so both
    stamps raised TypeError in production, swallowed by their `except Exception`: no interval stamp (a
    container per idle node per cycle) and no standing failure."""
    from services.redis_service import RedisService

    from core.config import settings as validator_settings

    class RecordingRedis:
        calls: list[tuple] = []

        async def set(self, key, value, ex=None):
            self.calls.append((key, value, ex))

    monkeypatch.setattr(validator_settings, "REDIS_COMMAND_LOCK_ENABLED", False)
    service = RedisService()  # builds pools from settings; nothing connects until a command runs
    service.redis = RecordingRedis()
    await service.set("rental_probe_ok:x", "1.0", ex=module._REDIS_STAMP_TTL_SECONDS)
    await service.set("plain", "v")
    assert RecordingRedis.calls == [
        ("rental_probe_ok:x", "1.0", module._REDIS_STAMP_TTL_SECONDS),
        ("plain", "v", None),
    ]


EGRESS_OK = "lium_egress dns=ok\nlium_egress tool=curl http=200\n"
EGRESS_DNS_FAIL = "lium_egress dns=fail\n"
EGRESS_NO_ANSWER = (
    "lium_egress dns=ok\nlium_egress tool=curl http=000\n"
    "curl: (28) Failed to connect to pypi.org port 443 after 10001 ms: Timeout was reached\n"
)


def _steps(result) -> dict[str, dict]:
    return {record["step"]: record for record in result.event.what_we_saw["probe_steps"]}


@pytest.mark.asyncio
async def test_egress_step_runs_the_script_inside_the_renter_container_and_passes_on_an_http_answer():
    """Regression: the egress script runs on the host instead of in the renter's container over its SSH
    session, so a pod network that drops the pod's traffic (ticket-0361) still passes."""
    ctx, _, _ = make_probe_context()
    with probe_settings(egress_check=True), renter_path(egress_stdout=EGRESS_OK) as seen:
        result = await RentalProbeCheck().run(ctx)
    assert result.passed and result.event.reason_code == Msg.PROBE_OK.reason
    assert _steps(result)[STEP_EGRESS]["ok"] is True
    egress_command = seen["conn"].commands[1]
    assert egress_command.startswith("sh -c ") and "getent hosts pypi.org" in egress_command
    # headers of a 325-byte file, not the 46 MB index (self-review)
    assert "https://pypi.org/robots.txt" in egress_command and "/simple" not in egress_command


@pytest.mark.asyncio
@pytest.mark.parametrize("egress_stdout", [EGRESS_DNS_FAIL, EGRESS_NO_ANSWER])
async def test_egress_failure_fails_the_probe_with_no_outbound_internet_when_enforced(
    egress_stdout,
):
    """Regression: a renter container with no DNS or no route out passes the probe, or fails it under the
    generic RENTAL_PROBE_FAILED instead of the reason the provider's panel explains."""
    redis = FakeRedis()
    ctx, _, _ = make_probe_context(redis=redis)
    with (
        probe_settings(egress_check=True, egress_enforced=True),
        renter_path(egress_stdout=egress_stdout),
    ):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed is False
    assert result.event.reason_code == "NO_OUTBOUND_INTERNET"
    assert result.event.what_we_saw["failed_step"] == STEP_EGRESS
    assert "FORWARD chain and DNS" in result.event.remediation
    assert result.updates["clear_verified_job_info"] is True
    assert redis.standing_failure() == STEP_EGRESS


@pytest.mark.asyncio
async def test_egress_failure_only_logs_while_enforcement_is_off():
    """Regression: the egress step hides hosts before anyone read the shadow rows, or the failure is not
    recorded at all."""
    redis = FakeRedis()
    ctx, _, _ = make_probe_context(redis=redis)
    with probe_settings(egress_check=True), renter_path(egress_stdout=EGRESS_DNS_FAIL):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed and result.event.reason_code == Msg.PROBE_OK.reason
    assert _steps(result)[STEP_EGRESS]["ok"] is False
    assert "dns_failed" in _steps(result)[STEP_EGRESS]["detail"]
    assert redis.standing_failure() is None


@pytest.mark.asyncio
async def test_egress_step_without_a_reading_records_nothing():
    """Regression: an image without curl or wget (no reading) fails the node as if it had no internet."""
    ctx, _, _ = make_probe_context()
    with (
        probe_settings(egress_check=True, egress_enforced=True),
        renter_path(egress_stdout="lium_egress dns=ok\nlium_egress tool=none http=000\n"),
    ):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed and result.event.reason_code == Msg.PROBE_OK.reason
    assert STEP_EGRESS not in _steps(result)


@pytest.mark.asyncio
async def test_a_standing_egress_failure_is_dropped_once_enforcement_is_off():
    """Regression: switching NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED off leaves nodes failing on the
    egress step they failed while it was on, until a probe passes."""
    ctx, _, _ = make_probe_context(
        redis=FakeRedis(failed_step=STEP_EGRESS), rented=rented_data(fillers=1)
    )
    with probe_settings(egress_check=True):
        result = await RentalProbeCheck().run(ctx)
    assert result.passed and result.event.reason_code == Msg.SKIPPED.reason
