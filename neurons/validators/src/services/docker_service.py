import asyncio
import contextlib
import enum
import ipaddress
import logging
import math
import random
import re
import secrets
import shlex
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

import aiohttp
import asyncssh
import bittensor
import redis.exceptions
from core.docker_utils import (
    ALPINE_HELPER_IMAGE,
    ContainerDeathDiagnostics,
    DockerCommand,
    collect_container_death_diagnostics,
    df_available_bytes,
)
from datura.requests.miner_requests import ExecutorSSHInfo
from fastapi import Depends
from payload_models.payloads import (
    AddSshPublicKeyRequest,
    BootstrapRestoreSpec,
    CacheVolume,
    ContainerBaseRequest,
    ContainerCreated,
    ContainerCreateRequest,
    ContainerDeleted,
    ContainerDeleteRequest,
    ContainerStartRequest,
    ContainerStopRequest,
    ContainerWarningCode,
    CustomOptions,
    ExternalVolumeInfo,
    FailedContainerErrorCodes,
    FailedContainerErrorTypes,
    FailedContainerRequest,
    InstallJupyterServerRequest,
    JupyterInstallationFailed,
    JupyterServerInstalled,
    PayloadPortMapping,
    ProfilerStep,
    ProfilerStepName,
    RemoveSshPublicKeysRequest,
    SshPubKeyAdded,
    SshPubKeyRemoved,
    VolumeEncryptionStatus,
    WorkloadKind,
    now_ms,
)
from services.attestation_service import AttestationError, AttestationService
from services.const import (
    FILLER_CACHE_VOLUME_PREFIXES,
    DPHN_CACHE_FREE_MARGIN_GB,
    DPHN_CACHE_LISTING_FLOOR_GB,
    DPHN_CACHE_SIZE_GB,
    DPHN_CACHE_VOLUME_PREFIX,
    FILLER_CONTAINER_PREFIX,
    GPU_WEDGE_SWEEP_SETTLE_SECONDS,
    MIN_PORT_COUNT,
    POD_CONTAINER_PREFIX,
    PREFERRED_POD_PORTS,
)
from services.cvm_quote_broker import ensure_quote_broker, quote_socket_pod_mount
from services.gpu_power_limit import (
    apply_filler_gpu_power_limits,
    raise_low_power_limits_to_default,
    restore_all_host_gpu_power_limits,
    restore_filler_pod_gpu_power_limits,
    restore_tracked_gpu_power_limits,
)
from services.gpu_wedge import cure_wedged_gpus, query_wedged_gpu_uuids
from services.nvidia_devices import build_gpu_docker_config_for_executor
from services.cluster_fabric import WIREGUARD_LISTEN_PORT, cluster_pod_networking
from services.redis_service import (
    STREAMING_LOG_CHANNEL,
    RedisService,
)
from services.rental_docker_observability import (
    exec_logged_rental_docker_sdk_operation,
    rental_run_spec_log_fields,
    run_logged_rental_docker_sdk_operation,
)
from services.rental_docker_sdk import (
    DEFAULT_DOCKER_PULL_TIMEOUT_SECONDS,
    ContainerExecSpec,
    ContainerRunSpec,
    ContainerUlimit,
    DeviceMount,
    PortBinding,
    RentalDockerConnectionError,
    RentalDockerOperationError,
    RentalDockerSdkClient,
    RentalDockerSdkClientFactory,
    VolumeMount,
    build_authorized_keys_exec_spec,
    build_container_command_argv,
    build_environment_exec_spec,
    build_remove_authorized_keys_exec_spec,
    require_rental_docker_ssh_host_key,
)
from services.ssh_connect_timing import connect_with_phase_timing
from services.storage_operations import (
    start_storage_operation,
    supports_storage_operation,
    wait_for_storage_operation,
)
from services.task.runner import SSHCommandRunner
from tenacity import RetryError

from core.config import settings
from core.utils import _m, _StructuredMessage, get_extra_info, retry_ssh_command
from services.ssh_service import SSHService
from services.volume_keys import VolumeKeyDeriver

logger = logging.getLogger(__name__)

REPOSITORIES = [
    "daturaai/compute-subnet-executor:latest",
    "daturaai/compute-subnet-executor-runner:latest",
    "nickfedor/watchtower",
    "daturaai/pytorch",
    "daturaai/ubuntu",
]

LOG_STREAM_INTERVAL = 0.5  # 500ms — keeps build-log p95 emit→publish under AC-3's
# 2000ms budget (DAH-2211). Was 5s; that was fine for slow `docker pull` lines
# but build phases emit dense progress (one line per layer) and a 5s batch
# would violate the SSE latency requirement.
IN_CONTAINER_SSH_BOOTSTRAP_PATH = "/tmp/lium-ssh-bootstrap.sh"

# DAH-2364: SIGTERM grace window (docker stop -t semantics) given to a container before
# forced removal. SIGKILL-only removal of GPU-heavy workloads repeatedly triggered a
# containerd/sysbox wedge ("could not kill container ... did not receive an exit event";
# root cause: sysbox-fs FUSE deadlock, fixed separately), leaving orphaned containers
# that hold GPUs and brick the executor.
CONTAINER_STOP_GRACE_SECONDS = 30

# Fillers get a shorter grace window than customer workloads. The backend preempts a filler
# before a customer rent with a total budget of FILLER_STOP_WAIT_TIMEOUT_SECONDS (30s in
# compute-app) and starts the rent anyway on timeout. This grace must stay strictly below that
# budget with room for the forced removal and the stopped-callback, so a SIGTERM-ignoring filler
# can never burn the whole budget inside docker stop and hold the GPUs into the customer rent.
# Half the budget leaves ~15s of headroom while still letting a well-behaved filler exit cleanly
# and avoid the containerd/sysbox wedge. Keep in sync with compute-app FILLER_STOP_WAIT_TIMEOUT_SECONDS.
FILLER_CONTAINER_STOP_GRACE_SECONDS = 15

# In-container port the cache-template images' start.sh hardcodes for Jupyter
# (`jupyter lab --port=8888`). It reads no port from the environment, so the
# image-managed Jupyter path is only safe while the mapped docker port is this one.
IMAGE_JUPYTER_DOCKER_PORT = 8888

DOCKER_VOLUME_PLUGINS = {
    "s3fs": "mochoa/s3fs-volume-plugin"
}

S3FS_PLUGIN_IMAGE = "mochoa/s3fs-volume-plugin"


LEGACY_S3FS_PLUGIN_ALIAS = "s3fs"


def _published_ports(
    port_maps: list[tuple[int, int, int]],
    cluster_udp_ports: tuple[int, ...],
) -> tuple[PortBinding, ...]:
    """The rental's own TCP ports, plus the UDP ports a cluster node needs on top.

    WireGuard's handshake is UDP and the fleet publishes only TCP by default, so a cluster node that
    got no UDP port here would raise its interface and never complete a handshake (DAH-2620).
    """
    return (
        *(
            PortBinding(container_port=docker_port, host_port=internal_port)
            for docker_port, internal_port, _ in port_maps
        ),
        *(
            PortBinding(container_port=udp_port, host_port=udp_port, protocol="udp")
            for udp_port in cluster_udp_ports
        ),
    )


def _s3fs_plugin_alias(volume_name: str) -> str:
    """DAH-2512: one plugin instance per volume.

    A shared instance holds a single credential pair and dies on `plugin disable`,
    so attaching or detaching one pod's volume breaks every other pod on the host.
    """
    return f"s3fs-{volume_name}"

# DAH-2496: sysbox installers reserve one 65536-wide subuid slice per host and map
# every container's root to its base. A wider range is handed out per container.
SYSBOX_SUBUID_SLICE_SIZE = 65536

# DAH-1991: tolerate concurrent health_check_* / container_* on the executor.
# Probe TTL is short (~30s); same-command retry within a 90s budget covers the
# documented race without regenerating port mappings.
_PORT_ALLOCATED_PHRASES = ("port is already allocated", "address already in use", "failed to bind host port")
_PORT_ALLOCATED_RETRY_BUDGET_SEC = 90
_PORT_ALLOCATED_RETRY_SLEEP_SEC = 5

_VLOOPBACK_MOUNT_ERROR_PHRASES = (
    "VolumeDriver.Mount",
    "cannot create mount point dir",
    "file exists",
)
_VLOOPBACK_DRIVER_PREFIX = "vloopback"
# Shared with core.docker_utils so exactly one helper image lands on nodes.
_VLOOPBACK_REPAIR_IMAGE = ALPINE_HELPER_IMAGE
_VLOOPBACK_REPAIR_COMMAND_TIMEOUT_SEC = 30
_DOCKER_VOLUME_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# DAH-2475: slack kept free above a rental's requested volume before we decide the DPHN filler cache
# has to go. Covers the image layers and scratch the pod needs beyond its own volume.
RENTAL_DISK_HEADROOM_GB = 20
_LOCAL_VOLUME_TIMEOUT_THRESHOLD_GB = 100
_LOCAL_VOLUME_TIMEOUT_BASE_SEC = 30
_LOCAL_VOLUME_TIMEOUT_GB_PER_SEC = 10
_LOCAL_VOLUME_TIMEOUT_MAX_SEC = 180
_FILLER_EXTERNAL_PORT_OFFSET = 20
_LIUM_CIPHER_MOUNT = "/lium-cipher"
_ENCRYPTED_VOLUME_IMAGE_LABEL = "lium.volume_encryption.enable"
# the path comes from a customer-authored template and the backend only requires a leading slash,
# so anything that is not a plain absolute path is refused here rather than mounted over
_PLAINTEXT_PATH_RE = re.compile(r"^(?:/(?!\.{1,2}(?:/|$))[A-Za-z0-9._-]+)+$")


class _VolumeEncryptionState(enum.Enum):
    ENCRYPTED = "encrypted"
    PLAIN = "plain"
    UNKNOWN = "unknown"


_DOCKER_NO_SUCH_CONTAINER_PHRASE = "No such container"
_DOCKER_REMOVAL_IN_PROGRESS_PHRASES = ("409", "removal", "already in progress")
HOST_KEY_REQUIRED_EXTRA = {
    "ssh_host_key_missing": True,
    "docker_sdk_host_key_required": True,
}
# Keep the rental create_container SSH session alive while long docker pulls
# are quiet. With 30s/4, AsyncSSH declares a dead peer after about 2 minutes.
_CREATE_CONTAINER_SSH_KEEPALIVE_INTERVAL_SEC = 30
_CREATE_CONTAINER_SSH_KEEPALIVE_COUNT_MAX = 4
_DOCKER_PULL_TIMEOUT_SECONDS = DEFAULT_DOCKER_PULL_TIMEOUT_SECONDS
_INSPECTOR_LIFECYCLE_TIMEOUT_SECONDS = 30


def _missing_rental_docker_host_key_log_text(
    default_extra: dict[str, Any],
    exc: RentalDockerConnectionError,
):
    return _m(
        "Missing executor SSH host key for rental Docker SDK operation",
        extra=get_extra_info({**default_extra, **HOST_KEY_REQUIRED_EXTRA, "error": str(exc)}),
    )


class _BoundLog:
    # structured logger bound to a default_extra, so a call site stays one line
    def __init__(self, base_extra: dict[str, Any]):
        self.base_extra = base_extra

    def _message(self, message: str, **fields: Any) -> _StructuredMessage:
        return _m(message, extra=get_extra_info({**self.base_extra, **fields}))

    # stacklevel=2 keeps the logged file/function/line pointing at the call site
    # rather than at this wrapper
    def info(self, message: str, **fields: Any) -> None:
        logger.info(self._message(message, **fields), stacklevel=2)

    def warning(self, message: str, **fields: Any) -> None:
        logger.warning(self._message(message, **fields), stacklevel=2)

    def error(self, message: str, *, exc_info: bool = False, **fields: Any) -> None:
        logger.error(self._message(message, **fields), exc_info=exc_info, stacklevel=2)


def _delete_container_log_extra(
    payload: ContainerDeleteRequest,
    executor_info: ExecutorSSHInfo,
) -> dict[str, Any]:
    return {
        "miner_hotkey": payload.miner_hotkey,
        "executor_uuid": payload.executor_id,
        "pod_id": payload.pod_id,
        "workload_kind": payload.workload_kind.value,
        "executor_ip_address": executor_info.address,
        "executor_port": executor_info.port,
        "executor_ssh_username": executor_info.ssh_username,
        "executor_ssh_port": executor_info.ssh_port,
    }


def _validate_delete_volume_names(payload: ContainerDeleteRequest) -> None:
    # raise ValueError on an unsafe volume name; the quoted result is unused, these names
    # reach the Docker SDK rather than a shell
    if payload.local_volume:
        _quote_safe_docker_volume_name(payload.local_volume, field_name="local_volume")
    if payload.external_volume:
        _quote_safe_docker_volume_name(payload.external_volume, field_name="external_volume")


@contextlib.contextmanager
def _best_effort_delete_step(log: _BoundLog, step: str, **fields: Any) -> Iterator[None]:
    # swallow a post-removal teardown failure: the container is already gone, so failing the
    # request here would make the backend retry a doomed undeploy and penalize the miner
    try:
        yield
    except Exception as exc:
        log.error(
            "delete_container post-teardown step failed (non-fatal)",
            step=step,
            error=str(exc),
            **fields,
        )


# DAH-2183: fresh vloopback sizing — compute effective volume/storage limits
# from on-host disk state when the backend sends disk_share.
_FRESH_SIZING_OVERHEAD_GB = 20   # reserved for system/docker overhead when reconstructing the pool
_FRESH_SIZING_HEADROOM_GB = 10   # min free space left on the fs after volume allocation
_FRESH_SIZING_GB_BYTES = 1024 ** 3
_VOLUME_SIZE_OPTION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kmgt]?)b?", re.IGNORECASE)
_VOLUME_SIZE_SUFFIX_MULTIPLIERS = {
    "": 1,
    "k": 1024,
    "m": 1024 ** 2,
    "g": 1024 ** 3,
    "t": 1024 ** 4,
}


class VolumeMinSizeError(Exception):
    """Fresh vloopback sizing produced a volume smaller than the requested minimum."""


@dataclass
class VolumeSizingResult:
    volume_limit_gb: int | None    # -> docker volume create -o size=
    storage_limit_gb: int | None   # -> docker run --storage-opt size=
    path: str                      # "fresh" | "legacy" | "fresh_fallback"
    capped_by: str | None = None   # "pool" | "request_cap" | "df_guard" (fresh path only)
    df_avail_bytes: int | None = None
    existing_volumes_bytes: int | None = None


def _parse_volume_size_to_bytes(value: str | None) -> int | None:
    """Parse a docker volume size into bytes.

    Accepts raw byte counts (vloopback ``Status.size-max``) and human size
    strings like ``19g`` / ``1t`` (vloopback ``Options.size``).
    """
    text = (value or "").strip()
    if not text:
        return None
    match = _VOLUME_SIZE_OPTION_RE.fullmatch(text)
    if not match:
        return None
    number, suffix = match.groups()
    return int(float(number) * _VOLUME_SIZE_SUFFIX_MULTIPLIERS[suffix.lower()])


def _exception_texts(exc: Exception) -> list[str]:
    texts = [str(exc)]
    if isinstance(exc, RetryError):
        last_exception = exc.last_attempt.exception()
        if last_exception is not None:
            texts.append(str(last_exception))
    return texts


class _CreateCancelledByDelete(Exception):
    """Raised at a create checkpoint when this pod's delete has already reported it gone."""


# DAH-2740: the name a pod's current container is parked under while an edit builds its replacement.
EDIT_PARKED_SUFFIX = "__prev"


class _EditSwap:
    """DAH-2740: keep the customer's container until its replacement runs, so a failed edit can be undone.

    The edit path used to force-remove the pod's own container in the stale sweep and only then
    create the new one, so anything that failed after the sweep — an unkillable container, a
    gocryptfs EPERM — left the customer with neither the old pod nor a new one (8 of 88 edits in
    one week). Now the container is renamed aside (instant, kills nothing) and stopped to free its
    ports and GPUs; the replacement is created under the original name; on success the parked one
    is removed, on any failure the half-built replacement is removed and the parked one is renamed
    back and started. If the container cannot even be stopped, the edit fails before anything was
    destroyed. Entered together with the SSH connection so the undo still has a live session.
    """

    def __init__(self, ssh_client: asyncssh.SSHClientConnection, container_name: str, default_extra: dict):
        self.ssh_client = ssh_client
        self.container_name = container_name
        self.parked_name: str | None = None
        self.default_extra = default_extra

    async def __aenter__(self) -> "_EditSwap":
        return self

    async def park(self) -> str | None:
        """Rename the running container aside and stop it. None when there is nothing to park."""
        listed = await self.ssh_client.run(
            f'/usr/bin/docker ps -a --format "{{{{.Names}}}}" --filter name=^{shlex.quote(self.container_name)}$'
        )
        if self.container_name not in (listed.stdout or "").split():
            return None
        parked = f"{self.container_name}{EDIT_PARKED_SUFFIX}"
        # a leftover from an earlier edit that never finished must not block the rename
        await self.ssh_client.run(f"/usr/bin/docker rm -fv {shlex.quote(parked)} 2>/dev/null || true")
        renamed = await self.ssh_client.run(
            f"/usr/bin/docker rename {shlex.quote(self.container_name)} {shlex.quote(parked)}"
        )
        if renamed.exit_status != 0:
            raise Exception(f"[park_current_container] docker rename failed: {(renamed.stderr or '').strip()}")
        self.parked_name = parked
        stopped = await self.ssh_client.run(f"/usr/bin/docker stop -t 10 {shlex.quote(parked)}")
        if stopped.exit_status != 0:
            # The container cannot be stopped: give it its name back and fail the edit with the
            # pod exactly as it was. This is the wedge that used to surface as RetryError[...]
            # after a destructive `docker rm -fv`. If even the rename back fails, parked_name
            # stays set so __aexit__ tries the full restore once more.
            renamed_back = await self.ssh_client.run(
                f"/usr/bin/docker rename {shlex.quote(parked)} {shlex.quote(self.container_name)}"
            )
            if renamed_back.exit_status == 0:
                self.parked_name = None
            raise Exception(
                "[park_current_container] the running container could not be stopped, the pod was left "
                f"as it was: {(stopped.stderr or '').strip()}"
            )
        logger.info(_m("Parked current container for edit", extra=get_extra_info({**self.default_extra, "parked": parked})))
        return parked

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if self.parked_name is None:
            return False
        if exc is None or isinstance(exc, _CreateCancelledByDelete):
            # Replacement is up (or the pod was deleted meanwhile): the old container is now the
            # stale one. Best effort — a wedged remove is left to the stale-container sweep.
            removed = await self.ssh_client.run(f"/usr/bin/docker rm -fv {shlex.quote(self.parked_name)}")
            if removed.exit_status != 0:
                logger.warning(
                    _m(
                        "Parked container could not be removed after edit; left for the stale sweep",
                        extra=get_extra_info({**self.default_extra, "parked": self.parked_name, "error": (removed.stderr or "").strip()}),
                    )
                )
            return False
        await self.restore()
        return False

    async def restore(self) -> None:
        """Undo the edit: drop the half-built replacement, give the parked container its name back, start it."""
        q_name, q_parked = shlex.quote(self.container_name), shlex.quote(self.parked_name)
        await self.ssh_client.run(f"/usr/bin/docker rm -fv {q_name} 2>/dev/null || true")
        renamed = await self.ssh_client.run(f"/usr/bin/docker rename {q_parked} {q_name}")
        started = await self.ssh_client.run(f"/usr/bin/docker start {q_name}") if renamed.exit_status == 0 else renamed
        extra = get_extra_info({**self.default_extra, "parked": self.parked_name, "restored": started.exit_status == 0})
        if started.exit_status == 0:
            logger.warning(_m("Edit failed; the pod's previous container was restored", extra=extra))
        else:
            logger.error(
                _m(
                    "Edit failed and the previous container could not be restored",
                    extra={**extra, "error": (started.stderr or "").strip()},
                )
            )


@dataclass
class _InflightCreate:
    # A retried rent can put a second create on the same pod while the first still runs, so the
    # entry outlives any single one of them and only the last to leave clears it.
    running: int = 0
    cancelled_by_delete: bool = False
    done: asyncio.Event = field(default_factory=asyncio.Event)


class _InflightCreateRegistry:
    """Creates running right now, and whether a delete has already cancelled them.

    DAH-2728: a delete that lands while the create for the same pod is still in flight finds no
    container yet ("No such container"), reports the pod deleted, and the create goes on to bind
    the host ports minutes later. That orphan then blocks the next paying rental with "port is
    already allocated". Only the create can undo itself, so the delete flips the flag here and the
    create tears itself down at its next checkpoint.
    """

    def __init__(self) -> None:
        self._creates_by_pod_id: dict[str, _InflightCreate] = {}

    @contextlib.contextmanager
    def track(self, pod_id: str) -> Iterator[None]:
        create = self._creates_by_pod_id.setdefault(pod_id, _InflightCreate())
        create.running += 1
        try:
            yield
        finally:
            create.running -= 1
            if create.running <= 0:
                self._creates_by_pod_id.pop(pod_id, None)
                create.done.set()

    def cancel(self, pod_id: str) -> bool:
        """Flag the in-flight create for this pod. False when no create is running."""
        create = self._creates_by_pod_id.get(pod_id)
        if create is None:
            return False
        create.cancelled_by_delete = True
        return True

    def is_cancelled(self, pod_id: str) -> bool:
        create = self._creates_by_pod_id.get(pod_id)
        return create is not None and create.cancelled_by_delete

    def is_running(self, pod_id: str) -> bool:
        return pod_id in self._creates_by_pod_id

    async def wait_until_done(self, pod_id: str, timeout: float) -> bool:
        """Wait for the create(s) on this pod to leave. False when the wait ran out."""
        create = self._creates_by_pod_id.get(pod_id)
        if create is None:
            return True
        try:
            await asyncio.wait_for(create.done.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return True


# In-process: a pod's create and delete are driven by the same validator event loop. Move it to
# Redis if the two ever land in different processes.
inflight_creates = _InflightCreateRegistry()

# How long a delete waits for the create it just cancelled. The create reads the flag at its next
# checkpoint, and the only checkpoint gap that can orphan a container is the short one before
# `docker run` — a pull-length wait would hold the customer's delete for nothing.
CANCELLED_CREATE_ABORT_TIMEOUT_SECONDS = 30.0


def _is_missing_docker_container_error(exc: Exception) -> bool:
    return any(_DOCKER_NO_SUCH_CONTAINER_PHRASE in text for text in _exception_texts(exc))


def _is_docker_container_removal_in_progress_error(exc: Exception) -> bool:
    return any(
        all(phrase in text.lower() for phrase in _DOCKER_REMOVAL_IN_PROGRESS_PHRASES)
        for text in _exception_texts(exc)
    )


def _is_stale_vloopback_mountpoint_error(error_text: str) -> bool:
    # Takes raw text so it serves both sources: a create-time exception, and the State.Error of a
    # container the host already tried to auto-restart.
    return all(phrase in error_text for phrase in _VLOOPBACK_MOUNT_ERROR_PHRASES)


def _is_safe_docker_volume_name(volume_name: str) -> bool:
    return bool(_DOCKER_VOLUME_NAME_RE.fullmatch(volume_name))


def _quote_safe_docker_volume_name(volume_name: str, *, field_name: str) -> str:
    if not _is_safe_docker_volume_name(volume_name):
        raise ValueError(f"Unsafe Docker volume name for {field_name}: {volume_name!r}")
    return shlex.quote(volume_name)


def _validate_cache_volume(cache_volume: CacheVolume) -> None:
    # raise ValueError on an unsafe FILLER cache volume. These mounts come from the backend, but a
    # `/`-prefixed source would be a HOST bind-mount and a `:` would corrupt the `src:target:rw` bind
    # spec — the name regex is the only guard, so validate defensively before it reaches the SDK.
    name = cache_volume.name
    if not _is_safe_docker_volume_name(name):
        raise ValueError(f"Unsafe cache volume name: {name!r}")
    # `volume_` marks the ephemeral run volumes every GC path deletes (clean_existing_containers,
    # clean_stale_vloopback_volumes); a persistent cache volume must never wear that prefix.
    if name.startswith("volume_"):
        raise ValueError(f"Cache volume name must not use the ephemeral 'volume_' prefix: {name!r}")
    target = cache_volume.target
    if not target.startswith("/") or target == "/" or ":" in target or ".." in target.split("/"):
        raise ValueError(f"Unsafe cache volume target: {target!r}")


def _build_cache_volume_mounts(payload: ContainerCreateRequest, occupied_targets: set[str]) -> list[VolumeMount]:
    # Persistent named cache volumes for a FILLER container (DPHN model/runtime cache). Empty for any
    # non-FILLER workload even when the field is set — a customer rental must never receive these
    # mounts. A cache entry whose target collides with an already-mounted path (the run's own volume or
    # /mnt) is skipped, since dockerd rejects a duplicate mount target and would fail the whole create.
    if payload.workload_kind != WorkloadKind.FILLER or not payload.cache_volumes:
        return []
    mounts: list[VolumeMount] = []
    for cache_volume in payload.cache_volumes:
        _validate_cache_volume(cache_volume)
        if cache_volume.target in occupied_targets:
            # Silent here would mean every start re-downloads into the ephemeral volume while the
            # named cache sits unmounted and unswept — indistinguishable from a working cache.
            logger.warning(
                _m(
                    "Cache volume skipped: its target is already mounted",
                    extra=get_extra_info({"volume": cache_volume.name, "target": cache_volume.target}),
                )
            )
            continue
        occupied_targets.add(cache_volume.target)
        mounts.append(VolumeMount(source=cache_volume.name, target=cache_volume.target))
    return mounts


def _wants_quote_socket(payload: ContainerCreateRequest, *, in_cvm: bool) -> bool:
    # customer rentals on a dstack CVM guest; a FILLER (DPHN/PEARL) has no attestation use and a
    # bare-metal node has no guest agent to broker
    return (
        settings.ENABLE_CVM_POD_QUOTE_SOCKET
        and in_cvm
        and payload.workload_kind == WorkloadKind.CUSTOMER_RENTAL
    )


def _is_vloopback_driver(driver: str) -> bool:
    return driver == _VLOOPBACK_DRIVER_PREFIX or driver.startswith(f"{_VLOOPBACK_DRIVER_PREFIX}:")


def _should_repair_stale_mountpoint(
    exc: Exception,
    local_volume: str | None,
    already_repaired: bool,
) -> bool:
    return (
        bool(local_volume)
        and not already_repaired
        and _is_stale_vloopback_mountpoint_error(str(exc))
    )

def _should_encrypt_local_volume(
    local_volume: str | None,
    workload_kind: WorkloadKind,
    is_sysbox: bool | None,
    enable_volume_encryption: bool | None,
) -> bool:
    return (
        bool(local_volume)
        and workload_kind != WorkloadKind.FILLER
        and bool(is_sysbox)
        and enable_volume_encryption
        and settings.ENABLE_VOLUME_ENCRYPTION
    )


def _opaque_shell_name() -> str:
    return f"_{secrets.token_hex(2)}"


def _xor_wrap_passphrase(passphrase: str) -> tuple[str, str]:
    raw = passphrase.encode("ascii")
    pad = secrets.token_bytes(len(raw))
    wrapped = bytes(a ^ b for a, b in zip(raw, pad, strict=True))
    return pad.hex(), wrapped.hex()


def _can_remount_encrypted_volume(local_volume_path: str | None) -> bool:
    # whether recovery has everything it needs to put a gocryptfs volume back the way the customer
    # had it. Checked before the container is started, because the remount only happens after
    # `docker start`: a missing piece would leave the pod flapping up and down once a cycle.
    # VOLUME_MASTER_SECRET must be the one the volume was created with — another validator's secret
    # derives a passphrase that simply will not mount. A plaintext path at or under the ciphertext
    # mount is refused as well: mounting the plaintext view inside the ciphertext it is decrypting
    # is not something recovery should attempt on customer data.
    if not (
        settings.RECOVER_ENCRYPTED_VOLUMES
        and settings.VOLUME_MASTER_SECRET
        and local_volume_path
    ):
        return False
    if local_volume_path == _LIUM_CIPHER_MOUNT or local_volume_path.startswith(
        f"{_LIUM_CIPHER_MOUNT}/"
    ):
        return False
    return _PLAINTEXT_PATH_RE.fullmatch(local_volume_path) is not None


def _shell_branch_when_gocryptfs_config_missing(allow_init: bool) -> str:
    if allow_init:
        return f'  gocryptfs -init {_LIUM_CIPHER_MOUNT} -passfile "$_pf"'
    return (
        f'  echo "gocryptfs.conf missing under {_LIUM_CIPHER_MOUNT};'
        ' refusing to re-initialise an existing rental volume" >&2\n'
        "  exit 1"
    )


def _build_gocryptfs_setup_and_mount_script(
    plaintext_path: str,
    *,
    pad_hex: str,
    wrapped_hex: str,
    pad_var: str,
    wrapped_var: str,
    passfile_path: str,
    allow_init: bool = True,
) -> str:
    # allow_init=False belongs to every start of a container that already exists — stale-mount
    # recovery and the backend's own start_container alike: such a volume was initialised at create
    # time, so a missing gocryptfs.conf means the ciphertext is gone. Re-initialising it there would
    # hand the customer an empty volume, report the start as a success and spare the miner the
    # penalty for data they destroyed. Only pod creation initialises.
    plaintext = shlex.quote(plaintext_path)
    passfile = shlex.quote(passfile_path)
    mount_check = (
        f"awk -v target={plaintext} "
        "'$2 == target && $3 == \"fuse.gocryptfs\" {found=1} END {exit !found}' /proc/mounts"
    )
    # Random pad XOR'd into the script as hex. Opaque names. No stdin.
    # ${v%${v#??}} / ${v#??} is portable sh for take/drop first two hex chars.
    return f"""set -e
{pad_var}={pad_hex}
{wrapped_var}={wrapped_hex}
_pf={passfile}
_d() {{ rm -f "$0" "$_pf" 2>/dev/null || true; unset {pad_var} {wrapped_var} _esc _x _y _pf; }}
trap _d EXIT
export PATH="/usr/local/bin:/usr/bin:/bin"
_esc=
while [ -n "${{{pad_var}}}" ]; do
  _x=${{{pad_var}%${{{pad_var}#??}}}}
  _y=${{{wrapped_var}%${{{wrapped_var}#??}}}}
  {pad_var}=${{{pad_var}#??}}
  {wrapped_var}=${{{wrapped_var}#??}}
  _esc=$_esc$(printf '\\\\%03o' $((0x$_x ^ 0x$_y)))
done
printf '%b' "$_esc" > "$_pf"
chmod 600 "$_pf"
unset _esc _x _y {pad_var} {wrapped_var}
mkdir -p {_LIUM_CIPHER_MOUNT} {plaintext}
if [ ! -f {_LIUM_CIPHER_MOUNT}/gocryptfs.conf ]; then
{_shell_branch_when_gocryptfs_config_missing(allow_init)}
fi
if ! {mount_check}; then
  gocryptfs {_LIUM_CIPHER_MOUNT} {plaintext} -passfile "$_pf" -o allow_other -nonempty
fi
"""


def build_startup_command_args(startup_commands: str | None) -> str:
    """Quote user-supplied startup_commands into a safe argv fragment.

    The fragment is appended to the host-side ``docker run ... <image>`` command
    that runs via ``/bin/sh -c`` over SSH as root on the executor. The user value
    is split into tokens (honouring its own quoting) and each token is
    ``shlex.quote``-d, so the host shell cannot interpret any metacharacter
    inside it: the tokens become the container's command + args, never a host
    command. Legitimate quoted commands such as ``bash -c "a && b"`` are
    preserved (the ``&&`` runs inside the container); break-out attempts such as
    a leading newline collapse to harmless container arguments. Unbalanced
    quotes or an empty value fall back to the image default command.
    """
    if not startup_commands or not startup_commands.strip():
        return ""
    try:
        tokens = shlex.split(startup_commands)
    except ValueError:
        # Unbalanced quotes etc. — don't risk a malformed/unsafe host command.
        return ""
    return " ".join(shlex.quote(token) for token in tokens)


class DockerService:
    def __init__(
        self,
        ssh_service: Annotated[SSHService, Depends(SSHService)],
        redis_service: Annotated[RedisService, Depends(RedisService)],
        attestation_service: Annotated[AttestationService, Depends(AttestationService)],
        rental_docker_client_factory: RentalDockerSdkClientFactory | None = None,
    ):
        self.ssh_service = ssh_service
        self.redis_service = redis_service
        self.attestation_service = attestation_service
        self.rental_docker_client_factory = (
            rental_docker_client_factory
            or RentalDockerSdkClientFactory(
                pull_timeout_seconds=_DOCKER_PULL_TIMEOUT_SECONDS,
            )
        )
        self.lock = asyncio.Lock()
        self.logs_queue: list[dict] = []
        self.log_task: asyncio.Task | None = None
        self.is_realtime_logging = False

    @staticmethod
    def get_container_name(payload: ContainerBaseRequest) -> str:
        if payload.workload_kind == WorkloadKind.FILLER:
            return f"{FILLER_CONTAINER_PREFIX}{payload.pod_id}"
        return f"{POD_CONTAINER_PREFIX}{payload.pod_id}"

    @staticmethod
    def _build_inspector_collector_command(
        executor_info: ExecutorSSHInfo,
        action: str,
    ) -> str:
        flag = {
            "start": "--start-collector",
            "stop": "--stop-collector",
        }[action]
        script = f"{executor_info.root_dir.rstrip('/')}/src/inspector_executor.py"
        command = f"{shlex.quote(executor_info.python_path)} {shlex.quote(script)} {flag}"
        if action == "start":
            command = f"nohup {command} >/dev/null 2>&1 &"
        return command

    async def _run_inspector_collector_lifecycle(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        executor_info: ExecutorSSHInfo,
        action: str,
        default_extra: dict,
    ) -> None:
        command = self._build_inspector_collector_command(executor_info, action)
        log_extra = {
            **default_extra,
            "command": command,
            "executor_uuid": executor_info.uuid,
            "inspector_action": action,
        }
        try:
            result = await ssh_client.run(
                command,
                timeout=_INSPECTOR_LIFECYCLE_TIMEOUT_SECONDS,
            )
            exit_status = getattr(result, "exit_status", 0)
            stderr = getattr(result, "stderr", "") or ""
            if exit_status != 0:
                raise RuntimeError(f"exit_status={exit_status} stderr={stderr}")
            outcome = "launched" if action == "start" else "succeeded"
            logger.info(
                _m(
                    f"Inspector collector {action} {outcome}",
                    extra=get_extra_info(log_extra),
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                _m(
                    f"Inspector collector {action} failed",
                    extra=get_extra_info({**log_extra, "error": str(exc)}),
                ),
            )

    async def _has_rented_containers(
        self,
        executor_info: ExecutorSSHInfo,
    ) -> bool:
        rented_machine = await self.redis_service.get_rented_machine(executor_info)
        return bool(rented_machine and rented_machine.get("containers"))

    def _ssh_bootstrap_script_path(self) -> Path:
        return Path(__file__).resolve().parent / "assets" / "sshd_bootstrap.sh"

    async def _run_docker_create_with_port_retry(
        self,
        ssh_client,
        command: str,
        container_name: str,
        log_tag: str,
        default_extra: dict,
        timeout: int,
        local_volume: str | None = None,
    ) -> None:
        """Run `docker run` with same-command retry on known Docker races.

        DAH-1991: backend-spawned `health_check_*` probes (TTL ~30s) can land
        on a port we already accepted into `port_maps` during the gap inside
        `create_container` (driven by `docker pull` and volume creation; the
        former 10s `clean_existing_containers` sleep was removed in DAH-1524).
        Wait through the probe's natural lifetime by retrying the same command
        on a 90s budget. Non-port-allocated errors propagate immediately.

        DAH-2018: Docker reserves the container name during command parse,
        before port-bind. A port-bind failure therefore leaves a Created-state
        container holding `pod_<id>`, and the next same-command attempt would
        otherwise collide with "container name already in use". Between
        attempts (after the backoff sleep, just before the next `docker run`)
        we issue `docker rm -fv <container_name>` so the rm→run window stays
        tight. Cleanup failures are warning-logged but do not abort the loop.

        DAH-2133: if Docker fails to mount an existing vloopback volume because
        an empty stale plugin mountpoint already exists, repair only that
        empty unmounted mountpoint and retry the same command once.
        """
        deadline = time.monotonic() + _PORT_ALLOCATED_RETRY_BUDGET_SEC
        attempt = 0
        vloopback_mount_repair_attempted = False
        while True:
            try:
                await self.execute_and_stream_logs(
                    ssh_client=ssh_client,
                    command=command,
                    log_tag=log_tag,
                    log_text="Creating docker container",
                    log_extra=default_extra,
                    timeout=timeout,
                )
                return
            except Exception as e:
                # Known Docker races:
                # repair stale vloopback mountpoints once;
                # retry port allocation failures until the short budget expires.
                if _should_repair_stale_mountpoint(e, local_volume, vloopback_mount_repair_attempted):
                    vloopback_mount_repair_attempted = True
                    if await self.repair_stale_vloopback_mountpoint(ssh_client, local_volume, default_extra):
                        logger.info(
                            _m(
                                "VLOOPBACK_STALE_MOUNTPOINT_RETRY",
                                extra=get_extra_info({**default_extra, "local_volume": local_volume}),
                            )
                        )
                        continue

                port_allocation_phrase = next((p for p in _PORT_ALLOCATED_PHRASES if p in str(e)), None)
                port_retry_deadline_expired = time.monotonic() >= deadline
                port_retry_needed = bool(port_allocation_phrase) and not port_retry_deadline_expired
                if port_retry_needed:
                    attempt += 1
                    logger.info(
                        _m(
                            "PORT_ALREADY_ALLOCATED_RETRY",
                            extra=get_extra_info({
                                **default_extra,
                                "attempt": attempt,
                                "remaining_sec": int(deadline - time.monotonic()),
                                "sleep_seconds": _PORT_ALLOCATED_RETRY_SLEEP_SEC,
                                "port_allocation_phrase": port_allocation_phrase,
                            }),
                        )
                    )
                    await asyncio.sleep(_PORT_ALLOCATED_RETRY_SLEEP_SEC)
                    await self._remove_failed_container_for_retry(
                        ssh_client=ssh_client,
                        container_name=container_name,
                        default_extra=default_extra,
                        warning_event="PORT_RETRY_STALE_RM_FAILED",
                    )

                    continue

                raise

    async def _run_rental_docker_create_with_port_retry(
        self,
        *,
        docker_client: RentalDockerSdkClient,
        ssh_client: asyncssh.SSHClientConnection,
        run_spec: ContainerRunSpec,
        container_name: str,
        default_extra: dict,
        local_volume: str | None = None,
        log_tag: str = "container_creation",
    ) -> None:
        deadline = time.monotonic() + _PORT_ALLOCATED_RETRY_BUDGET_SEC
        attempt = 0
        vloopback_mount_repair_attempted = False
        while True:
            try:
                await run_logged_rental_docker_sdk_operation(
                    operation="run_container",
                    log_extra=default_extra,
                    call=lambda: docker_client.run_container(run_spec),
                    attempt=attempt + 1,
                    **rental_run_spec_log_fields(run_spec),
                )
                return
            except Exception as exc:
                if _should_repair_stale_mountpoint(
                    exc,
                    local_volume,
                    vloopback_mount_repair_attempted,
                ):
                    vloopback_mount_repair_attempted = True
                    if await self.repair_stale_vloopback_mountpoint(
                        ssh_client,
                        local_volume,
                        default_extra,
                    ):
                        logger.info(
                            _m(
                                "VLOOPBACK_STALE_MOUNTPOINT_RETRY",
                                extra=get_extra_info(
                                    {**default_extra, "local_volume": local_volume}
                                ),
                            )
                        )
                        continue

                port_allocation_phrase = next(
                    (phrase for phrase in _PORT_ALLOCATED_PHRASES if phrase in str(exc)),
                    None,
                )
                port_retry_deadline_expired = time.monotonic() >= deadline
                port_retry_needed = bool(port_allocation_phrase) and not port_retry_deadline_expired
                if port_retry_needed:
                    attempt += 1
                    logger.info(
                        _m(
                            "PORT_ALREADY_ALLOCATED_RETRY",
                            extra=get_extra_info({
                                **default_extra,
                                "attempt": attempt,
                                "remaining_sec": int(deadline - time.monotonic()),
                                "sleep_seconds": _PORT_ALLOCATED_RETRY_SLEEP_SEC,
                                "port_allocation_phrase": port_allocation_phrase,
                            }),
                        )
                    )
                    await asyncio.sleep(_PORT_ALLOCATED_RETRY_SLEEP_SEC)
                    await self._remove_failed_rental_container_for_retry(
                        docker_client=docker_client,
                        container_name=container_name,
                        default_extra=default_extra,
                        warning_event="PORT_RETRY_STALE_RM_FAILED",
                    )
                    continue

                error_text = str(exc)
                logger.error(
                    _m(
                        "Docker SDK run container failed",
                        extra=get_extra_info(
                            {
                                **default_extra,
                                "container_name": container_name,
                                "error": error_text,
                            }
                        ),
                    ),
                    exc_info=True,
                )
                await self.stream_log(error_text, "error", log_tag)
                raise

    async def _remove_failed_rental_container_for_retry(
        self,
        *,
        docker_client: RentalDockerSdkClient,
        container_name: str,
        default_extra: dict,
        warning_event: str,
    ) -> None:
        try:
            await run_logged_rental_docker_sdk_operation(
                operation="remove_failed_container_for_retry",
                log_extra=default_extra,
                call=lambda: docker_client.remove_container(
                    container_name=container_name,
                    force=True,
                    remove_volumes=True,
                ),
                container_name=container_name,
                force=True,
                remove_volumes=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as rm_exc:
            logger.warning(
                _m(
                    warning_event,
                    extra=get_extra_info({
                        **default_extra,
                        "container_name": container_name,
                        "rm_error": str(rm_exc),
                    }),
                )
            )

    def _build_rental_container_run_spec(
        self,
        *,
        payload: ContainerCreateRequest,
        container_name: str,
        custom_options: CustomOptions,
        port_maps: list[tuple[int, int, int]],
        local_volume: str,
        local_volume_path: str,
        encrypted_local_volume: bool,
        external_volume_name: str | None,
        gpu_devices,
        effective_storage_limit_gb: int | None,
        cpu_count: int | None,
        quote_socket: bool = False,
    ) -> ContainerRunSpec:
        environment = {
            key: str(value)
            for key, value in (custom_options.environment or {}).items()
            if key and value and key.strip() and str(value).strip()
        }
        environment["NVIDIA_DRIVER_CAPABILITIES"] = "all"

        # DAH-2620: a node of a multi-node group rental gets its WireGuard overlay config injected and
        # the WireGuard UDP port published, so NCCL's socket bootstrap can reach the other nodes; the
        # tensors still travel over InfiniBand. Absent on an ordinary rental.
        cluster_udp_ports: tuple[int, ...] = ()
        if payload.cluster_membership is not None:
            cluster_networking = cluster_pod_networking(
                payload.cluster_membership.wireguard_conf,
                payload.cluster_membership.ssh_private_key,
                payload.cluster_membership.ssh_authorized_key,
            )
            environment.update(cluster_networking.environment)
            cluster_udp_ports = cluster_networking.published_udp_ports

        volume_target = _LIUM_CIPHER_MOUNT if encrypted_local_volume else local_volume_path
        volumes = [VolumeMount(source=local_volume, target=volume_target)]
        occupied_targets = {volume_target}
        if external_volume_name:
            volumes.append(VolumeMount(source=external_volume_name, target="/mnt"))
            occupied_targets.add("/mnt")
        # FILLER-only persistent cache volumes (DPHN model/runtime cache). No-op for customer rentals.
        volumes.extend(_build_cache_volume_mounts(payload, occupied_targets))
        # DAH-2828: the quote-only broker socket at the dstack SDK's default path, so a TEE
        # workload on a CVM node can take its own TDX quote from inside the pod.
        if quote_socket:
            volumes.append(quote_socket_pod_mount())

        device_mounts = [DeviceMount(path_on_host="/dev/net/tun", path_in_container="/dev/net/tun")]
        if encrypted_local_volume:
            device_mounts.append(DeviceMount(path_on_host="/dev/fuse", path_in_container="/dev/fuse"))
        device_mounts.extend(gpu_devices.device_mounts)
        devices = tuple(device_mounts)

        return ContainerRunSpec(
            image=payload.docker_image,
            name=container_name,
            command=build_container_command_argv(custom_options.startup_commands),
            environment=environment,
            ports=_published_ports(port_maps, cluster_udp_ports),
            volumes=tuple(volumes),
            restart_policy="unless-stopped",
            runtime="sysbox-runc" if payload.is_sysbox else None,
            cap_add=self._capabilities_for(devices),
            sysctls={"net.ipv4.conf.all.src_valid_mark": "1"},
            ulimits=self._memlock_ulimit_for(devices, payload.memory_gb),
            devices=devices,
            device_requests=gpu_devices.device_requests,
            cpu_count=cpu_count,
            memory_gb=payload.memory_gb,
            storage_limit_gb=effective_storage_limit_gb,
            shm_size=custom_options.shm_size,
            entrypoint=custom_options.entrypoint,
        )

    async def _ensure_pod_quote_socket(
        self,
        *,
        docker_client: RentalDockerSdkClient,
        ssh_client: asyncssh.SSHClientConnection,
        default_extra: dict,
        log_tag: str,
    ) -> bool:
        # best-effort: a guest whose broker cannot start still rents, the pod just gets no
        # socket and says so in its own log — a Docker Hub hiccup must not block every CVM rental
        try:
            await ensure_quote_broker(docker_client, ssh_client, log_extra=default_extra)
        except Exception as exc:
            logger.error(
                _m(
                    "CVM quote broker unavailable; creating the pod without /var/run/dstack.sock",
                    extra=get_extra_info({**default_extra, "error": str(exc)}),
                )
            )
            await self.stream_log(
                f"TDX quote socket unavailable in this pod (quote broker failed to start: {exc})",
                "warning",
                log_tag,
            )
            return False
        return True

    @staticmethod
    async def _assert_cluster_overlay_port_free(
        ssh_client: asyncssh.SSHClientConnection, default_extra: dict
    ) -> None:
        """A cluster node publishes the WireGuard port 1:1, so nothing else may hold it.

        Docker's own refusal is `Bind for 0.0.0.0:51820 failed: port is already allocated`, which
        says nothing about the overlay and sends whoever reads it hunting through the rental's TCP
        mappings. Checking first turns that into an answer that names the port and the holder
        (DAH-2620).
        """
        # UDP only: a co-tenant's TCP mapping can hold the same number, and a protocol-agnostic
        # filter would name that container as the holder and refuse a create nothing is blocking.
        result = await ssh_client.run(
            f"docker ps --filter publish={WIREGUARD_LISTEN_PORT}/udp --format '{{{{.Names}}}}'"
        )
        holders = [name.strip() for name in result.stdout.splitlines() if name.strip()]
        if not holders:
            return

        message = (
            f"UDP {WIREGUARD_LISTEN_PORT} is taken by {', '.join(holders)}, so the cluster overlay "
            "cannot bind. A node can carry one cluster pod at a time."
        )
        logger.error(_m("Cluster overlay port busy", extra=get_extra_info({**default_extra, "holders": holders})))
        raise RuntimeError(message)

    @staticmethod
    def _forwards_rdma(devices: tuple[DeviceMount, ...]) -> bool:
        return any(device.path_on_host.startswith("/dev/infiniband/") for device in devices)

    @classmethod
    def _capabilities_for(cls, devices: tuple[DeviceMount, ...]) -> tuple[str, ...]:
        """NET_ADMIN always, plus IPC_LOCK once the container holds verbs devices.

        Registering an RDMA memory region locks pages. Unlimited memlock covers a root process, but
        a workload that drops to an unprivileged user needs the capability as well, and without it
        `ibv_reg_mr` fails where every sysfs read still succeeds (DAH-2620).
        """
        if not cls._forwards_rdma(devices):
            return ("NET_ADMIN",)
        return ("NET_ADMIN", "IPC_LOCK")

    @classmethod
    def _memlock_ulimit_for(
        cls, devices: tuple[DeviceMount, ...], memory_gb: int | None
    ) -> tuple[ContainerUlimit, ...]:
        """Unlimited memlock, for a container that got RDMA devices AND a memory limit.

        RDMA pins the memory it registers and the default 64 KB is far below one queue pair, so the
        forwarded verbs devices are unusable without this (DAH-2571).

        Both conditions matter, though the limit is a bound, not safety. Locked pages are charged to
        the container's memory cgroup, so the tenant can pin at most `memory_gb` — but on a
        whole-host rental that is the machine: `ram_total` is host RAM less ~2 GiB. What the cgroup
        buys is a ceiling the kernel enforces and the OOM killer can act on. Without one —
        `mem_limit` is skipped for a falsy `memory_gb`, and a pod's `ram_total` defaults to 0 —
        there is no ceiling at all, and mlocked pages never reclaim.
        """
        forwards_rdma = any(
            device.path_on_host.startswith("/dev/infiniband/") for device in devices
        )
        if not forwards_rdma or not memory_gb:
            return ()
        return (ContainerUlimit(name="memlock", soft=-1, hard=-1),)

    async def _remove_failed_container_for_retry(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
        default_extra: dict,
        warning_event: str,
    ) -> None:
        try:
            # Reap the failed container together with its anonymous volumes (dind
            # images declare VOLUME /var/lib/docker); named volumes are unaffected by -v.
            await ssh_client.run(f"/usr/bin/docker rm -fv {shlex.quote(container_name)}")
        except asyncio.CancelledError:
            raise
        except Exception as rm_exc:
            logger.warning(
                _m(
                    warning_event,
                    extra=get_extra_info({
                        **default_extra,
                        "container_name": container_name,
                        "rm_error": str(rm_exc),
                    }),
                )
            )

    async def repair_stale_vloopback_mountpoint(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        local_volume: str,
        default_extra: dict,
    ) -> bool:
        if not _is_safe_docker_volume_name(local_volume):
            return False

        # Get the exact Docker-reported mountpoint path as the repair target.
        volume = shlex.quote(local_volume)
        inspect_result = await ssh_client.run(
            f"/usr/bin/docker volume inspect {volume} --format '{{{{.Driver}}}} {{{{.Mountpoint}}}}'",
            timeout=_VLOOPBACK_REPAIR_COMMAND_TIMEOUT_SEC,
        )
        if getattr(inspect_result, "exit_status", 0) != 0:
            return False

        driver, _, target = (inspect_result.stdout or "").strip().partition(" ")
        if not _is_vloopback_driver(driver) or target != f"/mnt/{local_volume}":
            return False
        plugin_result = await ssh_client.run(
            f"/usr/bin/docker plugin inspect {shlex.quote(driver)} --format '{{{{.Id}}}}'",
            timeout=_VLOOPBACK_REPAIR_COMMAND_TIMEOUT_SEC,
        )
        plugin_id = (plugin_result.stdout or "").strip()
        if getattr(plugin_result, "exit_status", 0) != 0 or not plugin_id:
            return False
        target = f"/var/lib/docker/plugins/{plugin_id}/propagated-mount/{local_volume}"

        # Recheck that the target is not currently mounted before removing it.
        mounted_result = await ssh_client.run(
            f"/usr/bin/findmnt {shlex.quote(target)} >/dev/null 2>&1",
            timeout=_VLOOPBACK_REPAIR_COMMAND_TIMEOUT_SEC,
        )
        mounted_exit_status = getattr(mounted_result, "exit_status", 1)
        if mounted_exit_status == 0:
            logger.warning(
                _m(
                    "VLOOPBACK_STALE_MOUNTPOINT_STILL_MOUNTED",
                    extra=get_extra_info({**default_extra, "local_volume": local_volume}),
                )
            )
            return False
        if mounted_exit_status != 1:
            return False

        # Repair by removing only the empty stale mountpoint directory.
        helper_cmd = (
            "/usr/bin/docker run --rm "
            f"-v {shlex.quote(str(Path(target).parent))}:/mnt "
            f"{_VLOOPBACK_REPAIR_IMAGE} rmdir /mnt/{shlex.quote(local_volume)}"
        )
        repair_result = await ssh_client.run(helper_cmd, timeout=_VLOOPBACK_REPAIR_COMMAND_TIMEOUT_SEC)
        if getattr(repair_result, "exit_status", 0) != 0:
            logger.warning(
                _m(
                    "VLOOPBACK_STALE_MOUNTPOINT_REPAIR_SKIPPED",
                    extra=get_extra_info({
                        **default_extra,
                        "local_volume": local_volume,
                        "exit_status": getattr(repair_result, "exit_status", None),
                        "stderr": getattr(repair_result, "stderr", ""),
                    }),
                )
            )
            return False

        logger.info(
            _m(
                "VLOOPBACK_STALE_MOUNTPOINT_REPAIRED",
                extra=get_extra_info({**default_extra, "local_volume": local_volume}),
            )
        )
        return True

    async def _prepare_known_hosts_policy(
        self,
        executor: ExecutorSSHInfo,
        miner_hotkey: str | None,
        log_context: dict,
    ) -> asyncssh.SSHKnownHosts | None:
        """Single chokepoint for every rental-path SSH host-key policy.

        N3 (CVM attestation gap remediation): with ENABLE_RENTAL_HOSTKEY_FAIL_CLOSED
        an unexpected error here fails closed — no host-key policy means no rental
        SSH — instead of silently falling back to unpinned SSH. Rental-path
        freshness rides the most recent validation-cycle attestation; no nonce is
        minted here by design.
        """
        try:
            host_policy = await self.attestation_service.prepare_host_policy(
                executor,
            )
            return host_policy.known_hosts
        except AttestationError:
            raise
        except Exception as exc:
            if settings.ENABLE_RENTAL_HOSTKEY_FAIL_CLOSED:
                logger.warning(
                    _m(
                        "Unable to prepare known_hosts policy — failing closed (no rental SSH)",
                        extra=get_extra_info({**log_context, "error": str(exc)}),
                    )
                )
                raise AttestationError(
                    f"Unable to prepare host-key policy for executor "
                    f"{executor.address}:{executor.port}; refusing unpinned rental SSH"
                ) from exc
            logger.warning(
                _m(
                    "Unable to prepare known_hosts policy",
                    extra=get_extra_info({**log_context, "error": str(exc)}),
                )
            )
            return None

    async def generate_portMappings(
        self,
        miner_hotkey: str,
        executor_id: str,
        pod_id: UUID,
        internal_ports: list[int] | None = None,
        initial_port_count: int | None = None,
        enable_jupyter: bool | None = False,
        available_ports_raw: list[PayloadPortMapping] | None = None,
        pod_mapping_raw: list[PayloadPortMapping] | None = None,
        workload_kind: WorkloadKind | None = None,
    ) -> tuple[list[tuple[int, int, int]], tuple[int, int] | None]:
        executor_uuid = UUID(executor_id)

        try:
            # Use distributed lock to prevent race conditions when allocating ports
            async with self.redis_service.acquire_executor_lock(executor_id):
                # Use port data from backend
                if available_ports_raw is not None and pod_mapping_raw is not None:
                    available_ports, pod_mapping = self._convert_payload_ports(available_ports_raw, pod_mapping_raw)
                    logger.info(f"Using port data from backend: {len(available_ports)} available, {len(pod_mapping)} pod mappings")
                else:
                    # No backend data provided - cannot proceed without port information
                    logger.error(f"No port data provided from backend for executor {executor_id}")
                    available_ports = {}
                    pod_mapping = {}

                if not pod_mapping and len(available_ports) < MIN_PORT_COUNT:
                    logger.warning(
                        f"Insufficient ports available ({len(available_ports)}/{MIN_PORT_COUNT}) "
                        f"for executor {executor_id}"
                    )
                    return [], None

                mappings = []
                reused_count = 0
                ssh_port = 22
                jupyter_port = 8888
                jupyter_port_map: tuple[int, int] | None = None

                user_defined = bool(internal_ports)
                docker_internal_ports = internal_ports or self._get_preferred_ports(initial_port_count)
                if ssh_port in docker_internal_ports:
                    docker_internal_ports.remove(ssh_port)
                docker_internal_ports.insert(0, ssh_port)

                if enable_jupyter:
                    if jupyter_port in docker_internal_ports:
                        docker_internal_ports.remove(jupyter_port)
                    docker_internal_ports.insert(1, jupyter_port)

                # Pre-strip every external_port that will be reused so the random.choice
                # / min branch below cannot pick a port still owned by pod_mapping.
                for port in docker_internal_ports:
                    if port in pod_mapping:
                        available_ports.pop(pod_mapping[port]["external_port"], None)

                for port in docker_internal_ports:
                    if port in pod_mapping:
                        port_mapping = pod_mapping[port]
                        mappings.append((port, port_mapping["internal_port"], port_mapping["external_port"]))
                        reused_count += 1
                        continue

                    if not len(available_ports):
                        break

                    filler_external_port = port + _FILLER_EXTERNAL_PORT_OFFSET
                    if (
                        workload_kind == WorkloadKind.FILLER
                        and user_defined
                        and port not in {ssh_port, jupyter_port}
                        and filler_external_port in available_ports
                    ):
                        docker_port = port
                        external_port = filler_external_port
                    elif port in available_ports:
                        docker_port = port
                        external_port = port
                    elif port == ssh_port or port == jupyter_port:
                        docker_port = port
                        external_port = max(available_ports.keys())
                    else:
                        external_port = random.choice(list(available_ports.keys())) if user_defined else min(available_ports.keys())
                        docker_port = port if user_defined else external_port

                    port_mapping = available_ports.pop(external_port)
                    mappings.append((docker_port, port_mapping["internal_port"], external_port))

                allocated_count = len(mappings) - reused_count
                logger.info(
                    f"Generated {len(mappings)} port mappings for pod {pod_id}: "
                    f"reused={reused_count}, allocated={allocated_count}, executor={executor_id}"
                )

                if enable_jupyter:
                    mapping = self._find_mapping_by_docker_port(mappings, jupyter_port)
                    if mapping:
                        jupyter_port_map = (mapping[0], mapping[2])

                # Port reservation now handled by backend

                return mappings, jupyter_port_map

        except (redis.exceptions.LockError, redis.exceptions.LockNotOwnedError) as e:
            logger.error(
                f"Failed to acquire or maintain lock for executor {executor_id} during port mapping generation: {e}",
                exc_info=True
            )
            # Return empty result to signal failure - caller should handle this case
            return [], None

    def _find_mapping_by_docker_port(self, mappings: list[tuple[int, int, int]], docker_port: int) -> tuple[int, int, int] | None:
        """Find a port mapping by docker port number."""
        return next((m for m in mappings if m[0] == docker_port), None)

    def _convert_payload_ports(
        self,
        available_ports_raw: list[PayloadPortMapping],
        pod_mapping_raw: list[PayloadPortMapping],
    ) -> tuple[dict[int, dict], dict[int, dict]]:
        """
        Convert payload port mappings to the format expected by generate_portMappings.

        Returns:
            - available_ports: dict[external_port, port_info_dict]
            - pod_mapping: dict[docker_port, port_info_dict]
        """
        available_ports: dict[int, dict] = {}
        for p in available_ports_raw:
            # Create a minimal port info dict with required fields
            port_info = {
                "internal_port": p.internal_port,
                "external_port": p.external_port,
                "docker_port": p.docker_port,
            }
            available_ports[p.external_port] = port_info

        pod_mapping: dict[int, dict] = {}
        for p in pod_mapping_raw:
            port_info = {
                "internal_port": p.internal_port,
                "external_port": p.external_port,
                "docker_port": p.docker_port,
            }
            # Use docker_port as key if available, otherwise fallback to external_port
            key = p.docker_port if p.docker_port is not None else p.external_port
            pod_mapping[key] = port_info

        return available_ports, pod_mapping

    async def execute_and_stream_logs(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        command: str,
        log_tag: str,
        log_text: str,
        log_extra: dict = {},
        timeout: int = 0,
        raise_exception: bool = True,
        stdin_data: str | None = None,
    ) -> tuple[bool, str]:
        logger.info(
            _m(
                log_text,
                extra=get_extra_info({
                    **log_extra,
                    "command": command,
                    "timeout_seconds": timeout,
                }),
            ),
        )

        await self.stream_log(log_text, "success", log_tag)

        status = True
        error = ''
        try:
            async with ssh_client.create_process(command) as process:
                if stdin_data is not None:
                    process.stdin.write(stdin_data)
                    process.stdin.write_eof()
                if timeout != 0:
                    status, error = await asyncio.wait_for(self._stream_process_output(process, log_tag), timeout=timeout)
                else:
                    status, error = await self._stream_process_output(process, log_tag)
        except TimeoutError:
            status = False
            error = "Process timed out"
            await self.stream_log(error, "error", log_tag)
            logger.warning(
                _m(
                    "Docker command timed out",
                    extra=get_extra_info({
                        **log_extra,
                        "command": command,
                        "timeout_seconds": timeout,
                        "log_text": log_text,
                    }),
                )
            )

        if not status and raise_exception:
            raise Exception(f"Failed {log_text}. command: {command} error: {error}")

        return status, error

    async def _stream_process_output(self, process, log_tag):
        status = True
        error = ''

        async for line in process.stdout:
            await self.stream_log(line.strip(), "success", log_tag)

        async for line in process.stderr:
            status = False
            error += line.strip() + "\n"
            await self.stream_log(line.strip(), "error", log_tag)

        return status, error

    async def handle_stream_logs(
        self,
        miner_hotkey,
        executor_id,
        pod_id,
    ):
        default_extra = {
            "miner_hotkey": miner_hotkey,
            "executor_uuid": executor_id,
            "pod_id": pod_id,
        }

        self.is_realtime_logging = True

        while True:
            await asyncio.sleep(LOG_STREAM_INTERVAL)

            async with self.lock:
                logs_to_process = self.logs_queue[:]
                self.logs_queue.clear()

            if logs_to_process:
                try:
                    await self.redis_service.publish(
                        STREAMING_LOG_CHANNEL,
                        {
                            "logs": logs_to_process,
                            "miner_hotkey": miner_hotkey,
                            "executor_uuid": executor_id,
                            "pod_id": pod_id,
                        },
                    )

                    logger.info(
                        _m(
                            f"Successfully published {len(logs_to_process)} logs",
                            extra=get_extra_info(default_extra),
                        )
                    )

                except Exception as e:
                    logger.error(
                        _m(
                            "Error publishing log stream",
                            extra=get_extra_info({**default_extra, "error": str(e)}),
                        ),
                        exc_info=True,
                    )

            if not self.is_realtime_logging:
                break

        logger.info(
            _m(
                "Exit handle_stream_logs",
                extra=get_extra_info(default_extra),
            )
        )

    async def finish_stream_logs(self):
        self.is_realtime_logging = False
        if self.log_task:
            await self.log_task

    async def check_container_running(
        self, ssh_client: asyncssh.SSHClientConnection, container_name: str, timeout: int = 10
    ):
        """Check if the container is running"""
        start_time = time.time()
        name_filter = shlex.quote(f"name={container_name}")
        while time.time() - start_time < timeout:
            result = await ssh_client.run(f"/usr/bin/docker ps -q --filter {name_filter}")
            if result.stdout.strip():
                return True
            await asyncio.sleep(1)
        return False

    async def wait_for_port_check_containers(
        self,
        executor_info: ExecutorSSHInfo,
        miner_hotkey: str,
        keypair: bittensor.Keypair,
        private_key: str,
        *,
        ssh_client: asyncssh.SSHClientConnection | None = None,
    ) -> tuple[bool, str]:
        """Force-remove lingering port-check / probe containers before a rental.

        Matches two prefix patterns:
        - 'container_{miner_hotkey}_*' — validator DinD/port-check probes (hotkey-scoped,
          preserved for cross-miner isolation on shared physical hosts)
        - 'health_check_*' — backend executor_health_check probes (hotkey-agnostic,
          backend creates these without a hotkey segment — see DAH-1991)

        DAH-2272: rentals take priority over background verification. Rather than
        WAITING for a lingering probe to clear (the old max_retries/retry_delay
        sleep loop, which absorbed 60-120s into "Started in subnet"), any match is
        force-removed IMMEDIATELY so the rental never blocks on a probe. Safe
        because:
        - A DinD probe killed mid-flight is tolerated on the verification side:
          ``PortConnectivityCheck`` treats a sysbox downgrade during
          ``renting_in_progress`` as inconclusive and keeps the last known value,
          so the miner is not penalised and the next verification cycle
          re-measures.
        - Any residual port-bind race is caught downstream by the 90s
          "port is already allocated" docker-create retry.

        DAH-2018: when the caller already holds an open SSH connection, pass it in
        via ``ssh_client`` to reuse the session (avoids a second connect and a
        wider TOCTOU gap); the late re-check inside ``create_container`` runs
        right before ``docker run``.

        Args:
            executor_info: Executor SSH connection info (ignored when
                ``ssh_client`` is provided).
            miner_hotkey: The miner's hotkey used to scope the container_* filter.
            keypair: Bittensor keypair for decrypting private key (ignored when
                ``ssh_client`` is provided).
            private_key: Encrypted SSH private key (ignored when ``ssh_client``
                is provided).
            ssh_client: Optional pre-opened SSH session to reuse.

        Returns:
            Tuple of (success: bool, message: str). Always succeeds — removal is
            best-effort and the rental proceeds regardless:
            - (True, "No port check containers found")
            - (True, "Port check containers forcefully removed")
        """
        container_prefix = f"container_{miner_hotkey}_"
        health_check_prefix = "health_check_"
        container_filter = shlex.quote(f"name=^{container_prefix}")
        health_check_filter = shlex.quote(f"name=^{health_check_prefix}")

        async def _run_checks(client: asyncssh.SSHClientConnection) -> tuple[bool, str]:
            # docker ps OR-s multiple --filter name= flags
            command = (
                '/usr/bin/docker ps --format "{{.Names}}" '
                f"--filter {container_filter} "
                f"--filter {health_check_filter}"
            )
            result = await client.run(command)

            if not result.stdout or not result.stdout.strip():
                return True, "No port check containers found"

            # Found lingering probe container(s). Force-remove IMMEDIATELY and let
            # the rental proceed — the customer-facing create request must not wait
            # on a verification/health-check probe (DAH-2272). Filter UNCHANGED:
            # targets BOTH container_{hotkey}_* AND health_check_*; do NOT narrow
            # to hotkey-only — the old retry loop could never clear a foreign
            # health_check_*, so removal is the only path that frees the port
            # (see DAH-2272 ADR).
            names = [n for n in result.stdout.strip().split("\n") if n]
            logger.warning(
                _m(
                    "port_check_force_removed",
                    extra=get_extra_info({
                        "miner_hotkey": miner_hotkey,
                        "context": "wait_for_port_check_containers",
                        "container_names": names,
                    }),
                )
            )

            remove_cmd = (
                "/usr/bin/docker ps -q "
                f"--filter {container_filter} "
                f"--filter {health_check_filter} "
                "| xargs -r /usr/bin/docker rm -fv"
            )
            await client.run(remove_cmd)

            logger.info("Forced removal of stale port check containers completed")
            return True, "Port check containers forcefully removed"

        if ssh_client is not None:
            try:
                return await _run_checks(ssh_client)
            except Exception as e:
                logger.error(f"Error checking for port check containers: {e}")
                return True, "Unable to check for port check containers, proceeding"

        # No reusable session — open a dedicated SSH connection.
        decrypted_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(decrypted_key)
        try:
            # DAH-2272: this early port-check connect is the one whose stall got
            # absorbed into "Started in subnet"; phase-time it so a future stall
            # is attributable to network (TCP) vs. remote sshd (login).
            async with connect_with_phase_timing(
                log_extra={
                    "miner_hotkey": miner_hotkey,
                    "executor_ip_address": executor_info.address,
                    "context": "wait_for_port_check_containers",
                },
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=None,
            ) as new_client:
                return await _run_checks(new_client)
        except Exception as e:
            logger.error(f"Error connecting to check for port check containers: {e}")
            # If we can't connect, assume it's safe to proceed
            return True, "Unable to check for port check containers, proceeding"

    async def clean_existing_containers(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        default_extra: dict,
        pod_name: str,
        sleep: int = 0,
        clear_volume: bool = True,
        active_container_names: list[str] | None = None,
        active_volume_names: list[str] | None = None,
    ):
        command = '/usr/bin/docker ps -a --format "{{.Names}}"'
        result = await ssh_client.run(command)
        if result.stdout.strip():
            # Optional pre-GC delay (default 0). DAH-1524 removed the 10s
            # deploy-path sleep; the port-race it hedged is now covered by
            # wait_for_port_check_containers + the 90s docker-run retry budget.
            if sleep:
                await asyncio.sleep(sleep)

            active_set = set(active_container_names) if active_container_names else set()
            active_volume_set = set(active_volume_names) if active_volume_names else set()
            pod_containers = [
                name for name in result.stdout.strip().split("\n")
                if name == pod_name
                or name.startswith(POD_CONTAINER_PREFIX)
                or name.startswith(FILLER_CONTAINER_PREFIX)
            ]
            stale_containers = []
            for name in pod_containers:
                if name in active_set:
                    continue
                stale_containers.append(name)
            container_names = " ".join(shlex.quote(name) for name in stale_containers)
            if not container_names:
                return

            logger.info(
                _m(
                    "Cleaning existing docker containers",
                    extra=get_extra_info({
                        **default_extra,
                        "container_names": container_names,
                        "active_containers": list(active_set),
                    }),
                ),
            )

            command = f'/usr/bin/docker rm -fv {container_names}'
            await retry_ssh_command(ssh_client, command, 'clean_existing_containers')

            if clear_volume:
                volumes_to_remove = []
                for name in stale_containers:
                    volume_id = name
                    for prefix in (POD_CONTAINER_PREFIX, FILLER_CONTAINER_PREFIX):
                        if name.startswith(prefix):
                            volume_id = name.removeprefix(prefix)
                            break
                    volume_name = f"volume_{volume_id}"
                    if volume_name not in active_volume_set:
                        volumes_to_remove.append(volume_name)
                if volumes_to_remove:
                    volumes = " ".join(shlex.quote(volume) for volume in volumes_to_remove)
                    command = f'/usr/bin/docker volume rm {volumes} 2>/dev/null || true'
                    await retry_ssh_command(ssh_client, command, 'clean_existing_containers')

    async def clean_stale_vloopback_volumes(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        default_extra: dict,
        skip_volume_names: list[str] | set[str] | None = None,
    ) -> None:
        skip_set = {name for name in (skip_volume_names or []) if name}
        list_volumes_cmd = '/usr/bin/docker volume ls --format "{{.Name}} {{.Driver}}"'
        mounted_volumes_cmd = (
            "/usr/bin/docker ps -a -q | xargs -r /usr/bin/docker inspect --format "
            "'{{range .Mounts}}{{if eq .Type \"volume\"}}{{.Name}}{{\"\\n\"}}{{end}}{{end}}'"
        )

        try:
            volume_result = await ssh_client.run(list_volumes_cmd)
            if getattr(volume_result, "exit_status", 0) != 0:
                logger.warning(
                    _m(
                        "Unable to list vloopback volumes",
                        extra=get_extra_info({
                            **default_extra,
                            "stderr": getattr(volume_result, "stderr", ""),
                        }),
                    )
                )
                return

            vloopback_volumes = set()
            for line in (volume_result.stdout or "").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    continue
                name, driver = parts
                if not (
                    name.startswith("volume_")
                    and (driver == "vloopback" or driver.startswith("vloopback:"))
                ):
                    continue
                vloopback_volumes.add(name)
            if not vloopback_volumes:
                return

            mounted_result = await ssh_client.run(mounted_volumes_cmd)
            if getattr(mounted_result, "exit_status", 0) != 0:
                logger.warning(
                    _m(
                        "Unable to inspect mounted Docker volumes",
                        extra=get_extra_info({
                            **default_extra,
                            "stderr": getattr(mounted_result, "stderr", ""),
                        }),
                    )
                )
                return

            mounted_volumes = {
                name.strip() for name in (mounted_result.stdout or "").splitlines() if name.strip()
            }
            stale_volumes = sorted(vloopback_volumes - mounted_volumes - skip_set)
            if not stale_volumes:
                return

            logger.info(
                _m(
                    "Cleaning stale vloopback Docker volumes",
                    extra=get_extra_info({
                        **default_extra,
                        "stale_volumes": stale_volumes,
                        "skipped_volumes": sorted(skip_set),
                    }),
                )
            )
            volumes = " ".join(shlex.quote(volume) for volume in stale_volumes)
            await retry_ssh_command(
                ssh_client,
                f"/usr/bin/docker volume rm {volumes} 2>/dev/null || true",
                "clean_stale_vloopback_volumes",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                _m(
                    "Failed to clean stale vloopback volumes",
                    extra=get_extra_info({**default_extra, "error": str(exc)}),
                ),
                exc_info=True,
            )

    async def _cache_rented_pod_best_effort(
        self,
        executor_info: ExecutorSSHInfo,
        pod_id: str,
        container_name: str,
        default_extra: dict,
    ) -> None:
        """Record the new container in the rented-machine cache; never fail the create over it.

        DAH-2475: this hash is a CACHE, not the source of truth — the backend rebuilds it wholesale
        every ~10 min (RentedMachineRequest -> delete(RENTED_MACHINE_PREFIX) + re-add) and already
        learns about this container from the ContainerCreated callback. Before this, a Redis blip at
        this last step raised and tripped cleanup_failed_container_creation, destroying a container
        that was already built and running: on 2026-07-22 a "Timeout connecting to server" here tore
        down a finished DPHN container and threw away its ~40 min of model download, then marked the
        run FAILED and put the node into launch backoff for a fault that was neither the node's nor
        the container's.
        """
        try:
            await self.redis_service.add_rented_pod(executor_info, pod_id, container_name)
        except Exception as exc:
            logger.warning(
                _m(
                    "Rented-pod cache write failed after the container was created; keeping the "
                    "container, the backend re-syncs this cache",
                    extra=get_extra_info({
                        **default_extra,
                        "container_name": container_name,
                        "pod_id": pod_id,
                        "error": str(exc),
                    }),
                ),
                exc_info=True,
            )

    async def reclaim_dphn_cache_for_rental(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        payload: ContainerCreateRequest,
        default_extra: dict,
    ) -> None:
        """Free the DPHN filler cache when a customer rental needs the disk it occupies.

        DAH-2475: the cache is filler property worth ~37 GB. Once a customer rents the node the disk
        belongs to the renter, so if what they asked for does not fit next to the cache, the cache
        goes — reclaiming here rather than at filler-stop is what stops the node re-downloading it
        every cycle (see sweep_stale_cache_volumes). Runs before volume sizing so the freed space is
        already visible when the pod's volume is measured and created.

        Only for CUSTOMER_RENTAL, only when the rental actually needs the room, and best-effort — a
        failure here must never break the rent.
        """
        if payload.workload_kind != WorkloadKind.CUSTOMER_RENTAL:
            return
        requested_gb: int | None = payload.volume_limit_gb
        try:
            cache_volumes: list[str] = await self._find_cache_volumes_to_sweep(ssh_client, set(), default_extra)
            if not cache_volumes:
                return

            # The backend sends volume_limit_gb=None for every executor whose docker lacks
            # --storage-opt support (calc_volume_storage_limit), so "no limit" does NOT mean "no disk
            # needed" — it means the renter may use the whole disk and no fit check is possible. Err on
            # the renter's side and give the cache back unconditionally; the filler re-downloads it
            # after the rental, which is the filler's cost to pay, not the customer's.
            if requested_gb:
                docker_root_dir = await self.get_docker_root_dir(ssh_client)
                free_bytes = await self._get_fs_available_bytes(ssh_client, docker_root_dir)
                free_gb = free_bytes / (1024**3)
                if free_gb >= requested_gb + RENTAL_DISK_HEADROOM_GB:
                    return

            logger.info(
                _m(
                    "Reclaiming DPHN cache volumes for the customer rental",
                    extra=get_extra_info({
                        **default_extra,
                        "volumes": cache_volumes,
                        "free_gb": round(free_gb, 1) if requested_gb else None,
                        "requested_gb": requested_gb,
                    }),
                )
            )
            await retry_ssh_command(
                ssh_client, DockerCommand.volume_remove(*cache_volumes), "reclaim_dphn_cache_for_rental"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                _m(
                    "DPHN cache reclaim for rental failed; continuing with the rent",
                    extra=get_extra_info({**default_extra, "error": str(exc)}),
                ),
                exc_info=True,
            )

    async def select_affordable_cache_volumes(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        payload: ContainerCreateRequest,
        default_extra: dict,
    ) -> list[CacheVolume]:
        """Which of the requested cache volumes this node can actually take.

        DAH-2475: the backend asks for the cache it WANTS and cannot know whether the host already has
        it — it only sees ~15-min-old telemetry. Deciding there meant charging the download on every
        launch, so a node granted the cache once dropped below the threshold by exactly that download
        and was refused it forever after: ~40 GB parked unmounted while every start re-downloaded it.

        Here the answer is a fact, not a projection. A volume that already exists costs nothing to
        mount, so it needs no check at all — that is what makes the decision idempotent. Only a volume
        that must be downloaded is measured, against live free space, so the node keeps enough room to
        stay in the rental listing (a delisted node is reachable by neither renters nor fillers).
        """
        if payload.workload_kind != WorkloadKind.FILLER or not payload.cache_volumes:
            return []
        requested_names: list[str] = [cache_volume.name for cache_volume in payload.cache_volumes]
        existing: set[str] = set(
            await self._find_cache_volumes_to_sweep(
                ssh_client, set(), default_extra, self._cache_volume_families(requested_names)
            )
        )
        present: list[CacheVolume] = [volume for volume in payload.cache_volumes if volume.name in existing]
        if len(present) == len(payload.cache_volumes):
            return list(payload.cache_volumes)

        try:
            docker_root_dir: str = await self.get_docker_root_dir(ssh_client)
            free_gb: float = await self._get_fs_available_bytes(ssh_client, docker_root_dir) / (1024**3)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Fail closed on the download, not on the launch: without a reading we cannot promise the
            # new volume fits, but what is already on the host is free to mount either way.
            logger.warning(
                _m(
                    "Could not measure free disk for the DPHN cache; granting only what already exists",
                    extra=get_extra_info({**default_extra, "error": str(exc)}),
                ),
            )
            return present

        floor_gb: int = DPHN_CACHE_LISTING_FLOOR_GB + DPHN_CACHE_FREE_MARGIN_GB
        if free_gb - DPHN_CACHE_SIZE_GB <= floor_gb:
            logger.info(
                _m(
                    "Node cannot afford the DPHN cache download; keeping only the volumes it already has",
                    extra=get_extra_info({
                        **default_extra,
                        "free_gb": round(free_gb, 1),
                        "kept_volumes": [volume.name for volume in present],
                    }),
                )
            )
            return present
        return list(payload.cache_volumes)

    @staticmethod
    def _cache_volume_families(volume_names: list[str]) -> tuple[str, ...]:
        """The cache prefixes these volume names belong to, e.g. `dphn_cache_` for a DPHN launch.

        DAH-2805: a launch may only sweep its OWN family. A node that moves from DPHN to ENGY asks
        for engy volumes, and sweeping every cache prefix there would delete the Dolphin cache the
        node needs for its next fast start — the very thing these volumes exist for.
        """
        return tuple(
            prefix
            for prefix in FILLER_CACHE_VOLUME_PREFIXES
            if any(name.startswith(prefix) for name in volume_names)
        )

    async def _find_cache_volumes_to_sweep(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        keep_names: set[str],
        default_extra: dict,
        name_prefixes: tuple[str, ...] = (DPHN_CACHE_VOLUME_PREFIX,),
    ) -> list[str]:
        # Cache volumes present on the host that this create no longer names, i.e. left by an older
        # model or runtime. Empty on any listing failure — sweeping is never worth failing a launch.
        if not name_prefixes:
            return []
        try:
            listed = await ssh_client.run('/usr/bin/docker volume ls --format "{{.Name}}"')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                _m(
                    "Failed to list filler cache volumes",
                    extra=get_extra_info({**default_extra, "error": str(exc)}),
                ),
                exc_info=True,
            )
            return []
        if listed.exit_status != 0:
            return []
        return sorted(
            name
            for name in (line.strip() for line in (listed.stdout or "").splitlines())
            if name.startswith(name_prefixes) and name not in keep_names
        )

    async def sweep_stale_cache_volumes(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        payload: ContainerCreateRequest,
        default_extra: dict,
    ) -> None:
        """Remove every DPHN cache volume on the host except the ones THIS create asks for.

        The cache volume name carries the model + runtime version, so a Dolphin model update makes the
        backend ask for a different name. Without this sweep the previous ~37 GB set would sit on the
        host forever (a named volume outside the `volume_` prefix is untouched by every other GC path)
        and each update would leak another set — the invariant is at most ONE model's cache per host.
        """
        if payload.workload_kind != WorkloadKind.FILLER or not payload.cache_volumes:
            # No cache requested -> this is version GC with nothing to compare against, NOT a reclaim.
            # Deleting here would thrash: the backend denies the cache to a node whose free disk sits
            # under its threshold, but a node's free disk is LOWER precisely because the cache is
            # already downloaded — so deny would delete it, free disk would rise back over the
            # threshold, the next cycle would grant it again, and the node would re-download ~37 GB
            # every cycle. Reclaiming a cache the node can no longer afford is the rental/disk-pressure
            # path's job, where the decision is made against real free space.
            return
        # A live filler SIBLING holds this node's cache and docker cannot remove an in-use volume, so
        # the listing round-trip would be a no-op. The run being created is already STARTING in the
        # backend by the time it builds this request, so its OWN container name is in the list too —
        # counting it made this guard always true and the sweep never ran (caught on staging).
        own_container_name: str = f"{FILLER_CONTAINER_PREFIX}{payload.pod_id}"
        if any(
            name.startswith(FILLER_CONTAINER_PREFIX) and name != own_container_name
            for name in payload.active_container_names or []
        ):
            return

        keep_names: set[str] = {cache_volume.name for cache_volume in payload.cache_volumes}
        stale_volumes: list[str] = await self._find_cache_volumes_to_sweep(
            ssh_client, keep_names, default_extra, self._cache_volume_families(sorted(keep_names))
        )
        if not stale_volumes:
            return
        logger.info(
            _m(
                "Sweeping stale filler cache volumes",
                extra=get_extra_info({
                    **default_extra,
                    "stale_volumes": stale_volumes,
                    "kept_volumes": sorted(keep_names),
                }),
            )
        )
        try:
            await retry_ssh_command(
                ssh_client, DockerCommand.volume_remove(*stale_volumes), "sweep_stale_cache_volumes"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Sweeping is never worth failing a launch: this runs inside create_container's try, so a
            # raise here becomes a container-create failure and costs the node a backoff strike for
            # housekeeping. The stale set survives until the next create or the disk-tight reclaim.
            logger.warning(
                _m(
                    "Stale DPHN cache sweep failed; continuing with the launch",
                    extra=get_extra_info({**default_extra, "stale_volumes": stale_volumes, "error": str(exc)}),
                )
            )

    async def capture_failed_container_diagnostics(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        default_extra: dict,
        container_name: str,
    ) -> ContainerDeathDiagnostics:
        """Read why a container died (inspect .State + logs tail + host context) and log it.

        Runs right before the failed container is force-removed: after that the
        OOMKilled/ExitCode verdict and the entrypoint output are gone, and the
        executor host belongs to the miner, so this log line is the only
        evidence left (DAH-2395: DPHN exit-137 deaths were undiagnosable).
        Best-effort — never raises (cancellation excepted), so the cleanup
        itself is never blocked by a diagnostics failure.
        """
        diagnostics = await collect_container_death_diagnostics(ssh_client, container_name)

        # Flat OOMKilled/ExitCode as top-level fields so Loki can filter on them
        # directly (`json | container_oom_killed="true"`); same shape as the
        # run-time pod-death path so both are queryable together.
        logger.warning(
            _m(
                "Failed container diagnostics before cleanup",
                extra=get_extra_info({
                    **default_extra,
                    "container_name": container_name,
                    **diagnostics.to_log_fields(),
                }),
            )
        )
        return diagnostics

    async def cleanup_failed_container_creation(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        default_extra: dict,
        container_name: str,
        volume_name: str | None = None,
        remove_volume: bool = False,
    ) -> bool:
        """Remove the failed container's artifacts; report whether it was already gone.

        DAH-2703: unproven cases (SSH dead, diagnostics failed) report False — never accuse a host
        on missing evidence.
        """
        container_missing = False
        try:
            # DAH-2395: read the death evidence before `rm -fv` destroys it —
            # without this line, "why did the container die" (OOM vs entrypoint
            # crash) is unanswerable: the host is the miner's, not ours.
            diagnostics = await self.capture_failed_container_diagnostics(
                ssh_client=ssh_client,
                default_extra=default_extra,
                container_name=container_name,
            )
            container_missing = diagnostics.container_missing

            container = shlex.quote(container_name)
            await retry_ssh_command(
                ssh_client,
                f"/usr/bin/docker rm -fv {container} 2>/dev/null || true",
                "cleanup_failed_container_creation",
            )

            if remove_volume and volume_name:
                volume = shlex.quote(volume_name)
                await retry_ssh_command(
                    ssh_client,
                    f"/usr/bin/docker volume rm {volume} 2>/dev/null || true",
                    "cleanup_failed_container_creation",
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                _m(
                    "Failed to clean up failed container creation artifacts",
                    extra=get_extra_info({
                        **default_extra,
                        "container_name": container_name,
                        "volume_name": volume_name,
                        "remove_volume": remove_volume,
                        "error": str(exc),
                    }),
                ),
                exc_info=True,
            )
        return container_missing

    async def _image_has_encrypted_volume_label(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        docker_image: str,
    ) -> bool:
        result = await ssh_client.run(
            "/usr/bin/docker image inspect "
            f"--format '{{{{index .Config.Labels \"{_ENCRYPTED_VOLUME_IMAGE_LABEL}\"}}}}' "
            f"{shlex.quote(docker_image)}",
            check=False,
        )
        if result.exit_status != 0:
            stderr = (result.stderr or "")[:200]
            stdout = (result.stdout or "")[:200]
            raise RuntimeError(
                "docker image inspect failed for volume-encryption label "
                f"(exit_status={result.exit_status}): stderr={stderr!r} stdout={stdout!r}"
            )
        return (result.stdout or "").strip() == "1"

    async def _encrypted_local_volume_name(
        self,
        docker_client: RentalDockerSdkClient,
        container_name: str,
    ) -> str | None:
        return await docker_client.mount_source_for_destination(
            container_name=container_name,
            destination=_LIUM_CIPHER_MOUNT,
        )

    async def setup_encrypted_local_volume(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
        plaintext_path: str,
        volume_name: str,
        pod_id: str,
        log_tag: str,
        log_extra: dict,
        allow_init: bool = True,
    ) -> None:
        passphrase = VolumeKeyDeriver.from_settings(settings).material(pod_id).passphrase

        container_q = shlex.quote(container_name)
        setup_script_path = f"/tmp/.x{uuid4().hex[:8]}"
        passfile_path = f"/tmp/.x{uuid4().hex[:8]}"
        pad_hex, wrapped_hex = _xor_wrap_passphrase(passphrase)
        pad_var = _opaque_shell_name()
        wrapped_var = _opaque_shell_name()
        while wrapped_var == pad_var:
            wrapped_var = _opaque_shell_name()

        async def wipe_tmp_files() -> None:
            await ssh_client.run(
                f"/usr/bin/docker exec -u 0 {container_q} rm -f "
                f"{shlex.quote(passfile_path)} {shlex.quote(setup_script_path)}",
                check=False,
            )

        async def fail_step(step: str, message: str, result: Any | None = None) -> None:
            exit_status = getattr(result, "exit_status", None)
            stdout = (getattr(result, "stdout", "") or "")[-2000:] if result else ""
            stderr = (getattr(result, "stderr", "") or "")[-2000:] if result else ""
            await self.stream_log(
                f"Encrypted volume setup failed ({step})",
                "error",
                log_tag,
            )
            logger.error(
                _m(
                    message,
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "plaintext_path": plaintext_path,
                        "cipher_mount": _LIUM_CIPHER_MOUNT,
                        "volume_name": volume_name,
                        "pod_id": pod_id,
                        "step": step,
                        "exit_status": exit_status,
                        "stdout": stdout,
                        "stderr": stderr,
                    }),
                )
            )
            raise RuntimeError(
                f"{message} (step={step}, exit_status={exit_status}, stderr={stderr or '<empty>'})"
            )

        await self.stream_log("Setting up encrypted local volume", "info", log_tag)
        logger.info(
            _m(
                "Encrypted volume setup started",
                extra=get_extra_info({
                    **log_extra,
                    "container_name": container_name,
                    "plaintext_path": plaintext_path,
                    "cipher_mount": _LIUM_CIPHER_MOUNT,
                    "volume_name": volume_name,
                    "pod_id": pod_id,
                }),
            )
        )

        setup_script = _build_gocryptfs_setup_and_mount_script(
            plaintext_path,
            pad_hex=pad_hex,
            wrapped_hex=wrapped_hex,
            pad_var=pad_var,
            wrapped_var=wrapped_var,
            passfile_path=passfile_path,
            allow_init=allow_init,
        )
        setup_heredoc = f"__SETUP_{uuid4().hex}__"
        upload_cmd = (
            f"/usr/bin/docker exec -u 0 -i {container_q} sh -c "
            f"\"cat > {setup_script_path}\" "
            f"<< '{setup_heredoc}'\n"
            f"{setup_script}\n"
            f"{setup_heredoc}"
        )
        logger.info(
            _m(
                "Uploading encrypted-volume setup script",
                extra=get_extra_info({**log_extra, "container_name": container_name, "pod_id": pod_id}),
            )
        )
        upload_result = await ssh_client.run(upload_cmd)
        if upload_result.exit_status != 0:
            await wipe_tmp_files()
            await fail_step(
                "upload_setup_script",
                "Failed to upload gocryptfs setup script into container",
                upload_result,
            )

        logger.info(
            _m(
                "Running gocryptfs init/mount",
                extra=get_extra_info({**log_extra, "container_name": container_name, "pod_id": pod_id}),
            )
        )
        mount_result = await ssh_client.run(
            f"/usr/bin/docker exec -u 0 {container_q} sh {shlex.quote(setup_script_path)}",
        )
        await wipe_tmp_files()
        if mount_result.exit_status != 0:
            await fail_step(
                "setup_or_mount",
                "Failed to initialize or mount gocryptfs inside container",
                mount_result,
            )

        verify_mount_script = (
            f"awk -v target={shlex.quote(plaintext_path)} "
            "'$2 == target && $3 == \"fuse.gocryptfs\" {found=1} END {exit !found}' /proc/mounts"
        )
        logger.info(
            _m(
                "Verifying gocryptfs mount",
                extra=get_extra_info({
                    **log_extra,
                    "container_name": container_name,
                    "plaintext_path": plaintext_path,
                    "pod_id": pod_id,
                }),
            )
        )
        verify_result = await ssh_client.run(
            f"/usr/bin/docker exec -u 0 {container_q} sh -lc {shlex.quote(verify_mount_script)}"
        )
        if verify_result.exit_status != 0:
            diagnostic_script = (
                'printf "%s\\n" "--- /proc/mounts ---"; '
                'cat /proc/mounts; '
                'printf "%s\\n" "--- gocryptfs ps ---"; '
                'ps aux | grep [g]ocryptfs || true'
            )
            diagnostic_result = await ssh_client.run(
                f"/usr/bin/docker exec -u 0 {container_q} sh -lc "
                f"{shlex.quote(diagnostic_script)}",
                check=False,
            )
            await fail_step(
                "verify_mount",
                "gocryptfs mount did not become visible inside container",
                diagnostic_result,
            )

        # The mount is created by root, so on an image whose USER is not root the
        # renter's own workload would own nothing inside its workspace and could
        # not write there (allow_other grants traversal, not permission).
        unwritable_workspace_error: str | None = await self._grant_workspace_to_container_user(
            ssh_client=ssh_client,
            container_q=container_q,
            plaintext_path=plaintext_path,
            log_extra={**log_extra, "container_name": container_name, "pod_id": pod_id},
        )
        if unwritable_workspace_error:
            await fail_step("chown_workspace", unwritable_workspace_error)

        await self.stream_log("Encrypted local volume mounted", "success", log_tag)

        logger.info(
            _m(
                "Encrypted local volume setup finished",
                extra=get_extra_info({
                    **log_extra,
                    "container_name": container_name,
                    "plaintext_path": plaintext_path,
                    "cipher_mount": _LIUM_CIPHER_MOUNT,
                    "volume_name": volume_name,
                    "pod_id": pod_id,
                }),
            ),
        )

    async def _grant_workspace_to_container_user(
        self,
        *,
        ssh_client: asyncssh.SSHClientConnection,
        container_q: str,
        plaintext_path: str,
        log_extra: dict,
    ) -> str | None:
        """Hand the freshly mounted workspace to the image's own user.

        The mount is created by root, so on an image whose USER is not root the
        renter would own nothing inside its own workspace. No-op for a root
        image. Returns an error message when the workspace is still not writable,
        so the caller fails the rental rather than billing for a pod whose
        encrypted volume the renter cannot use.

        The verdict is a real write probe run as the image's user, not a
        permission calculation: chown-ing the mountpoint says nothing about
        whether every parent directory on the way to it is traversable (a mount
        at ``/root/workspace`` under a 0700 ``/root`` is chown-ed and still
        unreachable).
        """
        # `docker inspect` on the host, not `id` in the container: a renter image
        # is not guaranteed to ship coreutils, and a probe we cannot run must not
        # be mistaken for a probe that passed.
        inspect_result = await ssh_client.run(
            f"/usr/bin/docker inspect -f '{{{{.Config.User}}}}' {container_q}",
            check=False,
        )
        if inspect_result.exit_status != 0:
            return (
                f"could not read the image USER of the rental container "
                f"(docker inspect exit={inspect_result.exit_status}, "
                f"stderr={(inspect_result.stderr or '')[-300:]!r})"
            )

        image_user: str = (inspect_result.stdout or "").strip()
        if image_user in ("", "0", "root", "0:0", "root:root"):
            return None

        user_q: str = shlex.quote(image_user)
        plaintext_q: str = shlex.quote(plaintext_path)
        chown_result = await ssh_client.run(
            f"/usr/bin/docker exec -u 0 {container_q} "
            f"sh -c {shlex.quote(f'chown {user_q} {plaintext_q}')}",
            check=False,
        )
        if chown_result.exit_status != 0:
            logger.warning(
                _m(
                    "Failed to chown encrypted workspace; probing writability anyway",
                    extra=get_extra_info({
                        **log_extra,
                        "image_user": image_user,
                        "stderr": (chown_result.stderr or "")[-500:],
                    }),
                )
            )

        # Unique name: the workspace may already hold renter data on a remount, and
        # a fixed probe path would delete a same-named file of theirs.
        probe_path_q: str = shlex.quote(f"{plaintext_path}/.lium-write-probe-{uuid4().hex}")
        probe_script: str = f"touch {probe_path_q} && rm -f {probe_path_q}"
        probe_result = await ssh_client.run(
            f"/usr/bin/docker exec -u {user_q} {container_q} sh -c {shlex.quote(probe_script)}",
            check=False,
        )
        if probe_result.exit_status != 0:
            return (
                f"encrypted workspace at {plaintext_path} is not writable by the image user "
                f"{image_user} (probe exit={probe_result.exit_status}, "
                f"stderr={(probe_result.stderr or '')[-300:]!r})"
            )
        return None

    async def install_open_ssh_server_and_start_ssh_service(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
        log_tag: str,
        log_extra: dict,
    ) -> bool:
        local_script_path = self._ssh_bootstrap_script_path()
        container_path = IN_CONTAINER_SSH_BOOTSTRAP_PATH
        success = True

        try:
            script_content = local_script_path.read_text()
        except Exception as exc:
            await self.stream_log("Failed to read SSH bootstrap script", "error", log_tag)
            logger.warning(
                _m(
                    "Failed to read SSH bootstrap script",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "local_script_path": str(local_script_path),
                        "error": str(exc),
                    }),
                ),
                exc_info=True,
            )
            return False

        try:
            container_name_quoted = shlex.quote(container_name)
            container_path_quoted = shlex.quote(container_path)
            heredoc = f"__LIUM_SSHD_BOOTSTRAP_{uuid4().hex}__"
            create_script_command = (
                f"/usr/bin/docker exec -i {container_name_quoted} sh -c "
                f"\"cat > {container_path_quoted} && chmod +x {container_path_quoted}\" "
                f"<< '{heredoc}'\n"
                f"{script_content}\n"
                f"{heredoc}"
            )
            logger.info(
                _m(
                    "Creating SSH bootstrap script inside container",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "local_script_path": str(local_script_path),
                        "container_path": container_path,
                    }),
                ),
            )
            create_result = await ssh_client.run(create_script_command)
            if create_result.exit_status != 0:
                await self.stream_log("Failed to create SSH bootstrap script in container", "error", log_tag)
                logger.warning(
                    _m(
                        "Failed to create SSH bootstrap script in container",
                        extra=get_extra_info({
                            **log_extra,
                            "container_name": container_name,
                            "container_path": container_path,
                            "exit_status": create_result.exit_status,
                            "stdout": create_result.stdout,
                            "stderr": create_result.stderr,
                        }),
                    )
                )
                return False
        except Exception as exc:
            await self.stream_log("Failed to create SSH bootstrap script in container", "error", log_tag)
            logger.warning(
                _m(
                    "Failed to create SSH bootstrap script in container",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "container_path": container_path,
                        "error": str(exc),
                    }),
                ),
                exc_info=True,
            )
            return False

        command = f"/usr/bin/docker exec {container_name_quoted} sh {container_path_quoted}"
        status, _ = await self.execute_and_stream_logs(
            ssh_client=ssh_client,
            command=command,
            log_tag=log_tag,
            log_text="Bootstrapping SSH daemon and watchdog",
            log_extra=log_extra,
            raise_exception=False,
        )
        success = success and status

        if not success:
            logger.warning(
                _m(
                    "SSH bootstrap script finished with errors",
                    extra=get_extra_info({**log_extra, "container_name": container_name}),
                )
            )

        return success

    async def install_open_ssh_server_and_start_ssh_service_with_rental_docker(
        self,
        docker_client: RentalDockerSdkClient,
        *,
        container_name: str,
        log_tag: str,
        log_extra: dict,
    ) -> bool:
        local_script_path = self._ssh_bootstrap_script_path()
        container_path = IN_CONTAINER_SSH_BOOTSTRAP_PATH

        try:
            script_content = local_script_path.read_text()
        except Exception as exc:
            await self.stream_log("Failed to read SSH bootstrap script", "error", log_tag)
            logger.warning(
                _m(
                    "Failed to read SSH bootstrap script",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "local_script_path": str(local_script_path),
                        "error": str(exc),
                    }),
                ),
                exc_info=True,
            )
            return False

        create_spec = ContainerExecSpec(
            container_name=container_name,
            argv=(
                "sh",
                "-c",
                f"cat > {shlex.quote(container_path)} "
                f"&& chmod +x {shlex.quote(container_path)}",
            ),
            stdin=script_content,
        )
        run_spec = ContainerExecSpec(
            container_name=container_name,
            argv=("sh", container_path),
        )

        try:
            create_result = await exec_logged_rental_docker_sdk_operation(
                docker_client=docker_client,
                operation="exec_create_ssh_bootstrap_script",
                exec_spec=create_spec,
                log_extra=log_extra,
            )
        except Exception as exc:
            await self.stream_log(
                "Failed to create SSH bootstrap script in container",
                "error",
                log_tag,
            )
            logger.warning(
                _m(
                    "Failed to create SSH bootstrap script in container",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "container_path": container_path,
                        "error": str(exc),
                    }),
                ),
                exc_info=True,
            )
            return False

        if create_result.exit_status != 0:
            await self.stream_log(
                "Failed to create SSH bootstrap script in container",
                "error",
                log_tag,
            )
            logger.warning(
                _m(
                    "Failed to create SSH bootstrap script in container",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "container_path": container_path,
                        "exit_status": create_result.exit_status,
                        "stdout": create_result.stdout,
                        "stderr": create_result.stderr,
                    }),
                )
            )
            return False

        run_result = await exec_logged_rental_docker_sdk_operation(
            docker_client=docker_client,
            operation="exec_run_ssh_bootstrap_script",
            exec_spec=run_spec,
            log_extra=log_extra,
        )
        if run_result.exit_status != 0:
            await self.stream_log(
                run_result.stderr or run_result.stdout or "SSH bootstrap script failed",
                "error",
                log_tag,
            )
            logger.warning(
                _m(
                    "SSH bootstrap script finished with errors",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "exit_status": run_result.exit_status,
                        "stdout": run_result.stdout,
                        "stderr": run_result.stderr,
                    }),
                )
            )
            return False

        return True

    async def add_ssh_public_keys_with_rental_docker(
        self,
        docker_client: RentalDockerSdkClient,
        *,
        container_name: str,
        public_keys: list[str] | tuple[str, ...],
        log_tag: str,
        log_extra: dict,
    ) -> None:
        exec_spec = build_authorized_keys_exec_spec(
            container_name=container_name,
            public_keys=public_keys,
        )
        result = await exec_logged_rental_docker_sdk_operation(
            docker_client=docker_client,
            operation="exec_add_authorized_keys",
            exec_spec=exec_spec,
            log_extra=log_extra,
        )
        if result.exit_status != 0:
            await self.stream_log(
                result.stderr or result.stdout or "Failed to add SSH public keys",
                "error",
                log_tag,
            )
            logger.warning(
                _m(
                    "Failed to add SSH public keys",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "exit_status": result.exit_status,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }),
                )
            )
            raise Exception(
                "Failed to add SSH public keys: "
                f"exit_status={result.exit_status}; "
                f"stderr={result.stderr}; stdout={result.stdout}"
            )

    async def remove_ssh_public_keys_with_rental_docker(
        self,
        docker_client: RentalDockerSdkClient,
        *,
        container_name: str,
        public_keys: list[str] | tuple[str, ...],
        log_tag: str,
        log_extra: dict,
    ) -> None:
        exec_spec = build_remove_authorized_keys_exec_spec(
            container_name=container_name,
            public_keys=public_keys,
        )
        result = await exec_logged_rental_docker_sdk_operation(
            docker_client=docker_client,
            operation="exec_remove_authorized_keys",
            exec_spec=exec_spec,
            log_extra=log_extra,
        )
        if result.exit_status != 0:
            await self.stream_log(
                result.stderr or result.stdout or "Failed to remove SSH public keys",
                "error",
                log_tag,
            )
            logger.warning(
                _m(
                    "Failed to remove SSH public keys",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "exit_status": result.exit_status,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }),
                )
            )
            raise Exception(
                "Failed to remove SSH public keys: "
                f"exit_status={result.exit_status}; "
                f"stderr={result.stderr}; stdout={result.stdout}"
            )

    async def add_environment_variables_with_rental_docker(
        self,
        docker_client: RentalDockerSdkClient,
        *,
        container_name: str,
        environment: dict[str, str] | None,
        log_tag: str,
        log_extra: dict,
    ) -> str | None:
        # returns the failure cause, or None when the environment was applied
        exec_spec = build_environment_exec_spec(
            container_name=container_name,
            environment=environment,
        )
        if exec_spec is None:
            return None

        try:
            result = await exec_logged_rental_docker_sdk_operation(
                docker_client=docker_client,
                operation="exec_append_environment",
                exec_spec=exec_spec,
                log_extra=log_extra,
            )
        except Exception as exc:
            await self.stream_log("Failed to set environment variables", "error", log_tag)
            logger.warning(
                _m(
                    "Failed to set environment variables",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "error": str(exc),
                    }),
                ),
                exc_info=True,
            )
            return str(exc)

        if result.exit_status != 0:
            await self.stream_log(
                result.stderr or result.stdout or "Failed to set environment variables",
                "error",
                log_tag,
            )
            logger.warning(
                _m(
                    "Failed to set environment variables",
                    extra=get_extra_info({
                        **log_extra,
                        "container_name": container_name,
                        "exit_status": result.exit_status,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }),
                )
            )
            return f"exit_status={result.exit_status}; stderr={result.stderr}; stdout={result.stdout}"

        return None

    async def resolve_sysbox_subuid_base(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        log_extra: dict,
    ) -> int | None:
        """Read the host uid that sysbox maps container root to, None if unusable.

        The SSH session lands inside the executor container, whose own /etc/subuid
        carries no sysbox entry; the executor shares the host PID namespace, so the
        host copy is reachable through the host init's root.
        """
        result = await ssh_client.run("cat /proc/1/root/etc/subuid")
        sysbox_entries: list[str] = [
            line for line in (result.stdout or "").splitlines()
            if line.startswith("sysbox:")
        ]
        if not sysbox_entries:
            return None

        try:
            _, subuid_base_text, slice_size_text = sysbox_entries[0].strip().split(":")
            subuid_base: int = int(subuid_base_text)
            slice_size: int = int(slice_size_text)
        except ValueError:
            return None

        if slice_size != SYSBOX_SUBUID_SLICE_SIZE:
            logger.warning(
                _m(
                    "sysbox subuid range is not a single slice; cannot align s3fs owner",
                    extra=get_extra_info({**log_extra, "slice_size": slice_size}),
                )
            )
            return None

        return subuid_base

    async def create_s3fs_volume(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        log_extra: dict,
        volume_info: ExternalVolumeInfo,
        log_tag: str,
        sysbox_subuid_base: int | None,
    ):
        # The name arrives from the backend and lands in a plugin alias, a volume
        # name and two shell commands — refuse it before any of that.
        if not _is_safe_docker_volume_name(volume_info.name):
            message = f"Unsafe external volume name: {volume_info.name!r}"
            logger.warning(_m(f"s3fs_volume failed. {message}", extra=get_extra_info({**log_extra})))
            return False, message

        plugin_alias: str = _s3fs_plugin_alias(volume_info.name)

        # DAH-2496: the kernel cannot ID-shift a FUSE mount, so under sysbox the
        # pod's root would see the bucket as nobody. s3fs keeps owner in object
        # metadata, so report the objects as owned by the pod's root instead.
        mount_options: str = "allow_other"
        if sysbox_subuid_base is not None:
            mount_options = f"allow_other,uid={sysbox_subuid_base},gid={sysbox_subuid_base}"

        setup_results: list[asyncssh.SSHCompletedProcess] = (
            await self._install_s3fs_plugin_instance(
                ssh_client=ssh_client,
                plugin_alias=plugin_alias,
                volume_info=volume_info,
                mount_options=mount_options,
            )
        )

        create_command: str = (
            f"/usr/bin/docker volume create -d {plugin_alias} {volume_info.name}"
        )
        result = await self.execute_and_stream_logs(
            ssh_client=ssh_client,
            command=create_command,
            log_tag=log_tag,
            log_text="Creating docker volume",
            log_extra=log_extra,
            raise_exception=False,
        )
        if not result[0]:
            setup_results.extend(
                await self._drop_legacy_shared_s3fs_volume(ssh_client, volume_info.name)
            )
            result = await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=create_command,
                log_tag=log_tag,
                log_text="Creating docker volume",
                log_extra=log_extra,
                raise_exception=False,
            )

        is_success, message = result
        if not is_success:
            responses_text = message
            for i, r in enumerate(setup_results):
                responses_text += f"|Step {i}: exit={r.exit_status}, stdout={r.stdout}, stderr={r.stderr}"
            logger.warning(_m(f"s3fs_volume failed. {responses_text}",extra=get_extra_info({**log_extra})))
        else:
            logger.info(_m("s3fs_volume success", extra=get_extra_info({**log_extra})))

        return result

    async def _install_s3fs_plugin_instance(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        plugin_alias: str,
        volume_info: ExternalVolumeInfo,
        mount_options: str,
    ) -> list[asyncssh.SSHCompletedProcess]:
        """Install and configure the plugin instance that serves this one volume."""
        commands: list[str] = [
            f"/usr/bin/docker plugin install {S3FS_PLUGIN_IMAGE} "
            f"--alias {plugin_alias} --grant-all-permissions --disable",
            f"/usr/bin/docker plugin disable {plugin_alias} -f",
            f"/usr/bin/docker plugin set {plugin_alias} "
            f"AWSACCESSKEYID={volume_info.iam_user_access_key} "
            f"AWSSECRETACCESSKEY={volume_info.iam_user_secret_key}",
            f'/usr/bin/docker plugin set {plugin_alias} DEFAULT_S3FSOPTS="{mount_options}"',
            f"/usr/bin/docker plugin enable {plugin_alias}",
        ]
        return [await ssh_client.run(command) for command in commands]

    async def _drop_legacy_shared_s3fs_volume(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        volume_name: str,
    ) -> list[asyncssh.SSHCompletedProcess]:
        """Free a volume name still held by the pre-DAH-2512 shared plugin instance.

        Docker refuses both create and rm of the name while that instance is
        disabled — it cannot query it — so enable it first and leave it enabled:
        pods still mounting through it keep working. Dropping the handle is safe,
        the data lives in the bucket.
        """
        return [
            await ssh_client.run(
                f"/usr/bin/docker plugin enable {LEGACY_S3FS_PLUGIN_ALIAS}"
            ),
            await ssh_client.run(f"/usr/bin/docker volume rm {volume_name}"),
        ]

    async def remove_s3fs_volume_plugin(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        volume_name: str,
    ) -> None:
        """Drop the plugin instance that served this volume, leaving the rest alone."""
        if not _is_safe_docker_volume_name(volume_name):
            return

        plugin_alias: str = _s3fs_plugin_alias(volume_name)
        await ssh_client.run(f"/usr/bin/docker plugin disable {plugin_alias} -f")
        await ssh_client.run(f"/usr/bin/docker plugin rm {plugin_alias}")

    async def run_jupyter(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
        jupyter_token: str,
        jupyter_port: int,
        log_tag: str,
        log_extra: dict,
        local_volume: str | None = None,
        local_volume_path: str = '/root',
        encrypted_local_volume: bool = False,
    ):
        if local_volume and not encrypted_local_volume:
            temp_container_name = f"temp_jupyter_copy_{uuid4()}"
            try:
                command = (
                    f"/usr/bin/docker run -d --rm -v {local_volume}:/mnt "
                    f"--name {temp_container_name} --entrypoint sh "
                    f"daturaai/compute-subnet-executor:latest -c 'cp /root/app/run_jupyter.sh /mnt/'"
                )
                await self.execute_and_stream_logs(
                    ssh_client=ssh_client,
                    command=command,
                    log_tag=log_tag,
                    log_text="Creating temporary container for script copy",
                    log_extra=log_extra,
                    raise_exception=True,
                )
            finally:
                command = f"/usr/bin/docker rm -fv {temp_container_name}"
                await self.execute_and_stream_logs(
                    ssh_client=ssh_client,
                    command=command,
                    log_tag=log_tag,
                    log_text="Removing temporary container",
                    log_extra=log_extra,
                    raise_exception=False,
                )

            # local_volume_path comes from the renter's template, so it reaches the
            # host shell only through shlex.quote — same as the branch below.
            container_q = shlex.quote(container_name)
            script_q = shlex.quote(f"{local_volume_path}/run_jupyter.sh")
            command = (
                f"/usr/bin/docker exec -u 0 {container_q} "
                f"sh -c {shlex.quote(f'chmod +x {script_q}')}"
            )
            await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="Making run_jupyter.sh executable",
                log_extra=log_extra,
                raise_exception=True,
            )

            command = (
                f"/usr/bin/docker exec -i -u 0 {container_q} sh -c "
                f"{shlex.quote(f'read JUPYTER_PASSWORD; {script_q} --password=$JUPYTER_PASSWORD --port={jupyter_port}')}"
            )
            status, error = await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="Running jupyter from volume",
                log_extra=log_extra,
                raise_exception=False,
                stdin_data=f"{jupyter_token}\n",
            )
        else:
            target_path = local_volume_path if encrypted_local_volume else "/root"
            container_q = shlex.quote(container_name)
            target_q = shlex.quote(target_path)
            command = (
                f"/usr/bin/docker exec -u 0 {container_q} "
                f"sh -c {shlex.quote(f'mkdir -p {target_q}')}"
            )
            await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="Preparing Jupyter script directory",
                log_extra=log_extra,
                raise_exception=True,
            )
            command = (
                f"/usr/bin/docker cp /root/app/run_jupyter.sh "
                f"{container_q}:/tmp/run_jupyter.sh"
            )
            await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="Copying run_jupyter.sh to container",
                log_extra=log_extra,
                raise_exception=True,
            )
            command = (
                f"/usr/bin/docker exec -u 0 {container_q} "
                f"sh -c {shlex.quote(f'cp /tmp/run_jupyter.sh {target_q}/run_jupyter.sh && chmod +x {target_q}/run_jupyter.sh')}"
            )
            await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="Installing run_jupyter.sh",
                log_extra=log_extra,
                raise_exception=True,
            )
            command = (
                f"/usr/bin/docker exec -i -u 0 {container_q} sh -c "
                f"{shlex.quote(f'read JUPYTER_PASSWORD; {target_q}/run_jupyter.sh --password=$JUPYTER_PASSWORD --port={jupyter_port}')}"
            )
            status, error = await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=command,
                log_tag=log_tag,
                log_text="Running jupyter",
                log_extra=log_extra,
                raise_exception=False,
                stdin_data=f"{jupyter_token}\n",
            )

        # Only raise exception for actual errors, not warnings or info messages
        if not status and error and any(keyword.lower() in error.lower() for keyword in [
            "Error", "FATAL", "CRITICAL", "Traceback", "Exception",
            "Permission denied", "Address already in use", "No such file or directory",
            "Connection refused", "Port already in use", "Failed to start"
        ]):
            raise Exception(error)

    async def get_docker_root_dir(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        timeout: int | None = None,
    ):
        """Get Docker storage info using docker info command"""
        command = "/usr/bin/docker info --format '{{.DockerRootDir}}'"
        if timeout is None:
            result = await ssh_client.run(command)
        else:
            result = await ssh_client.run(command, timeout=timeout)
        return result.stdout.strip()

    @staticmethod
    def _get_local_volume_create_timeout(limit: int | None, requested_timeout: int) -> int:
        if requested_timeout == 0:
            return requested_timeout
        if not limit or limit <= _LOCAL_VOLUME_TIMEOUT_THRESHOLD_GB:
            return requested_timeout

        scaled_timeout = _LOCAL_VOLUME_TIMEOUT_BASE_SEC + (
            (limit + _LOCAL_VOLUME_TIMEOUT_GB_PER_SEC - 1)
            // _LOCAL_VOLUME_TIMEOUT_GB_PER_SEC
        )
        return max(requested_timeout, min(scaled_timeout, _LOCAL_VOLUME_TIMEOUT_MAX_SEC))

    async def create_local_volume(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        docker_client: RentalDockerSdkClient,
        local_volume: str,
        log_tag: str,
        log_text: str,
        log_extra: dict,
        limit: int | None = None,
        timeout: int = 10,
        sparse: bool = False,
    ):
        requested_timeout = timeout
        _quote_safe_docker_volume_name(
            local_volume,
            field_name="local_volume",
        )
        if limit:
            # install loopback plugin
            loopback_plugin_name = "vloopback"

            docker_root_dir = await self.get_docker_root_dir(ssh_client)
            logger.info(_m(f"Docker data root: {docker_root_dir}", extra=get_extra_info(log_extra)))

            loopback_plugin_arg = shlex.quote(loopback_plugin_name)
            data_dir_arg = shlex.quote(f"DATA_DIR={docker_root_dir}/loopback")
            command = (
                "/usr/bin/docker plugin install ashald/docker-volume-loopback "
                f"--alias {loopback_plugin_arg} --grant-all-permissions {data_dir_arg}"
            )
            # TODO: migrate Docker plugin management if/when plugin setup becomes
            # part of the SDK migration scope. The user-controlled volume name is
            # not used in this shell command; volume creation below is SDK-backed.
            await ssh_client.run(command)

            # DAH-2265 Plan 3: the loopback plugin preallocates the whole backing file
            # by default (creation time scales with size). `sparse=true` writes a sparse
            # file (creation ~flat) while still capping the volume at `size`. Gated to
            # full-node rentals only — see the caller — because a sparse partial volume
            # leaves its unwritten space free in df AND still counts its declared size in
            # resolve_volume_sizing(), double-counting the pool and overcommitting host disk.
            volume_driver = loopback_plugin_name
            volume_driver_opts = {"size": f"{limit}g"}
            if sparse:
                volume_driver_opts["sparse"] = "true"
        else:
            loopback_plugin_name = None
            volume_driver = None
            volume_driver_opts = None

        timeout = self._get_local_volume_create_timeout(limit, timeout)
        volume_log_extra = {
            **log_extra,
            "local_volume": local_volume,
            "volume_limit_gb": limit,
            "requested_timeout_seconds": requested_timeout,
            "effective_timeout_seconds": timeout,
            "timeout_scaled": timeout != requested_timeout,
            "loopback_plugin": loopback_plugin_name,
            "sparse": bool(sparse and limit),
        }
        logger.info(
            _m(
                "Preparing local Docker volume creation",
                extra=get_extra_info(volume_log_extra),
            )
        )
        await self.stream_log(log_text, "success", log_tag)
        await run_logged_rental_docker_sdk_operation(
            operation="create_volume",
            log_extra=volume_log_extra,
            call=lambda: docker_client.create_volume(
                volume_name=local_volume,
                driver=volume_driver,
                driver_opts=volume_driver_opts,
                timeout=timeout,
            ),
            volume_name=local_volume,
            volume_driver=volume_driver,
            volume_driver_opts=volume_driver_opts,
            timeout_seconds=timeout,
        )

    async def _get_fs_available_bytes(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        docker_root_dir: str,
    ) -> int:
        # Delegates to the shared helper so the grant path (here) and the disk-tight reclaim
        # (container_cleanup) can never disagree on how free disk is measured — a df fix landing in
        # one copy only would make the validator grant a cache the backstop immediately reclaims.
        return await df_available_bytes(ssh_client, docker_root_dir)

    async def _get_existing_vloopback_bytes(
        self,
        ssh_client: asyncssh.SSHClientConnection,
    ) -> int:
        list_result = await ssh_client.run(
            '/usr/bin/docker volume ls --format "{{.Name}} {{.Driver}}"'
        )
        if getattr(list_result, "exit_status", 0) != 0:
            raise Exception(
                f"docker volume ls failed: {getattr(list_result, 'stderr', '')}"
            )

        volume_names = []
        for line in (list_result.stdout or "").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            name, driver = parts
            if _is_vloopback_driver(driver) and _is_safe_docker_volume_name(name):
                volume_names.append(name)
        if not volume_names:
            return 0

        names = " ".join(shlex.quote(name) for name in volume_names)
        inspect_result = await ssh_client.run(
            f"/usr/bin/docker volume inspect {names} "
            "--format '{{index .Status \"size-max\"}}|{{index .Options \"size\"}}'"
        )
        if getattr(inspect_result, "exit_status", 0) != 0:
            raise Exception(
                f"docker volume inspect failed: {getattr(inspect_result, 'stderr', '')}"
            )

        total_bytes = 0
        for volume_name, line in zip(volume_names, (inspect_result.stdout or "").splitlines()):
            line = line.strip()
            if not line:
                continue
            size_max_raw, _, size_option_raw = line.partition("|")
            size_bytes = _parse_volume_size_to_bytes(size_max_raw)
            if size_bytes is None:
                size_bytes = _parse_volume_size_to_bytes(size_option_raw)
            if size_bytes is not None:
                total_bytes += size_bytes
            else:
                # Undercounting existing volumes inflates the reconstructed pool.
                logger.info(
                    _m(
                        "vloopback_fresh_sizing_unparsable_volume_size",
                        extra=get_extra_info({
                            "volume_name": volume_name,
                            "inspect_line": line,
                        }),
                    )
                )
        return total_bytes

    async def resolve_volume_sizing(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        payload: ContainerCreateRequest,
        log_tag: str,
        log_extra: dict,
    ) -> VolumeSizingResult:
        """Resolve effective volume/storage limits for a new pod volume (DAH-2183).

        Legacy contract (payload.disk_share is None): backend-sent
        volume_limit_gb/storage_limit_gb are exact sizes and are returned
        untouched, without any SSH calls.

        Fresh contract (payload.disk_share set): measure fresh on-host disk
        state over the existing ssh_client and compute the pod's slice;
        backend-sent limits act only as upper-bound caps. Measurement
        failures fall back to legacy passthrough (never break the rent);
        a VolumeMinSizeError (effective volume below payload.min_volume_gb)
        propagates to the caller.

        Storage-opt gating: ``payload.storage_limit_gb is None`` is the
        backend's signal that the executor's filesystem cannot enforce
        ``--storage-opt size=`` (e.g. overlay2 without xfs+pquota /
        ext4+prjquota — see ``calc_volume_storage_limit`` in the backend
        which returns ``(None, None)`` in that case). Skip the fresh
        re-derivation entirely and pass the payload's limits through
        untouched; otherwise the fresh path would compute a non-None
        ``storage_limit_gb`` from ``disk_share`` and dockerd would reject
        the run with "supported only for overlay over xfs with 'pquota'".
        """
        if payload.storage_limit_gb is None:
            return VolumeSizingResult(
                volume_limit_gb=payload.volume_limit_gb,
                storage_limit_gb=payload.storage_limit_gb,
                path="storage_opt_unsupported",
            )

        if payload.disk_share is None:
            return VolumeSizingResult(
                volume_limit_gb=payload.volume_limit_gb,
                storage_limit_gb=payload.storage_limit_gb,
                path="legacy",
            )

        try:
            docker_root_dir = await self.get_docker_root_dir(ssh_client)
            df_avail_bytes = await self._get_fs_available_bytes(ssh_client, docker_root_dir)
            existing_volumes_bytes = await self._get_existing_vloopback_bytes(ssh_client)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                _m(
                    "vloopback_fresh_sizing_fallback",
                    extra=get_extra_info({**log_extra, "error": str(exc)}),
                ),
                exc_info=True,
            )
            return VolumeSizingResult(
                volume_limit_gb=payload.volume_limit_gb,
                storage_limit_gb=payload.storage_limit_gb,
                path="fresh_fallback",
            )

        # Clamp negative intermediates: a nearly-full disk must degrade to the
        # 1 GB floor (or a min-size rejection), never to a negative candidate
        # winning min().
        pool_bytes = max(
            df_avail_bytes
            + existing_volumes_bytes
            - _FRESH_SIZING_OVERHEAD_GB * _FRESH_SIZING_GB_BYTES,
            0,
        )
        slice_candidates = [("pool", payload.disk_share * pool_bytes)]
        if payload.volume_limit_gb is not None:
            slice_candidates.append(
                ("request_cap", payload.volume_limit_gb * _FRESH_SIZING_GB_BYTES * 1.5)
            )
        slice_candidates.append(
            (
                "df_guard",
                max(
                    (df_avail_bytes - _FRESH_SIZING_HEADROOM_GB * _FRESH_SIZING_GB_BYTES) * 1.5,
                    0,
                ),
            )
        )
        capped_by, slice_bytes = min(slice_candidates, key=lambda item: item[1])

        # -o size= requires a positive integer; floor to at least 1 GB before
        # the min-size check so the check runs against the final value.
        volume_limit_gb = max(math.floor(slice_bytes * 2 / 3 / _FRESH_SIZING_GB_BYTES), 1)
        storage_limit_gb = max(math.floor(slice_bytes / 3 / _FRESH_SIZING_GB_BYTES), 1)

        if payload.min_volume_gb is not None and volume_limit_gb < payload.min_volume_gb:
            logger.error(
                _m(
                    "vloopback_fresh_sizing_below_min",
                    extra=get_extra_info({
                        **log_extra,
                        "requested_volume_limit_gb": payload.volume_limit_gb,
                        "min_volume_gb": payload.min_volume_gb,
                        "computed_volume_limit_gb": volume_limit_gb,
                        "df_avail_bytes": df_avail_bytes,
                        "existing_volumes_bytes": existing_volumes_bytes,
                        "disk_share": payload.disk_share,
                        "docker_root_dir": docker_root_dir,
                    }),
                )
            )
            raise VolumeMinSizeError(
                f"Fresh vloopback sizing produced {volume_limit_gb}GB volume, "
                f"below required minimum {payload.min_volume_gb}GB"
            )

        logger.info(
            _m(
                "vloopback_fresh_sizing_calc",
                extra=get_extra_info({
                    **log_extra,
                    "disk_share": payload.disk_share,
                    "df_avail_bytes": df_avail_bytes,
                    "existing_volumes_bytes": existing_volumes_bytes,
                    "overhead_gb": _FRESH_SIZING_OVERHEAD_GB,
                    "headroom_gb": _FRESH_SIZING_HEADROOM_GB,
                    "pool_bytes": pool_bytes,
                    "slice_bytes": int(slice_bytes),
                    "slice_candidates": {name: int(value) for name, value in slice_candidates},
                    "capped_by": capped_by,
                    "effective_volume_limit_gb": volume_limit_gb,
                    "effective_storage_limit_gb": storage_limit_gb,
                    "requested_volume_limit_gb": payload.volume_limit_gb,
                    "requested_storage_limit_gb": payload.storage_limit_gb,
                    "docker_root_dir": docker_root_dir,
                }),
            )
        )

        if payload.volume_limit_gb is not None and volume_limit_gb < payload.volume_limit_gb:
            shrink_key = (
                "vloopback_fresh_sizing_severe_shrink"
                if volume_limit_gb < payload.volume_limit_gb / 2
                else "vloopback_fresh_sizing_shrink"
            )
            logger.warning(
                _m(
                    shrink_key,
                    extra=get_extra_info({
                        **log_extra,
                        "requested_volume_limit_gb": payload.volume_limit_gb,
                        "effective_volume_limit_gb": volume_limit_gb,
                        "capped_by": capped_by,
                    }),
                )
            )

        return VolumeSizingResult(
            volume_limit_gb=volume_limit_gb,
            storage_limit_gb=storage_limit_gb,
            path="fresh",
            capped_by=capped_by,
            df_avail_bytes=df_avail_bytes,
            existing_volumes_bytes=existing_volumes_bytes,
        )

    # ------------------------------------------------------------------
    # DAH-2211 — custom-dockerfile pod build helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _custom_build_image_tag(pod_id: str) -> str:
        return f"lium-build-{pod_id}"

    @staticmethod
    def _custom_build_scratch_dir(pod_id: str) -> str:
        return f"/tmp/lium-build-{pod_id}"

    @staticmethod
    def _dind_container_name(pod_id: str) -> str:
        return f"lium-dind-build-{pod_id}"

    # Build context written inside the throwaway DinD container.
    _DIND_BUILD_CONTEXT = "/build"

    async def _write_dockerfile_into_dind(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        dind_name: str,
        dockerfile_content: str,
    ) -> None:
        """Write the Dockerfile *inside* the DinD container without argv exposure.

        Streams user content via stdin into `cat` (through `docker exec -i`) so
        arbitrary content (`EOF` markers, backticks, `$(...)`) cannot escape into
        the shell. Only the controlled container name reaches argv.
        """
        ctx = self._DIND_BUILD_CONTEXT
        inner = f"mkdir -p {ctx} && cat > {ctx}/Dockerfile"
        command = (
            f"/usr/bin/docker exec -i {shlex.quote(dind_name)} "
            f"sh -c {shlex.quote(inner)}"
        )
        result = await ssh_client.run(command, input=dockerfile_content, check=False)
        if result.exit_status != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                f"Failed to write Dockerfile into {dind_name!r}: "
                f"exit={result.exit_status} stderr={stderr!r}"
            )

    @staticmethod
    def _parse_egress_block_cidrs(raw: str) -> list[str]:
        """Validate the configured egress-block CIDRs into shell-safe strings.

        Invalid entries are dropped with a warning. 169.254.0.0/16 (cloud
        metadata) is always included so a misconfiguration can never silently
        re-expose the credential-theft vector.
        """
        cidrs: list[str] = []
        for part in (raw or "").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                cidrs.append(str(ipaddress.ip_network(part, strict=False)))
            except ValueError:
                logger.warning(
                    _m(
                        "Ignoring invalid CUSTOM_DOCKERFILE_EGRESS_BLOCK_CIDRS entry",
                        extra=get_extra_info({"entry": part}),
                    )
                )
        metadata = "169.254.0.0/16"
        if metadata not in cidrs:
            cidrs.insert(0, metadata)
        return cidrs

    @staticmethod
    def _egress_filter_script(dind_ip: str, cidrs: list[str], apply: bool) -> str:
        """Backend-agnostic iptables script for the host DOCKER-USER chain.

        Runs inside a `--network=host --cap-add=NET_ADMIN` helper container so it
        edits the *host* netns regardless of where the validator SSH lands. The
        fleet is mixed nft/legacy, so we pick whichever backend actually owns the
        DOCKER-USER chain. Rules are scoped to the DinD container's source IP so
        DinD-internal docker networking (172.x bridges) is never affected.

        `dind_ip` and `cidrs` are pre-validated via `ipaddress`, so they are
        shell-safe to interpolate.
        """
        lines = [
            "IPT=iptables-nft",
            "$IPT -L DOCKER-USER -n >/dev/null 2>&1 || IPT=iptables-legacy",
        ]
        if apply:
            # No DOCKER-USER chain => egress filtering cannot be guaranteed. Fail
            # loudly so the caller aborts rather than running the build open.
            lines.append(
                '$IPT -L DOCKER-USER -n >/dev/null 2>&1 || '
                '{ echo "DOCKER-USER chain not found" >&2; exit 3; }'
            )
            for c in cidrs:
                lines.append(
                    f"$IPT -C DOCKER-USER -s {dind_ip} -d {c} -j DROP 2>/dev/null || "
                    f"$IPT -I DOCKER-USER -s {dind_ip} -d {c} -j DROP"
                )
        else:
            for c in cidrs:
                lines.append(
                    f"$IPT -D DOCKER-USER -s {dind_ip} -d {c} -j DROP 2>/dev/null || true"
                )
        return "; ".join(lines)

    async def _custom_build_image(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        payload: ContainerCreateRequest,
        log_tag: str,
        default_extra: dict,
    ) -> tuple[bool, str | None]:
        """Build a custom image from `payload.dockerfile_content` on the executor.

        Returns (success, failure_step). On success returns (True, None); the
        built image tag is `_custom_build_image_tag(pod_id)`. On failure returns
        (False, failure_step) — caller routes through the same CCF
        `UnknownError` path used by today's pull-failure.
        """
        from core.config import settings

        pod_id = payload.pod_id
        image_tag = self._custom_build_image_tag(pod_id)
        dind_name = self._dind_container_name(pod_id)
        ctx = self._DIND_BUILD_CONTEXT
        timeout_s = int(settings.CUSTOM_DOCKERFILE_BUILD_TIMEOUT_SECONDS)
        ready_timeout_s = int(settings.CUSTOM_DOCKERFILE_DIND_READY_TIMEOUT_SECONDS)
        dind_image = settings.CUSTOM_DOCKERFILE_DIND_IMAGE
        cidrs = self._parse_egress_block_cidrs(settings.CUSTOM_DOCKERFILE_EGRESS_BLOCK_CIDRS)

        # Defense-in-depth size cap. Route is authoritative; this is a wire-trust guard.
        max_bytes = int(settings.CUSTOM_DOCKERFILE_MAX_BYTES)
        content = payload.dockerfile_content or ""
        if len(content.encode("utf-8")) > max_bytes:
            await self.stream_log(
                f"Dockerfile exceeds {max_bytes} byte cap", "error", log_tag
            )
            logger.error(
                _m(
                    "Custom dockerfile exceeds size cap",
                    extra=get_extra_info({**default_extra, "size_bytes": len(content)}),
                )
            )
            return False, "build_input_oversize"

        # 1. Preflight: sysbox-runc MUST be available. Never fall back to runc —
        #    a silent fallback would run an internet-enabled build on the host
        #    daemon with no user-namespace containment, defeating the design.
        try:
            info = await ssh_client.run(
                "/usr/bin/docker info --format '{{json .Runtimes}}'", check=False
            )
            if info.exit_status != 0 or "sysbox-runc" not in (info.stdout or ""):
                await self.stream_log(
                    "sysbox-runc runtime unavailable on executor", "error", log_tag
                )
                logger.error(
                    _m(
                        "Custom build aborted: sysbox-runc unavailable",
                        extra=get_extra_info(
                            {**default_extra, "runtimes": (info.stdout or "").strip()}
                        ),
                    )
                )
                return False, "build_sysbox_unavailable"
        except Exception as exc:
            logger.error(
                _m(
                    "Custom build sysbox preflight failed",
                    extra=get_extra_info({**default_extra, "error": str(exc)}),
                ),
                exc_info=True,
            )
            return False, "build_sysbox_unavailable"

        dind_ip: str | None = None
        egress_applied = False
        try:
            # 2. Launch the throwaway DinD build container under sysbox-runc.
            await self.stream_log(
                f"Starting isolated build container {dind_name}", "success", log_tag
            )
            run_dind = (
                f"/usr/bin/docker run -d --runtime=sysbox-runc "
                f"--name {shlex.quote(dind_name)} "
                f"--cpus={shlex.quote(str(settings.CUSTOM_DOCKERFILE_DIND_CPUS))} "
                f"--memory={shlex.quote(str(settings.CUSTOM_DOCKERFILE_DIND_MEMORY))} "
                f"{shlex.quote(dind_image)}"
            )
            start_res = await ssh_client.run(run_dind, check=False)
            if start_res.exit_status != 0:
                logger.error(
                    _m(
                        "Custom build DinD start failed",
                        extra=get_extra_info(
                            {**default_extra, "stderr": (start_res.stderr or "").strip()}
                        ),
                    )
                )
                return False, "build_dind_start"

            # 3. Wait for the inner dockerd to accept connections.
            ready = False
            for _ in range(max(1, ready_timeout_s)):
                probe = await ssh_client.run(
                    f"/usr/bin/docker exec {shlex.quote(dind_name)} docker info",
                    check=False,
                )
                if probe.exit_status == 0:
                    ready = True
                    break
                await asyncio.sleep(1)
            if not ready:
                logger.error(
                    _m(
                        "Custom build DinD dockerd not ready",
                        extra=get_extra_info(
                            {**default_extra, "ready_timeout_s": ready_timeout_s}
                        ),
                    )
                )
                return False, "build_dind_unready"

            # 4. Resolve the DinD container IP and firewall its egress host-side
            #    (block cloud metadata + RFC1918; full public internet stays open).
            ip_res = await ssh_client.run(
                "/usr/bin/docker inspect -f "
                "'{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "
                f"{shlex.quote(dind_name)}",
                check=False,
            )
            raw_ip = (ip_res.stdout or "").strip()
            try:
                dind_ip = str(ipaddress.ip_address(raw_ip))
            except ValueError:
                logger.error(
                    _m(
                        "Custom build could not resolve DinD container IP",
                        extra=get_extra_info({**default_extra, "raw_ip": raw_ip}),
                    )
                )
                return False, "build_egress_setup"

            apply_script = self._egress_filter_script(dind_ip, cidrs, apply=True)
            egress_cmd = (
                f"/usr/bin/docker run --rm --network=host --cap-add=NET_ADMIN "
                f"--entrypoint /bin/sh {shlex.quote(dind_image)} "
                f"-c {shlex.quote(apply_script)}"
            )
            ok, err = await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=egress_cmd,
                log_tag=log_tag,
                log_text="Applying build egress firewall",
                log_extra={**default_extra, "dind_ip": dind_ip},
                raise_exception=False,
            )
            if not ok:
                # Cannot guarantee egress filtering -> never run the build open.
                logger.error(
                    _m(
                        "Custom build egress firewall failed",
                        extra=get_extra_info({**default_extra, "error": err}),
                    )
                )
                return False, "build_egress_setup"
            egress_applied = True

            # 5. Write the Dockerfile into the DinD container (stdin, not argv).
            await self.stream_log(
                f"Preparing build context in {dind_name}", "success", log_tag
            )
            try:
                await self._write_dockerfile_into_dind(ssh_client, dind_name, content)
            except Exception as exc:
                logger.error(
                    _m(
                        "Custom build Dockerfile write failed",
                        extra=get_extra_info({**default_extra, "error": str(exc)}),
                    ),
                    exc_info=True,
                )
                return False, "build_setup"

            # 6. Build INSIDE the DinD container WITH network enabled (no
            #    --network=none). BuildKit streams progress to stderr and
            #    `execute_and_stream_logs` treats any stderr as failure, so we
            #    redirect build output to stdout (streamed as success logs) and
            #    emit to stderr ONLY on non-zero exit — the streamer's signal.
            inner_build = (
                f"docker build --progress=plain --pull "
                f"-t {shlex.quote(image_tag)} {shlex.quote(ctx)} 2>&1; "
                f"rc=$?; [ $rc -ne 0 ] && echo BUILD_FAILED_RC=$rc >&2; exit $rc"
            )
            build_cmd = (
                f"/usr/bin/docker exec {shlex.quote(dind_name)} "
                f"sh -c {shlex.quote(inner_build)}"
            )
            try:
                ok, err = await self.execute_and_stream_logs(
                    ssh_client=ssh_client,
                    command=build_cmd,
                    log_tag=log_tag,
                    log_text=f"Building custom image {image_tag}",
                    log_extra={**default_extra, "build_image_tag": image_tag},
                    timeout=timeout_s,
                    raise_exception=False,
                )
            except Exception as exc:
                logger.error(
                    _m(
                        "Custom build invocation failed",
                        extra=get_extra_info({**default_extra, "error": str(exc)}),
                    ),
                    exc_info=True,
                )
                return False, "docker_build"
            if not ok:
                if "process timed out" in (err or "").lower():
                    return False, "build_timeout"
                return False, "docker_build"

            # 7. Export the image from DinD and load it onto the host daemon so
            #    the rental `docker run` (host-side) can use it.
            await self.stream_log(
                f"Loading built image {image_tag} onto host", "success", log_tag
            )
            export_cmd = (
                f"/usr/bin/docker exec {shlex.quote(dind_name)} "
                f"docker save {shlex.quote(image_tag)} | /usr/bin/docker load"
            )
            ok, err = await self.execute_and_stream_logs(
                ssh_client=ssh_client,
                command=export_cmd,
                log_tag=log_tag,
                log_text=f"Exporting custom image {image_tag}",
                log_extra={**default_extra, "build_image_tag": image_tag},
                timeout=timeout_s,
                raise_exception=False,
            )
            if not ok:
                if "process timed out" in (err or "").lower():
                    return False, "build_timeout"
                return False, "build_export"

            return True, None
        finally:
            # Always tear down the throwaway container + its egress rules. The
            # host-loaded image is removed later by _cleanup_custom_build_artifacts.
            await self._teardown_dind_build(
                ssh_client=ssh_client,
                dind_name=dind_name,
                dind_image=dind_image,
                dind_ip=dind_ip if egress_applied else None,
                cidrs=cidrs,
                default_extra=default_extra,
            )

    async def _teardown_dind_build(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        dind_name: str,
        dind_image: str,
        dind_ip: str | None,
        cidrs: list[str],
        default_extra: dict,
    ) -> None:
        """Best-effort teardown of the throwaway DinD container + its egress rules.

        Always called from `_custom_build_image`'s finally. Failures are logged,
        never raised — the rental flow must not break on cleanup.
        """
        if dind_ip:
            try:
                remove_script = self._egress_filter_script(dind_ip, cidrs, apply=False)
                await ssh_client.run(
                    f"/usr/bin/docker run --rm --network=host --cap-add=NET_ADMIN "
                    f"--entrypoint /bin/sh {shlex.quote(dind_image)} "
                    f"-c {shlex.quote(remove_script)}",
                    check=False,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    _m(
                        "Custom build egress rule teardown failed (non-fatal)",
                        extra=get_extra_info(
                            {**default_extra, "error": str(exc), "dind_ip": dind_ip}
                        ),
                    )
                )
        try:
            await ssh_client.run(
                f"/usr/bin/docker rm -fv {shlex.quote(dind_name)} 2>/dev/null || true",
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                _m(
                    "Custom build DinD container teardown failed (non-fatal)",
                    extra=get_extra_info(
                        {**default_extra, "error": str(exc), "dind_name": dind_name}
                    ),
                )
            )

    async def _cleanup_custom_build_artifacts(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        pod_id: str,
        default_extra: dict,
    ) -> None:
        """Always-on inline cleanup (Phase 3.4(i)).

        Removes the per-pod built image and scratch directory. Best-effort —
        any failure is logged but never raised. Safe to call for non-custom
        pods (no-op when the artifacts do not exist).
        """
        image_tag = self._custom_build_image_tag(pod_id)
        scratch_dir = self._custom_build_scratch_dir(pod_id)
        try:
            await ssh_client.run(
                f"/usr/bin/docker image rm {shlex.quote(image_tag)} 2>/dev/null || true",
                check=False,
            )
            await ssh_client.run(
                f"/usr/bin/rm -rf {shlex.quote(scratch_dir)}",
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                _m(
                    "Custom build artifact cleanup failed (non-fatal)",
                    extra=get_extra_info({**default_extra, "error": str(exc), "image_tag": image_tag}),
                )
            )

    async def _abort_if_cancelled_by_delete(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        payload: ContainerCreateRequest,
        default_extra: dict,
    ) -> None:
        """Tear this create down when its own delete has already reported the pod gone (DAH-2728)."""
        if not inflight_creates.is_cancelled(payload.pod_id):
            return

        container_name = self.get_container_name(payload)
        logger.warning(
            _m(
                "Create cancelled by an in-flight delete",
                extra=get_extra_info({**default_extra, "container_name": container_name}),
            )
        )
        if payload.workload_kind == WorkloadKind.FILLER:
            # Remove the container BEFORE lifting the cap. The enclosing handler removes it again
            # (both are idempotent), but restoring full power while a filler is still running is
            # the DAH-2356 hard-ban failure — an uncapped PEARL miner.
            await self.cleanup_failed_container_creation(
                ssh_client=ssh_client,
                default_extra=default_extra,
                container_name=container_name,
            )
            # DAH-2356: the cap is applied before `docker run`, and the delete that cancelled us
            # restored nothing — at that point this pod had no record yet.
            await restore_filler_pod_gpu_power_limits(
                ssh_client, self.redis_service, payload.pod_id, log_extra=default_extra
            )
        raise _CreateCancelledByDelete(
            f"delete for pod {payload.pod_id} arrived while {container_name} was being created"
        )

    async def create_container(
        self,
        payload: ContainerCreateRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        warnings = []
        local_volume = payload.local_volume
        external_volume_info = payload.external_volume_info

        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "pod_id": payload.pod_id,
            "workload_kind": payload.workload_kind.value,
            "executor_uuid": payload.executor_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
            "docker_image": payload.docker_image,
            "local_volume": local_volume,
            "edit_pod": True if local_volume else False,
            "external_volume": external_volume_info.name if external_volume_info else None,
            "enable_jupyter": payload.enable_jupyter,
        }

        # Deploy container profiler
        profilers = []
        # DAH-2458: seed with the backend's own pre-dispatch spans (measured before this request
        # reached the subnet, e.g. the filler-preemption wait) so the persisted profile is one
        # ordered backend -> subnet -> backend timeline. Unknown step names are dropped, never fatal.
        for wire_step in payload.pre_dispatch_profilers or []:
            seeded = ProfilerStep.from_wire(wire_step)
            if seeded is not None:
                profilers.append(seeded)
        if payload.timestamp:
            profilers.append(ProfilerStep(name=ProfilerStepName.REQUESTED_FROM_BACKEND, timestamp=payload.timestamp))
            prev_timestamp = payload.timestamp
        else:
            prev_timestamp = now_ms()
        profilers.append(ProfilerStep.since(ProfilerStepName.STARTED_IN_SUBNET, prev_timestamp))
        prev_timestamp = now_ms()

        logger.info(
            _m(
                "Edit Docker Container" if local_volume else "Create Docker Container",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        log_tag = "container_creation"
        current_step = "start"
        # DAH-2703: the container reached the host and then disappeared from it — the host-reaper
        # signature. `container_created` keeps it honest: before `docker run` succeeds there is
        # nothing to remove, so a missing container there is an ordinary create failure.
        container_created = False
        container_vanished = False
        login_error: str | None = None
        volume_encryption_status = VolumeEncryptionStatus.DISABLED

        # DAH-2211: a custom-build payload carries `dockerfile_content` (may be
        # `""`/whitespace if a broken caller bypassed the route XOR). Reject
        # empty/whitespace-only content BEFORE any SSH command runs. We treat
        # `is_custom_build` as truthy only when the field is a non-empty string
        # after strip().
        dockerfile_content_raw = payload.dockerfile_content
        is_custom_build = (
            dockerfile_content_raw is not None
            and dockerfile_content_raw.strip() != ""
        )
        if dockerfile_content_raw is not None and not is_custom_build:
            current_step = "build_input_empty"
            log_text = _m(
                "Custom dockerfile pod requested with empty content",
                extra=get_extra_info(default_extra),
            )
            logger.error(log_text)
            await self.redis_service.remove_pending_pod(
                payload.miner_hotkey, payload.executor_id, payload.pod_id
            )
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
                failure_step=current_step,
            )

        try:
            current_step = "prepare_request"
            custom_options = CustomOptions.sanitize(payload.custom_options)
            # generate port maps
            current_step = "port_mapping"
            port_maps, jupyter_port_map = await self.generate_portMappings(
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

            # Add profiler for port mappings generation
            profilers.append(ProfilerStep.since(ProfilerStepName.PORT_MAPPINGS_GENERATED, prev_timestamp))
            prev_timestamp = now_ms()

            if not port_maps:
                log_text = _m(
                    "No port mappings found",
                    extra=get_extra_info(default_extra),
                )
                logger.error(log_text)

                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    msg=str(log_text),
                    error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                    error_code=FailedContainerErrorCodes.NoPortMappings,
                    failure_step=current_step,
                )

            default_extra = {
                **default_extra,
                "jupyter_port_map": jupyter_port_map,
            }

            if payload.enable_jupyter and not jupyter_port_map:
                log_text = _m(
                    "No Jupyter port mapping found",
                    extra=get_extra_info(default_extra),
                )
                logger.error(log_text)

                # Port release now handled by backend

                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    msg=str(log_text),
                    error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                    error_code=FailedContainerErrorCodes.NoJupyterPortMapping,
                    failure_step=current_step,
                )

            current_step = "validate_request"
            if not payload.user_public_keys:
                log_text = _m(
                    "No public keys",
                    extra=get_extra_info(default_extra),
                )
                logger.error(log_text)

                # Port release now handled by backend

                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    msg=str(log_text),
                    error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                    error_code=FailedContainerErrorCodes.NoSshKeys,
                    failure_step=current_step,
                )

            # add executor in pending status dict
            current_step = "pending_pod"
            await self.redis_service.add_pending_pod(payload.miner_hotkey, payload.executor_id, payload.pod_id)

            current_step = "ssh_key_import"
            private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
            pkey = asyncssh.import_private_key(private_key)

            known_hosts_policy: asyncssh.SSHKnownHosts | None = None
            try:
                current_step = "attestation"
                known_hosts_policy = await self._prepare_known_hosts_policy(
                    executor_info,
                    payload.miner_hotkey,
                    default_extra,
                )
            except AttestationError as exc:
                log_text = _m(
                    "Attestation failed",
                    extra=get_extra_info({**default_extra, "error": str(exc)}),
                )
                logger.error(log_text)
                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    msg=str(log_text),
                    error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                    error_code=FailedContainerErrorCodes.UnknownError,
                    failure_step=current_step,
                )

            current_step = "docker_sdk_ssh_host_key"
            # Keep this immediately before the guard; the broad except uses it as failure_step.
            require_rental_docker_ssh_host_key(executor_info)

            current_step = "ssh_connect"
            # DAH-2272: connect_with_phase_timing logs the TCP-vs-SSH-login
            # split for this connect (host/network vs. remote sshd) without
            # changing how the connection is established.
            async with (
                connect_with_phase_timing(
                    log_extra=default_extra,
                    host=executor_info.address,
                    port=executor_info.ssh_port,
                    username=executor_info.ssh_username,
                    client_keys=[pkey],
                    known_hosts=known_hosts_policy,
                    keepalive_interval=_CREATE_CONTAINER_SSH_KEEPALIVE_INTERVAL_SEC,
                    keepalive_count_max=_CREATE_CONTAINER_SSH_KEEPALIVE_COUNT_MAX,
                ) as ssh_client,
                self.rental_docker_client_factory.connect(
                    executor_info=executor_info,
                    private_key=private_key,
                ) as docker_client,
                # DAH-2740: undoes a failed edit while this SSH session is still open
                _EditSwap(ssh_client, self.get_container_name(payload), default_extra) as edit_swap,
            ):
                # Add profiler for ssh connection
                profilers.append(ProfilerStep.since(ProfilerStepName.SSH_CONNECTION_ESTABLISHED, prev_timestamp))
                prev_timestamp = now_ms()

                # DAH-2728: cheapest place to notice the delete — the image pull/build below is
                # where a cancelled create spends its minutes, and nothing is on the host yet.
                await self._abort_if_cancelled_by_delete(ssh_client, payload, default_extra)

                # set real-time logging
                self.log_task = asyncio.create_task(
                    self.handle_stream_logs(
                        miner_hotkey=payload.miner_hotkey,
                        executor_id=payload.executor_id,
                        pod_id=payload.pod_id,
                    )
                )
                # No logout counterpart below: the SDK login is a POST /auth to the executor's
                # Docker daemon and the credential stays in this validator's client, so nothing is
                # written to the executor's ~/.docker/config.json for a `docker logout` to clear.
                #
                # The default cache-template images are public Docker Hub refs, so a
                # registry login buys nothing for them — the pull needs no auth and is
                # itself almost always skipped (the image is pre-cached on the executor).
                # The backend still attaches credentials whenever the renter has any
                # saved, which costs a round-trip to auth.docker.io on every such deploy
                # (~1.1s median in prod). Skip the login on that path.
                #
                # `ships_sshd` IS the "renter selected a default image" signal — the
                # backend sets it from the same check that resolves the recommended image
                # for this executor's GPU+driver (executor.py: `ships_sshd=is_cached`,
                # with `is_cached=False` forced for custom builds, so they keep this login
                # path as-is; the DinD `docker build` never sees this SDK login, and
                # passing credentials into it is a separate task). Don't re-derive it here:
                # a second, validator-side notion of "is this a default image?" could
                # disagree with the backend's and skip a login that was actually needed.
                has_credentials = bool(payload.docker_username and payload.docker_password)
                skip_login = not has_credentials or bool(payload.ships_sshd)
                if skip_login:
                    if has_credentials:
                        logger.info(
                            _m(
                                "Skipping docker login for default cache-template image",
                                extra=get_extra_info(default_extra),
                            )
                        )
                else:
                    current_step = "docker_login"
                    try:
                        await run_logged_rental_docker_sdk_operation(
                            operation="login",
                            log_extra=default_extra,
                            call=lambda: docker_client.login(
                                username=payload.docker_username,
                                password=payload.docker_password,
                                image=payload.docker_image,
                            ),
                            username_present=True,
                            username_len=len(payload.docker_username),
                        )
                    except Exception as exc:
                        login_error = str(exc)
                        logger.warning(
                            _m(
                                "Docker registry login failed",
                                extra=get_extra_info({**default_extra, "error": str(exc)}),
                            ),
                            exc_info=True,
                        )

                # Add profiler for docker login
                profilers.append(
                    ProfilerStep.since(
                        ProfilerStepName.DOCKER_LOGIN, prev_timestamp, skipped=skip_login
                    )
                )
                prev_timestamp = now_ms()

                # When `dockerfile_content` is set, build the image on the executor
                # instead of pulling. Otherwise pull the requested image through the
                # rental Docker SDK client.
                if is_custom_build:
                    current_step = "docker_build"
                    build_ok, build_failure_step = await self._custom_build_image(
                        ssh_client=ssh_client,
                        payload=payload,
                        log_tag=log_tag,
                        default_extra=default_extra,
                    )
                    if not build_ok:
                        # Best-effort scratch dir cleanup; image is most likely
                        # absent on failure but `image rm` is idempotent.
                        await self._cleanup_custom_build_artifacts(
                            ssh_client=ssh_client,
                            pod_id=payload.pod_id,
                            default_extra=default_extra,
                        )
                        # Raise so the existing except-Exception block in
                        # create_container emits the CCF FailedContainerRequest
                        # via the same path as today's pull failure.
                        current_step = build_failure_step or "docker_build"
                        raise Exception(
                            f"Custom dockerfile build failed (failure_step={current_step})"
                        )
                    # Override docker_image so the downstream `docker run` uses
                    # the locally built tag for this branch only.
                    effective_image = self._custom_build_image_tag(payload.pod_id)
                    default_extra = {**default_extra, "docker_image": effective_image}
                    payload.docker_image = effective_image
                    profilers.append(ProfilerStep.since(ProfilerStepName.CUSTOM_DOCKER_BUILD, prev_timestamp))
                    prev_timestamp = now_ms()
                else:
                    current_step = "docker_image_inspect"
                    image_present = False
                    try:
                        image_present = await run_logged_rental_docker_sdk_operation(
                            operation="inspect_image",
                            log_extra=default_extra,
                            call=lambda: docker_client.image_exists(
                                image=payload.docker_image
                            ),
                            image=payload.docker_image,
                        )
                    except Exception as exc:
                        logger.warning(
                            _m(
                                "Docker SDK image inspect probe failed; falling back to pull",
                                extra=get_extra_info({**default_extra, "error": str(exc)}),
                            )
                        )

                    current_step = "docker_pull"
                    if image_present:
                        logger.info(
                            _m(
                                "Skipping docker pull; image already present locally",
                                extra=get_extra_info(default_extra),
                            )
                        )
                        profilers.append(
                            ProfilerStep.since(
                                ProfilerStepName.DOCKER_PULL,
                                prev_timestamp,
                                skipped=True,
                            )
                        )
                        prev_timestamp = now_ms()
                    else:
                        await self.stream_log(
                            f"Pulling docker image {payload.docker_image}",
                            "success",
                            log_tag,
                        )
                        try:
                            await run_logged_rental_docker_sdk_operation(
                                operation="pull",
                                log_extra=default_extra,
                                call=lambda: docker_client.pull(image=payload.docker_image),
                                image=payload.docker_image,
                            )
                        except Exception as exc:
                            if login_error is None:
                                raise
                            # docker-py pulls anonymously after a failed login, so the
                            # pull failure is most likely the login's fault — attribute it
                            current_step = "docker_login"
                            raise RentalDockerOperationError(
                                f"{exc} (earlier login failure: {login_error})"
                            ) from exc

                        # Add profiler for docker pull
                        profilers.append(ProfilerStep.since(ProfilerStepName.DOCKER_PULL, prev_timestamp))
                        prev_timestamp = now_ms()

                # Get the container path from the first volume
                local_volume_path = custom_options.volumes[0].split(':')[-1] if custom_options.volumes else '/root'
                # DAH-2265: default-image / cached-template rentals set `ships_sshd`.
                # Those images run their own start.sh, which starts sshd unconditionally
                # and launches Jupyter itself. Rather than have the validator bootstrap
                # sshd and run run_jupyter post-create (see below), we let the image do
                # both and forward the Jupyter token as a container env var.
                #
                # The image's contract is JUPYTER_PASSWORD and nothing else: start.sh
                # gates on `if [[ $JUPYTER_PASSWORD ]]` and hands it straight to
                # `jupyter lab --ServerApp.token=$JUPYTER_PASSWORD`, with the in-container
                # port hardcoded to 8888. It reads neither ENABLE_JUPYTER nor JUPYTER_PORT
                # (no image in computenet-docker-images does), so forwarding only those
                # left start.sh's gate closed: Jupyter never started, yet the port was
                # still mapped, so the pod came up Running with a Jupyter URL that reset
                # the connection. JUPYTER_PASSWORD is also the name the validator's own
                # run_jupyter.sh takes (`--password=`), so both paths now agree.
                #
                # This MUST be set at `docker run` time — start.sh (the image entrypoint)
                # reads the environment once at container startup, so a post-create
                # /etc/environment write would be too late. It is injected into
                # custom_options.environment here so _build_rental_container_run_spec
                # carries it into the run spec (create-time env), not the post-create
                # add_environment_variables step.
                #
                # The image only runs its own start.sh — and therefore only starts
                # sshd and Jupyter itself — when we let its default CMD/ENTRYPOINT
                # run. A renter-supplied `startup_commands` is passed as the container
                # command and REPLACES the image CMD (`/start.sh`); a renter-supplied
                # `entrypoint` replaces `/pytorch-entrypoint.sh`, which is the script
                # that execs that CMD. In either case start.sh never runs, so the
                # image starts neither sshd nor Jupyter, and the validator must keep
                # managing both even though ships_sshd is set. commit 2345af85 reverted
                # this exact skip for precisely this reason (a custom startup_commands
                # replaces the image CMD); ships_sshd alone is not sufficient. Only take
                # the image-managed path when neither override is present.
                image_manages_services = bool(
                    payload.ships_sshd
                    and not (custom_options.startup_commands and custom_options.startup_commands.strip())
                    and not (custom_options.entrypoint and custom_options.entrypoint.strip())
                )
                # Guard on the mapped docker port: start.sh always binds 8888, so if the
                # mapping ever moved off that port the image's Jupyter would be
                # unreachable. Fall back to the validator's run_jupyter in that case
                # rather than silently serving nothing.
                image_managed_jupyter = bool(
                    image_manages_services
                    and payload.enable_jupyter
                    and jupyter_port_map
                    and jupyter_port_map[0] == IMAGE_JUPYTER_DOCKER_PORT
                )
                image_jupyter_token = (
                    secrets.token_hex(16) if image_managed_jupyter else None
                )
                if image_managed_jupyter:
                    custom_options.environment = {
                        **(custom_options.environment or {}),
                        "JUPYTER_PASSWORD": image_jupyter_token,
                    }

                container_name = self.get_container_name(payload)
                created_local_volume = False
                protected_volume_names = set(payload.active_volume_names or [])
                if local_volume:
                    protected_volume_names.add(local_volume)

                protected_container_names = list(payload.active_container_names or [])
                if local_volume:
                    # DAH-2740: an edit keeps its current container, parked, until the replacement runs;
                    # the sweep below must not treat the parked name as stale.
                    current_step = "park_current_container"
                    parked_name = await edit_swap.park()
                    if parked_name:
                        protected_container_names.append(parked_name)

                current_step = "container_cleanup"
                # DAH-1524: the GC below (force-removing stale pod_/filler_
                # containers + their volumes that aren't in active_*) stays on the
                # deploy path — it frees ports/volumes the new pod needs. The former
                # `sleep=10` here was a vestigial hedge against the port-allocation
                # race; that race is now handled by explicit, bounded mechanisms
                # (force_remove_health_checks at the probe spawn site,
                # wait_for_port_check_containers just before `docker run`, and the
                # 90s _run_docker_create_with_port_retry budget), so we no longer
                # block the critical path for ~10s. (sleep defaults to 0.)
                await self.clean_existing_containers(
                    ssh_client=ssh_client,
                    default_extra=default_extra,
                    pod_name=container_name,
                    clear_volume=False if local_volume else True,
                    active_container_names=protected_container_names,
                    active_volume_names=payload.active_volume_names,
                )

                await self.clean_stale_vloopback_volumes(
                    ssh_client=ssh_client,
                    default_extra=default_extra,
                    skip_volume_names=protected_volume_names,
                )

                # DAH-2475: sweep FIRST, against the names the backend REQUESTED. The stale old-version
                # cache is dead weight (its names will never be requested again), and it is often the
                # very thing that makes the node too tight to afford the new download — so removing it
                # must happen before affordability is judged, or a renamed cache strands the node in a
                # cold-start loop it can never leave. Sweeping requested-but-not-yet-granted names is
                # safe: they either do not exist yet or are the current version worth keeping.
                await self.sweep_stale_cache_volumes(
                    ssh_client=ssh_client,
                    payload=payload,
                    default_extra=default_extra,
                )

                # The backend asks for the cache it wants; only the host knows whether those volumes
                # already exist, and therefore whether mounting them costs a download at all.
                payload.cache_volumes = await self.select_affordable_cache_volumes(
                    ssh_client=ssh_client,
                    payload=payload,
                    default_extra=default_extra,
                )

                await self.reclaim_dphn_cache_for_rental(
                    ssh_client=ssh_client,
                    payload=payload,
                    default_extra=default_extra,
                )

                # Add profiler for docker volume creation
                profilers.append(ProfilerStep.since(ProfilerStepName.CONTAINER_CLEANING, prev_timestamp))
                prev_timestamp = now_ms()

                # Effective limits default to the backend-sent values (legacy /
                # restart-edit path); the fresh-sizing path overrides them below.
                effective_volume_limit_gb = payload.volume_limit_gb
                effective_storage_limit_gb = payload.storage_limit_gb

                if not local_volume:
                    # resolve effective sizing, then create docker volume
                    current_step = "volume_sizing"
                    sizing = await self.resolve_volume_sizing(
                        ssh_client=ssh_client,
                        payload=payload,
                        log_tag=log_tag,
                        log_extra=default_extra,
                    )
                    effective_volume_limit_gb = sizing.volume_limit_gb
                    effective_storage_limit_gb = sizing.storage_limit_gb

                    current_step = "volume_creation"
                    local_volume = f"volume_{payload.pod_id}"
                    # DAH-2265 Plan 3: only full-node rentals (disk_share >= 1.0) get a
                    # sparse loopback volume. A full-node pod is sole-tenant, so there is
                    # nothing to overcommit against; partial (< 1.0) and legacy (None)
                    # rentals must stay preallocated to keep the DAH-2183 fresh-sizing
                    # math (df_avail + existing declared sizes) balanced.
                    full_node_rental = (
                        payload.disk_share is not None and payload.disk_share >= 1.0
                    )
                    await self.create_local_volume(
                        ssh_client=ssh_client,
                        docker_client=docker_client,
                        local_volume=local_volume,
                        log_tag=log_tag,
                        log_text=f"Creating docker volume {local_volume}",
                        log_extra=default_extra,
                        limit=effective_volume_limit_gb,
                        sparse=full_node_rental,
                    )
                    created_local_volume = True

                    # DAH-1524: profile local volume sizing + creation on its own;
                    # otherwise this SSH-bound time hides inside the broad
                    # "container creation" bucket and looks like `docker run`.
                    profilers.append(ProfilerStep.since(ProfilerStepName.DOCKER_VOLUME_CREATION, prev_timestamp))
                    prev_timestamp = now_ms()

                external_volume_name = None
                use_encrypted_volume = _should_encrypt_local_volume(
                    local_volume,
                    payload.workload_kind,
                    payload.is_sysbox,
                    payload.enable_volume_encryption,
                )
                if use_encrypted_volume:
                    current_step = "encrypted_volume_image_inspect"
                    if not await self._image_has_encrypted_volume_label(
                        ssh_client,
                        payload.docker_image,
                    ):
                        use_encrypted_volume = False
                        volume_encryption_status = VolumeEncryptionStatus.UNSUPPORTED_IMAGE
                        await self.stream_log(
                            "Image missing lium.volume_encryption.enable=1; using plain local volume",
                            "warning",
                            log_tag,
                        )
                        logger.warning(
                            _m(
                                "Image missing volume-encryption label; falling back to plain volume",
                                extra=get_extra_info({
                                    **default_extra,
                                    "container_name": container_name,
                                    "docker_image": payload.docker_image,
                                    "image_label": _ENCRYPTED_VOLUME_IMAGE_LABEL,
                                }),
                            ),
                        )

                if payload.bootstrap_restore:
                    current_step = "bootstrap_restore"
                    await self._run_bootstrap_restore(
                        ssh_client=ssh_client,
                        executor_info=executor_info,
                        payload=payload,
                        restore=payload.bootstrap_restore,
                        local_volume=local_volume,
                        local_volume_path=local_volume_path,
                        encrypted=use_encrypted_volume,
                    )
                if external_volume_info:
                    current_step = "external_volume_creation"
                    sysbox_subuid_base: int | None = None
                    if payload.is_sysbox:
                        sysbox_subuid_base = await self.resolve_sysbox_subuid_base(
                            ssh_client=ssh_client, log_extra=default_extra
                        )
                    success, msg = await self.create_s3fs_volume(
                        ssh_client=ssh_client,
                        log_extra=default_extra,
                        volume_info=external_volume_info,
                        log_tag=log_tag,
                        sysbox_subuid_base=sysbox_subuid_base,
                    )
                    if success:
                        # Add profiler for docker volume creation
                        profilers.append(ProfilerStep.since(ProfilerStepName.DOCKER_VOLUME_CREATION, prev_timestamp))
                        prev_timestamp = now_ms()
                        external_volume_name = external_volume_info.name
                        if payload.is_sysbox and sysbox_subuid_base is None:
                            # Decided only once the volume is really attached: runc
                            # keeps /mnt writable, sysbox without the base does not.
                            payload.is_sysbox = False
                            await self.stream_log(
                                "Sysbox disabled: cannot align S3 volume owner with the executor's sysbox uid range",
                                "warning",
                                log_tag,
                            )
                    else:
                        warnings.append(ContainerWarningCode.ExternalVolumeFailed)
                        profilers.append(ProfilerStep.since(ProfilerStepName.DOCKER_VOLUME_CREATION_FAILED, prev_timestamp))
                        await self.stream_log("S3 volume setup failed", "error", log_tag)

                # GPU options. --gpus injects userspace libs (libnvidia-ml.so, nvidia-smi);
                # explicit --device entries persist the device cgroup across systemd
                # daemon-reload (cgroup v2 + systemd cgroup driver wipe the transient
                # nvidia hook program; HostConfig.Devices is reapplied by Docker).
                current_step = "gpu_flags"
                gpu_config = await build_gpu_docker_config_for_executor(
                    ssh_client,
                    payload.gpu_uuids,
                    executor_id=payload.executor_id,
                    default_extra=default_extra,
                )

                if payload.cluster_membership is not None:
                    current_step = "cluster_overlay_port"
                    await self._assert_cluster_overlay_port_free(ssh_client, default_extra)

                # DAH-2356: cap GPU power for the Lium PEARL filler (only PEARL carries
                # gpu_power_limits). Fail-closed: apply undoes any partial work on failure, and the
                # raise records this create FAILED (12h backoff) — never run the miner uncapped.
                if payload.workload_kind == WorkloadKind.FILLER and payload.gpu_power_limits:
                    current_step = "gpu_power_cap"
                    cap_applied = await apply_filler_gpu_power_limits(
                        ssh_client,
                        payload.gpu_power_limits,
                        self.redis_service,
                        payload.pod_id,
                        payload.executor_id,
                        log_extra=default_extra,
                    )
                    if not cap_applied:
                        raise RuntimeError("GPU power cap could not be applied; refusing to start PEARL filler uncapped")
                else:
                    # DAH-2356 safety net: restore any leftover pre-cap records BEFORE a container
                    # without its own cap starts, so a customer (or an uncapped filler) never
                    # inherits a reduced limit. Best-effort, never blocks the rental.
                    if payload.gpu_uuids:
                        await restore_tracked_gpu_power_limits(
                            ssh_client, self.redis_service, payload.gpu_uuids, log_extra=default_extra
                        )
                    else:
                        # empty gpu_uuids = whole-node container (--gpus all) → check every host GPU
                        await restore_all_host_gpu_power_limits(
                            ssh_client, self.redis_service, log_extra=default_extra
                        )
                    # State-free last-resort net: if a pre-cap record was lost, the record-based
                    # restore above did nothing — lift anything still below the check's floor back
                    # to the GPU's own default, so the customer never starts on a capped GPU.
                    await raise_low_power_limits_to_default(
                        ssh_client,
                        payload.executor_id,
                        payload.gpu_uuids or None,
                        log_extra=default_extra,
                    )

                # DAH-1524: build_gpu_flags issues 2-3 serial SSH probes (proc minor
                # map, shared nodes, and a slow nvidia-smi -q -x fallback). Profile it
                # apart from the docker run so a slow probe doesn't read as a slow run.
                profilers.append(ProfilerStep.since(ProfilerStepName.GPU_DEVICE_PROBE, prev_timestamp))
                prev_timestamp = now_ms()

                # CPU and memory restriction flags
                # --cpus flag isn't working inside cvm. skip to use it when tdx_quote is present
                # TODO: remove this when cvm is fixed
                in_cvm = bool(executor_info.tdx_quote)
                cpu_count = None if in_cvm else payload.cpu_count

                quote_socket = False
                if _wants_quote_socket(payload, in_cvm=in_cvm):
                    current_step = "cvm_quote_broker"
                    quote_socket = await self._ensure_pod_quote_socket(
                        docker_client=docker_client,
                        ssh_client=ssh_client,
                        default_extra=default_extra,
                        log_tag=log_tag,
                    )
                    # the broker cold start (image pull, socket wait) must not read as port-check wait
                    prev_timestamp = now_ms()
                run_spec = self._build_rental_container_run_spec(
                    payload=payload,
                    container_name=container_name,
                    custom_options=custom_options,
                    port_maps=port_maps,
                    local_volume=local_volume,
                    local_volume_path=local_volume_path,
                    encrypted_local_volume=use_encrypted_volume,
                    external_volume_name=external_volume_name,
                    gpu_devices=gpu_config,
                    effective_storage_limit_gb=effective_storage_limit_gb,
                    cpu_count=cpu_count,
                    quote_socket=quote_socket,
                )

                logger.info(
                    _m(
                        "Creating docker container with SDK",
                        extra=get_extra_info({**default_extra, "container_name": container_name}),
                    )
                )

                # DAH-2018/DAH-2272: force-remove any backend health_check_* /
                # validator port-test container immediately before `docker run`.
                # The early check in miner_service runs before the image pull, but
                # the backend's RentalVerificationCheck can spin up a health_check
                # container during the pull window and grab a host port from the
                # same verified-port pool the rental allocated. Reuse the open
                # ssh_client so we don't pay for a second connect (and don't widen
                # the TOCTOU gap). No wait — the rental takes priority; the
                # port-allocated retry loop + `docker rm -fv` are the backstop for
                # any residual race.
                current_step = "port_check_wait"
                wait_ok, wait_msg = await self.wait_for_port_check_containers(
                    executor_info=executor_info,
                    miner_hotkey=payload.miner_hotkey,
                    keypair=keypair,
                    private_key=private_key,
                    ssh_client=ssh_client,
                )
                logger.info(
                    _m(
                        f"Port check container pre-run wait result: {wait_msg}",
                        extra=get_extra_info({**default_extra, "ok": wait_ok}),
                    )
                )

                # DAH-1524/DAH-2272: keep the pre-run port-check step out of the
                # docker-run measurement. It no longer blocks (force-remove only),
                # but the SSH round-trip is still not docker-run time.
                profilers.append(ProfilerStep.since(ProfilerStepName.PORT_CHECK_WAIT, prev_timestamp))
                prev_timestamp = now_ms()

                try:
                    current_step = "docker_run"
                    # DAH-2728: last look before the host ports get bound.
                    await self._abort_if_cancelled_by_delete(ssh_client, payload, default_extra)
                    await self.stream_log("Creating docker container", "success", log_tag)
                    await self._run_rental_docker_create_with_port_retry(
                        docker_client=docker_client,
                        ssh_client=ssh_client,
                        run_spec=run_spec,
                        container_name=container_name,
                        default_extra=default_extra,
                        local_volume=local_volume,
                        log_tag=log_tag,
                    )

                    container_created = True
                    logger.info("Container creation step finished")

                    # DAH-1524: isolate the bare `docker run` (dominated by the NVIDIA
                    # --gpus prestart hook, +sysbox/storage-opt) from the post-run
                    # running-state poll below.
                    profilers.append(ProfilerStep.since(ProfilerStepName.DOCKER_RUN, prev_timestamp))
                    prev_timestamp = now_ms()

                    # check if the container is running correctly
                    current_step = "container_health_check"
                    if not await self.check_container_running(ssh_client, container_name):
                        # Capture the failure reason and check whether it points to our
                        # --device flags (DAH-1987). State.Error covers cgroup / device
                        # failures; logs --tail covers entrypoint failures.
                        failure_reason = ""
                        try:
                            container_name_quoted = shlex.quote(container_name)
                            inspect = await ssh_client.run(
                                f"/usr/bin/docker inspect -f '{{{{.State.Error}}}}' {container_name_quoted}"
                            )
                            logs_tail = await ssh_client.run(
                                f"/usr/bin/docker logs --tail 50 {container_name_quoted} 2>&1 || true"
                            )
                            failure_reason = (inspect.stdout or "") + "\n" + (logs_tail.stdout or "")
                        except Exception:
                            failure_reason = "(failure_reason capture failed)"

                        nvidia_signal = any(
                            marker in failure_reason.lower()
                            for marker in ("/dev/nvidia", "device cgroup", "no such device", "operation not permitted")
                        )
                        log_extra = get_extra_info({
                            **default_extra,
                            "container_name": container_name,
                            "gpu_device_mounts": [
                                device.path_on_host for device in gpu_config.device_mounts
                            ],
                            "gpu_device_request_ids": [
                                list(device_request.device_ids)
                                for device_request in gpu_config.device_requests
                            ],
                            "failure_reason": failure_reason[:2000],
                        })
                        if nvidia_signal:
                            logger.error(_m(
                                "docker run failed with NVIDIA-device-related error — "
                                "possible regression from GPU device configuration",
                                extra=log_extra,
                            ))
                        else:
                            logger.error(_m("docker run failed", extra=log_extra))

                        raise Exception("Run docker run command but container is not running")
                except Exception:
                    container_missing = await self.cleanup_failed_container_creation(
                        ssh_client=ssh_client,
                        default_extra=default_extra,
                        container_name=container_name,
                        volume_name=local_volume,
                        remove_volume=created_local_volume,
                    )
                    container_vanished = container_created and container_missing
                    # DAH-2211: inline cleanup of custom-build artifacts on docker_run failure.
                    if is_custom_build:
                        await self._cleanup_custom_build_artifacts(
                            ssh_client=ssh_client,
                            pod_id=payload.pod_id,
                            default_extra=default_extra,
                        )
                    raise

                # Add profiler for the post-run container running-state poll
                profilers.append(ProfilerStep.since(ProfilerStepName.CONTAINER_RUNNING_CHECK, prev_timestamp))
                prev_timestamp = now_ms()

                logger.info(
                    _m(
                        "Created Docker Container",
                        extra=get_extra_info({**default_extra, "container_name": container_name}),
                    ),
                )

                await self.stream_log("Created Docker Container", "success", log_tag)

                try:
                    if use_encrypted_volume:
                        current_step = "encrypted_volume_setup"
                        await self.setup_encrypted_local_volume(
                            ssh_client=ssh_client,
                            container_name=container_name,
                            plaintext_path=local_volume_path,
                            volume_name=local_volume,
                            pod_id=payload.pod_id,
                            log_tag=log_tag,
                            log_extra=default_extra,
                        )
                        volume_encryption_status = VolumeEncryptionStatus.ENABLED
                        profilers.append(ProfilerStep.since(ProfilerStepName.ENCRYPTED_VOLUME_SETUP, prev_timestamp))
                        prev_timestamp = now_ms()

                    # DAH-2341: inject the customer's public keys before the sshd
                    # bootstrap. The keys are plain data (mkdir + append) with no
                    # dependency on a running sshd, and the bootstrap may now spend
                    # a grace period waiting for an image-provided sshd — whichever
                    # sshd comes up first must already find the keys in place.
                    current_step = "add_public_keys"
                    await self.add_ssh_public_keys_with_rental_docker(
                        docker_client=docker_client,
                        container_name=container_name,
                        public_keys=payload.user_public_keys,
                        log_tag=log_tag,
                        log_extra=default_extra,
                    )

                    current_step = "ssh_bootstrap"
                    if image_manages_services:
                        # DAH-2265: the default image / cached template ships and
                        # starts sshd itself (its start.sh runs `service ssh start`
                        # unconditionally). Skip the validator's sshd bootstrap
                        # entirely. The public-key injection above already created
                        # ~/.ssh (mkdir -p /root/.ssh in the authorized_keys exec spec),
                        # so the skip path needs nothing further here.
                        # Note: gated on image_manages_services, not payload.ships_sshd
                        # alone — a renter startup_commands/entrypoint override replaces
                        # the CMD/ENTRYPOINT that runs start.sh, so the image would not
                        # start sshd and we must fall through to the bootstrap below.
                        logger.info(
                            _m(
                                "Skipping SSH-server bootstrap; template ships sshd",
                                extra=get_extra_info({**default_extra, "container_name": container_name}),
                            )
                        )
                    else:
                        await self.install_open_ssh_server_and_start_ssh_service_with_rental_docker(
                            docker_client=docker_client,
                            container_name=container_name,
                            log_tag=log_tag,
                            log_extra=default_extra,
                        )

                    jupyter_url = None
                    if payload.enable_jupyter and jupyter_port_map:
                        current_step = "jupyter_setup"
                        if image_managed_jupyter:
                            # DAH-2265: the image's start.sh already launched Jupyter from
                            # the JUPYTER_PASSWORD forwarded at `docker run` above, and
                            # serves it as that token. Don't run run_jupyter; just build
                            # the URL from the same token.
                            jupyter_token = image_jupyter_token
                        else:
                            jupyter_token = secrets.token_hex(16)
                            await self.run_jupyter(
                                ssh_client=ssh_client,
                                container_name=container_name,
                                jupyter_token=jupyter_token,
                                jupyter_port=jupyter_port_map[0],
                                log_tag=log_tag,
                                log_extra=default_extra,
                                local_volume=local_volume,
                                local_volume_path=local_volume_path,
                                encrypted_local_volume=use_encrypted_volume,
                            )
                        jupyter_url = f"http://{executor_info.address}:{jupyter_port_map[1]}/lab?token={jupyter_token}"

                    # Add profiler for ssh service installation (covers key
                    # injection + bootstrap + jupyter since the running check)
                    profilers.append(ProfilerStep.since(ProfilerStepName.SSH_SERVICE_INSTALLATION, prev_timestamp))
                    prev_timestamp = now_ms()

                    # add environment variables
                    current_step = "set_environment"
                    environment_error = await self.add_environment_variables_with_rental_docker(
                        docker_client=docker_client,
                        container_name=container_name,
                        environment=custom_options.environment if custom_options else None,
                        log_tag=log_tag,
                        log_extra=default_extra,
                    )
                    if environment_error:
                        raise RuntimeError(f"Failed to set environment variables: {environment_error}")

                    # Historical name — key injection moved before the bootstrap
                    # (DAH-2341), so this step now times the environment setup.
                    profilers.append(ProfilerStep.since(ProfilerStepName.ADDING_PUBLIC_KEYS, prev_timestamp))
                    prev_timestamp = now_ms()

                    await self.finish_stream_logs()

                    # DAH-2728: last call before the pod is cached as rented — a delete that landed
                    # during the run or the bootstrap above is holding ports it could not see.
                    await self._abort_if_cancelled_by_delete(ssh_client, payload, default_extra)

                    current_step = "finalize"
                    await self._cache_rented_pod_best_effort(
                        executor_info=executor_info,
                        pod_id=payload.pod_id,
                        container_name=container_name,
                        default_extra=default_extra,
                    )
                    if settings.ENABLE_INSPECTOR:
                        await self._run_inspector_collector_lifecycle(
                            ssh_client=ssh_client,
                            executor_info=executor_info,
                            action="start",
                            default_extra={
                                **default_extra,
                                "container_name": container_name,
                            },
                        )
                    profilers.append(
                        ProfilerStep.since(
                            ProfilerStepName.INSPECTOR_START,
                            prev_timestamp,
                            skipped=not settings.ENABLE_INSPECTOR,
                        )
                    )
                    prev_timestamp = now_ms()
                except Exception:
                    container_missing = await self.cleanup_failed_container_creation(
                        ssh_client=ssh_client,
                        default_extra=default_extra,
                        container_name=container_name,
                        volume_name=local_volume,
                        remove_volume=created_local_volume,
                    )
                    container_vanished = container_created and container_missing
                    # DAH-2211: inline cleanup of custom-build artifacts on post-run failure.
                    if is_custom_build:
                        await self._cleanup_custom_build_artifacts(
                            ssh_client=ssh_client,
                            pod_id=payload.pod_id,
                            default_extra=default_extra,
                        )
                    raise

                # DAH-2458: final step. Stamp the subnet's wall-clock finish time onto it (in
                # addition to its duration) so the backend derives its finalize span directly as
                # pending_finished_at - this timestamp, instead of reconstructing the subnet's
                # finish moment from the sum of every step's duration.
                finished_ms = now_ms()
                profilers.append(
                    ProfilerStep(
                        name=ProfilerStepName.FINISHED_IN_SUBNET,
                        duration=finished_ms - prev_timestamp,
                        timestamp=finished_ms,
                    )
                )

                if payload.workload_kind == WorkloadKind.FILLER:
                    await self.redis_service.remove_pending_pod(
                        payload.miner_hotkey,
                        payload.executor_id,
                        payload.pod_id,
                    )

                # DAH-1524: one structured, Loki-queryable summary of the deploy
                # profile. Reuses the `profilers` list (now typed `ProfilerStep`s).
                # Skipped steps appear at ~0ms (with "skipped": true) so before/after
                # is visible. Success path only (per spec). The anchor step has
                # `duration is None`, so it surfaces as `duration_ms: null` and is
                # excluded from the total.
                profile_steps = [
                    {
                        "name": p.name.value,
                        "duration_ms": p.duration,
                        "skipped": p.skipped,
                    }
                    for p in profilers
                ]
                # Sum of every step's duration. When the backend sends
                # `payload.timestamp`, the backend->subnet queue/transit leg is
                # captured inside the "Started in subnet" step (now - timestamp),
                # so this total is end-to-end; otherwise it is subnet-internal time.
                # The "Requested from backend" anchor has no duration and is excluded.
                total_duration_ms = sum(p.duration or 0 for p in profilers)
                logger.info(
                    _m(
                        "Deployment profile summary",
                        extra=get_extra_info({
                            **default_extra,
                            "container_name": container_name,
                            "profile_steps": profile_steps,
                            "total_duration_ms": total_duration_ms,
                        }),
                    )
                )

                return ContainerCreated(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    container_name=container_name,
                    volume_name=local_volume,
                    port_maps=[
                        (docker_port, external_port) for docker_port, _, external_port in port_maps
                    ],
                    profilers=profilers,
                    backup_log_id=payload.backup_log_id,
                    restore_path=payload.restore_path,
                    restore_log_id=payload.bootstrap_restore.restore_log_id if payload.bootstrap_restore else None,
                    jupyter_url=jupyter_url,
                    warnings=warnings,
                    storage_limit_gb=effective_storage_limit_gb,
                    volume_limit_gb=effective_volume_limit_gb,
                    local_volume_path=local_volume_path,
                    volume_encryption_status=volume_encryption_status,
                )
        except Exception as e:
            if isinstance(e, _CreateCancelledByDelete):
                current_step = "cancelled_by_delete"
            log_text = _m(
                "Failed create_container",
                extra=get_extra_info({
                    **default_extra,
                    # DAH-2740: a tenacity RetryError says nothing; the last attempt's text is the cause
                    "error": "; ".join(_exception_texts(e)),
                    "failure_step": current_step,
                }),
            )
            logger.error(log_text, exc_info=True)

            await self.finish_stream_logs()
            await self.redis_service.remove_pending_pod(payload.miner_hotkey, payload.executor_id, payload.pod_id)

            # Port release now handled by backend.
            # DAH-2475: msg carries the renter-safe headline; detail carries the FULL text (headline +
            # the extra dict holding the actual error) — it becomes filler_run.failure_reason on the
            # backend, where a bare "Failed create_container" told us nothing about why.
            failure_detail = log_text.to_full_string()
            if (
                current_step == "docker_sdk_ssh_host_key"
                and isinstance(e, RentalDockerConnectionError)
            ):
                failure_detail = f"{failure_detail}: {e}"

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                detail=failure_detail,
                error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                error_code=(
                    FailedContainerErrorCodes.ContainerVanished
                    if container_vanished
                    else FailedContainerErrorCodes.UnknownError
                ),
                failure_step=current_step,
                volume_encryption_status=(
                    VolumeEncryptionStatus.FAILED
                    if current_step
                    in {
                        "encrypted_volume_image_inspect",
                        "encrypted_volume_setup",
                    }
                    else None
                ),
            )

    async def _run_bootstrap_restore(
        self,
        *,
        ssh_client: asyncssh.SSHClientConnection,
        executor_info: ExecutorSSHInfo,
        payload: ContainerCreateRequest,
        restore: BootstrapRestoreSpec,
        local_volume: str,
        local_volume_path: str,
        encrypted: bool,
    ) -> None:
        if not await supports_storage_operation(ssh_client, restore.backup_engine):
            # Legacy archives must remain restorable while executor-image adoption
            # is gradual. Restic has no safe fallback without its pinned binary.
            if restore.backup_engine == "tar_aws_cli" and not encrypted:
                await self._run_legacy_bootstrap_restore(
                    ssh_client=ssh_client,
                    executor_info=executor_info,
                    restore=restore,
                    local_volume=local_volume,
                    local_volume_path=local_volume_path,
                )
                return
            raise RuntimeError(
                f"executor does not support bootstrap restore engine {restore.backup_engine}"
            )
        operation_id = UUID(restore.restore_log_id)
        workspace: dict[str, object] = {
            "mode": "encrypted_bootstrap" if encrypted else "plain_volume",
            "volume_name": local_volume,
            "volume_path": local_volume_path,
            "requested_path": restore.restore_path or local_volume_path,
        }
        if encrypted:
            workspace["volume_passphrase"] = VolumeKeyDeriver.from_settings(settings).material(
                payload.pod_id
            ).passphrase

        repository: dict[str, object] = {
            "bucket": restore.backup_volume_info.name,
            "access_key_id": restore.backup_volume_info.iam_user_access_key,
            "secret_access_key": restore.backup_volume_info.iam_user_secret_key,
            "session_token": restore.backup_volume_info.session_token,
            "password": restore.repository_password,
        }
        spec: dict[str, object] = {
            "operation_id": restore.restore_log_id,
            "pod_id": payload.pod_id,
            "repository_pod_id": restore.repository_pod_id,
            "action": "restore",
            "engine": restore.backup_engine,
            "repository": repository,
            "workspace": workspace,
            "snapshot_id": restore.snapshot_id,
            "legacy_object_key": restore.legacy_object_key,
            "legacy_object_size_bytes": restore.legacy_object_size_bytes,
            "reporter": {
                "api_url": settings.COMPUTE_REST_API_URL_EXTERNAL,
                "auth_token": restore.auth_token,
                "resource": "restore",
                "failure_timeout_seconds": restore.failure_timeout_seconds,
            },
        }
        files = await start_storage_operation(
            ssh_client,
            executor_info.python_path,
            operation_id,
            spec,
            retain_terminal_artifacts=True,
        )
        await wait_for_storage_operation(ssh_client, files)

    async def _run_legacy_bootstrap_restore(
        self,
        *,
        ssh_client: asyncssh.SSHClientConnection,
        executor_info: ExecutorSSHInfo,
        restore: BootstrapRestoreSpec,
        local_volume: str,
        local_volume_path: str,
    ) -> None:
        local_jobs = Path(__file__).resolve().parent.parent / "miner_jobs"
        remote_script = "/root/app/restore_storage.py"
        remote_helper = "/root/app/workspace_mount.py"
        async with ssh_client.start_sftp_client() as sftp:
            await sftp.put(str(local_jobs / "restore_storage.py"), remote_script)
            await sftp.put(str(local_jobs / "workspace_mount.py"), remote_helper)
        command = [
            executor_info.python_path,
            remote_script,
            "--api-url",
            settings.COMPUTE_REST_API_URL_EXTERNAL,
            "--target-volume",
            local_volume,
            "--restore-path",
            restore.restore_path or local_volume_path,
            "--backup-source-path",
            restore.legacy_object_key or "",
            "--auth-token",
            restore.auth_token,
            "--restore-log-id",
            restore.restore_log_id,
            "--backup-volume-name",
            restore.backup_volume_info.name,
            "--backup-volume-iam_user_access_key",
            restore.backup_volume_info.iam_user_access_key,
            "--backup-volume-iam_user_secret_key",
            restore.backup_volume_info.iam_user_secret_key,
            "--target-volume-path",
            local_volume_path,
        ]
        await ssh_client.run(shlex.join(command), check=True)

    async def stream_log(self, log_msg:str, log_status: str, log_tag: str):
        async with self.lock:
            self.logs_queue.append(
                {
                    "log_text": log_msg,
                    "log_status": log_status,
                    "log_tag": log_tag,
                }
            )

    async def stop_container(
        self,
        payload: ContainerStopRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_uuid": payload.executor_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Stop Docker Container", extra=get_extra_info({**default_extra, "payload": str(payload)})
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        known_hosts_policy: asyncssh.SSHKnownHosts | None = None
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerStopFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            require_rental_docker_ssh_host_key(executor_info)
        except RentalDockerConnectionError as exc:
            log_text = _missing_rental_docker_host_key_log_text(default_extra, exc)
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerStopFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            async with self.rental_docker_client_factory.connect(
                executor_info=executor_info,
                private_key=private_key,
            ) as docker_client:
                await run_logged_rental_docker_sdk_operation(
                    operation="stop_container",
                    log_extra=default_extra,
                    call=lambda: docker_client.stop(container_name=payload.container_name),
                    container_name=payload.container_name,
                )
        except Exception as exc:
            log_text = _m(
                "Failed stop_container",
                extra=get_extra_info(
                    {
                        **default_extra,
                        "container_name": payload.container_name,
                        "error": str(exc),
                    }
                ),
            )
            logger.error(log_text, exc_info=True)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerStopFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        logger.info(
            _m(
                "Stopped Docker Container",
                extra=get_extra_info(
                    {**default_extra, "container_name": payload.container_name}
                ),
            ),
        )

    async def start_container(
        self,
        payload: ContainerStartRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_uuid": payload.executor_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Restart Docker Container",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)

        known_hosts_policy: asyncssh.SSHKnownHosts | None = None
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerStartFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            require_rental_docker_ssh_host_key(executor_info)
        except RentalDockerConnectionError as exc:
            log_text = _missing_rental_docker_host_key_log_text(default_extra, exc)
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerStartFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            await self.start_existing_container(
                executor_info=executor_info,
                private_key=private_key,
                known_hosts_policy=known_hosts_policy,
                container_name=payload.container_name,
                local_volume_path=payload.local_volume_path,
                pod_id=payload.pod_id,
                default_extra=default_extra,
            )
        except Exception as exc:
            log_text = _m(
                "Failed start_container",
                extra=get_extra_info(
                    {
                        **default_extra,
                        "container_name": payload.container_name,
                        "error": str(exc),
                    }
                ),
            )
            logger.error(log_text, exc_info=True)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerStartFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

    async def start_existing_container(
        self,
        *,
        executor_info: ExecutorSSHInfo,
        private_key: str,
        known_hosts_policy: asyncssh.SSHKnownHosts | None,
        container_name: str,
        local_volume_path: str | None,
        pod_id: str,
        default_extra: dict[str, Any],
    ) -> None:
        # start a container that already exists and restore the two things a bare `docker start`
        # drops: the gocryptfs plaintext mount of an encrypted rental volume, and the sshd the
        # customer connects through. A failed remount raises and the caller decides how to report
        # it; a failed sshd bootstrap only warns, because the pod itself is up either way.
        # A falsy local_volume_path means the caller does not know the plaintext path, which is only
        # safe for a pod without an encrypted volume: mounting gocryptfs at a guessed path would
        # leave the customer's real path an ordinary container dir, writing plaintext to the host.
        pkey = asyncssh.import_private_key(private_key)
        async with self.rental_docker_client_factory.connect(
            executor_info=executor_info,
            private_key=private_key,
        ) as docker_client:
            await run_logged_rental_docker_sdk_operation(
                operation="start_container",
                log_extra=default_extra,
                call=lambda: docker_client.start(container_name=container_name),
                container_name=container_name,
            )
            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=known_hosts_policy,
            ) as ssh_client:
                encrypted_volume_name = await self._encrypted_local_volume_name(
                    docker_client,
                    container_name,
                )
                if encrypted_volume_name:
                    if not local_volume_path:
                        raise RuntimeError(
                            f"{container_name} holds an encrypted volume but no plaintext path "
                            "was supplied; refusing to remount it at a guessed path"
                        )
                    await self.setup_encrypted_local_volume(
                        ssh_client=ssh_client,
                        container_name=container_name,
                        plaintext_path=local_volume_path,
                        volume_name=encrypted_volume_name,
                        pod_id=pod_id,
                        log_tag=f"start_container_{pod_id}",
                        log_extra=default_extra,
                        allow_init=False,
                    )
            ssh_bootstrap_ok = await self.install_open_ssh_server_and_start_ssh_service_with_rental_docker(
                docker_client=docker_client,
                container_name=container_name,
                log_tag=f"start_container_{pod_id}",
                log_extra=default_extra,
            )
            if not ssh_bootstrap_ok:
                logger.warning(
                    _m(
                        "Docker container started but SSH bootstrap did not complete cleanly",
                        extra=get_extra_info(
                            {**default_extra, "container_name": container_name}
                        ),
                    )
                )
            logger.info(
                _m(
                    "Started Docker Container",
                    extra=get_extra_info(
                        {**default_extra, "container_name": container_name}
                    ),
                ),
            )

    async def recover_pod_after_stale_vloopback_mount(
        self,
        *,
        ssh_client: asyncssh.SSHClientConnection,
        executor_info: ExecutorSSHInfo,
        miner_hotkey: str,
        private_key: str,
        container_name: str,
        pod_id: str,
        container_error: str | None,
        local_volume_path: str | None,
        default_extra: dict[str, Any],
    ) -> bool:
        # DAH-2306: bring back a still-rented pod the host could not auto-restart after a reboot.
        # The reboot leaves the vloopback mountpoint dir behind, dockerd's `unless-stopped` restart
        # hits the plugin's "file exists" and the container stays down until the rental ends. Only
        # that exact failure is recovered: a pod stopped on purpose exits with an empty State.Error
        # and never matches, so a customer's stop is never overridden.
        if not container_error or not _is_stale_vloopback_mountpoint_error(container_error):
            return False

        log_extra = {**default_extra, "container_name": container_name}
        try:
            encryption_state = await self._local_volume_encryption_state(
                ssh_client, container_name
            )
            if encryption_state is _VolumeEncryptionState.UNKNOWN or (
                encryption_state is _VolumeEncryptionState.ENCRYPTED
                and not _can_remount_encrypted_volume(local_volume_path)
            ):
                logger.warning(
                    _m(
                        "POD_STALE_MOUNT_RECOVERY_SKIPPED_ENCRYPTED_VOLUME",
                        extra=get_extra_info({
                            **log_extra,
                            "volume_encryption_state": encryption_state.value,
                        }),
                    )
                )
                return False
            repaired_volume = await self._repair_stale_mountpoint_of_container_volume(
                ssh_client,
                container_name,
                log_extra,
            )
        except TimeoutError:
            # a command that did not finish in time is not a dead SSH transport: report it here so
            # the check keeps its POD_NOT_RUNNING verdict instead of turning it into
            # EXECUTOR_TRANSPORT_UNREACHABLE and dropping the penalty.
            logger.warning(
                _m(
                    "POD_STALE_MOUNT_RECOVERY_TIMED_OUT",
                    extra=get_extra_info(log_extra),
                )
            )
            return False
        if not repaired_volume:
            logger.warning(
                _m(
                    "POD_STALE_MOUNT_RECOVERY_REPAIR_FAILED",
                    extra=get_extra_info(log_extra),
                )
            )
            return False

        log_extra = {**log_extra, "local_volume": repaired_volume}
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                miner_hotkey,
                log_extra,
            )
            require_rental_docker_ssh_host_key(executor_info)
            await self.start_existing_container(
                executor_info=executor_info,
                private_key=private_key,
                known_hosts_policy=known_hosts_policy,
                container_name=container_name,
                local_volume_path=local_volume_path,
                pod_id=pod_id,
                default_extra=log_extra,
            )
        except asyncio.CancelledError:
            # the whole check runs under an outer timeout, so a cancellation can land between
            # `docker start` and the gocryptfs remount and leave the container up with its
            # plaintext path unmounted — exactly the state the Exception branch below stops. The
            # stop is shielded because an unshielded await in a cancelled task never runs; the
            # cancellation itself still propagates.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(
                    self._stop_half_recovered_container(ssh_client, container_name)
                )
            raise
        except Exception as exc:
            container_stopped = await self._stop_half_recovered_container(
                ssh_client, container_name
            )
            logger.error(
                _m(
                    "POD_STALE_MOUNT_RECOVERY_START_FAILED",
                    extra=get_extra_info({
                        **log_extra,
                        "error": str(exc),
                        "container_stopped": container_stopped,
                    }),
                )
            )
            return False

        logger.info(
            _m("POD_STALE_MOUNT_RECOVERED", extra=get_extra_info(log_extra))
        )
        return True

    async def _local_volume_encryption_state(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
    ) -> _VolumeEncryptionState:
        # whether the pod's rental volume is gocryptfs-encrypted, which recovery may only touch
        # when it knows the plaintext path. An encrypted volume is mounted as the ciphertext at
        # /lium-cipher and the plaintext path the customer actually uses is recorded nowhere on the
        # host; DAH-2545 carries it in the backend's rental-active response. Remounting at the
        # /root default would silently turn a custom path into an ordinary container dir, so
        # writes to it would land unencrypted on the miner's disk. An inspect that does not answer
        # is UNKNOWN rather than ENCRYPTED: recovery stands down either way, but a pod we could not
        # even look at must not be started on the strength of a path we cannot match to a volume.
        inspect_result = await ssh_client.run(
            f"/usr/bin/docker inspect {shlex.quote(container_name)} "
            '--format \'{{range .Mounts}}{{println .Destination}}{{end}}\'',
            timeout=_VLOOPBACK_REPAIR_COMMAND_TIMEOUT_SEC,
        )
        if getattr(inspect_result, "exit_status", 0) != 0:
            return _VolumeEncryptionState.UNKNOWN
        if _LIUM_CIPHER_MOUNT in (inspect_result.stdout or "").split():
            return _VolumeEncryptionState.ENCRYPTED
        return _VolumeEncryptionState.PLAIN

    async def _repair_stale_mountpoint_of_container_volume(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
        log_extra: dict[str, Any],
    ) -> str | None:
        # the volume whose stale mountpoint kept the container down, read off the container itself.
        # `docker inspect` reports mounts of a stopped container, so the real name is taken from the
        # host instead of guessed from pod_id — a pod created through the edit path carries a
        # backend-supplied volume name that no convention derives. Every mount can be offered
        # blindly: repair_stale_vloopback_mountpoint accepts only a vloopback volume whose stale
        # mountpoint dir is present and empty.
        inspect_result = await ssh_client.run(
            f"/usr/bin/docker inspect {shlex.quote(container_name)} "
            '--format \'{{range .Mounts}}{{if eq .Type "volume"}}{{println .Name}}{{end}}{{end}}\'',
            timeout=_VLOOPBACK_REPAIR_COMMAND_TIMEOUT_SEC,
        )
        if getattr(inspect_result, "exit_status", 0) != 0:
            return None

        for candidate_volume in (inspect_result.stdout or "").split():
            repaired = await self.repair_stale_vloopback_mountpoint(
                ssh_client,
                candidate_volume,
                {**log_extra, "local_volume": candidate_volume},
            )
            if repaired:
                return candidate_volume
        return None

    async def _stop_half_recovered_container(
        self,
        ssh_client: asyncssh.SSHClientConnection,
        container_name: str,
    ) -> bool:
        # a start that brought the container up but failed afterwards — a gocryptfs remount that did
        # not take — leaves the customer a running pod with an empty plaintext dir, and no later
        # cycle revisits it because recovery only fires on a stopped container. Put it back down so
        # the POD_NOT_RUNNING verdict keeps matching what the customer sees. A no-op when the start
        # itself was what failed.
        try:
            stop_result = await ssh_client.run(
                f"/usr/bin/docker stop {shlex.quote(container_name)}",
                timeout=_VLOOPBACK_REPAIR_COMMAND_TIMEOUT_SEC,
            )
            return getattr(stop_result, "exit_status", 1) == 0
        except asyncio.CancelledError:
            raise
        except Exception:
            return False

    def _container_deleted(self, payload: ContainerDeleteRequest) -> ContainerDeleted:
        return ContainerDeleted(
            miner_hotkey=payload.miner_hotkey,
            executor_id=payload.executor_id,
            pod_id=payload.pod_id,
            workload_kind=payload.workload_kind,
        )

    def _failed_delete(self, payload: ContainerDeleteRequest, msg: str) -> FailedContainerRequest:
        return FailedContainerRequest(
            miner_hotkey=payload.miner_hotkey,
            executor_id=payload.executor_id,
            pod_id=payload.pod_id,
            workload_kind=payload.workload_kind,
            msg=msg,
            error_type=FailedContainerErrorTypes.ContainerDeletionFailed,
            error_code=FailedContainerErrorCodes.UnknownError,
        )

    async def _stop_container_gracefully(
        self,
        docker_client: RentalDockerSdkClient,
        payload: ContainerDeleteRequest,
        log: _BoundLog,
    ) -> None:
        # DAH-2364: SIGTERM plus a grace window, so the workload exits cleanly before the forced
        # removal. Never fatal — the forced removal stays the single source of truth for deletion.
        # Called directly rather than through run_logged_rental_docker_sdk_operation so that an
        # expected failure — chiefly an already-absent container — is not logged at ERROR and does
        # not raise a false failed-deletion alert.
        # Fillers use a shorter grace window (see FILLER_CONTAINER_STOP_GRACE_SECONDS) so a
        # SIGTERM-ignoring filler cannot outlast the backend's preemption budget.
        stop_grace_seconds = (
            FILLER_CONTAINER_STOP_GRACE_SECONDS
            if payload.workload_kind == WorkloadKind.FILLER
            else CONTAINER_STOP_GRACE_SECONDS
        )

        stop_started = time.monotonic()
        try:
            await docker_client.stop(
                container_name=payload.container_name,
                stop_grace_seconds=stop_grace_seconds,
            )
        except Exception as exc:
            if _is_missing_docker_container_error(exc):
                log.info(
                    "Graceful stop skipped: container is already absent",
                    container_name=payload.container_name,
                )
            else:
                log.warning(
                    "Graceful container stop failed; proceeding to forced removal",
                    container_name=payload.container_name,
                    error=str(exc),
                )
            return

        log.info(
            "Graceful container stop completed",
            container_name=payload.container_name,
            stop_grace_seconds=stop_grace_seconds,
            duration_ms=int((time.monotonic() - stop_started) * 1000),
        )

    async def _force_remove_container(
        self,
        docker_client: RentalDockerSdkClient,
        payload: ContainerDeleteRequest,
        log: _BoundLog,
    ) -> FailedContainerRequest | None:
        try:
            await run_logged_rental_docker_sdk_operation(
                operation="remove_container",
                log_extra=log.base_extra,
                call=lambda: docker_client.remove_container(
                    container_name=payload.container_name,
                    force=True,
                    remove_volumes=True,
                ),
                container_name=payload.container_name,
                force=True,
                remove_volumes=True,
            )
        except Exception as exc:
            if _is_docker_container_removal_in_progress_error(exc):
                # Docker is still processing the original force-remove request;
                # let the backend keep polling without treating this as terminal.
                error_msg = str(exc)
                log.info(
                    "Container deletion is still in progress",
                    container_name=payload.container_name,
                    error=error_msg,
                )
                return FailedContainerRequest(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    msg=error_msg,
                    error_type=FailedContainerErrorTypes.ContainerDeletionFailed,
                    error_code=FailedContainerErrorCodes.DeletionInProgress,
                )

            # DAH-2345: deletion is idempotent for every workload kind — a container
            # that is already gone (e.g. removed by failed-create cleanup) must not
            # fail the delete, or the backend retries a doomed request until
            # force-removal and penalizes the miner.
            if not _is_missing_docker_container_error(exc):
                raise
            log.info(
                "Container is already absent",
                container_name=payload.container_name,
                error=str(exc),
            )
        return None

    async def delete_container(
        self,
        payload: ContainerDeleteRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = _delete_container_log_extra(payload, executor_info)
        log = _BoundLog(default_extra)

        # DAH-2728: a delete that races an in-flight create can arrive before any container name
        # exists. The create built its name from the pod id, so the same rule reconstructs it here
        # rather than letting the teardown run against an empty name.
        if not payload.container_name:
            payload.container_name = self.get_container_name(payload)
            log.info("Derived the container name for a delete that carried none",
                     container_name=payload.container_name)

        log.info("Deleting Docker Container", payload=str(payload))

        try:
            _validate_delete_volume_names(payload)
        except ValueError as exc:
            log.error("Invalid Docker volume name", error=str(exc))
            return self._failed_delete(payload, msg="Invalid Docker volume name")

        # DAH-2728: a create still in flight would otherwise finish behind our back and leave an
        # orphan container holding the ports. Past the validation above, so a delete this validator
        # refuses cannot kill a healthy create.
        if inflight_creates.cancel(payload.pod_id):
            log.info("Cancelled the in-flight create for this pod")
            # Let the create finish tearing itself down before this teardown runs, so the last
            # actor on the host is never a create that ran after we answered "deleted". On timeout
            # we tear the container down ourselves, which is what this delete did before DAH-2728.
            create_aborted = await inflight_creates.wait_until_done(
                payload.pod_id, CANCELLED_CREATE_ABORT_TIMEOUT_SECONDS
            )
            if not create_aborted:
                log.warning("The cancelled create is still running; deleting without it")

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log.error("Attestation failed", error=str(exc))
            return self._failed_delete(payload, msg="Attestation failed")

        try:
            require_rental_docker_ssh_host_key(executor_info)
        except RentalDockerConnectionError as exc:
            log_text = _missing_rental_docker_host_key_log_text(default_extra, exc)
            logger.error(log_text)
            return self._failed_delete(payload, msg=str(log_text))

        try:
            async with (
                asyncssh.connect(
                    host=executor_info.address,
                    port=executor_info.ssh_port,
                    username=executor_info.ssh_username,
                    client_keys=[pkey],
                    known_hosts=known_hosts_policy,
                ) as ssh_client,
                self.rental_docker_client_factory.connect(
                    executor_info=executor_info,
                    private_key=private_key,
                ) as docker_client,
            ):
                await self._stop_container_gracefully(docker_client, payload, log)

                # Fatal boundary: the forced removal is the only step whose failure fails the
                # undeploy. Every step below runs after the container is gone and is best-effort.
                try:
                    removal_failure = await self._force_remove_container(docker_client, payload, log)
                    if removal_failure is not None:
                        return removal_failure
                except Exception:
                    # DAH-2427: a failed force-remove (backend FAILED / STOP_FAILED) is the
                    # classic wedge path — sweep before propagating so a wedged card does not
                    # outlive the failed delete. No-op while a live compute app still exists.
                    if payload.workload_kind == WorkloadKind.FILLER:
                        with _best_effort_delete_step(log, "sweep_wedged_gpus_after_failed_remove"):
                            await _sweep_wedged_gpus_after_teardown(ssh_client, log)
                    raise

                # DAH-2211: always-on inline cleanup of custom-build artifacts
                # for this pod. No-op if the pod was not a custom build.
                # (self-guarding: logs and never raises)
                await self._cleanup_custom_build_artifacts(
                    ssh_client=ssh_client,
                    pod_id=payload.pod_id,
                    default_extra=default_extra,
                )

                # DAH-2356: restore the pre-cap GPU power limits after a Lium PEARL filler is
                # removed, so the next rental gets the machine at its original power.
                if payload.workload_kind == WorkloadKind.FILLER:
                    with _best_effort_delete_step(log, "restore_filler_gpu_power"):
                        await restore_filler_pod_gpu_power_limits(
                            ssh_client, self.redis_service, payload.pod_id, log_extra=default_extra
                        )
                    # DAH-2427: force-removing a CUDA workload can leave an orphaned kernel
                    # pinning the card (ghost GPU); cure it right here so the ghost never
                    # outlives the teardown and never reaches the next renter.
                    with _best_effort_delete_step(log, "sweep_wedged_gpus"):
                        await _sweep_wedged_gpus_after_teardown(ssh_client, log)

                with _best_effort_delete_step(log, "prune_images"):
                    await run_logged_rental_docker_sdk_operation(
                        operation="prune_images",
                        log_extra=default_extra,
                        call=docker_client.prune_images,
                    )

                if payload.local_volume:
                    with _best_effort_delete_step(
                        log, "remove_volume_local", volume_name=payload.local_volume
                    ):
                        await run_logged_rental_docker_sdk_operation(
                            operation="remove_volume",
                            log_extra=default_extra,
                            call=lambda: docker_client.remove_volume(
                                volume_name=payload.local_volume
                            ),
                            volume_name=payload.local_volume,
                            volume_role="local",
                        )

                if payload.external_volume:
                    with _best_effort_delete_step(
                        log, "remove_volume_external", volume_name=payload.external_volume
                    ):
                        await run_logged_rental_docker_sdk_operation(
                            operation="remove_volume",
                            log_extra=default_extra,
                            call=lambda: docker_client.remove_volume(
                                volume_name=payload.external_volume
                            ),
                            volume_name=payload.external_volume,
                            volume_role="external",
                        )
                        await self.remove_s3fs_volume_plugin(
                            ssh_client=ssh_client, volume_name=payload.external_volume
                        )

                log.info(
                    "Remove rented machine from redis",
                    container_name=payload.container_name,
                    local_volume=payload.local_volume,
                    external_volume=payload.external_volume,
                )
                with _best_effort_delete_step(log, "remove_rented_machine"):
                    await self.redis_service.remove_rented_machine(
                        executor_info, payload.container_name
                    )

                # Stop inspector only after the last rented pod leaves this executor.
                with _best_effort_delete_step(log, "inspector_stop"):
                    if (
                        settings.ENABLE_INSPECTOR
                        and not await self._has_rented_containers(executor_info)
                    ):
                        await self._run_inspector_collector_lifecycle(
                            ssh_client=ssh_client,
                            executor_info=executor_info,
                            action="stop",
                            default_extra={
                                **default_extra,
                                "container_name": payload.container_name,
                            },
                        )

                # Port release now handled by backend

                log.info("Deleted Docker Container", payload=str(payload))

                return self._container_deleted(payload)
        except Exception as e:
            log.error("Unknown Error delete_container", error=str(e), exc_info=True)

            # str(log_text) drops extra, hiding the cause from the backend
            return self._failed_delete(payload, msg=f"Unknown Error delete_container: {e}")

    async def install_jupyter_server(
        self,
        payload: InstallJupyterServerRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_uuid": payload.executor_id,
            "pod_id": payload.pod_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Install Jupyter server on pod",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)
        pkey = asyncssh.import_private_key(private_key)

        known_hosts_policy: asyncssh.SSHKnownHosts | None = None
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.ContainerCreationFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            async with asyncssh.connect(
                host=executor_info.address,
                port=executor_info.ssh_port,
                username=executor_info.ssh_username,
                client_keys=[pkey],
                known_hosts=known_hosts_policy,
            ) as ssh_client:
                jupyter_token = secrets.token_hex(16)
                jupyter_port = payload.jupyter_port_map[0]
                local_volume = payload.local_volume
                local_volume_path = payload.local_volume_path
                await self.run_jupyter(
                    ssh_client=ssh_client,
                    container_name=payload.container_name,
                    jupyter_token=jupyter_token,
                    jupyter_port=jupyter_port,
                    log_tag="jupyter",
                    log_extra=default_extra,
                    local_volume=local_volume,
                    local_volume_path=local_volume_path,
                )

                logger.info(
                    _m(
                        "Jupyter server installed",
                        extra=get_extra_info({
                            **default_extra,
                            "container_name": payload.container_name,
                            "jupyter_port": jupyter_port,
                        }),
                    ),
                )

                return JupyterServerInstalled(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    jupyter_url=f"http://{executor_info.address}:{payload.jupyter_port_map[1]}/lab?token={jupyter_token}",
                )
        except Exception as e:
            log_text = _m(
                "Failed install jupyter server",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return JupyterInstallationFailed(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
            )

    async def remove_ssh_keys(
        self,
        payload: RemoveSshPublicKeysRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "executor_uuid": payload.executor_id,
            "pod_id": payload.pod_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Remove ssh key(s) from pod",
                extra=get_extra_info({**default_extra}),
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)

        try:
            await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        if not payload.user_public_keys:
            log_text = _m(
                "ssh key Remove error: no public key",
                extra=get_extra_info({
                    **default_extra,
                    "container_name": payload.container_name,
                    "error": "No public keys",
                }),
            )
            logger.error(log_text)

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.NoSshKeys,
            )

        try:
            require_rental_docker_ssh_host_key(executor_info)
        except RentalDockerConnectionError as exc:
            log_text = _missing_rental_docker_host_key_log_text(default_extra, exc)
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            async with self.rental_docker_client_factory.connect(
                executor_info=executor_info,
                private_key=private_key,
            ) as docker_client:
                await self.remove_ssh_public_keys_with_rental_docker(
                    docker_client=docker_client,
                    container_name=payload.container_name,
                    public_keys=payload.user_public_keys,
                    log_tag=f"remove_ssh_keys_{payload.pod_id}",
                    log_extra=default_extra,
                )

                logger.info(
                    _m(
                        "Removed ssh key(s) from the container",
                        extra=get_extra_info({
                            **default_extra,
                            "container_name": payload.container_name,
                            "removed_keys": payload.user_public_keys,
                        }),
                    ),
                )

                return SshPubKeyRemoved(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    user_public_keys=payload.user_public_keys,
                )
        except Exception as e:
            log_text = _m(
                "Unknown Error remove_ssh_keys",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

    async def add_ssh_key(
        self,
        payload: AddSshPublicKeyRequest,
        executor_info: ExecutorSSHInfo,
        keypair: bittensor.Keypair,
        private_key: str,
    ):
        default_extra = {
            "miner_hotkey": payload.miner_hotkey,
            "pod_id": payload.pod_id,
            "executor_uuid": payload.executor_id,
            "executor_ip_address": executor_info.address,
            "executor_port": executor_info.port,
            "executor_ssh_username": executor_info.ssh_username,
            "executor_ssh_port": executor_info.ssh_port,
        }

        logger.info(
            _m(
                "Add ssh key to pod",
                extra=get_extra_info({**default_extra, "payload": str(payload)}),
            ),
        )

        private_key = self.ssh_service.decrypt_payload(keypair.ss58_address, private_key)

        known_hosts_policy: asyncssh.SSHKnownHosts | None = None
        try:
            known_hosts_policy = await self._prepare_known_hosts_policy(
                executor_info,
                payload.miner_hotkey,
                default_extra,
            )
        except AttestationError as exc:
            log_text = _m(
                "Attestation failed",
                extra=get_extra_info({**default_extra, "error": str(exc)}),
            )
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        if not payload.user_public_keys:
            log_text = _m(
                "ssh key Add error: no public key",
                extra=get_extra_info({
                    **default_extra,
                    "container_name": payload.container_name,
                    "error": "No public keys",
                }),
            )
            logger.error(log_text)

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.NoSshKeys,
            )

        try:
            require_rental_docker_ssh_host_key(executor_info)
        except RentalDockerConnectionError as exc:
            log_text = _missing_rental_docker_host_key_log_text(default_extra, exc)
            logger.error(log_text)
            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

        try:
            async with self.rental_docker_client_factory.connect(
                executor_info=executor_info,
                private_key=private_key,
            ) as docker_client:
                await self.add_ssh_public_keys_with_rental_docker(
                    docker_client=docker_client,
                    container_name=payload.container_name,
                    public_keys=payload.user_public_keys,
                    log_tag=f"add_ssh_key_{payload.pod_id}",
                    log_extra=default_extra,
                )

                logger.info(
                    _m(
                        "Added ssh key into Docker Container",
                        extra=get_extra_info({
                            **default_extra,
                            "container_name": payload.container_name
                        }),
                    ),
                )

                return SshPubKeyAdded(
                    miner_hotkey=payload.miner_hotkey,
                    executor_id=payload.executor_id,
                    pod_id=payload.pod_id,
                    workload_kind=payload.workload_kind,
                    user_public_keys=payload.user_public_keys,
                )
        except Exception as e:
            log_text = _m(
                "Failed add_ssh_key",
                extra=get_extra_info({**default_extra, "error": str(e)}),
            )
            logger.error(log_text, exc_info=True)

            return FailedContainerRequest(
                miner_hotkey=payload.miner_hotkey,
                executor_id=payload.executor_id,
                pod_id=payload.pod_id,
                workload_kind=payload.workload_kind,
                msg=str(log_text),
                error_type=FailedContainerErrorTypes.AddSSkeyFailed,
                error_code=FailedContainerErrorCodes.UnknownError,
            )

    def _get_preferred_ports(self, initial_port_count: int | None) -> list[int]:
        """Calculate preferred ports based on initial_port_count.

        - None: return all PREFERRED_POD_PORTS
        - Less than PREFERRED_POD_PORTS length: return limited list
        - More than PREFERRED_POD_PORTS length: return PREFERRED_POD_PORTS + sequential extras
        """
        if initial_port_count is None:
            return PREFERRED_POD_PORTS

        if initial_port_count <= len(PREFERRED_POD_PORTS):
            return PREFERRED_POD_PORTS[:initial_port_count]

        # Need more ports than available in PREFERRED_POD_PORTS
        max_port = max(PREFERRED_POD_PORTS)
        extra_count = initial_port_count - len(PREFERRED_POD_PORTS)
        extra_ports = [max_port + i for i in range(extra_count)]

        return list(PREFERRED_POD_PORTS) + extra_ports


async def _sweep_wedged_gpus_after_teardown(
    ssh_client: asyncssh.SSHClientConnection, log: "_BoundLog"
) -> None:
    """Cure any GPU the removed container left wedged; best-effort, caller guards errors.

    The signature is sampled twice, before and after a settle window, because the moment just
    after a SIGKILL is when a draining workload looks most like an orphaned kernel. A card that
    reads healthy immediately cannot be wedged, so the common teardown pays one cheap query
    pair and skips the wait entirely.
    """
    runner = SSHCommandRunner(ssh_client, max_retries=0)

    if not await query_wedged_gpu_uuids(runner):
        return

    await asyncio.sleep(GPU_WEDGE_SWEEP_SETTLE_SECONDS)
    wedged_uuids: list[str] = await query_wedged_gpu_uuids(runner)
    if not wedged_uuids:
        return

    for cure_outcome in await cure_wedged_gpus(runner, wedged_uuids):
        log.info(
            "Wedged GPU cure attempted after teardown",
            gpu_uuid=cure_outcome.gpu_uuid,
            cured=cure_outcome.cured,
            exit_code=cure_outcome.exit_code,
            output=cure_outcome.output,
            error=cure_outcome.error,
        )
