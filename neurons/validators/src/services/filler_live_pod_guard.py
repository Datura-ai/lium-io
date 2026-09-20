"""E-187 (DAH-3706 family): a filler never starts on GPUs a RUNNING customer pod holds — host truth.

The platform judges a filler launch against its own rows (lium-platform LivePodGuard). This is the
last check, on the host, right before the filler's `docker run`: every RUNNING `pod_*` container is
inspected for the GPUs it was given (HostConfig.DeviceRequests: pinned DeviceIDs, or Count=-1 for the
whole host) and the filler is refused when its GPU set intersects. It runs BEFORE the create's
container sweep, so a customer's container is never the thing removed to make room for a filler —
the case the platform cannot see is exactly a pod it has no row for (a host re-registered as a new
executor row, a row the platform lost).

Two outcomes, two `failure_step`s, because the platform (lium-platform#608 `handle_failed`) treats
them differently and either repo may deploy first:
- FILLER_LIVE_POD_GUARD_STEP ("filler_live_pod_guard"): a CONFIRMED overlap. The typed event
  FILLER_START_REFUSED_LIVE_POD is written; the platform closes the run STOPPED — a lost race, no
  FAILED row, no backoff — and counts `validator_refused`.
- FILLER_LIVE_POD_GUARD_UNREADABLE_STEP ("filler_live_pod_guard_unreadable"): the host could not be
  read — `docker ps` or `docker inspect` failed (non-zero exit), hung past the timeout, or printed a
  line the parser cannot trust. Still no filler this cycle (fail closed), but NO overlap event: the
  platform keeps today's FAILED + backoff path for it and counts `unreadable`, so a dockerd outage is
  neither a per-cycle retry loop nor a hit on the live-pod alert.

Two commands, not a pipeline: `docker ps -q … | xargs -r docker inspect …` reports xargs's exit (0)
when `docker ps` itself fails, and an empty stdout would read as "no pod" — fail OPEN. Each command
runs alone, bounded by the same timeout every neighbouring host read on the create path uses, and
its own exit status is checked.

The separator is emitted by a Go template ACTION, `{{"\\t"}}`: `docker inspect --format` copies text
outside actions verbatim, so a bare `\\t` in the template prints a backslash and a t (only `docker ps
--format` pre-processes `\\t`). Verified on docker 29.1.3: the r1 template printed `\\t` literally and
the parser saw no separator; a `pod_*` line WITHOUT the separator is therefore unreadable, never
"no GPU claim".
"""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass

import asyncssh
from services.const import POD_CONTAINER_PREFIX
from services.gpu_power_limit import NVIDIA_SMI_TIMEOUT_SECONDS

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

# The typed log event (`extra["event"]`) — the same name lium-platform writes for its refusals.
FILLER_START_REFUSED_LIVE_POD_EVENT = "FILLER_START_REFUSED_LIVE_POD"
# create_container's `failure_step` for a CONFIRMED overlap; the platform closes the run STOPPED on it.
FILLER_LIVE_POD_GUARD_STEP = "filler_live_pod_guard"
# create_container's `failure_step` when the host could not be read; the platform leaves that on the
# ordinary FAILED + backoff path (no FILLER_START_REFUSED_LIVE_POD event on either side).
FILLER_LIVE_POD_GUARD_UNREADABLE_STEP = "filler_live_pod_guard_unreadable"

# The bound every neighbouring host read on the create path uses (the prerun probe's timeout).
LIVE_POD_READ_TIMEOUT_SECONDS = NVIDIA_SMI_TIMEOUT_SECONDS

# Step 1: the ids of the RUNNING pod_* containers. The name filter is a substring match, so the
# prefix is re-checked on the parsed name after inspect. Runs ALONE so its exit status is seen.
LIVE_POD_IDS_CMD = "/usr/bin/docker ps -q --filter status=running --filter name=pod_"
# Step 2: one line per container: name, a TAB emitted by the template action, the device requests.
LIVE_POD_GPU_SETS_CMD_PREFIX = (
    "/usr/bin/docker inspect --format '{{.Name}}{{\"\\t\"}}{{json .HostConfig.DeviceRequests}}'"
)
_SEPARATOR = "\t"


class FillerRefusedLivePodError(RuntimeError):
    """A running pod_* holds GPUs the filler would take (a confirmed overlap)."""


class FillerLivePodListingUnreadableError(RuntimeError):
    """The host's running-pod listing could not be read or trusted; the filler is refused this cycle."""


@dataclass(frozen=True)
class LivePodGpuOverlap:
    pod_containers: list[str]
    gpu_overlap: list[str]  # physical GPU uuids; the pod's set when either side is whole-host


def live_pod_gpu_sets_cmd(container_ids: list[str]) -> str:
    return (
        LIVE_POD_GPU_SETS_CMD_PREFIX
        + " "
        + " ".join(shlex.quote(container_id) for container_id in container_ids)
    )


def parse_live_pod_gpu_sets(stdout: str) -> dict[str, frozenset[str] | None]:
    """`pod_*` name -> the GPU uuids it holds, or None when it holds the whole host.

    Non-pod names and pods created without `--gpus` (DeviceRequests null / []) are skipped. A `pod_*`
    line without the separator, or with a device set that cannot be parsed, is not "no claim" — it is a
    listing this guard cannot trust: FillerLivePodListingUnreadableError (fail closed).
    """
    pods: dict[str, frozenset[str] | None] = {}
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip("\r\n")
        if not line.strip():
            continue
        name, separator, requests_json = line.partition(_SEPARATOR)
        name = name.strip().lstrip("/")
        if not separator:
            if name.startswith(POD_CONTAINER_PREFIX):
                raise FillerLivePodListingUnreadableError(
                    f"Filler refused: the running pod listing has no separator on {line[:120]!r}"
                )
            continue  # not a pod, whatever the shape
        if not name.startswith(POD_CONTAINER_PREFIX):
            continue
        held = _gpu_set(requests_json.strip())
        if held is _UNPARSEABLE:
            raise FillerLivePodListingUnreadableError(
                f"Filler refused: the device requests of {name} could not be read: {requests_json[:120]!r}"
            )
        if held is not None and not held:
            continue  # created without --gpus: no GPU claim, nothing a filler could collide with
        pods[name] = held
    return pods


_UNPARSEABLE = object()


def _gpu_set(requests_json: str):
    """frozenset of uuids; None = the whole host; frozenset() = no GPU claim; _UNPARSEABLE."""
    try:
        requests = json.loads(requests_json or "null")
    except ValueError:
        return _UNPARSEABLE
    if requests is None or requests == []:
        # `null` / `[]`: created without --gpus — no GPU claim at all, not a whole-host one
        return frozenset()
    if not isinstance(requests, list):
        return _UNPARSEABLE
    uuids: set[str] = set()
    for request in requests:
        if not isinstance(request, dict):
            return _UNPARSEABLE
        device_ids = request.get("DeviceIDs") or []
        if device_ids:
            uuids.update(str(device_id) for device_id in device_ids)
        else:
            # Count=-1 (--gpus all) is every GPU on the host; a positive Count without ids means
            # docker picked the cards and we cannot tell which — the widest claim either way
            return None
    return frozenset(uuids)


def find_live_pod_gpu_overlap(
    filler_gpu_uuids: list[str] | None,
    live_pods: dict[str, frozenset[str] | None],
) -> LivePodGpuOverlap | None:
    """Which running pods hold GPUs the filler would take; None when the launch is clean.

    An empty / None filler set means the filler takes the whole host, so ANY pod with a GPU claim
    overlaps it; a whole-host pod (None) overlaps any filler.
    """
    planned: set[str] | None = set(filler_gpu_uuids) if filler_gpu_uuids else None
    pod_containers: list[str] = []
    overlap: set[str] = set()
    for name in sorted(live_pods):
        held = live_pods[name]
        if held is None:
            pod_containers.append(name)
            overlap.update(planned or ())
            continue
        if planned is None:
            if held:
                pod_containers.append(name)
                overlap.update(held)
            continue
        taken = planned & held
        if taken:
            pod_containers.append(name)
            overlap.update(taken)
    if not pod_containers:
        return None
    return LivePodGpuOverlap(pod_containers=pod_containers, gpu_overlap=sorted(overlap))


async def _read(ssh_client: asyncssh.SSHClientConnection, command: str, what: str) -> str:
    """Run one host read, bounded; a non-zero exit or a timeout is an unreadable listing."""
    try:
        result = await ssh_client.run(command, timeout=LIVE_POD_READ_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise FillerLivePodListingUnreadableError(
            f"Filler refused: {what} timed out after {LIVE_POD_READ_TIMEOUT_SECONDS}s"
        ) from exc
    if result.exit_status != 0:
        raise FillerLivePodListingUnreadableError(
            f"Filler refused: {what} could not be read (exit {result.exit_status}): "
            f"{(result.stderr or '').strip()[:300]}"
        )
    return result.stdout or ""


async def read_live_pod_gpu_sets(
    ssh_client: asyncssh.SSHClientConnection,
) -> dict[str, frozenset[str] | None]:
    """The RUNNING pod_* containers and their GPU sets, or FillerLivePodListingUnreadableError."""
    ids_stdout = await _read(ssh_client, LIVE_POD_IDS_CMD, "the running pod listing (docker ps)")
    container_ids = [line.strip() for line in ids_stdout.splitlines() if line.strip()]
    if not container_ids:
        return {}  # docker ps answered (exit 0) with no running pod_*: nothing to inspect
    inspect_stdout = await _read(
        ssh_client, live_pod_gpu_sets_cmd(container_ids), "the running pod inspect"
    )
    return parse_live_pod_gpu_sets(inspect_stdout)


async def assert_no_live_pod_on_filler_gpus(
    ssh_client: asyncssh.SSHClientConnection,
    *,
    filler_gpu_uuids: list[str] | None,
    executor_id: str,
    filler_pod_id: str,
    default_extra: dict,
) -> None:
    """Refuse the filler when a running pod_* holds its GPUs (event + FillerRefusedLivePodError).

    An unreadable host raises FillerLivePodListingUnreadableError instead — logged, no event.
    """
    try:
        live_pods = await read_live_pod_gpu_sets(ssh_client)
    except FillerLivePodListingUnreadableError as exc:
        logger.warning(
            _m(
                "Filler refused: the host's running pods could not be read",
                extra=get_extra_info(
                    {
                        **default_extra,
                        "executor_uuid": executor_id,
                        "filler_pod_id": filler_pod_id,
                        "filler_gpu_uuids": list(filler_gpu_uuids or []),
                        "detail": str(exc)[:300],
                    }
                ),
            )
        )
        raise
    overlap = find_live_pod_gpu_overlap(filler_gpu_uuids, live_pods)
    if overlap is None:
        return
    logger.warning(
        _m(
            "Filler start refused: a running customer pod holds its GPUs",
            extra=get_extra_info(
                {
                    **default_extra,
                    "event": FILLER_START_REFUSED_LIVE_POD_EVENT,
                    # the validator uuid — the ONE key both halves write for it (lium-platform#608
                    # writes executor_uuid too; its executor_id is the DB PK this side never sees)
                    "executor_uuid": executor_id,
                    "filler_pod_id": filler_pod_id,
                    "pod_containers": overlap.pod_containers,
                    "gpu_overlap": overlap.gpu_overlap,
                    "filler_gpu_uuids": list(filler_gpu_uuids or []),
                }
            ),
        )
    )
    raise FillerRefusedLivePodError(
        f"Filler refused: running {', '.join(overlap.pod_containers)} hold(s) GPU(s) "
        f"{', '.join(overlap.gpu_overlap) or 'the whole host'} the filler would take."
    )
