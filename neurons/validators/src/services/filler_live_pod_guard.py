"""E-187 (DAH-3706 family): a filler never starts on GPUs a LIVE customer pod holds — host truth.

The platform judges a filler launch against its own rows (lium-platform LivePodGuard). This is the
last check, on the host, right before the filler's `docker run`: every LIVE `pod_*` container —
docker state running, restarting or paused; every Lium pod runs with restart_policy unless-stopped,
so a crash-looping customer pod is `restarting` most of each cycle while still holding the rental
and its devices — is inspected for the GPUs it was given (HostConfig.DeviceRequests: pinned
DeviceIDs, or Count=-1 for the whole host) and the filler is refused when its GPU set intersects. It runs BEFORE the create's
container sweep, so a customer's container is never the thing removed to make room for a filler —
the case the platform cannot see is exactly a pod it has no row for (a host re-registered as a new
executor row, a row the platform lost).

Two outcomes, two `failure_step`s, because the platform (lium-platform#608 `handle_failed`) treats
them differently and either repo may deploy first:
- FILLER_LIVE_POD_GUARD_STEP ("filler_live_pod_guard"): a CONFIRMED overlap. The typed event
  FILLER_START_REFUSED_LIVE_POD is written; the platform closes the run STOPPED — a lost race, no
  FAILED row, no backoff — and counts `validator_refused`.
- FILLER_LIVE_POD_GUARD_UNREADABLE_STEP ("filler_live_pod_guard_unreadable"): the host could not be
  read — `docker ps` or `docker inspect` failed (non-zero exit), hung past the timeout, the SSH read
  itself raised (connection lost, session refused, socket error — anything that is not a verdict),
  or printed a line the parser cannot trust: no separator, an unparseable claim, or a `pod_*` whose
  GPU claim this guard cannot see (DeviceRequests null / [], or DeviceIDs that are not GPU-<uuid>s,
  e.g. a CDI `nvidia.com/gpu=0`). Lium's own create paths ALWAYS emit one DeviceRequest with GPU
  uuids or Count=-1 (rental_docker_sdk.build_gpu_docker_config; the legacy `--gpus` flag), so on a
  Lium host such a `pod_*` is a container whose GPU use is invisible to us, not a pod with no GPUs —
  fail closed. Still no filler this cycle, but NO overlap event: the platform keeps today's FAILED +
  backoff path for it and counts `unreadable`, so a dockerd / SSH outage is neither a per-cycle retry
  loop nor a hit on the live-pod alert.

Two commands, not a pipeline: `docker ps -q … | xargs -r docker inspect …` reports xargs's exit (0)
when `docker ps` itself fails, and an empty stdout would read as "no pod" — fail OPEN. Each command
runs alone, bounded by the same timeout every neighbouring host read on the create path uses, and
its own exit status is checked.

The separator is emitted by a Go template ACTION, `{{"\\t"}}`: `docker inspect --format` copies text
outside actions verbatim, so a bare `\\t` in the template prints a backslash and a t (only `docker ps
--format` pre-processes `\\t`). A `pod_*` line WITHOUT the separator is therefore unreadable, never
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

# Step 1: the ids of the LIVE pod_* containers: running, restarting (a crash-looping pod under
# unless-stopped takes its GPUs back within seconds) or paused — same-key filters OR. NOT exited /
# created: a stale exited pod_* the sweep would remove must not refuse the filler every cycle. The
# name filter is a substring match, so the prefix is re-checked on the parsed name after inspect.
# Runs ALONE so its exit status is seen.
LIVE_POD_IDS_CMD = (
    "/usr/bin/docker ps -q --filter status=running --filter status=restarting --filter status=paused"
    " --filter name=pod_"
)
# Step 2: one line per container: name, a TAB emitted by the template action, the device requests.
LIVE_POD_GPU_SETS_CMD_PREFIX = (
    "/usr/bin/docker inspect --format '{{.Name}}{{\"\\t\"}}{{json .HostConfig.DeviceRequests}}'"
)
_SEPARATOR = "\t"


class FillerRefusedLivePodError(RuntimeError):
    """A live pod_* holds GPUs the filler would take (a confirmed overlap)."""


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

    Non-pod names are skipped. A `pod_*` line without the separator, with a device set that cannot be
    parsed, with NO device request (null / []: a Lium pod always has one), or with device ids that are
    not GPU-<uuid>s (CDI) is not "no claim" — it is a claim this guard cannot read:
    FillerLivePodListingUnreadableError (fail closed).
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
                    f"Filler refused: the live pod listing has no separator on {line[:120]!r}"
                )
            continue  # not a pod, whatever the shape
        if not name.startswith(POD_CONTAINER_PREFIX):
            continue
        pods[name] = _gpu_set(name, requests_json.strip())
    return pods


_GPU_UUID_PREFIX = "GPU-"


def _gpu_set(name: str, requests_json: str) -> frozenset[str] | None:
    """frozenset of GPU uuids; None = the whole host; raises FillerLivePodListingUnreadableError when the claim cannot be read."""

    def unreadable() -> FillerLivePodListingUnreadableError:
        return FillerLivePodListingUnreadableError(
            f"Filler refused: the device requests of {name} could not be read: {requests_json[:120]!r}"
        )

    try:
        requests = json.loads(requests_json or "null")
    except ValueError:
        raise unreadable() from None
    if requests is None or requests == []:
        # `null` / `[]`: no DeviceRequest at all. A Lium pod always carries one (uuids or Count=-1),
        # so this is a pod_* whose GPU use we cannot see (env-only attachment) — not "no GPUs".
        raise unreadable()
    if not isinstance(requests, list):
        raise unreadable()
    uuids: set[str] = set()
    for request in requests:
        if not isinstance(request, dict):
            raise unreadable()
        device_ids = request.get("DeviceIDs") or []
        if device_ids:
            if not all(str(device_id).startswith(_GPU_UUID_PREFIX) for device_id in device_ids):
                # CDI names / indices can never intersect the filler's GPU uuids: unreadable, not disjoint
                raise unreadable()
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
    """Which live pods hold GPUs the filler would take; None when the launch is clean.

    An empty / None filler set means the filler takes the whole host, so ANY live pod overlaps it;
    a whole-host pod (None) overlaps any filler.
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
    """Run one host read, bounded; a non-zero exit, a timeout or a raising SSH read is unreadable.

    Anything the read raises is a host that could not be read (asyncssh.ConnectionLost /
    ChannelOpenError / DisconnectError, OSError, …) — the same rule as the prerun probe next door. It
    must never surface under the confirmed-overlap step, which the platform closes STOPPED and counts
    as a live-pod refusal.
    """
    try:
        result = await ssh_client.run(command, timeout=LIVE_POD_READ_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise FillerLivePodListingUnreadableError(
            f"Filler refused: {what} timed out after {LIVE_POD_READ_TIMEOUT_SECONDS}s"
        ) from exc
    except Exception as exc:
        raise FillerLivePodListingUnreadableError(
            f"Filler refused: {what} could not be read ({type(exc).__name__}: {str(exc)[:200]})"
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
    """The LIVE (running / restarting / paused) pod_* containers and their GPU sets, or unreadable."""
    ids_stdout = await _read(ssh_client, LIVE_POD_IDS_CMD, "the live pod listing (docker ps)")
    container_ids = [line.strip() for line in ids_stdout.splitlines() if line.strip()]
    if not container_ids:
        return {}  # docker ps answered (exit 0) with no live pod_*: nothing to inspect
    inspect_stdout = await _read(
        ssh_client, live_pod_gpu_sets_cmd(container_ids), "the live pod inspect"
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
    """Refuse the filler when a live pod_* holds its GPUs (event + FillerRefusedLivePodError).

    An unreadable host raises FillerLivePodListingUnreadableError instead — logged, no event.
    """
    try:
        live_pods = await read_live_pod_gpu_sets(ssh_client)
    except FillerLivePodListingUnreadableError as exc:
        logger.warning(
            _m(
                "Filler refused: the host's live pods could not be read",
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
            "Filler start refused: a live customer pod holds its GPUs",
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
        f"Filler refused: live {', '.join(overlap.pod_containers)} hold(s) GPU(s) "
        f"{', '.join(overlap.gpu_overlap) or 'the whole host'} the filler would take."
    )
