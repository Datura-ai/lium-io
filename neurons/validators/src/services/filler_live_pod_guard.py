"""E-187 (DAH-3706 family): a filler never starts on GPUs a RUNNING customer pod holds — host truth.

The platform judges a filler launch against its own rows (lium-platform LivePodGuard). This is the
last check, on the host, right before the filler's `docker run`: every RUNNING `pod_*` container is
inspected for the GPUs it was given (HostConfig.DeviceRequests: pinned DeviceIDs, or Count=-1 for the
whole host) and the filler is refused when its GPU set intersects. It runs BEFORE the create's
container sweep, so a customer's container is never the thing removed to make room for a filler —
the case the platform cannot see is exactly a pod it has no row for (a host re-registered as a new
executor row, a row the platform lost).

Fail closed: an unreadable listing refuses the filler (no filler this cycle) rather than guessing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import asyncssh
from services.const import POD_CONTAINER_PREFIX

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

# The typed log event (`extra["event"]`) — the same name lium-platform writes for its refusals.
FILLER_START_REFUSED_LIVE_POD_EVENT = "FILLER_START_REFUSED_LIVE_POD"
# create_container's `failure_step` for this refusal; the platform closes the run STOPPED on it.
FILLER_LIVE_POD_GUARD_STEP = "filler_live_pod_guard"

# One command: the RUNNING pod_* containers and the device requests each was created with. The name
# filter is a substring match, so the prefix is re-checked on the parsed name. `xargs -r` prints
# nothing when no pod runs.
LIVE_POD_GPU_SETS_CMD = (
    "/usr/bin/docker ps -q --filter status=running --filter name=pod_ "
    "| xargs -r /usr/bin/docker inspect --format '{{.Name}}\t{{json .HostConfig.DeviceRequests}}'"
)


class FillerRefusedLivePodError(RuntimeError):
    """A running pod_* holds GPUs the filler would take, or the host could not be read."""


@dataclass(frozen=True)
class LivePodGpuOverlap:
    pod_containers: list[str]
    gpu_overlap: list[str]  # physical GPU uuids; the pod's set when either side is whole-host


def parse_live_pod_gpu_sets(stdout: str) -> dict[str, frozenset[str] | None]:
    """`pod_*` name -> the GPU uuids it holds, or None when it holds the whole host.

    Non-pod names and pods with no GPU claim are skipped. A pod whose device set cannot be read
    (malformed JSON) counts as whole-host: an unknown claim is treated as the widest one.
    """
    pods: dict[str, frozenset[str] | None] = {}
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        name, _, requests_json = line.partition("\t")
        name = name.lstrip("/")
        if not name.startswith(POD_CONTAINER_PREFIX):
            continue
        held = _gpu_set(requests_json)
        if held is not None and not held:
            continue  # created without --gpus: no GPU claim, nothing a filler could collide with
        pods[name] = held
    return pods


def _gpu_set(requests_json: str) -> frozenset[str] | None:
    try:
        requests = json.loads(requests_json or "null")
    except ValueError:
        return None
    if not requests:
        # `null` / `[]`: created without --gpus — no GPU claim at all, not a whole-host one
        return frozenset()
    uuids: set[str] = set()
    for request in requests:
        if not isinstance(request, dict):
            return None
        device_ids = request.get("DeviceIDs") or []
        if device_ids:
            uuids.update(str(device_id) for device_id in device_ids)
        elif request.get("Count", 0) == -1 or request.get("Count") is None:
            return None  # every GPU on the host
        else:
            # a positive Count without ids: docker picked the cards, we cannot tell which
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


async def assert_no_live_pod_on_filler_gpus(
    ssh_client: asyncssh.SSHClientConnection,
    *,
    filler_gpu_uuids: list[str] | None,
    executor_id: str,
    filler_pod_id: str,
    default_extra: dict,
) -> None:
    """Refuse the filler (event + FillerRefusedLivePodError) when a running pod_* holds its GPUs."""
    result = await ssh_client.run(LIVE_POD_GPU_SETS_CMD)
    if result.exit_status != 0:
        raise FillerRefusedLivePodError(
            "Filler refused: the running pod listing could not be read "
            f"(exit {result.exit_status}): {(result.stderr or '').strip()[:300]}"
        )
    overlap = find_live_pod_gpu_overlap(
        filler_gpu_uuids, parse_live_pod_gpu_sets(result.stdout or "")
    )
    if overlap is None:
        return
    logger.warning(
        _m(
            "Filler start refused: a running customer pod holds its GPUs",
            extra=get_extra_info(
                {
                    **default_extra,
                    "event": FILLER_START_REFUSED_LIVE_POD_EVENT,
                    "executor_id": executor_id,
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
