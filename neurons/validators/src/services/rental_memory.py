"""DAH-3798 — the memory limit a rental container gets: the host's RAM less a reserve for the host.

The backend sends ``memory_gb`` = (host RAM − 2 GiB) × the pod's GPU share, or 0 for a pod row that
never got a size. A flat 2 GiB is nothing on a large guest: in ticket-0355 (21 Sep 2026) a renter's
job on an 8x H200 CVM took the guest to 99.7 % RAM with 5.6 GB left, the guest's own agent and the
executor starved beside it, the validator's scrape timed out and the node went dark for 25 h while
the pod still read RUNNING. The host keeps RENTAL_MEMORY_RESERVE_PERCENT of its RAM (at least
RENTAL_MEMORY_RESERVE_MIN_GB) and a container gets at most its GPU share of the rest, so the
container's own cgroup runs out first and the kernel's OOM killer acts inside it.

The RAM is read from the host at create time (``/proc/meminfo`` of the machine the container runs on,
which on a CVM is the guest). A read that fails keeps the backend's value, as before this module.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from core.config import settings
from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

KIB_PER_GIB = 1024 * 1024
# The smallest limit ever set: below this a container cannot start a shell.
MIN_RENTAL_MEMORY_GB = 1
# MemTotal in KiB, then the number of GPU device nodes (/dev/nvidia0, /dev/nvidia1, …).
HOST_MEMORY_PROBE_CMD = (
    "awk '/^MemTotal:/ {print $2}' /proc/meminfo; ls -d /dev/nvidia[0-9]* 2>/dev/null | wc -l"
)

SOURCE_CAP_DISABLED = "cap_disabled"
SOURCE_HOST_UNKNOWN = "host_unknown"
SOURCE_REQUESTED = "requested"
SOURCE_CLAMPED = "clamped_to_host_reserve"
SOURCE_HOST_MINUS_RESERVE = "host_minus_reserve"


@dataclass(frozen=True)
class RentalMemoryLimit:
    """What `docker run` gets for memory, and why.

    ``limit_gb`` is the ``--memory`` value (None = no limit). ``cap_enabled`` is False only when
    RENTAL_MEMORY_CAP_ENABLED is off; the container then gets exactly the flags it got before DAH-3798.
    """

    limit_gb: int | None
    requested_gb: int | None
    host_total_gb: float | None
    reserve_gb: int | None
    ceiling_gb: int | None
    gpu_share: float
    source: str
    cap_enabled: bool = True

    @property
    def swap_off(self) -> bool:
        return self.cap_enabled and self.limit_gb is not None

    def describe(self) -> str:
        if self.limit_gb is None:
            return "Memory limit: none (the host's RAM could not be read and no size was requested)"
        if self.host_total_gb is None or self.reserve_gb is None:
            return f"Memory limit: {self.limit_gb} GiB"
        swap = ", swap off" if self.swap_off else ""
        return (
            f"Memory limit: {self.limit_gb} GiB of the host's {self.host_total_gb:.0f} GiB "
            f"({self.reserve_gb} GiB kept for the host{swap})"
        )

    def log_fields(self) -> dict:
        return {
            "memory_limit_gb": self.limit_gb,
            "memory_requested_gb": self.requested_gb,
            "host_memory_gb": round(self.host_total_gb, 2)
            if self.host_total_gb is not None
            else None,
            "host_memory_reserve_gb": self.reserve_gb,
            "memory_ceiling_gb": self.ceiling_gb,
            "gpu_share": self.gpu_share,
            "memory_limit_source": self.source,
            "swap_off": self.swap_off,
        }


def host_memory_reserve_gb(
    host_total_gb: float, *, reserve_min_gb: int, reserve_percent: float
) -> int:
    """The RAM the host keeps: ``reserve_percent`` of it, rounded up to a whole GiB, never under ``reserve_min_gb``."""
    percent = max(0.0, float(reserve_percent))
    return max(int(reserve_min_gb), math.ceil(host_total_gb * percent / 100))


def gpu_share_of_host(gpu_uuids: list[str] | None, host_gpu_count: int | None) -> float:
    """The pod's share of the host's GPUs; a whole-host rental (no UUIDs) or an unknown count is 1."""
    requested = len(gpu_uuids or [])
    if requested <= 0 or not host_gpu_count or requested >= host_gpu_count:
        return 1.0
    return requested / host_gpu_count


def rental_memory_limit(
    requested_gb: int | None,
    host_total_kib: int | None,
    *,
    gpu_share: float = 1.0,
    reserve_min_gb: int,
    reserve_percent: float,
) -> RentalMemoryLimit:
    """The limit: the backend's ``requested_gb``, never above the GPU share of (host RAM − reserve).

    A missing or zero ``requested_gb`` gets that ceiling, so no rental container runs without a limit
    once the host's RAM is known. An unknown host RAM keeps ``requested_gb`` unchanged.
    """
    requested = int(requested_gb) if requested_gb and int(requested_gb) > 0 else None
    share = gpu_share if 0 < gpu_share <= 1 else 1.0
    if not host_total_kib or host_total_kib <= 0:
        return RentalMemoryLimit(
            limit_gb=requested,
            requested_gb=requested,
            host_total_gb=None,
            reserve_gb=None,
            ceiling_gb=None,
            gpu_share=share,
            source=SOURCE_HOST_UNKNOWN,
        )

    host_total_gb = host_total_kib / KIB_PER_GIB
    reserve_gb = host_memory_reserve_gb(
        host_total_gb, reserve_min_gb=reserve_min_gb, reserve_percent=reserve_percent
    )
    ceiling_gb = max(MIN_RENTAL_MEMORY_GB, math.floor((host_total_gb - reserve_gb) * share))
    if requested is None:
        limit_gb, source = ceiling_gb, SOURCE_HOST_MINUS_RESERVE
    elif requested > ceiling_gb:
        limit_gb, source = ceiling_gb, SOURCE_CLAMPED
    else:
        limit_gb, source = requested, SOURCE_REQUESTED
    return RentalMemoryLimit(
        limit_gb=limit_gb,
        requested_gb=requested,
        host_total_gb=host_total_gb,
        reserve_gb=reserve_gb,
        ceiling_gb=ceiling_gb,
        gpu_share=share,
        source=source,
    )


def parse_host_memory_probe(stdout: str) -> tuple[int | None, int | None]:
    """``HOST_MEMORY_PROBE_CMD`` output → (MemTotal KiB, GPU device count); None for a line that is not a number."""
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]

    def _int(index: int) -> int | None:
        if index >= len(lines) or not lines[index].isdigit():
            return None
        return int(lines[index])

    return _int(0), _int(1)


async def resolve_rental_memory_limit(
    ssh_client,
    *,
    requested_gb: int | None,
    gpu_uuids: list[str] | None,
    log_extra: dict,
) -> RentalMemoryLimit:
    """Read the host's RAM over the rent's SSH connection and compute the limit for this container."""
    if not settings.RENTAL_MEMORY_CAP_ENABLED:
        requested = requested_gb or None
        return RentalMemoryLimit(
            limit_gb=requested,
            requested_gb=requested,
            host_total_gb=None,
            reserve_gb=None,
            ceiling_gb=None,
            gpu_share=1.0,
            source=SOURCE_CAP_DISABLED,
            cap_enabled=False,
        )

    host_total_kib: int | None = None
    host_gpu_count: int | None = None
    try:
        result = await ssh_client.run(HOST_MEMORY_PROBE_CMD)
        if result.exit_status == 0:
            host_total_kib, host_gpu_count = parse_host_memory_probe(result.stdout or "")
    except Exception as exc:  # a failed read keeps the backend's value; the rent goes on
        logger.warning(
            _m(
                "rental_memory_probe_failed",
                extra=get_extra_info({**log_extra, "error": str(exc)}),
            )
        )

    limit = rental_memory_limit(
        requested_gb,
        host_total_kib,
        gpu_share=gpu_share_of_host(gpu_uuids, host_gpu_count),
        reserve_min_gb=settings.RENTAL_MEMORY_RESERVE_MIN_GB,
        reserve_percent=settings.RENTAL_MEMORY_RESERVE_PERCENT,
    )
    logger.info(
        _m("rental_memory_limit", extra=get_extra_info({**log_extra, **limit.log_fields()}))
    )
    return limit
