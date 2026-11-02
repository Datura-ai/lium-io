"""DAH-5269: record an AWS EFA fabric next to the RoCE measurement.

EFA (libfabric, SRD) is a third cluster fabric beside InfiniBand and RoCE. Its ports report neither
an InfiniBand nor an Ethernet link layer, so nothing in `roce_link_probe` sees them; the kernel
driver the scrape now reports per port (`driver == "efa"`) is what names them.

Runs where `roce_link_probe.measure_and_attach` runs, over the same results, and like it never
touches a rented host and never lets an error reach the caller.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from services.task.models import JobResult

from core.utils import _m, get_extra_info

logger = logging.getLogger(__name__)

EFA_DRIVER = "efa"
SPEC_KEY = "efa_fabric"


def _is_active(port: dict) -> bool:
    return str(port.get("state") or "").split(":")[-1].strip().upper() == "ACTIVE"


def _is_efa(port: dict) -> bool:
    return str(port.get("driver") or "").strip().lower() == EFA_DRIVER


def efa_device_count(result: JobResult) -> int:
    """How many EFA devices this host holds with a live port; 0 when it holds none."""
    return len(
        {
            port.get("device")
            for port in (result.spec or {}).get("infiniband_ports") or []
            if _is_active(port) and _is_efa(port)
        }
    )


def measure_and_attach(results: list[JobResult], log_extra: dict) -> None:
    """Write the EFA fact into the specs of every free host that carries one.

    shortcut: presence and live device count only, no pair bandwidth. Upgrade to an fi_pingpong /
    2-node NCCL pair run (shaped like the RoCE pair sweep) once a probe image ships libfabric and an
    EFA pair is available to test against; until then `gigabits_per_second` stays None.
    """
    try:
        measured_at: str = datetime.now(UTC).isoformat()
        hosts: int = 0
        for result in results:
            if result.is_rented:
                continue
            device_count = efa_device_count(result)
            if not device_count:
                continue
            result.spec[SPEC_KEY] = {
                "device_count": device_count,
                "gigabits_per_second": None,
                "measured_at": measured_at,
            }
            hosts += 1
        if hosts:
            logger.info(
                _m("EFA fabric recorded", extra=get_extra_info({**log_extra, "hosts": hosts})),
            )
    except Exception as error:
        logger.warning(
            _m("EFA link probe failed", extra=get_extra_info({**log_extra, "error": str(error)})),
        )
