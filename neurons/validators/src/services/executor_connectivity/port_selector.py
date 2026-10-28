from collections.abc import Iterable

from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import PORT_RANGE_BUCKET_WIDTH
from services.executor_connectivity.models import PortPair, PortRangeResult
from services.port_utils import get_all_ports

MIN_PORT = 1
MAX_PORT = 65535


def declared_ports(executor_info: ExecutorSSHInfo) -> list[PortPair]:
    """Every port pair the executor declares, ssh port excluded, sorted by internal port."""
    return [
        PortPair(internal, external)
        for internal, external in get_all_ports(
            executor_info.port_range, executor_info.port_mappings, executor_info.ssh_port
        )
    ]


def tally_port_ranges(
    declared: Iterable[PortPair],
    probed: Iterable[PortPair],
    answered: Iterable[PortPair],
) -> tuple[PortRangeResult, ...]:
    """Declared, probed and answered counts per declared range, ascending.

    Counts distinct external ports in 1-65535, and only answers among the probes, so a
    duplicated port or two mappings sharing one external port are counted once and
    `answered <= probed <= declared` holds in every tally. Ports are grouped into
    PORT_RANGE_BUCKET_WIDTH-wide buckets, and each tally names the lowest and highest declared port
    in its bucket: a range narrower than a bucket is one tally, a wide range is split so a forward
    that covers only part of it shows which part.
    """
    probed_ext = {p.external for p in probed}
    answered_ext = {p.external for p in answered} & probed_ext
    buckets: dict[int, list[int]] = {}
    for external in sorted({p.external for p in declared if MIN_PORT <= p.external <= MAX_PORT}):
        buckets.setdefault(external // PORT_RANGE_BUCKET_WIDTH, []).append(external)

    def tally(ports: list[int]) -> PortRangeResult:
        return PortRangeResult(
            first=ports[0],
            last=ports[-1],
            declared=len(ports),
            probed=sum(e in probed_ext for e in ports),
            answered=sum(e in answered_ext for e in ports),
        )

    return tuple(tally(ports) for _, ports in sorted(buckets.items()))


class PortSelector:
    """Selects which ports to verify: the lowest `size` free declared ports."""

    def select(
        self,
        executor_info: ExecutorSSHInfo,
        size: int,
        unavailable_ports: set[int],
        declared: list[PortPair],
    ) -> list[PortPair]:
        """Select ports to check from `declared_ports(executor_info)`, skipping external ports
        already taken by pods or fillers."""
        available_ports = [p for p in declared if p.external not in unavailable_ports]
        return available_ports[:size]
