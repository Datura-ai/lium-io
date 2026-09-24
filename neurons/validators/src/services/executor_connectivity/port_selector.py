from collections.abc import Iterable

from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import PORT_RANGE_BUCKET_WIDTH
from services.executor_connectivity.models import PortPair, PortRangeResult
from services.port_utils import get_all_ports


def declared_ports(executor_info: ExecutorSSHInfo) -> list[PortPair]:
    """Every port pair the executor declares, ssh port excluded, sorted by internal port."""
    return [
        PortPair(internal, external)
        for internal, external in get_all_ports(
            executor_info.port_range, executor_info.port_mappings, executor_info.ssh_port
        )
    ]


def tally_port_ranges(
    declared: Iterable[PortPair], probed: Iterable[PortPair], answered: Iterable[PortPair]
) -> tuple[PortRangeResult, ...]:
    """Declared, probed and answered counts per declared range, ascending.

    Ports are grouped by external port into PORT_RANGE_BUCKET_WIDTH-wide buckets, and each tally
    names the lowest and highest declared port in its bucket: a range narrower than a bucket is
    one tally, a wide range is split so a forward that covers only part of it shows which part.
    """
    probed_ext = {p.external for p in probed}
    answered_ext = {p.external for p in answered}
    buckets: dict[int, list[int]] = {}
    for p in declared:
        buckets.setdefault(p.external // PORT_RANGE_BUCKET_WIDTH, []).append(p.external)
    return tuple(
        PortRangeResult(
            first=min(ports),
            last=max(ports),
            declared=len(ports),
            probed=sum(e in probed_ext for e in ports),
            answered=sum(e in answered_ext for e in ports),
        )
        for _, ports in sorted(buckets.items())
    )


def sample_ports(ports: list[PortPair], size: int) -> list[PortPair]:
    """Pick at most `size` ports to probe from `ports` (sorted ascending), spread over the whole list.

    A declaration of `size` ports or fewer is probed in full, exactly as before. A larger one keeps
    its lowest `size // 2` ports (so a host whose forwarded ports sit at the bottom of a wide range
    still passes as it does today) and fills the rest of the budget with ports evenly spaced across
    the remainder, always including the highest declared port. A host that declares 40000-65535
    but forwards only 60000 and up is therefore probed where its ports are open.

    Same probe count as the lowest-`size` cut it replaces, and deterministic: an unchanged
    declaration is probed on the same ports every cycle, so a node does not flap between cycles.
    """
    if size <= 0:
        return []
    if len(ports) <= size:
        return list(ports)
    head = size // 2
    tail = size - head
    rest = ports[head:]
    if tail == 1:
        return ports[:head] + [rest[0]]
    last = len(rest) - 1
    return ports[:head] + [rest[i * last // (tail - 1)] for i in range(tail)]


class PortSelector:
    """Selects which ports to verify."""

    def select(self, executor_info: ExecutorSSHInfo, size: int, unavailable_ports: set[int]) -> list[PortPair]:
        """Select ports to check, skipping external ports already taken by pods or fillers."""
        all_ports = get_all_ports(executor_info.port_range, executor_info.port_mappings, executor_info.ssh_port)
        available_ports = [
            PortPair(internal, external) for internal, external in all_ports if external not in unavailable_ports
        ]
        return sample_ports(available_ports, size)
