from collections.abc import Iterable

from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import PORT_RANGE_BUCKET_WIDTH, PORT_RANGE_MAX_ENTRIES
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
    declared: Iterable[PortPair], probed: Iterable[PortPair], answered: Iterable[PortPair]
) -> tuple[PortRangeResult, ...]:
    """Declared, probed and answered counts per declared range, ascending.

    Counts distinct external ports in 1-65535, so a duplicated port or two mappings sharing one
    external port are counted once and `answered <= probed <= declared` holds in every tally.
    Ports are grouped into PORT_RANGE_BUCKET_WIDTH-wide buckets, and each tally names the lowest
    and highest declared port in its bucket: a range narrower than a bucket is one tally, a wide
    range is split so a forward that covers only part of it shows which part. Past
    PORT_RANGE_MAX_ENTRIES buckets the rest are summed into one final `other` tally.
    """
    probed_ext = {p.external for p in probed}
    answered_ext = {p.external for p in answered}
    buckets: dict[int, list[int]] = {}
    for external in sorted({p.external for p in declared if MIN_PORT <= p.external <= MAX_PORT}):
        buckets.setdefault(external // PORT_RANGE_BUCKET_WIDTH, []).append(external)
    groups = [ports for _, ports in sorted(buckets.items())]
    kept = groups if len(groups) <= PORT_RANGE_MAX_ENTRIES else groups[: PORT_RANGE_MAX_ENTRIES - 1]
    overflow = [e for ports in groups[len(kept) :] for e in ports]

    def tally(ports: list[int], other: bool = False) -> PortRangeResult:
        return PortRangeResult(
            first=ports[0],
            last=ports[-1],
            declared=len(ports),
            probed=sum(e in probed_ext for e in ports),
            answered=sum(e in answered_ext for e in ports),
            other=other,
        )

    return tuple(tally(ports) for ports in kept) + ((tally(overflow, other=True),) if overflow else ())


def sample_ports(ports: list[PortPair], size: int) -> list[PortPair]:
    """Pick at most `size` ports to probe from `ports` (sorted ascending), spread over the whole list.

    A list of `size` ports or fewer is probed in full, exactly as before. A longer one keeps its
    lowest `size // 2` ports and fills the rest of the budget with ports evenly spaced across the
    remainder, including the highest available port whenever `size` is 3 or more. A host that
    declares 40000-65535 but forwards only 60000 and up is therefore probed where its ports are open.

    Limits, on 40000-65535 with a budget of 300: the spread probes about every 170th port, so a
    block forwarded at the top verifies 3 ports only when it is about 342 ports or wider (513 when
    the DinD probe on one of them fails). A block of 3-150 open ports lying only at positions
    151-299 of a wide declaration is found by the lowest-300 cut but not by this one, and a low
    block of 151-300 ports verifies about 151 ports instead of up to 300.

    Same probe count as the lowest-`size` cut it replaces, and deterministic: for the same
    declaration and rental set the same ports are probed every cycle, so a node does not flap.
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

    def select(
        self,
        executor_info: ExecutorSSHInfo,
        size: int,
        unavailable_ports: set[int],
        declared: list[PortPair] | None = None,
    ) -> list[PortPair]:
        """Select ports to check, skipping external ports already taken by pods or fillers.

        `declared` is `declared_ports(executor_info)` when the caller has already parsed it."""
        if declared is None:
            declared = declared_ports(executor_info)
        available_ports = [p for p in declared if p.external not in unavailable_ports]
        return sample_ports(available_ports, size)
