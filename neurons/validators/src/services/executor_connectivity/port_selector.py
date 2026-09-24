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
    declared: Iterable[PortPair],
    probed: Iterable[PortPair],
    answered: Iterable[PortPair],
    pass_number: int = 1,
) -> tuple[PortRangeResult, ...]:
    """Declared, probed and answered counts per declared range for one pass, ascending.

    Counts distinct external ports in 1-65535, and only answers among this pass's probes, so a
    duplicated port or two mappings sharing one external port are counted once and
    `answered <= probed <= declared` holds in every tally. Ports are grouped into
    PORT_RANGE_BUCKET_WIDTH-wide buckets, and each tally names the lowest and highest declared port
    in its bucket: a range narrower than a bucket is one tally, a wide range is split so a forward
    that covers only part of it shows which part. Past PORT_RANGE_MAX_ENTRIES buckets the rest are
    summed into one final `other` tally.
    """
    probed_ext = {p.external for p in probed}
    answered_ext = {p.external for p in answered} & probed_ext
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
            pass_number=pass_number,
        )

    return tuple(tally(ports) for ports in kept) + (
        (tally(overflow, other=True),) if overflow else ()
    )


def spread_ports(ports: list[PortPair], size: int) -> list[PortPair]:
    """At most `size` of `ports` (sorted ascending), evenly spaced, always including the last one.

    All of them when they fit. Deterministic: the same list gives the same picks.
    """
    if size <= 0:
        return []
    if len(ports) <= size:
        return list(ports)
    if size == 1:
        return [ports[-1]]
    last = len(ports) - 1
    return [ports[i * last // (size - 1)] for i in range(size)]


class PortSelector:
    """Selects which ports to verify.

    Pass one (`select`) is the lowest `size` free declared ports, the check every host gets. Pass
    two (`select_spread`) runs only when pass one verified fewer than MIN_PORT_COUNT, counted after
    the DinD probe has taken its port: up to `size`
    ports spread evenly over the free declared ports pass one did not test, always including the
    highest. On 40000-65535 pass two probes every 84th or 85th port from 40300 up, so a block
    forwarded at the top verifies 3 ports when it is 170 ports or wider (255 when the DinD probe on
    one of them fails), and a block anywhere above pass one's ports when it is 254 ports or wider
    (338). For the same declaration and rental set both passes probe the same ports every cycle, so
    a node does not flap.
    """

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
        return available_ports[:size]

    def select_spread(
        self,
        declared: list[PortPair],
        size: int,
        unavailable_ports: set[int],
        tested: Iterable[PortPair],
    ) -> list[PortPair]:
        """Pass two: up to `size` free declared ports that pass one did not test, spread evenly."""
        tested_ext = {p.external for p in tested}
        remaining = [
            p
            for p in declared
            if p.external not in unavailable_ports and p.external not in tested_ext
        ]
        return spread_ports(remaining, size)
