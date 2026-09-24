import hashlib
import random
from dataclasses import dataclass, field

from datura.requests.miner_requests import ExecutorSSHInfo

from services.executor_connectivity.models import PortPair
from services.port_utils import get_all_ports

SELECTION_ALL = "all"
SELECTION_STRATIFIED = "stratified"
# the stratified sample verified fewer than SAMPLED_PORTS_LOWEST_PASS_BELOW, so the lowest free ports were probed too
SELECTION_STRATIFIED_LOWEST = "stratified+lowest"


@dataclass(frozen=True)
class PortSample:
    """The ports to probe this cycle, with the counts they were drawn from."""

    ports: list[PortPair]
    # len(get_all_ports(...)): every port the executor declares, ssh port excluded
    declared_count: int
    # declared ports minus those held by the node's own pods and fillers; the population sampled
    free_count: int
    selection: str
    # the lowest `size` free ports, the set probed before sampling; empty when `ports` is every free port
    lowest_ports: list[PortPair] = field(default_factory=list)


class PortSelector:
    """Selects which ports to verify."""

    def select(
        self,
        executor_info: ExecutorSSHInfo,
        size: int,
        unavailable_ports: set[int],
        *,
        seed: str | None = None,
    ) -> PortSample:
        """Select up to `size` ports, skipping external ports already taken by pods or fillers.

        A free set no larger than `size` is probed whole, in ascending order. A larger one is
        sampled across its whole span: the free ports are cut into `size` equal strata and one
        port is drawn from each, so every part of the range is probed and a host whose low ports
        are busy or unforwarded is measured on the rest. The draw is seeded by `seed` (the
        executor and the cycle), so a cycle is reproducible and successive cycles probe different
        ports. The result is shuffled so any prefix the published-port tiers take is spread too.
        `lowest_ports` keeps the pre-sampling probe set for the orchestrator's below-floor pass.
        """
        all_ports = get_all_ports(
            executor_info.port_range, executor_info.port_mappings, executor_info.ssh_port
        )
        free = [
            PortPair(internal, external)
            for internal, external in all_ports
            if external not in unavailable_ports
        ]
        if len(free) <= size:
            return PortSample(free, len(all_ports), len(free), SELECTION_ALL)

        rng = random.Random(_seed_int(seed if seed is not None else str(executor_info.uuid)))
        n = len(free)
        picks = [free[rng.randrange(k * n // size, (k + 1) * n // size)] for k in range(size)]
        rng.shuffle(picks)
        return PortSample(picks, len(all_ports), n, SELECTION_STRATIFIED, lowest_ports=free[:size])


def estimate_usable_ports(verified: int, probed: int, free: int) -> int:
    """verified/probed scaled to the `free` ports sampled from, floored and capped at `free`.

    Equals `verified` when every free port was probed; never below `verified`.
    """
    if probed <= 0:
        return 0
    return min(free, verified * free // probed)


def _seed_int(seed: str) -> int:
    # hashlib rather than hash(): str hashing is salted per process (PYTHONHASHSEED)
    return int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
