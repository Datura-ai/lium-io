
from datura.requests.miner_requests import ExecutorSSHInfo

from services.executor_connectivity.models import PortPair
from services.port_utils import get_all_ports


class PortSelector:
    """Selects which ports to verify."""

    def select(self, executor_info: ExecutorSSHInfo, size: int, unavailable_ports: set[int]) -> list[PortPair]:
        """Select ports to check, skipping external ports already taken by pods or fillers."""
        all_ports = get_all_ports(executor_info.port_range, executor_info.port_mappings, executor_info.ssh_port)
        available_ports = [
            PortPair(internal, external) for internal, external in all_ports if external not in unavailable_ports
        ]
        if len(available_ports) <= size:
            return available_ports
        # the lowest half stays as before (the semi-batch and sequential tiers probe only the first 50);
        # the rest is spread evenly up to the highest declared port, so a wide range forwarded only
        # at its top is still found without probing more ports
        head = size // 2
        span, picks = len(available_ports) - 1 - head, size - head
        return available_ports[:head] + [
            available_ports[head + i * span // max(picks - 1, 1)] for i in range(picks)
        ]

