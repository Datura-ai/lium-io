
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
        return available_ports[:size]

