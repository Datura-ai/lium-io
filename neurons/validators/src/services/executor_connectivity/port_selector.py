import json
import random

from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import PREFERRED_POD_PORTS
from services.executor_connectivity.models import PortPair
from services.port_utils import get_all_ports


class PortSelector:
    """Selects which ports to verify."""

    def select(self, executor_info: ExecutorSSHInfo, size: int, rented: set[int]) -> list[PortPair]:
        """Select ports to check, based on executor params and info about rented external ports."""
        all_ports = get_all_ports(executor_info.port_range, executor_info.port_mappings, executor_info.ssh_port)
        available_ports = [PortPair(internal, external) for internal, external in all_ports if external not in rented]
        return available_ports[:size]

