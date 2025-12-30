import json
import random

from datura.requests.miner_requests import ExecutorSSHInfo

from services.const import PREFERRED_POD_PORTS
from services.executor_connectivity.models import PortPair


class PortSelector:
    """Selects which ports to verify."""

    def select(self, executor_info: ExecutorSSHInfo, size: int, rented: set[int]) -> list[PortPair]:
        """Select available ports with preference for common ports."""
        if executor_info.port_mappings:
            return self._from_mappings(executor_info, size, rented)
        return self._from_range(executor_info, size, rented)

    def _from_mappings(self, info: ExecutorSSHInfo, size: int, rented: set[int]) -> list[PortPair]:
        """Select from existing mappings."""
        mappings = json.loads(info.port_mappings)
        available = [
            PortPair(i, e) for i, e in mappings
            if i != info.ssh_port and e != info.ssh_port and e not in rented
        ]
        preferred = [p for p in available if p.internal in PREFERRED_POD_PORTS or p.external in PREFERRED_POD_PORTS]
        remaining = [p for p in available if p not in preferred]

        result = preferred[:size]
        if len(result) < size and remaining:
            result.extend(random.sample(remaining, min(size - len(result), len(remaining))))
        return result[:size]

    def _from_range(self, info: ExecutorSSHInfo, size: int, rented: set[int]) -> list[PortPair]:
        """Select from port range."""
        if info.port_range:
            if "-" in info.port_range:
                min_p, max_p = map(int, (part.strip() for part in info.port_range.split("-")))
                ports = list(range(min_p, max_p + 1))
            else:
                ports = list(map(int, (part.strip() for part in info.port_range.split(","))))
        else:
            ports = list(range(20000, 65535))

        ports = [p for p in ports if p != info.ssh_port and p not in rented]
        if not ports:
            return []

        preferred = [p for p in PREFERRED_POD_PORTS if p in ports]
        remaining = [p for p in ports if p not in PREFERRED_POD_PORTS]

        selected = preferred[:size]
        if len(selected) < size and remaining:
            selected.extend(random.sample(remaining, min(size - len(selected), len(remaining))))
        return [PortPair(p, p) for p in selected[:size]]
