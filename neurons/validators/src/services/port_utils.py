import json

DEFAULT_PORT_RANGE = (20000, 65536)


def get_all_ports(port_range: str | None, port_mappings: str | None, ssh_port: int) -> list[tuple[int, int]]:
    """Get all available ports sorted ascending, excluding ssh_port."""
    if port_mappings:
        all_ports: list[tuple[int, int]] = [tuple(p) for p in json.loads(port_mappings)]
    elif port_range:
        if "-" in port_range:
            min_port, max_port = map(int, (p.strip() for p in port_range.split("-")))
            all_ports = [(p, p) for p in range(min_port, max_port + 1)]
        else:
            all_ports = [(int(p.strip()), int(p.strip())) for p in port_range.split(",")]
    else:
        all_ports = [(p, p) for p in range(*DEFAULT_PORT_RANGE)]
    return sorted([(i, e) for i, e in all_ports if i != ssh_port and e != ssh_port], key=lambda x: x[0])
