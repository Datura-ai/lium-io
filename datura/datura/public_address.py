"""The one rule for whether an address is on the public internet, shared by every app that dials an address someone
outside Lium supplied (a node, a miner's axon). A dial helper refuses any address this rule does not call public."""

import ipaddress

NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")  # RFC 6052 well-known prefix: the IPv4 address in the last 32 bits
# 6to4 and Teredo tunnel to an IPv4 address the prefix hides; no node needs them, and Python versions differ on
# whether they are global, so they are never public
TUNNEL_PREFIXES = (ipaddress.ip_network("2002::/16"), ipaddress.ip_network("2001::/32"))


def carried_ipv4(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 address stands for, when it is one of the forms that carry one."""
    if ip.version == 4:
        return None
    if ip.ipv4_mapped:
        return ip.ipv4_mapped
    if ip in NAT64_PREFIX:  # a DNS64 answer: what an IPv6-only host gets for a public A record
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Global by the stdlib, and none of what Python before 3.13 also calls global. An IPv4-mapped address and a NAT64
    address are judged as the IPv4 address they carry; 6to4, Teredo, multicast, IPv6 site-local (fec0::/10) and the
    other reserved ranges are not public."""
    if ip.version == 6 and any(ip in prefix for prefix in TUNNEL_PREFIXES):
        return False
    ip = carried_ipv4(ip) or ip
    return ip.is_global and not ip.is_multicast and not ip.is_reserved and not getattr(ip, "is_site_local", False)


def is_public_address(address: str) -> bool:
    """``is_public_ip`` for an address as text: ``[v6]`` brackets and a ``%scope`` suffix are dropped; text that is
    not an IP address is not public."""
    try:
        ip = ipaddress.ip_address(address.strip("[]").split("%", 1)[0])
    except ValueError:
        return False
    return is_public_ip(ip)
