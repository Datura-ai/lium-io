import ipaddress

import pytest

from datura.public_address import is_public_address, is_public_ip

# the addresses every app's dial helper is checked against (validator, miner, the outside prober's copy)
TABLE = [
    ("8.8.8.8", True),
    ("2001:4860:4860::8888", True),
    ("[2001:4860:4860::8888]", True),
    ("64:ff9b::808:808", True),  # DNS64 answer for a public A record
    ("::ffff:8.8.8.8", True),
    ("127.0.0.1", False),
    ("10.0.0.1", False),  # secret-scan: allow
    ("169.254.169.254", False),
    ("100.64.0.1", False),  # secret-scan: allow
    ("0.0.0.0", False),
    ("224.0.0.1", False),  # secret-scan: allow
    ("240.0.0.1", False),  # secret-scan: allow
    ("::1", False),
    ("fe80::1%eth0", False),
    ("fd00::1", False),
    ("fec0::1", False),
    ("ff0e::1", False),
    ("::ffff:127.0.0.1", False),
    ("64:ff9b::a9fe:a9fe", False),  # NAT64 form of the metadata address
    ("2002:808:808::1", False),  # 6to4
    ("2002:7f00:1::1", False),
    ("2001:0:4136:e378:8000:63bf:3fff:fdd2", False),  # Teredo
    ("not-an-address", False),
    ("", False),
]


@pytest.mark.parametrize("address, public", TABLE)
def test_is_public_address(address, public):
    assert is_public_address(address) is public


PARSED = [(a, p) for a, p in TABLE if a and "%" not in a and "[" not in a and a != "not-an-address"]


@pytest.mark.parametrize("address, public", PARSED)
def test_is_public_ip_agrees_on_parsed_addresses(address, public):
    assert is_public_ip(ipaddress.ip_address(address)) is public
