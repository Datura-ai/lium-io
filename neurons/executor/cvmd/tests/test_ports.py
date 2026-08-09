"""Forwarded ports — parsed as dstack parses them, and never handed out twice."""

import socket

import pytest
from cvmd.cvm.ports import PortError, assert_free, parse, parse_all


class TestParsing:
    def test_three_parts_default_to_loopback(self):
        """dstack's own default. A mapping with no address is not a public one."""
        mapping = parse("tcp:2200:22")
        assert (mapping.protocol, mapping.address) == ("tcp", "127.0.0.1")
        assert (mapping.host_port, mapping.guest_port) == (2200, 22)

    def test_four_parts_carry_the_bind_address(self):
        mapping = parse("tcp:0.0.0.0:12200:2200")
        assert mapping.address == "0.0.0.0"
        assert (mapping.host_port, mapping.guest_port) == (12200, 2200)

    @pytest.mark.parametrize(
        "spec",
        ["tcp:22", "tcp:a:b:c:d:e", "tcp:0.0.0.0:notaport:22", "tcp:0:22", "tcp:70000:22"],
        ids=["too-few", "too-many", "non-numeric", "port-zero", "port-too-high"],
    )
    def test_a_mapping_cvmd_cannot_parse_is_refused(self, spec):
        with pytest.raises(PortError):
            parse(spec)

    def test_the_same_host_port_twice_is_refused(self):
        """QEMU would take the first and drop the second silently; only one could ever work."""
        with pytest.raises(PortError, match="mapped twice"):
            parse_all(["tcp:0.0.0.0:12200:2200", "tcp:0.0.0.0:12200:8001"])

    def test_the_same_host_port_on_different_protocols_is_fine(self):
        assert len(parse_all(["tcp:0.0.0.0:12200:2200", "udp:0.0.0.0:12200:2200"])) == 2


class TestAvailability:
    def test_free_ports_pass(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        # The socket is closed, so the port is free again — this is the happy path.
        assert_free(parse_all([f"tcp:127.0.0.1:{free_port}:22"]))

    def test_a_port_someone_else_holds_refuses_the_launch(self):
        """The check that stops a second CVM answering on a live one's address."""
        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            held.listen()
            port = held.getsockname()[1]

            with pytest.raises(PortError, match=f"{port}"):
                assert_free(parse_all([f"tcp:127.0.0.1:{port}:22"]))

    def test_every_taken_port_is_named_at_once(self):
        """One message, not one per relaunch: each retry is a chance to leave a CVM half-built."""
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            second.bind(("127.0.0.1", 0))
            first.listen()
            second.listen()
            ports = [first.getsockname()[1], second.getsockname()[1]]

            with pytest.raises(PortError) as raised:
                assert_free(parse_all([f"tcp:127.0.0.1:{p}:22" for p in ports]))

        message = str(raised.value)
        assert all(str(port) in message for port in ports)

    def test_a_loopback_holder_blocks_a_wildcard_bind(self):
        """The realistic collision: a live CVM forwards 127.0.0.1:P, a new one wants 0.0.0.0:P.

        SO_REUSEADDR is deliberately not set, so this is caught rather than allowed through to
        two processes fighting over the same port.
        """
        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            held.listen()
            port = held.getsockname()[1]

            with pytest.raises(PortError):
                assert_free(parse_all([f"tcp:0.0.0.0:{port}:22"]))
