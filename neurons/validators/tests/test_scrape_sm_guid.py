"""The scrape asks each InfiniBand port's subnet manager for its GUID (DAH-6480): the one fact that tells
two hosts on one fabric from two hosts each running its own subnet manager at the same LID."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

from neurons.validators.tests.helpers import build_scrape_namespace

SRC = Path(__file__).resolve().parents[1] / "src"
CONSTANTS = {
    "INFINIBAND_MAD_SYSFS_PATH",
    "INFINIBAND_DEV_PATH",
    "UMAD_HEADER_BYTES",
    "MAD_BYTES",
    "SMP_LID_ROUTED_CLASS",
    "SM_INFO_ATTRIBUTE",
    "MAD_METHOD_GET",
    "MAD_METHOD_GET_RESPONSE",
    "SMP_DATA_OFFSET",
    "SM_INFO_TIMEOUT_MS",
    "SM_INFO_RETRIES",
}


def _scrape(*names: str) -> dict:
    return build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py", {*names, *CONSTANTS, "read_sysfs_value"}, {"os": os, "sys": sys}
    )


def _answer(request: bytes, guid: bytes, status: int = 0) -> bytes:
    # what the kernel hands back: the header with the request's status, the MAD turned into a GetResponse
    header = request[:4] + status.to_bytes(4, sys.byteorder) + request[8:56]
    mad = bytearray(request[56:])
    mad[3] = 0x81
    mad[64:72] = guid
    return header + bytes(mad)


def test_the_request_is_a_lid_routed_get_sminfo_to_the_sm():
    request = _scrape("sm_info_request")["sm_info_request"](7, 0x1A)

    assert len(request) == 56 + 256
    assert int.from_bytes(request[0:4], sys.byteorder) == 7
    assert request[28:30] == (0x1A).to_bytes(2, "big")
    assert request[56:60] == bytes([1, 0x01, 1, 0x01])
    assert request[56 + 16 : 56 + 18] == (0x0020).to_bytes(2, "big")


def test_the_answer_gives_the_sm_guid_as_sysfs_writes_a_guid():
    ns = _scrape("sm_info_request", "sm_guid_of_response")
    request = ns["sm_info_request"](7, 1)

    guid = ns["sm_guid_of_response"](_answer(request, bytes.fromhex("b8cef60300a41d2c")))

    assert guid == "b8ce:f603:00a4:1d2c"


def test_no_guid_when_the_request_timed_out_or_came_back_empty():
    ns = _scrape("sm_info_request", "sm_guid_of_response")
    request = ns["sm_info_request"](7, 1)

    assert ns["sm_guid_of_response"](_answer(request, bytes.fromhex("b8cef60300a41d2c"), status=110)) == ""
    assert ns["sm_guid_of_response"](_answer(request, bytes(8))) == ""
    assert ns["sm_guid_of_response"](request) == ""  # still a Get: nothing answered
    assert ns["sm_guid_of_response"](b"") == ""


def test_the_umad_device_is_found_by_its_device_and_port(tmp_path: Path):
    ns = _scrape("umad_device_of")
    for name, device, port in (("umad0", "mlx5_0", "1"), ("umad1", "mlx5_4", "1")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "ibdev").write_text(device + "\n")
        (tmp_path / name / "port").write_text(port + "\n")
    ns["INFINIBAND_MAD_SYSFS_PATH"] = str(tmp_path)

    assert ns["umad_device_of"]("mlx5_4", "1") == "/dev/infiniband/umad1"
    assert ns["umad_device_of"]("mlx5_9", "1") == ""


def test_an_issm_device_of_the_same_port_is_never_picked(tmp_path: Path):
    # issmN sorts before umadN and shares its ibdev/port; opening it would claim the port's IsSM bit
    ns = _scrape("umad_device_of")
    for name in ("issm0", "umad0"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "ibdev").write_text("mlx5_0\n")
        (tmp_path / name / "port").write_text("1\n")
    ns["INFINIBAND_MAD_SYSFS_PATH"] = str(tmp_path)

    assert ns["umad_device_of"]("mlx5_0", "1") == "/dev/infiniband/umad0"


def test_only_active_infiniband_ports_with_an_sm_are_asked():
    ns = _scrape("query_sm_guids")
    asked = []
    ns["umad_device_of"] = lambda device, port: f"/dev/infiniband/{device}"
    ns["query_sm_guid"] = lambda path, sm_lid: asked.append((path, sm_lid)) or "b8ce:f603:00a4:1d2c"

    def port(device, link_layer="InfiniBand", state="4: ACTIVE", sm_lid="0x1"):
        return SimpleNamespace(device=device, port="1", link_layer=link_layer, state=state, sm_lid=sm_lid, sm_guid="")

    ports = [port("ib"), port("eth", link_layer="Ethernet"), port("down", state="1: DOWN"), port("nosm", sm_lid="0x0")]
    ns["query_sm_guids"](ports)

    assert asked == [("/dev/infiniband/ib", 1)]
    assert [p.sm_guid for p in ports] == ["b8ce:f603:00a4:1d2c", "", "", ""]
