"""DAH-3490: the Xid table that says who broke a card, on the lines the kernel really writes."""

from datetime import UTC, datetime, timedelta

import pytest
from neurons.validators.src.services.gpu_xid_attribution import (
    ATTRIBUTION_HARDWARE,
    ATTRIBUTION_NONE,
    ATTRIBUTION_WORKLOAD,
    HARDWARE_XIDS,
    WORKLOAD_XIDS,
    attribute,
    node_answers,
    parse_docker_started_at,
    parse_ecc_uncorrected,
    parse_xid_lines,
)

START = datetime(2026, 9, 16, 9, 0, tzinfo=UTC)
NOW = datetime(2026, 9, 16, 10, 30, tzinfo=UTC)


def iso(stamp: datetime, code: int, *, pid: int | None = 4242, pci: str = "0000:81:00") -> str:
    # `dmesg --time-format=iso` on util-linux 2.39: `2026-09-16T09:30:00,123456+00:00 NVRM: Xid (PCI:...): 31, pid=..., ...`
    pid_part = f"pid={pid}, name=python, " if pid is not None else ""
    return f"{stamp.strftime('%Y-%m-%dT%H:%M:%S')},123456+00:00 NVRM: Xid (PCI:{pci}): {code}, {pid_part}Ch 00000008"


def test_the_table_is_rustams_split():
    # the ticket's words: 13/31/43/45 are the workload's; 48/79/94/95 and the other hardware codes are the provider's
    assert WORKLOAD_XIDS == {13, 31, 43, 45}
    assert {48, 79, 94, 95} <= HARDWARE_XIDS
    assert not (WORKLOAD_XIDS & HARDWARE_XIDS)


def test_iso_and_ctime_stamps_are_both_read():
    lines = parse_xid_lines(
        iso(START + timedelta(minutes=30), 31)
        + "\n[Wed Sep 16 09:45:00 2026] NVRM: Xid (PCI:0000:81:00): 79, GPU has fallen off the bus.\n"
        + "NVRM: Xid (PCI:0000:81:00): 13, pid=7, Graphics Exception\n"
    )
    assert [line.code for line in lines] == [31, 79, 13]
    assert lines[0].timestamp == datetime(2026, 9, 16, 9, 30, 0, 123456, tzinfo=UTC)
    assert lines[0].pid == 4242 and lines[0].pci == "0000:81:00"
    assert lines[1].timestamp == datetime(2026, 9, 16, 9, 45, tzinfo=UTC) and lines[1].pid is None
    # no stamp at all: the line is kept (it is evidence) but can never be placed in a window
    assert lines[2].timestamp is None


@pytest.mark.parametrize("code", sorted(WORKLOAD_XIDS))
def test_an_application_xid_inside_the_rental_from_the_renters_process_is_the_workloads(code):
    verdict = attribute(
        parse_xid_lines(iso(START + timedelta(minutes=10), code)),
        window_start=START,
        window_end=NOW,
        container_pids={4242, 4243},
    )
    assert verdict.attribution == ATTRIBUTION_WORKLOAD
    assert len(verdict.workload) == 1 and verdict.hardware == []


@pytest.mark.parametrize("code", [48, 62, 63, 64, 74, 79, 92, 94, 95, 119])
def test_a_hardware_xid_inside_the_rental_is_the_providers_even_beside_a_workload_xid(code):
    text = iso(START + timedelta(minutes=10), 31) + "\n" + iso(START + timedelta(minutes=11), code, pid=None)
    verdict = attribute(parse_xid_lines(text), window_start=START, window_end=NOW, container_pids={4242})
    assert verdict.attribution == ATTRIBUTION_HARDWARE
    assert len(verdict.workload) == 1 and len(verdict.hardware) == 1


def test_lines_outside_the_rental_window_attribute_nothing():
    text = iso(START - timedelta(minutes=5), 31) + "\n" + iso(NOW + timedelta(minutes=1), 79, pid=None)
    verdict = attribute(parse_xid_lines(text), window_start=START, window_end=NOW)
    assert verdict.attribution == ATTRIBUTION_NONE
    assert verdict.outside_window == 2


def test_an_unknown_window_start_attributes_nothing():
    verdict = attribute(parse_xid_lines(iso(START + timedelta(minutes=10), 31)), window_start=None, window_end=NOW)
    assert verdict.attribution == ATTRIBUTION_NONE


def test_another_containers_application_xid_is_not_this_renters():
    # mid-rental: the PID on the line is not in the renter's container -> another tenant's, not counted
    verdict = attribute(
        parse_xid_lines(iso(START + timedelta(minutes=10), 31, pid=9999)),
        window_start=START,
        window_end=NOW,
        container_pids={4242},
    )
    assert verdict.attribution == ATTRIBUTION_NONE
    assert verdict.other_container == 1


def test_at_rental_end_the_timestamp_alone_places_the_line():
    # the container is gone, so no PID filter: container_pids=None accepts the renter's dead process
    verdict = attribute(
        parse_xid_lines(iso(START + timedelta(minutes=10), 31, pid=9999)),
        window_start=START,
        window_end=NOW,
        container_pids=None,
    )
    assert verdict.attribution == ATTRIBUTION_WORKLOAD


def test_uncorrected_ecc_makes_it_hardware_even_with_only_workload_lines():
    verdict = attribute(
        parse_xid_lines(iso(START + timedelta(minutes=10), 31)),
        window_start=START,
        window_end=NOW,
        container_pids={4242},
        ecc_uncorrected={"GPU-aaaa": 3},
    )
    assert verdict.attribution == ATTRIBUTION_HARDWARE
    assert verdict.as_report()["ecc_uncorrected"] == {"GPU-aaaa": 3}


def test_ecc_rows_keep_only_cards_with_uncorrected_errors():
    text = "00000000:81:00.0, GPU-aaaa, 0\n00000000:82:00.0, GPU-bbbb, 2\n00000000:83:00.0, GPU-cccc, [N/A]\n"
    assert parse_ecc_uncorrected(text) == {"GPU-bbbb": 2}


def test_docker_started_at_is_read_with_nanoseconds_and_zone():
    assert parse_docker_started_at("2026-09-16T09:00:00.123456789Z\n") == START.replace(microsecond=123456)
    assert parse_docker_started_at("2026-09-16T11:00:00+02:00") == START
    # a never-started container reports the zero time
    assert parse_docker_started_at("0001-01-01T00:00:00Z") is None
    assert parse_docker_started_at("") is None


def test_node_answers_is_false_when_nvidia_smi_lost_a_card():
    assert node_answers(0, "00000000:81:00.0, GPU-aaaa, 0\n", "")
    assert not node_answers(15, "", "Unable to determine the device handle for GPU 0000:81:00.0: Unknown Error")
    assert not node_answers(0, "GPU is lost. Reboot the system to recover this GPU", "")


def test_the_report_is_bounded():
    text = "\n".join(iso(START + timedelta(seconds=i), 31) for i in range(60))
    report = attribute(parse_xid_lines(text), window_start=START, window_end=NOW, container_pids={4242}).as_report()
    assert len(report["workload_xids"]) == 20
    assert all(len(line) <= 200 for line in report["workload_xids"])
