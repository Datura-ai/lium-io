"""SpecFloorCheck: hard floors on the scraped specs a node is listed with.

RED→GREEN: before the check, a node reporting 0 CPU cores, a fraction of a GB of RAM per GPU, more GPUs than it
lists cards for, or placeholder GPU UUIDs passed validation, because each of those fields was only read, never
bounded. Every healthy listed shape stays above these floors.
"""

from __future__ import annotations

import pytest
from neurons.validators.src.services.task.checks.spec_floor import SpecFloorCheck
from neurons.validators.src.services.task.messages import SpecFloorMessages as Msg

from tests.helpers import build_state
from tests.test_gpu_fingerprint_check import _live_rental

KIB_PER_GB = 1024 * 1024


def _uuid(i: int) -> str:
    return f"GPU-{i:08x}-1a2b-3c4d-5e6f-0123456789ab"


def _specs(*, cpu=128, ram_gb=2048, gpu_count=8, uuids=None):
    uuids = [_uuid(i + 1) for i in range(gpu_count)] if uuids is None else uuids
    return {
        "cpu": {"count": cpu},
        "ram": {"total": ram_gb * KIB_PER_GB},
        "gpu": {
            "count": gpu_count,
            "details": [{"name": "NVIDIA H100 80GB HBM3", "uuid": u} for u in uuids],
        },
    }


async def _run(context_factory, rented=None, **kwargs):
    ctx = context_factory(state=build_state(specs=_specs(**kwargs), rented_data=rented))
    return ctx, await SpecFloorCheck().run(ctx)


@pytest.mark.asyncio
async def test_healthy_h100x8_passes(context_factory):
    _, result = await _run(context_factory)

    assert result.passed is True
    assert result.event.reason_code == Msg.OK.reason


@pytest.mark.parametrize(
    ("kwargs", "finding"),
    [
        ({"cpu": 0}, "cpu_count"),
        ({"cpu": None}, "cpu_count"),
        ({"ram_gb": 12}, "ram_per_gpu"),
        ({"gpu_count": 8, "uuids": [_uuid(1)] * 1}, "gpu_count_vs_cards"),
        (
            {"gpu_count": 2, "uuids": ["GPU-00000000-0000-0000-0000-000000000000"] * 2},
            "gpu_uuid_format",
        ),
        ({"gpu_count": 2, "uuids": ["not-a-uuid", _uuid(2)]}, "gpu_uuid_format"),
        ({"cpu": float("inf")}, "cpu_count"),
        ({"ram_gb": float("inf")}, "ram_per_gpu"),
    ],
)
@pytest.mark.asyncio
async def test_implausible_spec_fails_idle_node_fatally(kwargs, finding, context_factory):
    _, result = await _run(context_factory, **kwargs)

    assert (result.passed, result.fatal) == (False, None)
    assert result.event.reason_code == Msg.REJECTED.reason
    assert finding in result.event.what_we_saw["findings"]


_REAL = "0" * 31 + "1:" + "a" * 64
_EMPTY_MD5 = "d41d8cd98f00b204e9800998ecf8427e"
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _checksums(**overrides):
    return {"docker": _REAL, "nvidia_smi": _REAL, "libnvidia_ml": _REAL, **overrides}


@pytest.mark.parametrize(
    "checksums",
    [
        # md5 of zero bytes in both halves
        _checksums(docker=f"{_EMPTY_MD5}:{_EMPTY_MD5}", nvidia_smi=f"{_EMPTY_MD5}:{_EMPTY_MD5}"),
        _checksums(nvidia_smi=f"{_EMPTY_MD5}:{_EMPTY_SHA256}"),
        _checksums(docker="0" * 31 + "1:" + _EMPTY_SHA256),
        _checksums(docker=""),
        {"libnvidia_ml": _REAL},
        "not-a-dict",
    ],
)
@pytest.mark.asyncio
async def test_file_checksums_no_real_host_reports_fail_the_node(checksums, context_factory):
    specs = {**_specs(), "md5_checksums": checksums}
    ctx = context_factory(state=build_state(specs=specs))

    result = await SpecFloorCheck().run(ctx)

    assert result.passed is False
    assert result.event.what_we_saw["findings"] == ["file_checksums"]


@pytest.mark.asyncio
async def test_real_file_checksums_pass(context_factory):
    specs = {**_specs(), "md5_checksums": _checksums()}
    ctx = context_factory(state=build_state(specs=specs))

    assert (await SpecFloorCheck().run(ctx)).passed is True


@pytest.mark.asyncio
async def test_one_card_read_twice_in_a_scrape_passes(context_factory):
    # A healthy node's scrape can list one card twice; the kernel-count check owns a short set.
    _, result = await _run(context_factory, gpu_count=2, uuids=[_uuid(1), _uuid(1)])

    assert result.passed is True


@pytest.mark.asyncio
async def test_a_live_rental_is_not_halted(context_factory):
    _, result = await _run(context_factory, rented=_live_rental(), cpu=0)

    assert (result.passed, result.fatal) == (False, False)
    assert result.updates == {"spec_floor_passed": False}
