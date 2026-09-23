"""The parallel-stream Cloudflare capacity is THE network number (DAH-2774).

Fixture-driven, no network: every payload below is the `network_execution` block exactly as
celium-gpu-verifier's `execute_network_challenge` serializes it (verifyx/src/challenge/network.rs
at a69d6a3b: `speedtest{download_mbps, upload_mbps}` is the 3-stream × 75 MB × 2-round Cloudflare
capacity, `download{…, speed_mbps}` is the one-object single-stream package download). The payload
goes through the real `_perform_verification_checks` and then the real `VerifyXCheck`, so what is
asserted is what reaches `specs.network` — the keys lium-platform's `best_download_speed_sql` /
`effective_network_speeds` chain reads (`ema_verifyx_download_speed`, then `verifyx_download_speed`).

Three properties:
1. the capacity, not the single-stream figure, is published and gated (the B300 SXM6 PC hosts,
   owner 22 Sep 2026: "show the accurate total bandwidth capacity of all the nodes");
2. a probe that could not reach Cloudflare falls back to the package download instead of
   feeding the EMA a zero, so an outage does not delist an honest host;
3. the EMA handoff: a host whose stored EMA came from the single-stream era moves toward the
   capacity on its first parallel-stream sample, and the scrape's empty `network` block leaves
   only VerifyX keys behind.
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from neurons.validators.src.services.task.checks.network_ema import compute_ema
from neurons.validators.src.services.task.checks.verifyx import (
    MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
    VerifyXCheck,
)
from neurons.validators.src.services.task.messages import VerifyXMessages as Msg
from neurons.validators.src.services.verifyx_validation_service import (
    _perform_verification_checks,
    _verify_network_test,
    settings,
)
from tests.helpers import build_context_config, build_services, build_state
from tests.test_verifyx_check import DummyVerifyXService, _rented_data_with_ema

SERVICE = "neurons.validators.src.services.verifyx_validation_service"

# One 8× B300 SXM6 PC host (22 Sep 2026): a single HF object through one TCP stream reads far
# under the link, the 3-stream Cloudflare probe reads what the link can carry.
B300_PC_SINGLE_STREAM_MBPS = 180.0
B300_PC_CAPACITY_MBPS = 2400.0
B300_PC_UPLOAD_MBPS = 1900.0

PACKAGE = {"pkg": "distilbert-base-uncased.tar", "size": 268_000_000, "hash": "sha256:abc"}


# Capacity is the only gated number. There is no off/shadow/enforce flag.


def _challenge_data() -> dict:
    return {"network_challenge": {"download": dict(PACKAGE), "timeout_seconds": 120}}


def _probe_payload(
    *,
    capacity_mbps: float = B300_PC_CAPACITY_MBPS,
    upload_mbps: float = B300_PC_UPLOAD_MBPS,
    single_stream_mbps: float = B300_PC_SINGLE_STREAM_MBPS,
) -> dict:
    """`NetworkTestExecution` for a run where every direction finished (network.rs:99-107)."""
    return {
        "speedtest": {"download_mbps": capacity_mbps, "upload_mbps": upload_mbps},
        "download": {
            **PACKAGE,
            "status": "success",
            "speed_mbps": single_stream_mbps,
            "time_ms": 11_900,
            "error": None,
        },
        "success": True,
        "error": "",
        "execution_time_ms": 24_300,
    }


def _cloudflare_unreachable_payload(single_stream_mbps: float = B300_PC_SINGLE_STREAM_MBPS) -> dict:
    """The package downloaded, then `execute_speedtest` could not reach speed.cloudflare.com:
    network.rs:632-641 reports both directions as 0.0 and the probe as failed."""
    return {
        "speedtest": {"download_mbps": 0.0, "upload_mbps": 0.0},
        "download": {
            **PACKAGE,
            "status": "success",
            "speed_mbps": single_stream_mbps,
            "time_ms": 11_900,
            "error": None,
        },
        "success": False,
        "error": (
            "Cloudflare download failed: error sending request for url "
            "(https://speed.cloudflare.com/__down?bytes=75000000): connection refused"
        ),
        "execution_time_ms": 131_200,
    }


def _judge(network_execution: dict, *, network_flag: bool = False) -> dict:
    """What `evaluate_verifyx_capture` hands the check: the real `_perform_verification_checks`
    over this payload, with memory and storage passing."""
    payload = {
        "challenge_data": _challenge_data(),
        "response_data": {"network_execution": copy.deepcopy(network_execution)},
    }
    with (
        patch.dict(
            f"{SERVICE}.settings.FEATURE_FLAGS", {"verifyx_network_validation": network_flag}
        ),
        patch(f"{SERVICE}._verify_memory_test", return_value=({"success": True}, [])),
        patch(f"{SERVICE}._verify_storage_test", return_value=({"success": True}, [])),
    ):
        return _perform_verification_checks(payload)


async def _run_check(context_factory, verification_result: dict, *, prev_ema=None, specs=None):
    # The double spreads `updated_specs` over its own `success: True`, so the judged result's
    # `success` (True or False) is what the check reads, as after `evaluate_verifyx_capture`.
    verifyx_service = DummyVerifyXService(success=True, updated_specs=verification_result)
    services = build_services(verifyx=verifyx_service)
    config = build_context_config(verifyx_enabled=True)
    state = build_state(
        specs=specs if specs is not None else {"gpu": {"count": 8}, "network": {}},
        rented_data=_rented_data_with_ema("executor-123", download=prev_ema),
    )
    ctx = context_factory(services=services, config=config, state=state)
    return await VerifyXCheck().run(ctx)


# 1. capacity, not the single-stream figure ---------------------------------------------------


def test_service_reads_the_capacity_and_the_single_stream_figure_into_their_own_keys():
    stats, errors = _verify_network_test(_challenge_data(), {"network_execution": _probe_payload()})

    assert errors == []
    assert stats == {
        "download_speed": B300_PC_CAPACITY_MBPS,
        "upload_speed": B300_PC_UPLOAD_MBPS,
        "package_download_speed": B300_PC_SINGLE_STREAM_MBPS,
        "capacity_download_speed": B300_PC_CAPACITY_MBPS,
        "success": True,
        "execution_time_ms": 24_300,
    }


@pytest.mark.asyncio
async def test_b300_pc_host_publishes_the_parallel_stream_capacity_not_the_single_stream_speed(
    context_factory,
):
    """A never-measured B300 PC host: the specs the platform lists carry 2400 (capacity), the
    EMA seeds from it and the gate passes; 180 (one stream) reaches no `specs.network` key."""
    verification = _judge(_probe_payload())
    assert verification["success"] is True
    assert verification["errors"] == []

    result = await _run_check(context_factory, verification, prev_ema=None)

    assert result.passed is True
    assert result.event.reason_code == Msg.VERIFY_SUCCESS.reason
    net = result.updates["state"].specs["network"]
    assert net["verifyx_download_speed"] == B300_PC_CAPACITY_MBPS
    assert net["ema_verifyx_download_speed"] == B300_PC_CAPACITY_MBPS
    assert net["verifyx_upload_speed"] == B300_PC_UPLOAD_MBPS
    assert net["ema_verifyx_upload_speed"] == B300_PC_UPLOAD_MBPS
    assert B300_PC_SINGLE_STREAM_MBPS not in net.values()
    assert set(net) == {
        "verifyx_download_speed",
        "ema_verifyx_download_speed",
        "verifyx_upload_speed",
        "ema_verifyx_upload_speed",
    }


@pytest.mark.asyncio
async def test_single_stream_speed_under_the_gate_does_not_fail_a_host_whose_capacity_clears_it(
    context_factory,
):
    """The B300 PC symptom in the extreme: one stream reads 60 Mbps (under the 100 Mbps EMA gate,
    above the 50 Mbps package floor), the parallel probe reads 2400. The gate reads 2400."""
    verification = _judge(_probe_payload(single_stream_mbps=60.0))
    assert verification["success"] is True

    result = await _run_check(context_factory, verification, prev_ema=None)

    assert result.passed is True
    net = result.updates["state"].specs["network"]
    assert net["ema_verifyx_download_speed"] == B300_PC_CAPACITY_MBPS
    assert net["ema_verifyx_download_speed"] >= MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS


@pytest.mark.asyncio
async def test_capacity_under_the_gate_fails_the_host_even_when_one_stream_reads_fast(
    context_factory,
):
    """The mirror image: a CDN-flattered host — one stream 900 Mbps, capacity 80 Mbps — fails
    the fatal gate on its capacity (Risks bullet 1)."""
    verification = _judge(_probe_payload(capacity_mbps=80.0, single_stream_mbps=900.0))
    assert verification["success"] is True  # 80 ≥ NETWORK_MIN_DOWNLOAD_SPEED_MBPS (50)

    result = await _run_check(context_factory, verification, prev_ema=None)

    assert result.passed is False
    assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    assert result.event.what_we_saw["ema_verifyx_download_speed"] == 80.0


def test_package_floor_reads_the_single_stream_figure_only():
    """The package floor (50 Mbps, its own setting) judges the one-stream package download; the
    capacity floor judges the Cloudflare figure. Neither reads the other's number."""
    verification = _judge(_probe_payload(single_stream_mbps=40.0), network_flag=True)

    assert verification["success"] is False
    assert verification["errors"] == [
        "Package download speed inadequate: 40.00 Mbps achieved, "
        f"{settings.verifyx.NETWORK_MIN_PACKAGE_DOWNLOAD_SPEED_MBPS:.0f} Mbps required"
    ]
    assert verification["network"]["download_speed"] == B300_PC_CAPACITY_MBPS


# 2. Cloudflare unreachable ---------------------------------------------------------------------


def test_cloudflare_unreachable_falls_back_to_the_package_reading():
    stats, errors = _verify_network_test(
        _challenge_data(), {"network_execution": _cloudflare_unreachable_payload()}
    )

    assert stats["download_speed"] == B300_PC_SINGLE_STREAM_MBPS
    assert stats["package_download_speed"] == B300_PC_SINGLE_STREAM_MBPS
    assert stats["capacity_download_speed"] is None
    assert stats["cloudflare_fallback"] is True
    assert stats["success"] is True
    assert stats["upload_speed"] == 0.0
    assert errors == [
        "Network execution failed: Cloudflare download failed: error sending request for url "
        "(https://speed.cloudflare.com/__down?bytes=75000000): connection refused"
    ]


@pytest.mark.asyncio
async def test_cloudflare_unreachable_uses_the_package_reading_and_does_not_feed_the_ema_a_zero(
    context_factory,
):
    """A Cloudflare outage is our probe's third party, not the host's. The package download
    becomes the gated number (180), never a zero: 2400 → 1290, still over 100."""
    verification = _judge(_cloudflare_unreachable_payload())
    assert verification["success"] is True
    assert verification["network"]["success"] is True
    assert verification["network"]["cloudflare_fallback"] is True

    result = await _run_check(context_factory, verification, prev_ema=B300_PC_CAPACITY_MBPS)

    assert result.passed is True
    net = result.updates["state"].specs["network"]
    assert net["verifyx_download_speed"] == B300_PC_SINGLE_STREAM_MBPS
    assert net["ema_verifyx_download_speed"] == pytest.approx(
        compute_ema(B300_PC_CAPACITY_MBPS, B300_PC_SINGLE_STREAM_MBPS)
    )
    assert net["ema_verifyx_download_speed"] == pytest.approx(1290.0)
    assert net["ema_verifyx_download_speed"] > MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS


@pytest.mark.asyncio
async def test_cloudflare_unreachable_for_five_cycles_does_not_delist_an_honest_host(
    context_factory,
):
    """Five Cloudflare outages feed the package reading, not zeros. The host stays above the gate."""
    ema = B300_PC_CAPACITY_MBPS
    outcomes = []
    for _ in range(5):
        verification = _judge(_cloudflare_unreachable_payload())
        result = await _run_check(context_factory, verification, prev_ema=ema)
        ema = result.updates["state"].specs["network"]["ema_verifyx_download_speed"]
        outcomes.append(result.passed)

    assert all(outcomes)
    assert ema > MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS


@pytest.mark.asyncio
async def test_cloudflare_unreachable_still_passes_when_the_network_flag_is_on(
    context_factory,
):
    """The package fallback is the gated number, so the network flag does not fail an honest host
    for a Cloudflare outage."""
    verification = _judge(_cloudflare_unreachable_payload(), network_flag=True)
    assert verification["success"] is True
    assert verification["network"]["cloudflare_fallback"] is True

    result = await _run_check(context_factory, verification, prev_ema=B300_PC_CAPACITY_MBPS)

    assert result.passed is True
    net = result.updates["state"].specs["network"]
    assert net["verifyx_download_speed"] == B300_PC_SINGLE_STREAM_MBPS


# 3. EMA handoff -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_capacity_sample_moves_a_single_stream_era_ema_toward_the_capacity(
    context_factory,
):
    """A host measured under the old library carries a stored EMA of 150 (single-stream samples).
    Its first parallel-stream sample moves it half-way: compute_ema(150, 2400) = 1275, the gate
    passes at once, and the raw sample published is the capacity."""
    verification = _judge(_probe_payload())

    result = await _run_check(context_factory, verification, prev_ema=150.0)

    assert result.passed is True
    net = result.updates["state"].specs["network"]
    assert net["verifyx_download_speed"] == B300_PC_CAPACITY_MBPS
    assert net["ema_verifyx_download_speed"] == pytest.approx(compute_ema(150.0, 2400.0))
    assert net["ema_verifyx_download_speed"] == pytest.approx(1275.0)


@pytest.mark.asyncio
async def test_ema_converges_on_the_capacity_over_repeated_samples(context_factory):
    """Six identical capacity samples from a 150 Mbps single-stream-era EMA: 1275, 1837.5, …,
    within 2 % of 2400 by the sixth cycle — the listed figure settles on the capacity."""
    ema = 150.0
    trail = []
    for _ in range(6):
        verification = _judge(_probe_payload())
        result = await _run_check(context_factory, verification, prev_ema=ema)
        ema = result.updates["state"].specs["network"]["ema_verifyx_download_speed"]
        trail.append(ema)

    assert trail[:2] == [pytest.approx(1275.0), pytest.approx(1837.5)]
    assert trail == sorted(trail)
    assert abs(trail[-1] - B300_PC_CAPACITY_MBPS) / B300_PC_CAPACITY_MBPS < 0.02


@pytest.mark.asyncio
async def test_scrape_network_block_is_empty_and_only_verifyx_keys_leave_the_check(
    context_factory,
):
    """`machine_scrape.py` sends `network: {}` since this PR (no Ookla `download_speed` /
    `upload_speed`, no scrape `ema_*`), so after the check the block holds the four VerifyX keys
    and nothing the platform chain's `ema_download_speed` / `download_speed` rungs could read."""
    verification = _judge(_probe_payload())
    scraped = {"gpu": {"count": 8}, "cpu": {"cores": 128}, "network": {}}

    result = await _run_check(context_factory, verification, prev_ema=None, specs=scraped)

    specs = result.updates["state"].specs
    assert specs["gpu"] == scraped["gpu"] and specs["cpu"] == scraped["cpu"]
    assert set(specs["network"]) == {
        "verifyx_download_speed",
        "ema_verifyx_download_speed",
        "verifyx_upload_speed",
        "ema_verifyx_upload_speed",
    }
    for stale_key in ("download_speed", "upload_speed", "ema_download_speed", "ema_upload_speed"):
        assert stale_key not in specs["network"]
