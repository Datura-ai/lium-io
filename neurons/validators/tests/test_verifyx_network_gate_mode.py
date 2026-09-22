"""DAH-2774: VERIFYX_NETWORK_GATE_MODE ships the capacity reading shadow-first.

The network number moves from the one-object package download of libverifyx.so (main's build) to
the Cloudflare capacity probe of libverifyx_capacity.so (celium-gpu-verifier#25); two earlier builds
of that library were wrong, one read every host's download as 0, and it fails probes (Cloudflare
HTTP errors, rate limits, slow uploads) where main's library passes. So both are vendored and the
reading that gates is a setting:

- off: libverifyx.so's package download gates and is published; nothing else runs;
- shadow (the default): exactly off, then one light libverifyx_capacity.so run whose capacity
  reading and the EMA it would give ride on the event (`network_gate`) and in the cycle summary;
- enforce: libverifyx_capacity.so's capacity reading gates and is published.

Each payload goes through the real `_perform_verification_checks` and the real `VerifyXCheck`, so
what is asserted is the verdict and the `specs.network` keys lium-platform lists from. The shadow
end-to-end tests run the real service over a fake SSH shell, one answer per library.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from neurons.validators.src.core.config import VerifyXSettings
from neurons.validators.src.services.task.checks import verifyx as check_module
from neurons.validators.src.services.task.checks.network_ema import compute_ema
from neurons.validators.src.services.task.checks.verifyx import (
    MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
    VerifyXCheck,
)
from neurons.validators.src.services.task.messages import VerifyXMessages as Msg
from neurons.validators.src.services.verifyx_validation_service import (
    CAPACITY_LIB_PATH,
    LIB_PATH,
    NetworkGateTally,
    VerifyXValidationService,
    _verify_network_test,
    settings,
)
from tests.helpers import build_context_config, build_services, build_state
from tests.test_incentive_flow import _run_sync_with_jobs
from tests.test_verifyx_capacity_probe import (
    _challenge_data,
    _cloudflare_unreachable_payload,
    _judge,
    _probe_payload,
)
from tests.test_verifyx_check import DummyVerifyXService, _rented_data_with_ema
from tests.test_verifyx_two_libraries import (
    TODAYS_EXECUTOR,
    UPDATED_EXECUTOR,
    asked_for,
    commands,
    executor_shell,
    patched_libraries,
)

pytest_plugins = ["fixtures.incentive_fixtures"]

# A library that reads every host's capacity as 0 (the F8 build) on a host whose package download
# clears the floor today.
ZERO_CAPACITY = {"capacity_mbps": 0.0, "upload_mbps": 900.0, "single_stream_mbps": 300.0}
# A host whose one-stream package download sits under the floor while its link carries 2.4 Gbps.
FAST_LINK_SLOW_OBJECT = {"capacity_mbps": 2400.0, "upload_mbps": 1900.0, "single_stream_mbps": 80.0}


@pytest.fixture(autouse=True)
def tally():
    # The process-wide tally the check records into (imported the way the check imports it).
    _zero(check_module.NETWORK_GATE_TALLY)
    yield check_module.NETWORK_GATE_TALLY
    _zero(check_module.NETWORK_GATE_TALLY)


@pytest.fixture
def libraries():
    with patched_libraries() as library:
        yield library


@pytest.fixture
def mode(monkeypatch):
    def set_mode(value: str) -> None:
        monkeypatch.setattr(settings.verifyx, "NETWORK_GATE_MODE", value)

    return set_mode


def _measured(
    capacity_mbps: float | None, *, success: bool = True, upload_mbps: float = 900.0
) -> dict:
    """A shadow libverifyx_capacity.so run that measured (`measure_capacity_shadow`)."""
    return {
        "status": "measured",
        "capacity_download_speed": capacity_mbps,
        "upload_speed": upload_mbps,
        "package_download_speed": 300.0,
        "success": success,
        "errors": [],
    }


async def _run_check(
    context_factory, verification_result: dict, *, prev_ema=None, capacity_run=None, **ctx_extra
):
    tally = _zero(check_module.NETWORK_GATE_TALLY)
    verifyx_service = DummyVerifyXService(success=True, updated_specs=verification_result)
    verifyx_service.capacity_run = capacity_run
    ctx = context_factory(
        services=build_services(verifyx=verifyx_service),
        config=build_context_config(verifyx_enabled=True),
        state=build_state(
            specs={"gpu": {"count": 8}, "network": {}},
            rented_data=_rented_data_with_ema("executor-123", download=prev_ema),
        ),
        **ctx_extra,
    )
    result = await VerifyXCheck().run(ctx)
    return result, NetworkGateTally(**_counts(tally))


async def _run_real(context_factory, shell, *, prev_ema):
    """The real service and check over a fake SSH shell (the `libraries` fixture answers)."""
    tally = _zero(check_module.NETWORK_GATE_TALLY)
    ctx = context_factory(
        services=build_services(verifyx=VerifyXValidationService(), shell=shell),
        config=build_context_config(verifyx_enabled=True),
        state=build_state(
            specs={"gpu": {"count": 8, "details": [{"uuid": "u", "name": "H100"}]}, "network": {}},
            rented_data=_rented_data_with_ema("executor-123", download=prev_ema),
        ),
    )
    result = await VerifyXCheck().run(ctx)
    return result, NetworkGateTally(**_counts(tally))


def _outcome(result) -> tuple:
    """What main's validator decides and stores: the verdict, the reason and `specs.network`."""
    specs = result.updates["state"].specs if result.updates else None
    return result.passed, result.event.reason_code, (specs or {}).get("network")


def _counts(tally) -> dict[str, int]:
    return {name: getattr(tally, name) for name in tally.__dataclass_fields__}


def _zero(tally):
    for name in tally.__dataclass_fields__:
        setattr(tally, name, 0)
    return tally


def test_the_default_is_shadow_and_an_unknown_mode_refuses_to_start(monkeypatch):
    monkeypatch.delenv("VERIFYX_NETWORK_GATE_MODE", raising=False)
    assert VerifyXSettings().NETWORK_GATE_MODE == "shadow"
    monkeypatch.setenv("VERIFYX_NETWORK_GATE_MODE", "enforce")
    assert VerifyXSettings().NETWORK_GATE_MODE == "enforce"
    monkeypatch.setenv("VERIFYX_NETWORK_GATE_MODE", "enforced")
    with pytest.raises(ValidationError):
        VerifyXSettings()


def test_env_template_documents_shadow_and_that_enforce_is_a_flip():
    text = Path(__file__).resolve().parents[1].joinpath(".env.template").read_text()
    assert "VERIFYX_NETWORK_GATE_MODE=shadow" in text
    assert "flip this to enforce" in text
    assert "No date is set" in text


@pytest.mark.asyncio
async def test_off_gates_the_package_download_and_records_no_capacity(context_factory, mode):
    mode("off")
    stats, errors = _verify_network_test(
        _challenge_data(), {"network_execution": _probe_payload(**ZERO_CAPACITY)}
    )
    assert errors == []
    assert stats == {
        "download_speed": 300.0,
        "upload_speed": 900.0,
        "package_download_speed": 300.0,
        "success": True,
        "execution_time_ms": 24_300,
    }

    result, tally = await _run_check(context_factory, _judge(_probe_payload(**ZERO_CAPACITY)))

    assert result.passed is True
    assert "network_gate" not in result.event.what_we_saw
    network = result.updates["state"].specs["network"]
    assert network["verifyx_download_speed"] == network["ema_verifyx_download_speed"] == 300.0
    assert set(_counts(tally).values()) == {0}


@pytest.mark.asyncio
async def test_shadow_keeps_todays_verdict_and_records_the_capacity_next_to_it(
    context_factory, mode
):
    mode("shadow")

    zero, zero_tally = await _run_check(
        context_factory,
        _judge(_probe_payload(**ZERO_CAPACITY)),
        prev_ema=400.0,
        capacity_run=_measured(0.0),
    )
    slow_object, slow_tally = await _run_check(
        context_factory,
        _judge(_probe_payload(**FAST_LINK_SLOW_OBJECT)),
        prev_ema=90.0,
        capacity_run=_measured(2400.0),
    )

    # the zero-capacity library changes nothing a renter or the score sees
    assert zero.passed is True
    network = zero.updates["state"].specs["network"]
    assert network["verifyx_download_speed"] == 300.0
    assert network["ema_verifyx_download_speed"] == 350.0
    assert zero.event.what_we_saw["network_gate"] == {
        "mode": "shadow",
        "floor_mbps": MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
        "package_download_speed": 300.0,
        "capacity_download_speed": 0.0,
        "ema_package": 350.0,
        "ema_capacity": 200.0,
        "capacity_run": _measured(0.0),
    }
    assert _counts(zero_tally) == {
        **dict.fromkeys(_counts(zero_tally), 0),
        "package_pass": 1,
        "capacity_pass": 1,
    }

    # today's verdict stands for a host the capacity reading would pass
    assert slow_object.passed is False
    assert slow_object.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    assert slow_object.updates["state"].specs["network"]["ema_verifyx_download_speed"] == 85.0
    assert slow_object.event.what_we_saw["network_gate"]["ema_capacity"] == 1245.0
    assert slow_tally.newly_pass == 1 and slow_tally.package_fail == 1


@pytest.mark.asyncio
async def test_shadow_counts_a_host_the_capacity_reading_would_fail(context_factory, mode):
    mode("shadow")

    result, tally = await _run_check(
        context_factory, _judge(_probe_payload(**ZERO_CAPACITY)), capacity_run=_measured(0.0)
    )

    assert result.passed is True
    assert result.event.what_we_saw["network_gate"]["ema_package"] == 300.0
    assert result.event.what_we_saw["network_gate"]["ema_capacity"] == 0.0
    assert (tally.package_pass, tally.capacity_fail, tally.newly_fail) == (1, 1, 1)


@pytest.mark.parametrize(
    "speedtest",
    [
        None,
        # main's library divides its requested bytes by the elapsed time whatever came back: a
        # 403 in 17 ms reads 48 000 Mbps
        {"download_mbps": 48_000.0, "upload_mbps": 310.0},
    ],
)
def test_shadow_judges_the_gated_answer_exactly_as_off(mode, speedtest):
    payload = _probe_payload(single_stream_mbps=300.0)
    if speedtest is None:
        del payload["speedtest"]
    else:
        payload["speedtest"] = speedtest

    mode("off")
    off = _verify_network_test(_challenge_data(), {"network_execution": payload})
    mode("shadow")
    shadow = _verify_network_test(_challenge_data(), {"network_execution": payload})

    assert shadow == off
    assert shadow[0]["download_speed"] == 300.0
    assert "capacity_download_speed" not in shadow[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("gate_mode", ["off", "shadow"])
async def test_five_failed_probes_decay_and_fail_the_node_exactly_as_on_main(
    context_factory, mode, libraries, gate_mode
):
    # libverifyx.so downloads the package at 900 Mbps and its probe fails as a whole. main's
    # _verify_network_test returns {"success": False} with no download_speed, so the check feeds
    # 0.0 and the EMA halves each cycle from 2000: 1000, 500, 250, 125, 62.5, and the fifth cycle
    # fails the 100 Mbps floor. In shadow libverifyx_capacity.so reads 2400 Mbps on the same host
    # every cycle; that lands on the record only.
    mode(gate_mode)
    libraries.answers[LIB_PATH] = _cloudflare_unreachable_payload(single_stream_mbps=900.0)
    libraries.answers[CAPACITY_LIB_PATH] = _probe_payload(capacity_mbps=2400.0)
    main_emas = [1000.0, 500.0, 250.0, 125.0, 62.5]
    main_passed = [True, True, True, True, False]

    ema, emas, passed, gate_records = 2000.0, [], [], []
    for _ in range(5):
        shell = executor_shell(UPDATED_EXECUTOR)
        result, tally = await _run_real(context_factory, shell, prev_ema=ema)
        network = result.updates["state"].specs["network"]
        assert "verifyx_download_speed" not in network
        ema = network["ema_verifyx_download_speed"]
        emas.append(ema)
        passed.append(result.passed)
        gate_records.append(result.event.what_we_saw.get("network_gate"))
        if gate_mode == "off":
            assert asked_for(shell) == [LIB_PATH] and len(commands(shell)) == 1
        else:
            assert asked_for(shell) == [LIB_PATH, CAPACITY_LIB_PATH]
            assert f"--lib {CAPACITY_LIB_PATH}" in commands(shell)[1]

    assert emas == main_emas
    assert passed == main_passed
    assert result.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    if gate_mode == "off":
        assert gate_records == [None] * 5
        return
    assert [record["ema_package"] for record in gate_records] == main_emas
    assert [record["capacity_download_speed"] for record in gate_records] == [2400.0] * 5
    assert [record["ema_capacity"] for record in gate_records] == [
        compute_ema(prev, 2400.0) for prev in [2000.0, *main_emas[:4]]
    ]
    assert tally.newly_pass == 1 and tally.package_fail == 1


# The Cloudflare probe of libverifyx_capacity.so failing where main's library passes
# (celium-gpu-verifier#25): a rate limit reads 0 both ways; a slow upload fails the probe and
# keeps the download reading.
RATE_LIMITED = {
    **_cloudflare_unreachable_payload(single_stream_mbps=420.0),
    "error": "Cloudflare download failed: HTTP status client error (429 Too Many Requests)",
}
SLOW_UPLOAD = {
    **_probe_payload(capacity_mbps=2400.0, upload_mbps=22.0, single_stream_mbps=420.0),
    "success": False,
    "error": "Cloudflare upload failed: 22.0 Mbps under the 30 Mbps floor",
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capacity_answer", "capacity_mbps", "ema_capacity"),
    [(RATE_LIMITED, None, 200.0), (SLOW_UPLOAD, 2400.0, 1400.0)],
    ids=["cloudflare_429", "slow_upload"],
)
async def test_shadow_keeps_mains_outcome_when_the_capacity_library_fails_its_probe(
    context_factory, mode, libraries, capacity_answer, capacity_mbps, ema_capacity
):
    # main's library passes: package 420 Mbps, and its unchecked 48 000 Mbps speedtest figure
    # never reaches anything (main never read it)
    libraries.answers[LIB_PATH] = {
        **_probe_payload(single_stream_mbps=420.0),
        "speedtest": {"download_mbps": 48_000.0, "upload_mbps": 310.0},
    }
    libraries.answers[CAPACITY_LIB_PATH] = capacity_answer

    mode("off")
    main, _ = await _run_real(context_factory, executor_shell(UPDATED_EXECUTOR), prev_ema=400.0)
    mode("shadow")
    shadow, tally = await _run_real(
        context_factory, executor_shell(UPDATED_EXECUTOR), prev_ema=400.0
    )

    assert _outcome(shadow) == _outcome(main)
    assert main.passed is True
    assert main.updates["state"].specs["network"]["verifyx_download_speed"] == 420.0
    assert main.updates["state"].specs["network"]["ema_verifyx_download_speed"] == 410.0
    record = shadow.event.what_we_saw["network_gate"]
    assert record["package_download_speed"] == 420.0
    assert record["capacity_download_speed"] == capacity_mbps
    assert record["ema_package"] == 410.0
    assert record["ema_capacity"] == ema_capacity
    assert record["capacity_run"]["status"] == "measured"
    assert record["capacity_run"]["success"] is False
    assert record["capacity_run"]["errors"] == [
        f"Network execution failed: {capacity_answer['error']}"
    ]
    # one failed probe on a 400 Mbps EMA still clears the floor either way
    assert (tally.package_pass, tally.capacity_pass, tally.newly_fail) == (1, 1, 0)


@pytest.mark.asyncio
async def test_shadow_on_todays_executor_keeps_mains_outcome_and_counts_the_missing_library(
    context_factory, mode, libraries
):
    libraries.answers[LIB_PATH] = _probe_payload(single_stream_mbps=420.0)
    shell = executor_shell(TODAYS_EXECUTOR)
    mode("off")
    main, _ = await _run_real(context_factory, executor_shell(TODAYS_EXECUTOR), prev_ema=400.0)
    mode("shadow")

    shadow, tally = await _run_real(context_factory, shell, prev_ema=400.0)

    assert _outcome(shadow) == _outcome(main)
    assert asked_for(shell) == [LIB_PATH, CAPACITY_LIB_PATH] and len(commands(shell)) == 1
    record = shadow.event.what_we_saw["network_gate"]
    assert record["capacity_run"]["status"] == "library_missing"
    assert record["capacity_download_speed"] is None and record["ema_capacity"] is None
    assert (tally.capacity_library_missing, tally.unmeasured) == (1, 1)


@pytest.mark.asyncio
async def test_the_shadow_run_is_skipped_on_a_first_pass_and_fits_the_executor_task_budget(
    context_factory, mode
):
    mode("shadow")
    verification = _judge(_probe_payload(single_stream_mbps=300.0))

    async def run(*, first_pass: bool = False, started_seconds_ago: float = 0.0):
        service = DummyVerifyXService(success=True, updated_specs=verification)
        service.capacity_run = _measured(2400.0)
        ctx = context_factory(
            services=build_services(verifyx=service),
            config=build_context_config(verifyx_enabled=True, first_pass=first_pass),
            state=build_state(specs={"gpu": {"count": 8}, "network": {}}),
            started_at_monotonic=time.monotonic() - started_seconds_ago,
        )
        result = await VerifyXCheck().run(ctx)
        return result, getattr(service, "capacity_timeout_seconds", None)

    first_pass, first_pass_timeout = await run(first_pass=True)
    fresh, fresh_timeout = await run()
    # the executor task (JOB_TIME_OUT - 120 = 780 s) started 250 s ago: 780 - 250 - 300 s reserve
    later, later_timeout = await run(started_seconds_ago=250)
    # 780 - 400 - 300 = 80 s, under the 150 s a run needs
    late, late_timeout = await run(started_seconds_ago=400)

    assert first_pass.event.what_we_saw["network_gate"]["capacity_run"] == {
        "status": "skipped_first_pass"
    }
    assert first_pass_timeout is None
    assert fresh_timeout == check_module.CAPACITY_SHADOW_TIMEOUT_SECONDS == 240
    assert later_timeout == pytest.approx(230.0, abs=1.0)
    assert late.event.what_we_saw["network_gate"]["capacity_run"]["status"] == "skipped_no_budget"
    assert late_timeout is None
    assert first_pass.passed is fresh.passed is later.passed is late.passed is True


@pytest.mark.asyncio
async def test_a_shadow_run_that_raises_is_recorded_and_the_verdict_stands(context_factory, mode):
    mode("shadow")
    verification = _judge(_probe_payload(single_stream_mbps=300.0))
    service = DummyVerifyXService(success=True, updated_specs=verification)
    service.measure_capacity_shadow = AsyncMock(side_effect=RuntimeError("native crash"))
    ctx = context_factory(
        services=build_services(verifyx=service),
        config=build_context_config(verifyx_enabled=True),
        state=build_state(
            specs={"gpu": {"count": 8}, "network": {}},
            rented_data=_rented_data_with_ema("executor-123", download=400.0),
        ),
    )

    result = await VerifyXCheck().run(ctx)

    assert result.passed is True
    assert result.updates["state"].specs["network"]["ema_verifyx_download_speed"] == 350.0
    assert result.event.what_we_saw["network_gate"]["capacity_run"] == {
        "status": "run_failed",
        "error": "RuntimeError: native crash",
    }


@pytest.mark.asyncio
async def test_enforce_gates_and_publishes_the_capacity(context_factory, mode):
    mode("enforce")

    zero, zero_tally = await _run_check(context_factory, _judge(_probe_payload(**ZERO_CAPACITY)))
    fast_link, _ = await _run_check(
        context_factory, _judge(_probe_payload(**FAST_LINK_SLOW_OBJECT)), prev_ema=90.0
    )

    assert zero.passed is False
    assert zero.event.reason_code == Msg.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW.reason
    assert zero.updates["state"].specs["network"]["ema_verifyx_download_speed"] == 0.0
    assert zero.event.what_we_saw["network_gate"]["mode"] == "enforce"
    assert (zero_tally.package_pass, zero_tally.capacity_fail, zero_tally.newly_fail) == (1, 1, 1)

    assert fast_link.passed is True
    network = fast_link.updates["state"].specs["network"]
    assert network["verifyx_download_speed"] == 2400.0
    assert network["ema_verifyx_download_speed"] == 1245.0


def test_the_cycle_summary_counts_both_readings_once_and_resets(mode, caplog):
    mode("shadow")
    tally = NetworkGateTally()
    floor = MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS
    tally.record(350.0, 200.0, floor)
    tally.record(300.0, 0.0, floor)
    tally.record(85.0, 1245.0, floor)
    tally.record(300.0, None, floor, capacity_library_missing=True)
    tally.record(None, None, floor)
    tally.record_probe_failed()

    with caplog.at_level(logging.INFO):
        counts = tally.log_and_reset(floor, {"job_batch_id": "batch-1"})

    assert counts == {
        "package_pass": 2,
        "package_fail": 1,
        "capacity_pass": 2,
        "capacity_fail": 1,
        "newly_fail": 1,
        "newly_pass": 1,
        "unmeasured": 2,
        "capacity_library_missing": 1,
        "probe_failed": 1,
    }
    assert any(
        "VerifyX network gate summary mode=shadow floor_mbps=100 package_pass=2 package_fail=1 "
        "capacity_pass=2 capacity_fail=1 newly_fail=1 newly_pass=1 unmeasured=2 "
        "capacity_library_missing=1 probe_failed=1" in record.getMessage()
        for record in caplog.records
    )
    assert set(_counts(tally).values()) == {0}


@pytest.mark.asyncio
@pytest.mark.parametrize("gate_mode", ["shadow", "enforce", "off"])
async def test_each_validation_cycle_logs_one_summary_and_the_next_starts_from_zero(
    gate_mode, mode, tally, validator_with_mocks, create_neuron_info, caplog
):
    mode(gate_mode)
    tally.record(300.0, 0.0, MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS)

    with caplog.at_level(logging.INFO):
        await _run_sync_with_jobs(
            validator_with_mocks, [create_neuron_info(uid=100, hotkey="burner1")], {}
        )

    summaries = [
        record.getMessage()
        for record in caplog.records
        if "VerifyX network gate summary" in record.getMessage()
    ]
    if gate_mode == "off":
        assert summaries == []
        return
    assert len(summaries) == 1
    assert f"mode={gate_mode} floor_mbps=100 package_pass=1 package_fail=0" in summaries[0]
    assert "newly_fail=1" in summaries[0]
    assert set(_counts(tally).values()) == {0}
