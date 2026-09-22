"""DAH-2774: VERIFYX_NETWORK_GATE_MODE ships the capacity reading shadow-first.

The network number moves from the one-object package download to the Cloudflare capacity probe of
a vendored libverifyx.so (celium-gpu-verifier#25); two earlier builds of that library were wrong,
one read every host's download as 0. So the reading that gates is a setting:

- off: the package download gates and is published; the capacity reading is not recorded;
- shadow (the default): the package download gates and is published, exactly as off; the capacity
  reading and the EMA it would give ride on the event (`network_gate`) and in the cycle summary;
- enforce: the capacity reading gates and is published.

Each payload goes through the real `_perform_verification_checks` and the real `VerifyXCheck`, so
what is asserted is the verdict and the `specs.network` keys lium-platform lists from.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from neurons.validators.src.core.config import VerifyXSettings
from neurons.validators.src.services.task.checks import verifyx as check_module
from neurons.validators.src.services.task.checks.verifyx import (
    MIN_VERIFYX_EMA_DOWNLOAD_SPEED_MBPS,
    VerifyXCheck,
)
from neurons.validators.src.services.task.messages import VerifyXMessages as Msg
from neurons.validators.src.services.verifyx_validation_service import (
    NetworkGateTally,
    _verify_network_test,
    settings,
)
from tests.helpers import build_context_config, build_services, build_state
from tests.test_incentive_flow import _run_sync_with_jobs
from tests.test_verifyx_capacity_probe import _challenge_data, _judge, _probe_payload
from tests.test_verifyx_check import DummyVerifyXService, _rented_data_with_ema

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
def mode(monkeypatch):
    def set_mode(value: str) -> None:
        monkeypatch.setattr(settings.verifyx, "NETWORK_GATE_MODE", value)

    return set_mode


async def _run_check(context_factory, verification_result: dict, *, prev_ema=None):
    tally = _zero(check_module.NETWORK_GATE_TALLY)
    verifyx_service = DummyVerifyXService(success=True, updated_specs=verification_result)
    ctx = context_factory(
        services=build_services(verifyx=verifyx_service),
        config=build_context_config(verifyx_enabled=True),
        state=build_state(
            specs={"gpu": {"count": 8}, "network": {}},
            rented_data=_rented_data_with_ema("executor-123", download=prev_ema),
        ),
    )
    result = await VerifyXCheck().run(ctx)
    return result, NetworkGateTally(**_counts(tally))


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
        context_factory, _judge(_probe_payload(**ZERO_CAPACITY)), prev_ema=400.0
    )
    slow_object, slow_tally = await _run_check(
        context_factory, _judge(_probe_payload(**FAST_LINK_SLOW_OBJECT)), prev_ema=90.0
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

    result, tally = await _run_check(context_factory, _judge(_probe_payload(**ZERO_CAPACITY)))

    assert result.passed is True
    assert result.event.what_we_saw["network_gate"]["ema_package"] == 300.0
    assert result.event.what_we_saw["network_gate"]["ema_capacity"] == 0.0
    assert (tally.package_pass, tally.capacity_fail, tally.newly_fail) == (1, 1, 1)


def test_shadow_gates_like_off_without_a_cloudflare_reading(mode):
    mode("shadow")
    payload = _probe_payload(single_stream_mbps=300.0)
    del payload["speedtest"]

    stats, errors = _verify_network_test(_challenge_data(), {"network_execution": payload})

    assert errors == []
    assert stats["success"] is True
    assert stats["download_speed"] == 300.0
    assert stats["capacity_download_speed"] is None


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
    tally.record(85.0, 1245.0, floor, previous_library=True)
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
        "unmeasured": 1,
        "previous_library": 1,
        "probe_failed": 1,
    }
    assert any(
        "VerifyX network gate summary mode=shadow floor_mbps=100 package_pass=2 package_fail=1 "
        "capacity_pass=2 capacity_fail=1 newly_fail=1 newly_pass=1 unmeasured=1 "
        "previous_library=1 probe_failed=1" in record.getMessage()
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
