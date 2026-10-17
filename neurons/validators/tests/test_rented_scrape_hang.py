"""A rented node whose scrape hangs on the host for 6 h in a row is reported SCRAPE_HUNG_ON_HOST and reset.

A cycle counts only when the scrape timed out and the host then answered a probe on the same connection; a
probe that fails (the validator's own network, the host's link) is undetermined and ends the run.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from neurons.validators.src.services.task.checks import machine_spec_scrape
from neurons.validators.src.services.task.checks.machine_spec_scrape import MachineSpecScrapeCheck
from neurons.validators.src.services.task.messages import MachineSpecMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult
from test_rented_machine_check import build_rented_data

from helpers import build_context_config, build_services, build_state

EXECUTOR = "executor-123"
RENTAL_OWNER = "test-miner"
CYCLE_S = 15 * 60
T0 = 1_800_000_000.0


def _result(command: str, *, success: bool, error_type: str | None = None) -> SSHCommandResult:
    now = datetime.now(UTC)
    return SSHCommandResult(
        command=command,
        command_id="cmd",
        exit_code=0 if success else -1,
        stdout="",
        stderr="",
        duration_ms=300_000 if error_type == "timeout" else 10,
        started_at=now,
        finished_at=now,
        success=success,
        error_type=error_type,
    )


class _Runner:
    """The scrape times out; the probe after it answers or not."""

    def __init__(self, probe_answers: bool):
        self.probe_answers = probe_answers

    async def run(self, command, timeout=300, retryable=False, stdin_text=None):
        if command == "true":
            return _result(command, success=self.probe_answers, error_type=None if self.probe_answers else "timeout")
        return _result(command, success=False, error_type="timeout")


def _ctx(context_factory, services, runner, *, rented: bool = True):
    containers = [{"name": "tenant-123", "pod_id": "pod-1"}] if rented else []
    return context_factory(
        miner_hotkey=RENTAL_OWNER,
        services=services,
        config=build_context_config(machine_scrape_filename="scrape.sh", obfuscation_keys={}),
        state=build_state(
            remote_dir="/remote",
            rented_data=build_rented_data(EXECUTOR, {"containers": containers, "owner_flag": False}),
        ),
        runner=runner,
    )


async def _cycles(context_factory, services, offsets_s, probe_answers):
    """Runs one scrape per offset; probe_answers(offset) says whether the host answered after the timeout."""
    result = None
    for offset in offsets_s:
        ctx = _ctx(context_factory, services, _Runner(probe_answers(offset)))
        with patch.object(machine_spec_scrape.time, "time", return_value=T0 + offset):
            result = await MachineSpecScrapeCheck().run(ctx)
    return result


def _every_cycle_until(last_s: int) -> list[int]:
    return [*range(0, last_s, CYCLE_S), last_s]


@pytest.mark.asyncio
async def test_six_hours_of_host_side_hangs_report_a_host_fault_and_reset(context_factory):
    result = await _cycles(context_factory, build_services(), _every_cycle_until(6 * 3600), lambda _: True)

    assert result.passed is False
    assert result.event.reason_code == Msg.SCRAPE_HUNG_ON_HOST.reason
    assert result.event.what_we_saw["host_side_hang_s"] == 6 * 3600
    assert result.updates["clear_verified_job_info"] is True
    assert result.updates["clear_verified_job_evidence"]["reason_code"] == Msg.SCRAPE_HUNG_ON_HOST.reason
    assert result.updates["clear_verified_job_evidence"]["rented_pod_ids"] == ["pod-1"]


@pytest.mark.asyncio
async def test_one_minute_short_of_six_hours_stays_undetermined(context_factory):
    result = await _cycles(context_factory, build_services(), _every_cycle_until(6 * 3600 - 60), lambda _: True)

    assert result.event.reason_code == Msg.SCRAPE_TIMEOUT.reason
    assert result.event.what_we_saw["host_side_hang_s"] == 6 * 3600 - 60
    assert not result.updates


@pytest.mark.asyncio
async def test_timeouts_the_host_did_not_answer_after_are_never_converted(context_factory):
    result = await _cycles(context_factory, build_services(), _every_cycle_until(12 * 3600), lambda _: False)

    assert result.event.reason_code == Msg.SCRAPE_TIMEOUT.reason
    assert not result.updates


@pytest.mark.asyncio
async def test_one_validator_side_cycle_restarts_the_run(context_factory):
    # the probe fails once at 3 h, so at 6 h the host-side run is only 3 h long
    result = await _cycles(
        context_factory, build_services(), _every_cycle_until(6 * 3600), lambda offset: offset != 3 * 3600
    )

    assert result.event.reason_code == Msg.SCRAPE_TIMEOUT.reason
    assert result.event.what_we_saw["host_side_hang_s"] == 3 * 3600 - CYCLE_S
    assert not result.updates


@pytest.mark.asyncio
async def test_an_unrented_node_is_left_alone(context_factory):
    services = build_services()
    result = None
    for offset in _every_cycle_until(6 * 3600):
        ctx = _ctx(context_factory, services, _Runner(True), rented=False)
        with patch.object(machine_spec_scrape.time, "time", return_value=T0 + offset):
            result = await MachineSpecScrapeCheck().run(ctx)

    assert result.event.reason_code == Msg.SCRAPE_TIMEOUT.reason
    assert not result.updates


@pytest.mark.asyncio
async def test_the_kill_switch_keeps_the_timeout(context_factory):
    with patch.object(machine_spec_scrape.settings, "RENTED_SCRAPE_HANG_RESET_ENABLED", False):
        result = await _cycles(context_factory, build_services(), _every_cycle_until(6 * 3600), lambda _: True)

    assert result.event.reason_code == Msg.SCRAPE_TIMEOUT.reason
    assert not result.updates


@pytest.mark.asyncio
async def test_a_cycle_without_a_rental_restarts_the_run_for_the_next_rental(context_factory):
    services = build_services()
    # 5 h of host-side hangs on one rental, then one unrented cycle (clean or not)
    await _cycles(context_factory, services, _every_cycle_until(5 * 3600), lambda _: True)
    ctx = _ctx(context_factory, services, _Runner(True), rented=False)
    with patch.object(machine_spec_scrape.time, "time", return_value=T0 + 5 * 3600 + CYCLE_S):
        await MachineSpecScrapeCheck().run(ctx)

    # a new rental 1 h 15 min later would be past 6 h on the old start; it counts from its own first cycle
    start = 5 * 3600 + 2 * CYCLE_S
    result = await _cycles(context_factory, services, [start, start + 3600], lambda _: True)

    assert result.event.reason_code == Msg.SCRAPE_TIMEOUT.reason
    assert result.event.what_we_saw["host_side_hang_s"] == 3600
    assert not result.updates
