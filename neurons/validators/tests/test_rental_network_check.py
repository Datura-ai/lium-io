"""Validation lists only a node whose lium-rentals network the rental create accepts."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from services.rental_docker_sdk import RentalDockerOperationError, _require_icc_off
from services.task.availability import AvailabilityErrorCode, run_availability_errors
from services.task.checks.rental_network import RentalNetworkCheck
from services.task.checks.score import ScoreCheck
from services.task.pipeline import LoggerSink, Pipeline
from services.task.pipeline_factory import PipelineFactory
from services.task.runner import SSHCommandRunner

from tests.helpers import build_services, build_state, make_context
from tests.test_sysbox_required_check import _rented_data

ICC = "com.docker.network.bridge.enable_icc"
REFUSED = [
    {"Driver": "bridge", "Options": {}},  # created by hand without the option: ICC on
    {"Driver": "bridge", "Options": {ICC: "true"}},
    {"Driver": "overlay", "Options": {ICC: "false"}},
]
ACCEPTED = [
    {"Driver": "bridge", "Options": {ICC: "false"}},
    {"Driver": "bridge", "Options": {ICC: "false", "com.docker.network.driver.mtu": "1450"}},
]


def _ctx(ssh, rented: bool = False):
    state = build_state(rented_data=_rented_data() if rented else None)
    return make_context(ssh=ssh, runner=SSHCommandRunner(ssh), state=state)


def _ssh(exit_status: int, stdout: str):
    ssh = AsyncMock()
    ssh.run = AsyncMock(return_value=Mock(exit_status=exit_status, stdout=stdout, stderr=""))
    return ssh


def _inspect(network: dict):
    return _ssh(0, json.dumps([{"Name": "lium-rentals", **network}]))


@pytest.mark.asyncio
@pytest.mark.parametrize("network", REFUSED)
async def test_an_unrented_node_with_a_refused_network_is_hidden(network):
    result = await RentalNetworkCheck().run(_ctx(_inspect(network)))

    assert result.passed is False
    assert result.fatal is None and RentalNetworkCheck.fatal is False
    assert result.event.reason_code == AvailabilityErrorCode.RENTAL_NETWORK_UNUSABLE
    assert result.event.is_availability_error
    assert result.updates["state"].rental_network_usable is False


@pytest.mark.asyncio
async def test_a_rented_node_is_hidden_and_its_rental_runs_on():
    result = await RentalNetworkCheck().run(_ctx(_inspect(REFUSED[0]), rented=True))

    assert result.passed is False
    assert result.fatal is None and RentalNetworkCheck.fatal is False


class _Log:
    def info(self, *a, **k): ...
    warning = error = debug = info


@pytest.mark.asyncio
@pytest.mark.parametrize("rented", [False, True])
async def test_a_refused_network_hides_the_node_and_keeps_its_score(rented):
    """Review (lio#2529 at 02a1530): a fatal stop before ScoreCheck left an idle node at score 0 as well as hidden."""

    async def run(network):
        ssh = _inspect(network)
        state = build_state(rented_data=_rented_data() if rented else None)
        services = build_services(score_calculator=lambda ctx, is_rented: (0.7, 0.4, ""))
        ctx = make_context(ssh=ssh, runner=SSHCommandRunner(ssh), state=state, services=services, rented=rented)
        ok, events, end = await Pipeline([RentalNetworkCheck(), ScoreCheck()], sink=LoggerSink(_Log())).run(ctx)
        return ok, run_availability_errors(ok, events), (end.score, end.job_score)

    accepted, refused = await run(ACCEPTED[0]), await run(REFUSED[0])

    assert accepted[2] == refused[2] == (0.7, 0.4)
    assert accepted[1] == []
    assert [e["reason_code"] for e in refused[1]] == [AvailabilityErrorCode.RENTAL_NETWORK_UNUSABLE]


@pytest.mark.asyncio
@pytest.mark.parametrize("network", ACCEPTED)
async def test_an_isolated_bridge_passes(network):
    result = await RentalNetworkCheck().run(_ctx(_inspect(network)))

    assert result.passed is True
    assert result.event.reason_code == "RENTAL_NETWORK_OK"


@pytest.mark.asyncio
async def test_a_host_without_the_network_passes_the_create_makes_it():
    result = await RentalNetworkCheck().run(_ctx(_ssh(1, "[]\n")))

    assert result.passed is True
    assert result.event.reason_code == "RENTAL_NETWORK_OK"


@pytest.mark.asyncio
@pytest.mark.parametrize(("exit_status", "stdout"), [(1, ""), (127, "sh: docker: not found\n"), (0, "garbage")])
async def test_an_answer_that_says_nothing_passes(exit_status, stdout):
    result = await RentalNetworkCheck().run(_ctx(_ssh(exit_status, stdout)))

    assert result.passed is True
    assert result.event.reason_code == "RENTAL_NETWORK_NOT_READ"


@pytest.mark.asyncio
async def test_a_read_with_no_answer_passes():
    ssh = AsyncMock()
    ssh.run = AsyncMock(side_effect=asyncio.TimeoutError)

    result = await RentalNetworkCheck().run(_ctx(ssh))

    assert result.passed is True
    assert result.event.reason_code == "RENTAL_NETWORK_NOT_READ"


@pytest.mark.asyncio
@pytest.mark.parametrize("network", REFUSED + ACCEPTED)
async def test_validation_and_the_create_agree_on_every_network(network):
    try:
        _require_icc_off("lium-rentals", network)
        create_refuses = False
    except RentalDockerOperationError:
        create_refuses = True

    listed = (await RentalNetworkCheck().run(_ctx(_inspect(network)))).passed

    assert listed is not create_refuses


@pytest.mark.parametrize(
    "build",
    [PipelineFactory.build_checks, PipelineFactory.build_fast_path_checks, PipelineFactory.build_dry_run_checks],
)
def test_every_pipeline_reads_it_beside_the_pid_max_read(build):
    names = [type(c).__name__ for c in build()]
    assert names.index("RentalNetworkCheck") == names.index("HostPidMaxCheck") + 1
