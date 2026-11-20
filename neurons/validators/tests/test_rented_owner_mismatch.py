"""G15: a rental is the node's only when the backend's rental names the miner being validated.

The backend keys rentals by executor UUID, which the miner reports. A box listed under another miner's
rented UUID would halt as rented and keep its score without the matmul ever running.
"""

import logging

import pytest

from neurons.validators.src.services.task.checks import registry_pull, rental_owner
from neurons.validators.src.services.task.checks.port_count import PortCountCheck
from neurons.validators.src.services.task.checks.rented_machine import TenantEnforcementCheck
from neurons.validators.src.services.task.messages import TenantEnforcementMessages as Msg
from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse, RentedPod
from test_rented_machine_check import DummyScoreCalculator, DummySSHClient, MockContainerCleanup

from helpers import build_context_config, build_services, build_state

EXECUTOR_UUID = "executor-123"


def _rented_by(miner_hotkey: str, port: str = "22") -> RentedExecutorsResponse:
    # The context's executor is 127.0.0.1:22 (helpers.default_executor).
    return RentedExecutorsResponse(
        executors={
            EXECUTOR_UUID: RentedExecutor(
                miner_hotkey=miner_hotkey,
                executor_ip_address="127.0.0.1",
                executor_ip_port=port,
                pods=[RentedPod(pod_id="pod-1", container_name="tenant-1", rented_ports=[])],
            )
        },
        banned_guids=[],
    )


def _ctx(context_factory, rental_miner_hotkey: str, rental_port: str = "22"):
    services = build_services(
        score_calculator=DummyScoreCalculator(actual_score=1.0, job_score=1.0, warning=""),
        container_cleanup=MockContainerCleanup(),
    )
    state = build_state(
        gpu_processes=[],
        gpu_details=[],
        gpu_model="NVIDIA RTX 4090",
        rented_data=_rented_by(rental_miner_hotkey, rental_port),
    )
    return context_factory(
        services=services,
        config=build_context_config(),
        state=state,
        ssh=DummySSHClient(pod_running=True),
        miner_hotkey="miner-hotkey",
        collateral_deposited=True,
        is_rental_succeed=True,
        contract_version="v1.0.0",
    )


def _mismatch_logs(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if "RENTED_OWNER_MISMATCH" in record.getMessage()]


@pytest.mark.asyncio
async def test_own_rental_logs_no_mismatch(context_factory, caplog):
    ctx = _ctx(context_factory, rental_miner_hotkey="miner-hotkey")

    with caplog.at_level(logging.WARNING, logger=rental_owner.logger.name):
        result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code != Msg.NOT_RENTED.reason
    assert _mismatch_logs(caplog) == []


@pytest.mark.asyncio
async def test_another_miners_rental_is_logged_and_validated_as_not_rented(context_factory, caplog):
    ctx = _ctx(context_factory, rental_miner_hotkey="other-miner-hotkey", rental_port="2222")

    with caplog.at_level(logging.WARNING, logger=rental_owner.logger.name):
        result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code == Msg.NOT_RENTED.reason
    assert result.updates["rented"] is False
    assert len(_mismatch_logs(caplog)) == 1


@pytest.mark.asyncio
async def test_another_miners_rental_on_same_endpoint_stays_rented(context_factory, caplog):
    """Same IP and port: the renter's pod is on this box, so the probes must not run beside it."""
    ctx = _ctx(context_factory, rental_miner_hotkey="other-miner-hotkey")

    with caplog.at_level(logging.WARNING, logger=rental_owner.logger.name):
        result = await TenantEnforcementCheck().run(ctx)

    assert result.event.reason_code != Msg.NOT_RENTED.reason
    assert len(_mismatch_logs(caplog)) == 1


@pytest.mark.asyncio
async def test_port_count_floor_applies_beside_another_miners_rental(context_factory, caplog):
    ctx = _ctx(context_factory, rental_miner_hotkey="other-miner-hotkey", rental_port="2222")

    with caplog.at_level(logging.WARNING, logger=rental_owner.logger.name):
        result = await PortCountCheck().run(ctx)

    # not rented, so the empty port range fails the floor; only TenantEnforcementCheck logs the mismatch
    assert result.passed is False
    assert _mismatch_logs(caplog) == []


def test_registry_pull_runs_beside_another_miners_rental(context_factory):
    # RegistryPullCheck skips a rented node; another miner's rental on another endpoint is not this node's
    assert registry_pull._is_rented(_ctx(context_factory, rental_miner_hotkey="miner-hotkey")) is True
    assert registry_pull._is_rented(_ctx(context_factory, "other-miner-hotkey", rental_port="2222")) is False


# DAH-4263 review on #1834: every other rental reader goes through rental_of_this_node too.
OTHER = {"rental_miner_hotkey": "other-miner-hotkey", "rental_port": "2222"}


@pytest.mark.asyncio
async def test_capability_re_read_ignores_another_miners_rental(context_factory):
    from unittest.mock import AsyncMock

    from neurons.validators.src.services.task.checks.capability import lium_workload_live_now

    ctx = _ctx(context_factory, **OTHER)
    ctx.services.backend.get_rented_executors_now = AsyncMock(return_value=_rented_by("other-miner-hotkey", "2222"))
    assert await lium_workload_live_now(ctx) is None
    ctx.services.backend.get_rented_executors_now = AsyncMock(return_value=_rented_by("miner-hotkey"))
    assert (await lium_workload_live_now(ctx)).kind == "pod"


@pytest.mark.asyncio
async def test_pid_host_fails_beside_another_miners_rental(context_factory):
    from neurons.validators.src.services.task.checks.pid_host_required import PidHostRequiredCheck

    def run(rental_miner_hotkey, rental_port="22"):
        state = build_state(
            specs={"executor_pid_mode": "private"}, rented_data=_rented_by(rental_miner_hotkey, rental_port)
        )
        return PidHostRequiredCheck().run(context_factory(state=state, miner_hotkey="miner-hotkey"))

    assert (await run("miner-hotkey")).passed is True
    assert (await run(**OTHER)).passed is False


@pytest.mark.asyncio
async def test_container_download_speed_measures_beside_another_miners_rental(context_factory, monkeypatch):
    from neurons.validators.src.services.task.checks import container_download_speed as cds

    monkeypatch.setattr(cds.settings, "CONTAINER_DOWNLOAD_SPEED_CHECK_ENABLED", True)

    async def unreadable(*_args):
        raise RuntimeError("redis down")

    monkeypatch.setattr(cds, "get_or_adopt_legacy", unreadable)
    own = await cds.ContainerDownloadSpeedCheck().run(_ctx(context_factory, rental_miner_hotkey="miner-hotkey"))
    other = await cds.ContainerDownloadSpeedCheck().run(_ctx(context_factory, **OTHER))
    assert "rented" in str(own.event) and "rented" not in str(other.event)


def test_gpu_usage_backend_answer_is_not_this_nodes_under_another_miners_rental(context_factory):
    from neurons.validators.src.services.task.checks.gpu_usage import _the_answer_is_about_this_node

    assert _the_answer_is_about_this_node(_ctx(context_factory, rental_miner_hotkey="miner-hotkey"), EXECUTOR_UUID)
    assert not _the_answer_is_about_this_node(_ctx(context_factory, **OTHER), EXECUTOR_UUID)
    assert not _the_answer_is_about_this_node(_ctx(context_factory, **OTHER), None)


@pytest.mark.asyncio
async def test_provider_side_load_excuses_no_build_of_another_miners_pod(context_factory, monkeypatch):
    from neurons.validators.src.services.task.checks import provider_side_load as psl

    async def none_late(*_args):
        return set()

    monkeypatch.setattr(psl, "late_started_containers_the_backend_owns", none_late)
    build = f"{psl.BUILD_DIND_PREFIX}pod-1"
    assert build in await psl.lium_containers(_ctx(context_factory, rental_miner_hotkey="miner-hotkey"))
    assert build not in await psl.lium_containers(_ctx(context_factory, **OTHER))


@pytest.mark.asyncio
async def test_rental_probe_runs_beside_another_miners_rental(context_factory):
    from neurons.validators.src.services.task.checks.rental_probe import _skip_reason

    assert (await _skip_reason(_ctx(context_factory, rental_miner_hotkey="miner-hotkey")))[0] == "rented"
    assert (await _skip_reason(_ctx(context_factory, **OTHER)))[0] != "rented"


def test_banned_provider_live_rental_gate_ignores_another_miners_rental(context_factory):
    from neurons.validators.src.services.task.checks.banned_provider import has_live_rental

    assert has_live_rental(_ctx(context_factory, rental_miner_hotkey="miner-hotkey")) is True
    assert has_live_rental(_ctx(context_factory, **OTHER)) is False


# a filler listed under this UUID for another miner is not this node's
def _filler_owned_by(owner_hotkey: str) -> RentedExecutorsResponse:
    return RentedExecutorsResponse(
        executors={},
        all_filler_containers_by_executor={EXECUTOR_UUID: ["filler_1"]},
        filler_ports_by_executor={EXECUTOR_UUID: [2001, 2002]},
        executor_owner_hotkeys={EXECUTOR_UUID: owner_hotkey},
    )


@pytest.mark.asyncio
async def test_capability_re_read_ignores_another_miners_filler(context_factory):
    from unittest.mock import AsyncMock

    from neurons.validators.src.services.task.checks.capability import lium_workload_live_now

    ctx = _ctx(context_factory, **OTHER)
    ctx.services.backend.get_rented_executors_now = AsyncMock(return_value=_filler_owned_by("other-miner-hotkey"))
    assert await lium_workload_live_now(ctx) is None
    ctx.services.backend.get_rented_executors_now = AsyncMock(return_value=_filler_owned_by("miner-hotkey"))
    assert (await lium_workload_live_now(ctx)).kind == "filler"


@pytest.mark.asyncio
async def test_port_count_ignores_another_miners_filler_ports(context_factory):
    def run(owner_hotkey: str):
        state = build_state(
            verified_port_count=0, specs={"verified_ports": []}, rented_data=_filler_owned_by(owner_hotkey)
        )
        return PortCountCheck().run(context_factory(state=state))

    assert (await run("other-miner-hotkey")).updates["port_count"] == 0
    assert (await run("miner-hotkey")).updates["port_count"] == 2
