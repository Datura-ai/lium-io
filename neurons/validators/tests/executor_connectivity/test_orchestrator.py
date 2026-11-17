import pytest

from services.executor_connectivity.models import DindProbeResult, PortPair, PortProbeResult
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.dind_probe import DindProbe
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_selector import PortSelector


@pytest.mark.asyncio
async def test_orchestrator_success(sample_executor_info, mock_ssh_client, mocker):
    selected_ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected_ports

    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(
        return_value=PortProbeResult(successful=tuple(selected_ports), failed=tuple())
    )

    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(
            success=True,
            sysbox_runtime=True,
            port=selected_ports[0],
            log_text="ok",
        )
    )

    orchestrator = ConnectivityOrchestrator(
        port_selector,
        port_probe,
        dind_probe,
    )

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=False,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.status == "ok"
    assert result.dind_ok is True
    assert result.sysbox_runtime is True
    assert len(result.successful_ports) == 2

    port_selector.select.assert_called_once()
    port_probe.probe.assert_called_once()
    dind_probe.verify.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_no_ports(sample_executor_info, mock_ssh_client, mocker):
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = []

    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock()

    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock()

    orchestrator = ConnectivityOrchestrator(
        port_selector,
        port_probe,
        dind_probe,
    )

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=False,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.status == "no_ports"
    assert result.successful_ports == tuple()
    assert result.failed_ports == tuple()
    assert result.dind_port is None
    assert result.dind_ok is False

    port_selector.select.assert_called_once()
    port_probe.probe.assert_not_called()
    dind_probe.verify.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_probe_fails_dind_ok(sample_executor_info, mock_ssh_client, mocker):
    selected_ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected_ports

    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(
        return_value=PortProbeResult(successful=tuple(), failed=tuple(selected_ports))
    )

    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(
            success=True,
            sysbox_runtime=True,
            port=selected_ports[0],
            log_text="ok",
        )
    )

    orchestrator = ConnectivityOrchestrator(
        port_selector,
        port_probe,
        dind_probe,
    )

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=False,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.status == "ok"
    assert result.dind_ok is True
    assert len(result.successful_ports) == 1
    assert result.successful_ports[0] in result.selected_ports

    port_selector.select.assert_called_once()
    port_probe.probe.assert_called_once()
    dind_probe.verify.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_probe_ok_dind_fails(sample_executor_info, mock_ssh_client, mocker):
    selected_ports = [PortPair(9000, 9000), PortPair(9001, 9001)]

    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected_ports

    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(
        return_value=PortProbeResult(successful=tuple(selected_ports), failed=tuple())
    )

    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(
            success=False,
            sysbox_runtime=False,
            port=selected_ports[0],
            log_text="fail",
        )
    )

    orchestrator = ConnectivityOrchestrator(
        port_selector,
        port_probe,
        dind_probe,
    )

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.status == "ok"
    assert result.dind_ok is False
    assert result.sysbox_runtime is False
    assert len(result.failed_ports) == 1
    assert result.failed_ports[0] in result.selected_ports

    port_selector.select.assert_called_once()
    port_probe.probe.assert_called_once()
    dind_probe.verify.assert_called_once()


@pytest.mark.parametrize(
    "requested, dind_success, dind_sysbox, measured",
    [
        (True, True, True, True),  # reached the container, sysbox-runc worked
        (True, True, False, False),  # reached the container, sysbox-runc failed: a verdict
        (True, False, True, None),  # never reached the container: measured nothing
        (False, True, False, None),  # sysbox-runc not requested: measured nothing
    ],
)
@pytest.mark.asyncio
async def test_orchestrator_reports_the_probes_own_sysbox_reading(
    sample_executor_info, mock_ssh_client, mocker, requested, dind_success, dind_sysbox, measured
):
    """ticket-0494: sysbox_measured is the DinD probe's reading, None unless it reached its container
    with sysbox-runc requested."""
    selected_ports = [PortPair(9000, 9000), PortPair(9001, 9001), PortPair(9002, 9002), PortPair(9003, 9003)]
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected_ports
    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(
        return_value=PortProbeResult(successful=tuple(selected_ports), failed=tuple())
    )
    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(success=dind_success, sysbox_runtime=dind_sysbox, port=selected_ports[0])
    )

    result = await ConnectivityOrchestrator(port_selector, port_probe, dind_probe).verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=requested,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.sysbox_measured is measured


@pytest.mark.asyncio
async def test_orchestrator_a_timed_out_sysbox_test_measured_nothing(sample_executor_info, mock_ssh_client, mocker):
    """30 Oct, pod c0877f60: the probe reached its container with sysbox-runc, hello-world timed out."""
    selected_ports = [PortPair(9000, 9000), PortPair(9001, 9001)]
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected_ports
    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(
        return_value=PortProbeResult(successful=tuple(selected_ports), failed=tuple())
    )
    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(
            success=True, sysbox_runtime=False, port=selected_ports[0], sysbox_timed_out=True
        )
    )

    result = await ConnectivityOrchestrator(port_selector, port_probe, dind_probe).verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.dind_ok is True
    assert result.sysbox_measured is None


@pytest.mark.asyncio
async def test_orchestrator_dind_skips_port_a_pod_published_mid_job(
    sample_executor_info, mock_ssh_client, mocker
):
    selected_ports = [PortPair(40000, 40000), PortPair(40001, 40001)]
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected_ports
    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(
        return_value=PortProbeResult(successful=tuple(selected_ports), failed=tuple())
    )
    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(success=True, sysbox_runtime=True, port=selected_ports[1], log_text="ok")
    )
    # nothing held when the job starts; a rental created mid-job holds 40000 in an exited container
    mock_ssh_client.run = mocker.AsyncMock(
        side_effect=[
            mocker.MagicMock(exit_status=0, stdout="", stderr=""),
            mocker.MagicMock(exit_status=0, stdout="/pod_bac6 40000 40002 \n/port_test_pub_ab 40001 \n", stderr=""),
        ]
    )
    orchestrator = ConnectivityOrchestrator(port_selector, port_probe, dind_probe)

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert dind_probe.verify.call_args.args[0] == PortPair(40001, 40001)
    assert result.dind_port == PortPair(40001, 40001)


@pytest.mark.asyncio
async def test_orchestrator_does_not_select_ports_held_by_a_stopped_container_at_start(
    sample_executor_info, mock_ssh_client, mocker
):
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = [PortPair(40001, 40001)]
    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(return_value=PortProbeResult(successful=(PortPair(40001, 40001),), failed=()))
    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(success=True, sysbox_runtime=True, port=PortPair(40001, 40001), log_text="ok")
    )
    mock_ssh_client.run = mocker.AsyncMock(
        return_value=mocker.MagicMock(exit_status=0, stdout="/pod_bac6 40000 \n/container_probe 40005 \n", stderr="")
    )
    orchestrator = ConnectivityOrchestrator(port_selector, port_probe, dind_probe)

    await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[40100],
        ssh_client=mock_ssh_client,
    )

    # our own probe containers' ports are not a pod's, so only the pod's port is excluded
    assert port_selector.select.call_args.args[2] == {40100}
    assert 40000 not in {p.internal for p in port_selector.select.call_args.kwargs["declared"]}


@pytest.mark.asyncio
async def test_orchestrator_drops_a_held_host_port_from_a_mapped_pair(sample_executor_info, mock_ssh_client, mocker):
    """DAH-6376 review: the inventory lists host (internal) ports, so a held 46681 drops the pair [46681, 56681],
    not a pair whose external port is 46681."""
    sample_executor_info.port_mappings = "[[46681, 56681], [46682, 56682], [46683, 46681]]"
    sample_executor_info.port_range = None
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = []
    mock_ssh_client.run = mocker.AsyncMock(
        return_value=mocker.MagicMock(exit_status=0, stdout="/pod_bac6 46681 \n", stderr="")
    )
    orchestrator = ConnectivityOrchestrator(port_selector, mocker.Mock(spec=PortProbe), mocker.Mock(spec=DindProbe))

    await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[40100],
        ssh_client=mock_ssh_client,
    )

    assert port_selector.select.call_args.args[2] == {40100}
    assert {p.internal for p in port_selector.select.call_args.kwargs["declared"]} == {46682, 46683}


def _orchestrator_with_reads(mocker, mock_ssh_client, reads):
    selected_ports = [PortPair(40000, 40000), PortPair(40001, 40001)]
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected_ports
    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(
        return_value=PortProbeResult(successful=tuple(selected_ports), failed=tuple())
    )
    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(success=True, sysbox_runtime=True, port=selected_ports[0], log_text="ok")
    )
    mock_ssh_client.run = mocker.AsyncMock(side_effect=reads)
    return ConnectivityOrchestrator(port_selector, port_probe, dind_probe), dind_probe


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reads",
    [
        # every candidate is held by the time the DinD port is chosen
        "all_held_at_second_read",
        "read_raises",
        "read_exits_non_zero",
        "read_unparsable",
        "read_partly_unparsable",
    ],
)
async def test_orchestrator_skips_dind_when_no_candidate_port_is_known_free(
    sample_executor_info, mock_ssh_client, mocker, reads
):
    ok = mocker.MagicMock(exit_status=0, stdout="", stderr="")
    second = {
        "all_held_at_second_read": mocker.MagicMock(exit_status=0, stdout="/pod_a 40000 40001 \n", stderr=""),
        "read_raises": ConnectionError("channel closed"),
        "read_exits_non_zero": mocker.MagicMock(exit_status=1, stdout="", stderr="permission denied"),
        "read_unparsable": mocker.MagicMock(exit_status=0, stdout="Cannot connect to the Docker daemon\n", stderr=""),
        "read_partly_unparsable": mocker.MagicMock(exit_status=0, stdout="/pod_a 40002 \n/pod_b <no value>\n", stderr=""),
    }[reads]
    orchestrator, dind_probe = _orchestrator_with_reads(mocker, mock_ssh_client, [ok, second])

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    dind_probe.verify.assert_not_called()
    assert result.dind_skipped is True
    assert result.dind_port is None
    assert result.dind_ok is False
    # neutral: the batch ports that answered count, the skipped probe adds neither a pass nor a failure
    assert result.status == "ok"
    assert len(result.successful_ports) == 2
    assert result.failed_ports == ()


@pytest.mark.asyncio
async def test_orchestrator_skips_dind_when_the_inventory_is_unknown_at_start(
    sample_executor_info, mock_ssh_client, mocker
):
    # the second read is never made: an unknown inventory at the start is not trusted later
    orchestrator, dind_probe = _orchestrator_with_reads(
        mocker, mock_ssh_client, [mocker.MagicMock(exit_status=1, stdout="", stderr="")]
    )

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    dind_probe.verify.assert_not_called()
    assert result.dind_skipped is True
    assert mock_ssh_client.run.call_count == 1


class _Claims:
    def __init__(self, pending=frozenset()):
        self.pending = set(pending)
        self.held: dict[str, set[int]] = {}
        self.released: list[str] = []

    async def pending_pod_ports(self, executor_id):
        return self.pending

    async def hold_probe_ports(self, executor_id, external_ports, ttl_seconds):
        self.held[executor_id] = set(external_ports)

    async def release_probe_ports(self, executor_id):
        self.released.append(executor_id)


@pytest.mark.asyncio
async def test_orchestrator_holds_its_ports_skips_a_new_pods_ports_and_releases(
    sample_executor_info, mock_ssh_client, mocker
):
    """DAH-6376 (58760e62, port 19898): a pod given its ports after the backend's rental list was read is skipped,
    and the batch test publishes the ports it binds until it ends."""
    selected = [PortPair(9001, 9001), PortPair(9002, 9002)]
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected
    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(return_value=PortProbeResult(successful=tuple(selected), failed=()))
    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(success=True, sysbox_runtime=True, port=selected[1], log_text="ok")
    )
    mock_ssh_client.run = mocker.AsyncMock(return_value=mocker.MagicMock(exit_status=0, stdout="", stderr=""))
    claims = _Claims(pending={19898})
    orchestrator = ConnectivityOrchestrator(port_selector, port_probe, dind_probe, port_claims=claims)

    await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[40100],
        ssh_client=mock_ssh_client,
    )

    assert port_selector.select.call_args.args[2] == {40100, 19898}
    assert claims.held[sample_executor_info.uuid] == {9001, 9002}
    assert claims.released == [sample_executor_info.uuid]


@pytest.mark.asyncio
async def test_orchestrator_dind_skips_a_port_a_new_pod_was_given_during_the_batch_test(
    sample_executor_info, mock_ssh_client, mocker
):
    selected = [PortPair(9001, 9001), PortPair(9002, 9002)]
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = selected
    port_probe = mocker.Mock(spec=PortProbe)
    port_probe.probe = mocker.AsyncMock(return_value=PortProbeResult(successful=tuple(selected), failed=()))
    dind_probe = mocker.Mock(spec=DindProbe)
    dind_probe.verify = mocker.AsyncMock(
        return_value=DindProbeResult(success=True, sysbox_runtime=True, port=selected[1], log_text="ok")
    )
    mock_ssh_client.run = mocker.AsyncMock(return_value=mocker.MagicMock(exit_status=0, stdout="", stderr=""))
    claims = _Claims()
    reads = iter([set(), {9001}])
    claims.pending_pod_ports = mocker.AsyncMock(side_effect=lambda _: next(reads))
    orchestrator = ConnectivityOrchestrator(port_selector, port_probe, dind_probe, port_claims=claims)

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert dind_probe.verify.call_args.args[0] == PortPair(9002, 9002)
    assert result.dind_port == PortPair(9002, 9002)


@pytest.mark.asyncio
async def test_orchestrator_a_failing_claims_store_never_fails_the_check(sample_executor_info, mock_ssh_client, mocker):
    port_selector = mocker.Mock(spec=PortSelector)
    port_selector.select.return_value = []
    claims = mocker.Mock()
    claims.pending_pod_ports = mocker.AsyncMock(side_effect=ConnectionError("redis down"))
    claims.hold_probe_ports = mocker.AsyncMock(side_effect=ConnectionError("redis down"))
    claims.release_probe_ports = mocker.AsyncMock(side_effect=ConnectionError("redis down"))
    orchestrator = ConnectivityOrchestrator(
        port_selector, mocker.Mock(spec=PortProbe), mocker.Mock(spec=DindProbe), port_claims=claims
    )

    result = await orchestrator.verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.status == "no_ports"
