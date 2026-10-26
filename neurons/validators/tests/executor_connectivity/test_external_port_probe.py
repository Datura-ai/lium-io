import importlib.util
import json
from pathlib import Path

import pytest
from aiohttp import web
from core.config import settings
from services.executor_connectivity.external_probe import ExternalPortProbe
from services.executor_connectivity.models import (
    BatchResult,
    ContainerStartResult,
    DindProbeResult,
    PortPair,
    PortProbeResult,
)
from services.executor_connectivity.orchestrator import ConnectivityOrchestrator
from services.executor_connectivity.port_probe import PortProbe
from services.executor_connectivity.port_verifiers import BatchVerifier

PORTS = [PortPair(9000 + i, 9000 + i) for i in range(5)]


def _verifier(mocker, reached):
    runner = mocker.Mock()
    runner.run = mocker.AsyncMock(return_value=ContainerStartResult(ok=True, container_id="c", status="Up"))
    runner.cleanup = mocker.AsyncMock()
    port_tester = mocker.Mock()
    port_tester.test_many = mocker.AsyncMock(side_effect=lambda session, host, tested, token, log_ctx=None: (tested, []))
    external = mocker.Mock(spec=ExternalPortProbe)
    external.reached = mocker.AsyncMock(side_effect=lambda host, sample, token: reached(sample))
    return BatchVerifier(port_tester, runner, external), port_tester


@pytest.mark.asyncio
async def test_batch_flags_ports_that_answer_only_the_validator(mocker):
    mocker.patch.object(settings, "EXTERNAL_PORT_PROBE_ENFORCE", True)
    verifier, _ = _verifier(mocker, lambda sample: [])

    result = await verifier.verify(PORTS, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert result.external_blocked is True


@pytest.mark.asyncio
async def test_batch_only_logs_ports_that_answer_only_the_validator_until_enforced(mocker, caplog):
    mocker.patch.object(settings, "EXTERNAL_PORT_PROBE_ENFORCE", False)
    verifier, _ = _verifier(mocker, lambda sample: [])

    with caplog.at_level("WARNING"):
        result = await verifier.verify(PORTS, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert result.external_blocked is False
    assert result.successful == PORTS
    assert "not the outside prober (log-only)" in caplog.text


@pytest.mark.asyncio
async def test_batch_counts_ports_the_outside_prober_reached_without_fetching_them_again(mocker):
    verifier, port_tester = _verifier(mocker, lambda sample: sample)

    result = await verifier.verify(PORTS, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert result.external_blocked is False
    assert sorted(result.successful, key=lambda p: p.internal) == PORTS
    assert len(port_tester.test_many.call_args.args[2]) == len(PORTS) - 3


@pytest.mark.asyncio
async def test_batch_skips_the_validator_fetch_when_the_prober_reached_every_port(mocker):
    verifier, port_tester = _verifier(mocker, lambda sample: sample)

    result = await verifier.verify(PORTS[:3], ssh_client=mocker.Mock(), host="1.2.3.4")

    assert sorted(result.successful, key=lambda p: p.internal) == PORTS[:3]
    port_tester.test_many.assert_not_called()


@pytest.mark.asyncio
async def test_a_failed_prober_never_flags_the_node(mocker):
    verifier, _ = _verifier(mocker, lambda sample: None)

    result = await verifier.verify(PORTS, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert result.external_blocked is False
    assert result.successful == PORTS


@pytest.mark.asyncio
async def test_port_probe_does_not_fall_back_to_validator_only_tiers_when_blocked(mocker):
    batch = mocker.Mock()
    batch.verify = mocker.AsyncMock(return_value=BatchResult(PORTS, [], external_blocked=True))
    semi, fallback = mocker.Mock(), mocker.Mock()

    result = await PortProbe(batch, semi, fallback).probe(PORTS, ssh_client=mocker.Mock(), host="1.2.3.4")

    assert result.tier == "external_blocked"
    assert result.successful == ()
    semi.verify.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_leaves_an_externally_unreachable_node_no_verified_ports(
    sample_executor_info, mock_ssh_client, mocker
):
    selector = mocker.Mock()
    selector.select.return_value = PORTS
    probe = mocker.Mock()
    probe.probe = mocker.AsyncMock(return_value=PortProbeResult((), tuple(PORTS), "external_blocked"))
    dind = mocker.Mock()
    dind.verify = mocker.AsyncMock(
        return_value=DindProbeResult(success=True, sysbox_runtime=True, port=PORTS[0], log_text="ok")
    )

    result = await ConnectivityOrchestrator(selector, probe, dind).verify(
        executor_info=sample_executor_info,
        miner_hotkey="miner",
        sysbox_runtime=True,
        unavailable_ports=[],
        ssh_client=mock_ssh_client,
    )

    assert result.status == "externally_unreachable"
    assert result.successful_ports == ()
    assert result.sysbox_runtime is True


@pytest.mark.asyncio
async def test_external_probe_counts_only_answers_carrying_the_token():
    async def prober(request):
        assert request.headers["Authorization"] == "Bearer secret"
        targets = (await request.json())["targets"]
        bodies = ["tok:9000", "tok:other", None]
        return web.json_response({"results": [{**t, "body": b} for t, b in zip(targets, bodies)]})

    app = web.Application()
    app.router.add_post("/", prober)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        reached = await ExternalPortProbe(f"http://127.0.0.1:{port}/", "secret").reached("1.2.3.4", PORTS[:3], "tok")
    finally:
        await runner.cleanup()

    assert reached == [PORTS[0]]


@pytest.mark.asyncio
async def test_external_probe_returns_none_when_the_prober_is_down():
    assert await ExternalPortProbe("http://127.0.0.1:9", "secret").reached("1.2.3.4", PORTS[:1], "tok") is None


def _lambda_handler(monkeypatch):
    monkeypatch.setenv("PROBE_TOKEN", "secret")
    path = Path(__file__).parents[2] / "external_prober" / "handler.py"
    spec = importlib.util.spec_from_file_location("external_prober_handler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.handler


def test_lambda_refuses_a_wrong_token_and_non_public_targets(monkeypatch):
    handler = _lambda_handler(monkeypatch)

    def call(auth, host):
        event = {"headers": {"Authorization": auth}, "body": json.dumps({"targets": [{"host": host, "port": 22}]})}
        return handler(event, None)["statusCode"]

    assert call("Bearer wrong", "8.8.8.8") == 401
    assert call("Bearer secret", "10.0.0.1") == 400
    assert call("Bearer secret", "169.254.169.254") == 400
    assert call("Bearer secret", "example.com") == 400
