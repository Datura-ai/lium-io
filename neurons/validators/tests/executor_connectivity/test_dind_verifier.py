import json

import pytest

from services.executor_connectivity.dind_probe import DindVerifier
from services.executor_connectivity.models import PortPair


def _run_result(mocker, exit_status=0, stdout="", stderr=""):
    result = mocker.Mock()
    result.exit_status = exit_status
    result.stdout = stdout
    result.stderr = stderr
    return result


def _setup_ssh_session(mocker, *, info_result):
    """Wire DinD-create + cleanup on the host SSH client, and a single
    `docker info` call on the inner SSH session inside the DinD container."""
    ssh_service = mocker.Mock()
    ssh_service.generate_keypair.return_value = ("priv", "pub")

    ssh_client = mocker.AsyncMock()
    ssh_client.run = mocker.AsyncMock(
        side_effect=[
            _run_result(mocker, exit_status=0, stdout="container_id"),
            _run_result(mocker, exit_status=0, stdout=""),
        ]
    )

    mocker.patch("services.executor_connectivity.dind_probe.asyncio.sleep", new=mocker.AsyncMock())
    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.import_private_key",
        return_value=mocker.Mock(),
    )

    ssh_session = mocker.AsyncMock()
    ssh_session.run = mocker.AsyncMock(return_value=info_result)

    connect = mocker.patch("services.executor_connectivity.dind_probe.asyncssh.connect")
    connect.return_value.__aenter__.return_value = ssh_session
    connect.return_value.__aexit__.return_value = mocker.AsyncMock()

    return ssh_service, ssh_client, connect


@pytest.mark.asyncio
async def test_dind_verifier_sysbox_runtime_present_sets_true(mocker):
    port = PortPair(9000, 9000)
    runtimes_json = json.dumps(
        {"runc": {"path": "runc"}, "sysbox-runc": {"path": "/usr/bin/sysbox-runc"}}
    )
    ssh_service, ssh_client, connect = _setup_ssh_session(
        mocker, info_result=_run_result(mocker, exit_status=0, stdout=runtimes_json)
    )

    verifier = DindVerifier(ssh_service)

    result = await verifier.verify(
        port,
        ssh_client=ssh_client,
        host="127.0.0.1",
        container_name_prefix="container_miner",
        sysbox=True,
    )

    assert result.success is True
    assert result.sysbox_runtime is True
    assert "check ok" in (result.log_text or "")
    connect.assert_called_once()


@pytest.mark.asyncio
async def test_dind_verifier_sysbox_runtime_missing_sets_false(mocker):
    port = PortPair(9000, 9000)
    runtimes_json = json.dumps({"runc": {"path": "runc"}})
    ssh_service, ssh_client, connect = _setup_ssh_session(
        mocker, info_result=_run_result(mocker, exit_status=0, stdout=runtimes_json)
    )

    verifier = DindVerifier(ssh_service)

    result = await verifier.verify(
        port,
        ssh_client=ssh_client,
        host="127.0.0.1",
        container_name_prefix="container_miner",
        sysbox=True,
    )

    assert result.success is True
    assert result.sysbox_runtime is False
    assert "check ok" in (result.log_text or "")
    connect.assert_called_once()


@pytest.mark.asyncio
async def test_dind_verifier_sysbox_docker_info_unreachable_sets_false(mocker):
    """Inner Docker daemon is unreachable — sysbox falls back to False, but
    DinD wrapper itself is still considered up (we did SSH in)."""
    port = PortPair(9000, 9000)
    ssh_service, ssh_client, connect = _setup_ssh_session(
        mocker,
        info_result=_run_result(
            mocker,
            exit_status=1,
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
        ),
    )

    verifier = DindVerifier(ssh_service)

    result = await verifier.verify(
        port,
        ssh_client=ssh_client,
        host="127.0.0.1",
        container_name_prefix="container_miner",
        sysbox=True,
    )

    assert result.success is True
    assert result.sysbox_runtime is False
    assert "check ok" in (result.log_text or "")
    connect.assert_called_once()


@pytest.mark.asyncio
async def test_dind_verifier_does_not_call_dockerhub(mocker):
    """Regression guard: probe must not pull from any external registry.
    Anyone changing the probe to `docker pull <image>` will fail this test."""
    port = PortPair(9000, 9000)
    runtimes_json = json.dumps({"runc": {"path": "runc"}, "sysbox-runc": {"path": "/usr/bin/sysbox-runc"}})
    ssh_service, ssh_client, connect = _setup_ssh_session(
        mocker, info_result=_run_result(mocker, exit_status=0, stdout=runtimes_json)
    )
    ssh_session = connect.return_value.__aenter__.return_value

    verifier = DindVerifier(ssh_service)

    await verifier.verify(
        port,
        ssh_client=ssh_client,
        host="127.0.0.1",
        container_name_prefix="container_miner",
        sysbox=True,
    )

    issued_commands = [call.args[0] for call in ssh_session.run.call_args_list]
    for cmd in issued_commands:
        assert "docker pull" not in cmd, f"DinD probe must not call docker pull (saw: {cmd!r})"


@pytest.mark.asyncio
async def test_dind_verifier_docker_run_fails(mocker):
    port = PortPair(9000, 9000)

    ssh_service = mocker.Mock()
    ssh_service.generate_keypair.return_value = ("priv", "pub")

    ssh_client = mocker.AsyncMock()
    ssh_client.run = mocker.AsyncMock(
        side_effect=[
            _run_result(mocker, exit_status=1, stderr="boom"),
            _run_result(mocker, exit_status=0, stdout=""),
        ]
    )

    connect = mocker.patch("services.executor_connectivity.dind_probe.asyncssh.connect")

    verifier = DindVerifier(ssh_service)

    result = await verifier.verify(
        port,
        ssh_client=ssh_client,
        host="127.0.0.1",
        container_name_prefix="container_miner",
        sysbox=True,
    )

    assert result.success is False
    assert "check failed" in (result.log_text or "")
    connect.assert_not_called()
