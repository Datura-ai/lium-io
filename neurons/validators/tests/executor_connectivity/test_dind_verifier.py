import pytest

from services.executor_connectivity.dind_probe import DindVerifier
from services.executor_connectivity.models import PortPair


def _run_result(mocker, exit_status=0, stdout="", stderr=""):
    result = mocker.Mock()
    result.exit_status = exit_status
    result.stdout = stdout
    result.stderr = stderr
    return result


@pytest.mark.asyncio
async def test_dind_verifier_sysbox_failure_sets_false(mocker):
    port = PortPair(9000, 9000)

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
    ssh_session.run = mocker.AsyncMock(return_value=_run_result(mocker, exit_status=1, stderr="fail"))

    connect = mocker.patch("services.executor_connectivity.dind_probe.asyncssh.connect")
    connect.return_value.__aenter__.return_value = ssh_session
    connect.return_value.__aexit__.return_value = mocker.AsyncMock()

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
async def test_dind_verifier_sysbox_dockerhub_ratelimit_treated_as_ok(mocker):
    port = PortPair(9000, 9000)

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
    ssh_session.run = mocker.AsyncMock(
        return_value=_run_result(
            mocker,
            exit_status=1,
            stderr=(
                "Error response from daemon: toomanyrequests: You have reached "
                "your unauthenticated pull rate limit. https://www.docker.com/increase-rate-limit"
            ),
        )
    )

    connect = mocker.patch("services.executor_connectivity.dind_probe.asyncssh.connect")
    connect.return_value.__aenter__.return_value = ssh_session
    connect.return_value.__aexit__.return_value = mocker.AsyncMock()

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
