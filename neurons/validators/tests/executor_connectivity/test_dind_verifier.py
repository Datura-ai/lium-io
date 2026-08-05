import asyncio

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

    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.import_private_key",
        return_value=mocker.Mock(),
    )

    ssh_session = mocker.AsyncMock()
    ssh_session.run = mocker.AsyncMock(return_value=_run_result(mocker, exit_status=1, stderr="fail"))

    connect = mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect", new=mocker.AsyncMock()
    )
    connect.return_value.__aenter__.return_value = ssh_session

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

    connect = mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect", new=mocker.AsyncMock()
    )

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


@pytest.mark.asyncio
async def test_dind_verifier_retries_connect_until_sshd_is_up(mocker):
    """DAH-2588: sshd inside a fresh container is not listening immediately, so a refused
    connection must be retried under the readiness deadline instead of failing the probe."""
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

    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.import_private_key",
        return_value=mocker.Mock(),
    )

    ssh_session = mocker.AsyncMock()
    ssh_session.run = mocker.AsyncMock(return_value=_run_result(mocker, exit_status=0))
    connection = mocker.MagicMock()
    connection.__aenter__.return_value = ssh_session

    connect = mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect",
        new=mocker.AsyncMock(
            side_effect=[
                ConnectionRefusedError("[Errno 111] Connect call failed"),
                ConnectionRefusedError("[Errno 111] Connect call failed"),
                connection,
            ]
        ),
    )
    sleep = mocker.patch(
        "services.executor_connectivity.dind_probe.asyncio.sleep", new=mocker.AsyncMock()
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
    assert connect.await_count == 3
    assert sleep.await_args_list == [mocker.call(1.5), mocker.call(1.5)]


@pytest.mark.asyncio
async def test_dind_verifier_hung_connect_fails_within_timeout(mocker):
    """DAH-2272: a hung asyncssh.connect must fail within connect/login timeout,
    not stall the probe indefinitely."""
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

    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.import_private_key",
        return_value=mocker.Mock(),
    )

    # Simulate asyncssh's own connect/login timeout firing (a hung connect
    # never resolves the underlying TCP/SSH handshake within connect_timeout/
    # login_timeout, which asyncssh surfaces as a TimeoutError).
    connect = mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect",
        new=mocker.AsyncMock(side_effect=TimeoutError("connect timed out")),
    )
    # Deadline already spent, so the probe gives up on the first attempt instead of polling.
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_READY_TIMEOUT", 0)

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
    connect.assert_called_once()
    # The connect call must carry the bounded timeouts, not rely on an
    # unbounded default.
    _, kwargs = connect.call_args
    assert kwargs["connect_timeout"] == 12
    assert kwargs["login_timeout"] == 12


@pytest.mark.asyncio
async def test_dind_verifier_gives_up_after_deadline(mocker):
    """DAH-2588: a container that never comes up must still fail, and fail with the underlying
    connection error rather than a generic timeout — the two have different diagnoses."""
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

    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.import_private_key",
        return_value=mocker.Mock(),
    )

    # Real (tiny) sleeps, so the loop's own clock advances and the deadline is what stops it.
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_READY_TIMEOUT", 0.05)
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_POLL_INTERVAL", 0.01)
    connect = mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect",
        new=mocker.AsyncMock(side_effect=ConnectionRefusedError("[Errno 111] Connect call failed")),
    )

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
    assert connect.await_count >= 2


@pytest.mark.asyncio
async def test_dind_verifier_hung_inner_docker_run_degrades_sysbox(mocker):
    """DAH-2272: a hung inner `docker run --rm hello-world` must degrade to
    sysbox_ok=False via asyncio.wait_for(timeout=30) instead of hanging the
    probe forever, matching the existing exit-status-nonzero fallback."""
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

    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.import_private_key",
        return_value=mocker.Mock(),
    )

    # The inner `docker run --rm hello-world` never returns (simulating a
    # wedged remote dockerd). We don't want the test to actually burn the
    # real 30s timeout, so patch asyncio.wait_for as seen from the dind_probe
    # module to immediately raise TimeoutError, closing the pending
    # coroutine to avoid an "was never awaited" warning.
    async def _hangs_forever():
        await asyncio.Event().wait()

    ssh_session = mocker.AsyncMock()
    ssh_session.run = mocker.Mock(return_value=_hangs_forever())

    connect = mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect", new=mocker.AsyncMock()
    )
    connect.return_value.__aenter__.return_value = ssh_session

    seen_timeouts = []

    async def _fast_timeout(coro, timeout):
        seen_timeouts.append(timeout)
        coro.close()
        raise asyncio.TimeoutError()

    mocker.patch("services.executor_connectivity.dind_probe.asyncio.wait_for", side_effect=_fast_timeout)

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
    assert seen_timeouts == [30]
    connect.assert_called_once()
