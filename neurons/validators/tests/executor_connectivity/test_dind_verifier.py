import pytest

from services.const import DIND_PROBE_IMAGE
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


def test_dind_probe_image_is_pinned_ecr_mirror():
    """Guard against regressions to Docker Hub. The original DAH-1959 outage was caused by
    anonymous-pull rate limits on docker.io; we mitigate by using AWS public ECR's mirror
    pinned by digest. If someone changes this constant back to docker.io or drops the digest,
    they need to revisit DAH-1959 first.
    """
    assert DIND_PROBE_IMAGE.startswith("public.ecr.aws/docker/library/hello-world@sha256:")
    assert "docker.io" not in DIND_PROBE_IMAGE
    assert ":latest" not in DIND_PROBE_IMAGE


@pytest.mark.asyncio
async def test_dind_verifier_uses_pinned_ecr_image_in_run_command(mocker):
    """End-to-end: the verifier issues `docker run --rm <pinned-ecr-image>` inside the inner SSH session."""
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
    ssh_session.run = mocker.AsyncMock(return_value=_run_result(mocker, exit_status=0, stdout=""))

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
    ssh_session.run.assert_awaited_once()
    cmd = ssh_session.run.await_args.args[0]
    assert cmd == f"docker run --rm {DIND_PROBE_IMAGE}"


@pytest.mark.asyncio
async def test_dind_verifier_retries_on_ratelimit_then_succeeds(mocker):
    """When inner docker run returns 'toomanyrequests', retry once after a backoff;
    the second attempt's success keeps sysbox_runtime=True. This covers the ECR
    public 1 pull/sec/IP limit hit by miners with many executors behind one NAT."""
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

    sleep_mock = mocker.AsyncMock()
    mocker.patch("services.executor_connectivity.dind_probe.asyncio.sleep", new=sleep_mock)
    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.import_private_key",
        return_value=mocker.Mock(),
    )

    # First inner run returns toomanyrequests; second succeeds.
    ssh_session = mocker.AsyncMock()
    ssh_session.run = mocker.AsyncMock(
        side_effect=[
            _run_result(
                mocker,
                exit_status=1,
                stderr="docker: toomanyrequests: Rate exceeded",
            ),
            _run_result(mocker, exit_status=0, stdout=""),
        ]
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

    # Probe succeeded on retry, so sysbox stays True.
    assert result.success is True
    assert result.sysbox_runtime is True
    # Inner ssh.run was called twice: probe + retry.
    assert ssh_session.run.await_count == 2
    # Backoff was within 2..10 s window.
    sleep_calls = [c.args[0] for c in sleep_mock.await_args_list if c.args]
    assert any(2 <= s <= 10 for s in sleep_calls), f"expected a 2-10s backoff, got {sleep_calls}"


@pytest.mark.asyncio
async def test_dind_verifier_does_not_retry_on_non_ratelimit_failure(mocker):
    """Real failures (e.g. inner dockerd dead) should fail fast without burning a retry."""
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

    sleep_mock = mocker.AsyncMock()
    mocker.patch("services.executor_connectivity.dind_probe.asyncio.sleep", new=sleep_mock)
    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.import_private_key",
        return_value=mocker.Mock(),
    )

    ssh_session = mocker.AsyncMock()
    ssh_session.run = mocker.AsyncMock(
        return_value=_run_result(
            mocker,
            exit_status=1,
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock.",
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

    # Probe failed with non-ratelimit error → sysbox False, no retry.
    assert result.success is True
    assert result.sysbox_runtime is False
    assert ssh_session.run.await_count == 1
    # Only the 5-second post-creation sleep should fire — no extra retry backoff sleep.
    assert sleep_mock.await_count == 1, (
        f"expected 1 sleep (post-create only), got {sleep_mock.await_count} "
        f"calls={[c.args for c in sleep_mock.await_args_list]}"
    )
