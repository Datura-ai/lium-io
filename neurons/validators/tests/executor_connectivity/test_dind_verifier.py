import asyncio
from typing import Any, NamedTuple

import pytest

from services.executor_connectivity.dind_probe import DindVerifier, diagnose_dind_log, diagnose_docker_run_error
from services.executor_connectivity.models import DIND_INNER_DOCKERD_IPTABLES, DindLogCause, PortPair


def _run_result(mocker, exit_status=0, stdout="", stderr=""):
    result = mocker.Mock()
    result.exit_status = exit_status
    result.stdout = stdout
    result.stderr = stderr
    return result


class StartedDind(NamedTuple):
    verifier: DindVerifier
    ssh_client: Any


def _executor_host(mocker, container_log: str = ""):
    """The executor host's ssh session: `docker run` succeeds, the DAH-2856 diagnostics read returns
    `container_log`, everything else (the remove) succeeds silently."""
    async def run(cmd: str):
        if "docker run" in cmd:
            return _run_result(mocker, exit_status=0, stdout="container_id")
        if "docker logs" in cmd:
            return _run_result(mocker, exit_status=0, stdout=container_log)
        return _run_result(mocker, exit_status=0, stdout="")

    ssh_client = mocker.AsyncMock()
    ssh_client.run = mocker.AsyncMock(side_effect=run)
    return ssh_client


def _build_started_dind(mocker, container_log: str = "") -> StartedDind:
    # a probe whose `docker run` already succeeded, so the test starts at the SSH step
    ssh_service = mocker.Mock()
    ssh_service.generate_keypair.return_value = ("priv", "pub")

    ssh_client = _executor_host(mocker, container_log)

    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.import_private_key",
        return_value=mocker.Mock(),
    )
    return StartedDind(DindVerifier(ssh_service), ssh_client)


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
    verifier, ssh_client = _build_started_dind(mocker)

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

    ssh_client = _executor_host(mocker)

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
    # One poll interval outlasts the whole deadline, so the probe gives up on the first attempt
    # while that attempt still gets the full connect budget the assertions below pin.
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_POLL_INTERVAL_SECONDS", 60)

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
    verifier, ssh_client = _build_started_dind(mocker)

    # Real (tiny) sleeps, so the loop's own clock advances and the deadline is what stops it.
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_READY_TIMEOUT_SECONDS", 0.05)
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_POLL_INTERVAL_SECONDS", 0.01)
    connect = mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect",
        new=mocker.AsyncMock(side_effect=ConnectionRefusedError("[Errno 111] Connect call failed")),
    )

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
async def test_dind_verifier_hanging_attempts_stay_inside_the_deadline(mocker):
    """DAH-2588: attempts that hang for their whole budget must not push the probe past the
    readiness deadline — the last attempt gets what is left of it, not a fresh full budget."""
    port = PortPair(9000, 9000)
    verifier, ssh_client = _build_started_dind(mocker)

    # Same 30/12/1.5 ratio as production, scaled down 100x so the test runs in real time.
    ready_timeout_seconds = 0.3
    mocker.patch(
        "services.executor_connectivity.dind_probe.DIND_SSH_READY_TIMEOUT_SECONDS",
        ready_timeout_seconds,
    )
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_CONNECT_TIMEOUT_SECONDS", 0.12)
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_POLL_INTERVAL_SECONDS", 0.015)

    granted_timeouts = []

    async def hang_for_the_whole_budget(**kwargs):
        granted_timeouts.append(kwargs["connect_timeout"])
        await asyncio.sleep(kwargs["connect_timeout"])
        raise TimeoutError("connect timed out")

    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect",
        new=mocker.AsyncMock(side_effect=hang_for_the_whole_budget),
    )

    started_at = asyncio.get_running_loop().time()

    result = await verifier.verify(
        port,
        ssh_client=ssh_client,
        host="127.0.0.1",
        container_name_prefix="container_miner",
        sysbox=True,
    )

    elapsed = asyncio.get_running_loop().time() - started_at

    assert result.success is False
    assert elapsed <= ready_timeout_seconds * 1.2
    assert granted_timeouts[-1] < granted_timeouts[0]


@pytest.mark.asyncio
async def test_dind_verifier_late_poll_wakeup_starts_no_further_attempt(mocker):
    """DAH-2588: when the loop resumes the poll past the deadline there is no budget left, and an
    attempt started anyway could only time out — reporting a timeout instead of the connection
    error that actually kept sshd unreachable."""
    port = PortPair(9000, 9000)
    verifier, ssh_client = _build_started_dind(mocker)

    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_READY_TIMEOUT_SECONDS", 0.05)
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_POLL_INTERVAL_SECONDS", 0.01)

    granted_timeouts = []

    async def refuse_immediately(**kwargs):
        granted_timeouts.append(kwargs["connect_timeout"])
        raise ConnectionRefusedError("[Errno 111] Connect call failed")

    connect = mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect",
        new=mocker.AsyncMock(side_effect=refuse_immediately),
    )

    # A poll that oversleeps its interval by more than the deadline had left.
    real_sleep = asyncio.sleep

    async def oversleep(_):
        await real_sleep(0.2)

    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncio.sleep",
        new=mocker.AsyncMock(side_effect=oversleep),
    )

    result = await verifier.verify(
        port,
        ssh_client=ssh_client,
        host="127.0.0.1",
        container_name_prefix="container_miner",
        sysbox=True,
    )

    assert result.success is False
    assert connect.await_count == 1
    assert all(timeout > 0 for timeout in granted_timeouts)


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


# DAH-2856 — a container that started but whose sshd never answered names its real cause.

# verbatim from daturaai/dind:0.0.1 on an nftables-mode host (EC2 run 20260917T180704Z-2185): 333 chars
NFT_HOST_DOCKERD_LOG = (
    "[INFO] [/usr/local/bin/start-docker.sh] dockerd is running\n"
    'time="2026-09-17T18:09:34.568850375Z" level=info msg="Loading containers: start."\n'
    "failed to start daemon: Error initializing network controller: error obtaining controller instance: "
    'failed to register "bridge" driver: failed to create NAT chain DOCKER: iptables failed: '
    "iptables --wait -t nat -N DOCKER: iptables v1.8.10 (legacy): can't initialize iptables table `nat': "
    "Table does not exist (do you need to insmod?)\n"
    "Perhaps iptables or your kernel needs to be upgraded.\n"
    " (exit status 3)\n"
)


def test_diagnose_dind_log_names_the_iptables_cause_and_quotes_dockerd():
    cause = diagnose_dind_log(NFT_HOST_DOCKERD_LOG)
    code, words = cause.code, cause.message
    assert code == "DIND_INNER_DOCKERD_IPTABLES"
    assert "nf_tables" in words and "modprobe" in words
    # the provider sees dockerd's own line, not only the validator's reading of it
    assert "dockerd said: failed to start daemon" in words
    assert "Table does not exist (do you need to insmod?)" in words  # the whole 333-char line survives the cap
    assert "\n" not in words
    # the read line travels on its own too, so the sysbox verdict knows the cause was measured
    assert cause.dockerd_line and cause.dockerd_line.startswith("failed to start daemon")


def test_diagnose_dind_log_generic_when_nothing_matches():
    cause = diagnose_dind_log("")
    assert cause.code == "DIND_SSHD_NOT_READY" and "no dockerd error" in cause.message
    assert cause.dockerd_line is None
    assert diagnose_dind_log(None).code == "DIND_SSHD_NOT_READY"
    cause = diagnose_dind_log("failed to start daemon: something else")
    assert cause.code == "DIND_INNER_DOCKERD_DOWN"
    assert "dockerd said: failed to start daemon: something else" in cause.message
    assert cause.dockerd_line == "failed to start daemon: something else"


def test_diagnose_dind_log_caps_the_quoted_line_head_first():
    cause = diagnose_dind_log("failed to start daemon: " + "x" * 2000)
    assert cause.code == "DIND_INNER_DOCKERD_DOWN"
    assert "dockerd said: failed to start daemon: xxx" in cause.message
    assert len(cause.message) < 500


@pytest.mark.asyncio
async def test_dind_verifier_reads_the_container_log_before_removing_it(mocker):
    """DAH-2856: sshd never answers → the probe reads the container's logs while it still exists,
    the result carries the cause, and the container is removed afterwards (ticket-0309 got
    "install sysbox" for a host whose inner dockerd could not use legacy iptables)."""
    port = PortPair(9000, 9000)
    verifier, ssh_client = _build_started_dind(mocker, container_log=NFT_HOST_DOCKERD_LOG)
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_READY_TIMEOUT_SECONDS", 0.05)
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_POLL_INTERVAL_SECONDS", 0.01)
    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect",
        new=mocker.AsyncMock(side_effect=ConnectionRefusedError("[Errno 111] Connect call failed")),
    )

    result = await verifier.verify(
        port, ssh_client=ssh_client, host="127.0.0.1", container_name_prefix="container_miner", sysbox=True
    )

    assert result.success is False
    assert isinstance(result.error, DindLogCause)
    assert result.error.code == DIND_INNER_DOCKERD_IPTABLES
    assert result.error.text.startswith("DIND_INNER_DOCKERD_IPTABLES: ")
    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    logs_index = next(i for i, c in enumerate(commands) if "docker logs" in c)
    remove_index = next(i for i, c in enumerate(commands) if "docker rm -fv" in c)
    assert logs_index < remove_index
    assert "container_miner_9000" in commands[logs_index]
    assert "/var/log/dockerd.err.log" in commands[logs_index]


@pytest.mark.asyncio
async def test_dind_verifier_no_diagnosis_when_sysbox_was_not_requested(mocker):
    """Rustam's review (21 Sep): without sysbox the probe runs plain runc, where the inner dockerd
    fails by design, so its log names no host fault. The cause is read only when sysbox was
    requested; otherwise the result carries no error and nothing tells the provider to fix a host."""
    port = PortPair(9000, 9000)
    verifier, ssh_client = _build_started_dind(mocker, container_log=NFT_HOST_DOCKERD_LOG)
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_READY_TIMEOUT_SECONDS", 0.05)
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_POLL_INTERVAL_SECONDS", 0.01)
    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect",
        new=mocker.AsyncMock(side_effect=ConnectionRefusedError("[Errno 111] Connect call failed")),
    )

    result = await verifier.verify(
        port, ssh_client=ssh_client, host="127.0.0.1", container_name_prefix="container_miner", sysbox=False
    )

    assert result.success is False
    assert result.error is None
    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert not any("docker logs" in c for c in commands)
    assert any("docker rm -fv" in c for c in commands)


@pytest.mark.asyncio
async def test_dind_verifier_no_diagnosis_when_docker_run_itself_failed(mocker):
    """No container, nothing to read: the docker-run failure path stays as it was."""
    port = PortPair(9000, 9000)
    ssh_service = mocker.Mock()
    ssh_service.generate_keypair.return_value = ("priv", "pub")
    ssh_client = mocker.AsyncMock()
    ssh_client.run = mocker.AsyncMock(
        side_effect=[
            _run_result(mocker, exit_status=125, stderr="docker: Error response from daemon: port is already allocated"),
            _run_result(mocker, exit_status=0, stdout=""),
        ]
    )

    result = await DindVerifier(ssh_service).verify(
        port, ssh_client=ssh_client, host="127.0.0.1", container_name_prefix="container_miner", sysbox=True
    )

    assert result.success is False
    assert result.error is None
    assert not any("docker logs" in call.args[0] for call in ssh_client.run.await_args_list)


@pytest.mark.asyncio
async def test_dind_verifier_diagnosis_read_failure_still_returns_a_cause(mocker):
    """The log read is best-effort: a host that refuses it still yields the generic cause and the
    container is still removed."""
    port = PortPair(9000, 9000)
    verifier, ssh_client = _build_started_dind(mocker)

    async def run(cmd: str):
        if "docker run" in cmd:
            return _run_result(mocker, exit_status=0, stdout="container_id")
        if "docker logs" in cmd:
            raise OSError("ssh channel closed")
        return _run_result(mocker, exit_status=0, stdout="")

    ssh_client.run = mocker.AsyncMock(side_effect=run)
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_READY_TIMEOUT_SECONDS", 0.05)
    mocker.patch("services.executor_connectivity.dind_probe.DIND_SSH_POLL_INTERVAL_SECONDS", 0.01)
    mocker.patch(
        "services.executor_connectivity.dind_probe.asyncssh.connect",
        new=mocker.AsyncMock(side_effect=ConnectionRefusedError("[Errno 111] Connect call failed")),
    )

    result = await verifier.verify(
        port, ssh_client=ssh_client, host="127.0.0.1", container_name_prefix="container_miner", sysbox=True
    )

    assert result.success is False
    assert isinstance(result.error, DindLogCause) and result.error.code == "DIND_SSHD_NOT_READY"
    assert "Connect call failed" in result.error.message  # the ssh error itself is the only fact left
    assert any("docker rm -fv" in call.args[0] for call in ssh_client.run.await_args_list)


@pytest.mark.asyncio
async def test_dind_verifier_removes_the_container_when_cancelled_mid_probe(mocker):
    # The validation fast path cancels this check when the sibling lane stops; the probe
    # container must not stay behind with its port bound until the next wave's cleanup.
    port = PortPair(9000, 9000)
    verifier, ssh_client = _build_started_dind(mocker)
    hang = asyncio.Event()

    async def connect_forever(*args, **kwargs):
        await hang.wait()

    mocker.patch("services.executor_connectivity.dind_probe.asyncssh.connect", side_effect=connect_forever)

    task = asyncio.ensure_future(
        verifier.verify(port, ssh_client=ssh_client, host="127.0.0.1", container_name_prefix="container_miner", sysbox=True)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert [("docker run" in c, "docker rm -fv" in c) for c in commands] == [(True, False), (False, True)], commands
    assert "container_miner_9000" in commands[1]


@pytest.mark.asyncio
async def test_dind_verifier_removes_the_container_once_on_every_way_out(mocker):
    port = PortPair(9000, 9000)
    # docker run fails; docker run ok but the ssh connect raises
    for docker_run_ok, connect_side_effect in ((False, None), (True, RuntimeError("ssh exploded"))):
        verifier, ssh_client = _build_started_dind(mocker)
        if not docker_run_ok:
            ssh_client.run = mocker.AsyncMock(
                side_effect=[_run_result(mocker, exit_status=1, stderr="boom"), _run_result(mocker, exit_status=0)]
            )
        if connect_side_effect is not None:
            mocker.patch("services.executor_connectivity.dind_probe.asyncssh.connect", side_effect=connect_side_effect)
        result = await verifier.verify(
            port, ssh_client=ssh_client, host="127.0.0.1", container_name_prefix="container_miner", sysbox=True
        )
        assert result.success is False
        removals = [c for c in ssh_client.run.await_args_list if "docker rm -fv" in str(c.args[0])]
        assert len(removals) == 1, (docker_run_ok, connect_side_effect)


# DAH-3634 — `docker run` refused by the NVIDIA container hook names the real cause.

# verbatim from the validator log (`DinD creation failed`)
NVIDIA_MISMATCH_STDERR = (
    "docker: Error response from daemon: failed to create task for container: failed to create shim task: "
    "OCI runtime create failed: runc create failed: unable to start container process: error during container "
    "init: error running prestart hook #0: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'\n"
    "nvidia-container-cli: initialization error: nvml error: driver/library version mismatch\n\n"
    "Run 'docker run --help' for more information"
)
NVIDIA_GPU_RESET_STDERR = (
    "docker: Error response from daemon: failed to create task for container: failed to create shim task: "
    "OCI runtime create failed: container_linux.go:439: starting container process caused: process_linux.go:608: "
    "container init caused: Running hook #0:: error running hook: exit status 1, stdout: , stderr: Using requested "
    "mode 'legacy'\nnvidia-container-cli: detection error: nvml error: gpu requires reset\n\n"
    "Run 'docker run --help' for more information"
)
PORT_ALLOCATED_STDERR = (
    "docker: Error response from daemon: failed to set up container networking: driver failed programming external "
    "connectivity on endpoint container_miner_9000 (07f1): Bind for 0.0.0.0:9000 failed: port is already allocated\n\n"
    "Run 'docker run --help' for more information"
)
# sysbox's shiftfs fallback (nvidia_docker_sysbox_setup.sh --check names it): a sysbox problem, not an NVML one
SYSBOX_SHIFTFS_MOUNT_STDERR = (
    "docker: Error response from daemon: OCI runtime create failed: error running hook #0: exit status 1, stdout: , "
    "stderr: nvidia-container-cli: mount error: file lookup failed: "
    "/var/lib/sysbox/shiftfs/<eight hex characters>/merged/proc/driver/nvidia: no such file or directory\n\n"
    "Run 'docker run --help' for more information"
)


def test_diagnose_docker_run_error_names_the_driver_mismatch_and_quotes_the_hook_line():
    cause = diagnose_docker_run_error(NVIDIA_MISMATCH_STDERR)
    assert isinstance(cause, DindLogCause)
    assert cause.code == "NVIDIA_RUNTIME_MISMATCH"
    assert "kernel driver" in cause.message and "different versions" in cause.message
    # the provider sees the hook's own line, not only the validator's reading of it
    assert cause.message.endswith(
        "docker said: nvidia-container-cli: initialization error: nvml error: driver/library version mismatch"
    )
    assert "\n" not in cause.message
    assert cause.text.startswith("NVIDIA_RUNTIME_MISMATCH: ")


def test_diagnose_docker_run_error_generic_nvml_failure_and_none_for_the_rest():
    cause = diagnose_docker_run_error(NVIDIA_GPU_RESET_STDERR)
    assert cause is not None and cause.code == "NVIDIA_CONTAINER_HOOK_FAILED"
    assert cause.message.endswith("docker said: nvidia-container-cli: detection error: nvml error: gpu requires reset")
    # a bound host port, a name conflict or sysbox's own mount error say nothing about NVML: no cause
    assert diagnose_docker_run_error(PORT_ALLOCATED_STDERR) is None
    assert diagnose_docker_run_error(SYSBOX_SHIFTFS_MOUNT_STDERR) is None
    assert diagnose_docker_run_error("unknown error") is None
    assert diagnose_docker_run_error(None) is None


@pytest.mark.parametrize(
    "hook_line",
    [
        "nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: "
        "cannot open shared object file: no such file or directory: unknown",
        "nvidia-container-cli: initialization error: nvml error: driver not loaded: unknown",
        "nvidia-container-cli: initialization error: driver error: failed to process request",
        "nvidia-container-cli: device error: GPU-d2d12cfb-eb9e-03f0-b007-785d32aed1b2: unknown device",
        "nvidia-container-cli: container error: cgroup subsystem devices not found: unknown",
    ],
)
def test_diagnose_docker_run_error_names_every_nvidia_hook_error(hook_line: str):
    stderr = (
        "docker: Error response from daemon: OCI runtime create failed: error running prestart hook #0: "
        f"exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'\n{hook_line}\n\n"
        "Run 'docker run --help' for more information"
    )

    cause = diagnose_docker_run_error(stderr)

    assert cause is not None and cause.code == "NVIDIA_CONTAINER_HOOK_FAILED"
    assert cause.message.endswith(f"docker said: {hook_line}")


def test_diagnose_docker_run_error_quotes_from_the_hook_and_caps_head_first():
    # a daemon prefix longer than the cap on the same line must not push the hook's words out
    stderr = "docker: " + "p" * 1000 + " stderr: nvidia-container-cli: detection error: nvml error: " + "x" * 2000
    cause = diagnose_docker_run_error(stderr)
    assert cause is not None and cause.code == "NVIDIA_CONTAINER_HOOK_FAILED"
    assert "docker said: nvidia-container-cli: detection error: nvml error: xxx" in cause.message
    assert "ppp" not in cause.message
    assert len(cause.message) < 600


@pytest.mark.asyncio
async def test_dind_verifier_docker_run_refused_by_the_nvidia_hook_carries_the_cause(mocker):
    """DAH-3634: the hook refused the container → the result names the cause; no container log is
    read (there is no container) and the name is still removed."""
    port = PortPair(9000, 9000)
    ssh_service = mocker.Mock()
    ssh_service.generate_keypair.return_value = ("priv", "pub")
    ssh_client = mocker.AsyncMock()
    ssh_client.run = mocker.AsyncMock(
        side_effect=[
            _run_result(mocker, exit_status=125, stderr=NVIDIA_MISMATCH_STDERR),
            _run_result(mocker, exit_status=0, stdout=""),
        ]
    )

    result = await DindVerifier(ssh_service).verify(
        port, ssh_client=ssh_client, host="127.0.0.1", container_name_prefix="container_miner", sysbox=True
    )

    assert result.success is False
    assert result.sysbox_runtime is True  # the orchestrator downgrades it; the probe measured nothing
    assert isinstance(result.error, DindLogCause) and result.error.code == "NVIDIA_RUNTIME_MISMATCH"
    assert "driver/library version mismatch" in result.error.message
    commands = [call.args[0] for call in ssh_client.run.await_args_list]
    assert not any("docker logs" in c for c in commands)
    assert any("docker rm -fv container_miner_9000" in c for c in commands)


@pytest.mark.asyncio
async def test_dind_verifier_carries_the_cause_when_sysbox_was_not_requested(mocker):
    """The executor's own sysbox self-report (machine_scrape.check_sysbox_gpu_compatibility) runs
    the same hook on the same host and is refused too, so the probe runs without sysbox-runc on
    exactly these nodes. The cause must be carried either way or the fix reaches nobody."""
    port = PortPair(9000, 9000)
    ssh_service = mocker.Mock()
    ssh_service.generate_keypair.return_value = ("priv", "pub")
    ssh_client = mocker.AsyncMock()
    ssh_client.run = mocker.AsyncMock(
        side_effect=[
            _run_result(mocker, exit_status=125, stderr=NVIDIA_MISMATCH_STDERR),
            _run_result(mocker, exit_status=0, stdout=""),
        ]
    )

    result = await DindVerifier(ssh_service).verify(
        port, ssh_client=ssh_client, host="127.0.0.1", container_name_prefix="container_miner", sysbox=False
    )

    assert result.success is False
    assert result.sysbox_runtime is False
    assert isinstance(result.error, DindLogCause) and result.error.code == "NVIDIA_RUNTIME_MISMATCH"
