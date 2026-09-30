"""A command that ends without an exit status from the host is not a success.

asyncssh's conn.run() does not raise when the connection drops or the channel closes after it
opened: it returns whatever output arrived with exit_status None. The runner used to read that as
exit 0, so a scrape cut off mid-run landed on a host-side code.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace

import asyncssh
import pytest
from neurons.validators.src.services.task.checks.machine_spec_scrape import MachineSpecScrapeCheck
from neurons.validators.src.services.task.messages import MachineSpecMessages as Msg
from neurons.validators.src.services.task.runner import NO_EXIT_STATUS, SSHCommandRunner

from tests.helpers import build_context_config, build_services, build_state

ProcessHandler = Callable[[asyncssh.SSHServerProcess], Awaitable[None]]


class _FakeSSH:
    def __init__(self, *completed: SimpleNamespace):
        self.completed = list(completed)
        self.calls = 0

    async def run(self, cmd, input=None):
        self.calls += 1
        return self.completed.pop(0)


def _completed(exit_status, stdout="", stderr="", exit_signal=None) -> SimpleNamespace:
    return SimpleNamespace(
        exit_status=exit_status, exit_signal=exit_signal, stdout=stdout, stderr=stderr
    )


@pytest.mark.asyncio
async def test_no_exit_status_is_a_failure_that_keeps_what_arrived():
    runner = SSHCommandRunner(_FakeSSH(_completed(None, stdout="partial line\n")))

    result = await runner.run("scrape", retryable=False)

    assert result.success is False
    assert result.exit_code == -1
    assert result.error_type == NO_EXIT_STATUS
    assert result.stdout == "partial line"


@pytest.mark.asyncio
async def test_no_exit_status_is_retried_like_an_ssh_error_when_retryable():
    ssh = _FakeSSH(_completed(None), _completed(0, stdout="ok"))
    runner = SSHCommandRunner(ssh, max_retries=1)

    result = await runner.run("echo ok", retryable=True)

    assert ssh.calls == 2
    assert result.success is True
    assert result.error_type is None


@pytest.mark.parametrize(
    "exit_status,success",
    [
        pytest.param(0, True, id="exit-0"),
        pytest.param(3, False, id="exit-3"),
        pytest.param(-1, False, id="killed-by-signal"),
    ],
)
@pytest.mark.asyncio
async def test_an_exit_status_from_the_host_leaves_error_type_unset(exit_status, success):
    runner = SSHCommandRunner(_FakeSSH(_completed(exit_status)))

    result = await runner.run("scrape", retryable=False)

    assert result.exit_code == exit_status
    assert result.success is success
    assert result.error_type is None


class _NoAuthServer(asyncssh.SSHServer):
    def begin_auth(self, username: str) -> bool:
        return False


@asynccontextmanager
async def _local_ssh(handler: ProcessHandler) -> AsyncIterator[asyncssh.SSHClientConnection]:
    # a real asyncssh server on an ephemeral loopback port, and a client connected to it
    server = await asyncssh.create_server(
        _NoAuthServer,
        "127.0.0.1",
        0,
        server_host_keys=[asyncssh.generate_private_key("ssh-ed25519")],
        process_factory=handler,
    )
    port = server.sockets[0].getsockname()[1]
    try:
        async with asyncssh.connect(
            "127.0.0.1",
            port,
            username="validator",
            known_hosts=None,
            client_keys=None,
            agent_path=None,
            config=None,
        ) as conn:
            yield conn
    finally:
        server.close()
        await server.wait_closed()


def _drop_connection_mid_command(stdout: str) -> ProcessHandler:
    async def handler(process: asyncssh.SSHServerProcess) -> None:
        if stdout:
            process.stdout.write(stdout)
            await process.stdout.drain()
        await asyncio.sleep(0.05)
        process.channel.get_connection().abort()

    return handler


def _close_channel_without_exit_status(stdout: str) -> ProcessHandler:
    async def handler(process: asyncssh.SSHServerProcess) -> None:
        if stdout:
            process.stdout.write(stdout)
            await process.stdout.drain()
        process.channel.close()

    return handler


def _exit_with(status: int) -> ProcessHandler:
    async def handler(process: asyncssh.SSHServerProcess) -> None:
        process.stderr.write("IndexError: list index out of range\n")
        process.exit(status)

    return handler


def _killed_by(signal: str) -> ProcessHandler:
    async def handler(process: asyncssh.SSHServerProcess) -> None:
        process.exit_with_signal(signal)

    return handler


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(_drop_connection_mid_command(""), id="tcp-abort-no-output"),
        pytest.param(
            _drop_connection_mid_command("gAAAAABtruncated"), id="tcp-abort-partial-output"
        ),
        pytest.param(_close_channel_without_exit_status(""), id="channel-closed-no-output"),
    ],
)
@pytest.mark.asyncio
async def test_real_ssh_drop_mid_command_comes_back_as_no_exit_status(handler):
    async with _local_ssh(handler) as conn:
        result = await SSHCommandRunner(conn).run("scrape", timeout=10, retryable=False)

    assert result.success is False
    assert result.exit_code == -1
    assert result.error_type == NO_EXIT_STATUS


async def _scrape_over(conn: asyncssh.SSHClientConnection, context_factory):
    ctx = context_factory(
        services=build_services(),
        config=build_context_config(machine_scrape_filename="scrape.sh", machine_scrape_timeout=10),
        state=build_state(remote_dir="/remote/path"),
        runner=SSHCommandRunner(conn),
        ssh=conn,
        encrypt_key="test-encrypt-key",
    )
    return await MachineSpecScrapeCheck().run(ctx)


@pytest.mark.parametrize(
    "handler,expected_reason,expected_error_type",
    [
        # before: exit 0 with empty stdout -> SCRAPE_FAILED_ON_HOST
        pytest.param(
            _drop_connection_mid_command(""),
            Msg.SCRAPE_TRANSPORT_FAILED.reason,
            NO_EXIT_STATUS,
            id="tcp-abort-no-output",
        ),
        # before: exit 0 with a cut token -> SCRAPE_PARSE_FAILED
        pytest.param(
            _drop_connection_mid_command("gAAAAABtruncated"),
            Msg.SCRAPE_TRANSPORT_FAILED.reason,
            NO_EXIT_STATUS,
            id="tcp-abort-partial-output",
        ),
        pytest.param(
            _close_channel_without_exit_status(""),
            Msg.SCRAPE_TRANSPORT_FAILED.reason,
            NO_EXIT_STATUS,
            id="channel-closed-no-output",
        ),
        pytest.param(_exit_with(1), Msg.SCRAPE_FAILED_ON_HOST.reason, None, id="host-exit-1"),
        pytest.param(
            _killed_by("KILL"), Msg.SCRAPE_FAILED_ON_HOST.reason, None, id="host-killed-by-signal"
        ),
    ],
)
@pytest.mark.asyncio
async def test_real_ssh_scrape_outcome_maps_to_its_code(
    handler, expected_reason, expected_error_type, context_factory
):
    async with _local_ssh(handler) as conn:
        result = await _scrape_over(conn, context_factory)

    assert result.passed is False
    assert result.event.reason_code == expected_reason
    assert result.event.what_we_saw.get("error_type") == expected_error_type
