"""DAH-5729: NodeAccountCheck over real SSH handshakes with two machines on two loopback addresses.

Machine A holds the host key the portal recorded when the account registered the node; machine B (another
account's) holds its own. Host keys come from the executor's real ssh_host_keys.sh, and machine A serves the key of
its executor container after the container was recreated. The validator's SSH session is a real asyncssh connection.
"""

import subprocess
from pathlib import Path

import asyncssh
import pytest
import pytest_asyncio
from datura.requests.miner_requests import ExecutorSSHInfo

from services.task.checks.node_account import NodeAccountCheck
from tests.helpers import make_context

IP_A, IP_B = "127.0.0.1", "127.0.0.2"
HOST_KEYS_SCRIPT = Path(__file__).resolve().parents[2] / "executor" / "ssh_host_keys.sh"

pytestmark = pytest.mark.asyncio


class _OpenServer(asyncssh.SSHServer):
    def begin_auth(self, username):
        return False  # no auth needed: the test only needs the key exchange


def _executor_container_start(container: Path, volume: Path, key_type: str = "ed25519") -> asyncssh.SSHKey:
    """The executor container's start (run.sh runs ssh_host_keys.sh); the host key of `key_type` its sshd serves."""
    env = {"PATH": "/usr/bin:/bin", "SSH_ROOT": str(container), "SSH_HOST_KEYS_DIR": str(volume)}
    subprocess.run(["sh", str(HOST_KEYS_SCRIPT)], env=env, check=True, capture_output=True)
    return asyncssh.read_private_key(container / f"etc/ssh/ssh_host_{key_type}_key")


@pytest_asyncio.fixture
async def two_machines(tmp_path):
    # machine A registers, then its executor container is recreated (fresh filesystem, same volume)
    registered = _executor_container_start(tmp_path / "a-container-1", tmp_path / "a-volume")
    key_a = _executor_container_start(tmp_path / "a-container-2", tmp_path / "a-volume")
    key_b = _executor_container_start(tmp_path / "b-container", tmp_path / "b-volume")
    server_a = await asyncssh.listen(IP_A, 0, server_host_keys=[key_a], server_factory=_OpenServer)
    port = server_a.sockets[0].getsockname()[1]
    server_b = await asyncssh.listen(IP_B, port, server_host_keys=[key_b], server_factory=_OpenServer)
    conns = []

    async def dial(ip):
        conn = await asyncssh.connect(ip, port, known_hosts=None, username="root")
        conns.append(conn)
        return conn

    registered_key = registered.export_public_key("openssh").decode().strip() + " root@registered"
    yield dial, registered_key
    for conn in conns:
        conn.close()
    server_a.close()
    server_b.close()


def _ctx(address: str, ssh, expected_ssh_host_key: str | None):
    executor = ExecutorSSHInfo(
        uuid="node-1", address=address, port=8001, ssh_username="root", ssh_port=22,
        python_path="/usr/bin/python3", root_dir="/root", expected_ssh_host_key=expected_ssh_host_key,
    )
    return make_context(executor=executor, ssh=ssh)


async def test_hostname_node_on_its_registered_machine_passes_after_recreation(two_machines):
    dial, registered_key = two_machines
    result = await NodeAccountCheck().run(_ctx("node.example.com", await dial(IP_A), registered_key))
    assert result.passed
    assert result.event.reason_code == "NODE_ACCOUNT_OK"


async def test_hostname_re_pointed_to_another_machine_is_refused(two_machines):
    dial, registered_key = two_machines
    result = await NodeAccountCheck().run(_ctx("node.example.com", await dial(IP_B), registered_key))
    assert not result.passed
    assert result.event.reason_code == "NODE_ACCOUNT_MISMATCH"
    assert result.updates == {"node_account_passed": False, "clear_verified_job_info": True}


async def test_ip_addressed_node_is_unchanged(two_machines):
    dial, registered_key = two_machines
    result = await NodeAccountCheck().run(_ctx(IP_A, await dial(IP_B), registered_key))
    assert result.passed
    assert result.event.reason_code == "NODE_ACCOUNT_SKIPPED"


async def test_missing_expected_host_key_is_not_failed(two_machines):
    dial, _ = two_machines
    result = await NodeAccountCheck().run(_ctx("node.example.com", await dial(IP_B), None))
    assert result.passed
    assert result.event.reason_code == "NODE_ACCOUNT_SKIPPED"


def _public(key: asyncssh.SSHKey) -> str:
    return key.export_public_key("openssh").decode().strip()


async def _check_against(key: asyncssh.SSHKey, recorded: str):
    """NodeAccountCheck over a real handshake with a server that presents only `key`."""
    server = await asyncssh.listen(IP_A, 0, server_host_keys=[key], server_factory=_OpenServer)
    try:
        port = server.sockets[0].getsockname()[1]
        async with asyncssh.connect(IP_A, port, known_hosts=None, username="root") as conn:
            return await NodeAccountCheck().run(_ctx("node.example.com", conn, recorded))
    finally:
        server.close()


KEY_TYPES = ["ed25519", "ecdsa", "rsa"]


@pytest.fixture
def recorded_machine(tmp_path):
    """The node's host keys by type, and the portal's record of them: sorted `<type> <base64>` lines."""
    _executor_container_start(tmp_path / "container", tmp_path / "volume")
    keys = {t: asyncssh.read_private_key(tmp_path / "volume" / f"ssh_host_{t}_key") for t in KEY_TYPES}
    return keys, "\n".join(sorted(_public(k) for k in keys.values()))


@pytest.mark.parametrize("key_type", KEY_TYPES)
async def test_every_recorded_host_key_passes(recorded_machine, key_type):
    keys, recorded = recorded_machine
    result = await _check_against(keys[key_type], recorded)
    assert result.passed
    assert result.event.reason_code == "NODE_ACCOUNT_OK"


async def test_unrecorded_host_key_fails(recorded_machine, tmp_path):
    _, recorded = recorded_machine
    other = _executor_container_start(tmp_path / "other-container", tmp_path / "other-volume")
    result = await _check_against(other, recorded)
    assert not result.passed
    assert result.event.reason_code == "NODE_ACCOUNT_MISMATCH"


# DAH-6539: an IP node behind a relay forwarding to another account's machine
POOL, CLAIM = "5Pool", "claim-of-this-account"


class _HostFiles:
    """The validator's SSH runner on a machine whose /etc/liumd files read `stdout` (fixed command only)."""

    def __init__(self, stdout: str, success: bool = True):
        self.stdout, self.success, self.commands = stdout, success, []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):  # noqa: ASYNC109
        from datetime import UTC, datetime

        from services.task.runner import SSHCommandResult

        self.commands.append(cmd)
        now = datetime.now(UTC)
        return SSHCommandResult(
            command=cmd, command_id="c", exit_code=0 if self.success else 1, stdout=self.stdout, stderr="",
            duration_ms=1, started_at=now, finished_at=now, success=self.success,
        )


def _relay_ctx(runner, hotkeys=("5Account", POOL), claim=CLAIM, address="203.0.113.7"):
    executor = ExecutorSSHInfo(
        uuid="node-1", address=address, port=8001, ssh_username="root", ssh_port=22, python_path="/usr/bin/python3",
        root_dir="/root", expected_node_hotkeys=list(hotkeys) if hotkeys else None, expected_account_claim=claim,
    )
    return make_context(executor=executor, runner=runner, miner_hotkey=POOL)  # the validator reaches it via the pool hotkey


@pytest.mark.parametrize(
    "files, reason",
    [
        ("hotkey=5Victim\nclaim=\n", "NODE_ACCOUNT_MISMATCH"),  # another provider's own-key machine
        (f"hotkey={POOL}\nclaim=claim-of-another-account\n", "NODE_ACCOUNT_MISMATCH"),  # another account's pool node
        (f"hotkey={POOL}\nclaim={CLAIM}\n", "NODE_ACCOUNT_SKIPPED"),  # its own pool node
        ("hotkey=5Account\nclaim=\n", "NODE_ACCOUNT_SKIPPED"),  # its own own-key node
        (f"hotkey={POOL}\nclaim=\n", "NODE_ACCOUNT_SKIPPED"),  # a pool node from before the claim file: no evidence
        ("hotkey=\nclaim=\n", "NODE_ACCOUNT_SKIPPED"),  # an executor without the host files: no evidence
    ],
)
async def test_ip_node_is_refused_on_a_machine_set_up_for_another_account(files, reason):
    runner = _HostFiles(files)
    result = await NodeAccountCheck().run(_relay_ctx(runner))
    assert result.event.reason_code == reason
    assert result.passed is (reason != "NODE_ACCOUNT_MISMATCH")
    if not result.passed:
        assert result.updates == {"node_account_passed": False, "clear_verified_job_info": True}
        assert "hostname" not in result.event.remediation
    assert runner.commands == [
        "printf 'hotkey=%s\\nclaim=%s\\n' \"$(cat /etc/liumd/miner_hotkey 2>/dev/null)\" "
        "\"$(cat /etc/liumd/account_claim 2>/dev/null)\""
    ]


@pytest.mark.parametrize(
    "files, success, unread",
    [("hotkey=\nclaim=\n", True, "hotkey"), (f"hotkey={POOL}\nclaim=\n", True, "claim")],
)
async def test_no_evidence_passes_and_is_logged(files, success, unread, caplog):
    """Missing files or an empty pool claim (a read that worked) prove nothing: the node passes, the read is logged."""
    result = await NodeAccountCheck().run(_relay_ctx(_HostFiles(files, success=success)))
    assert result.passed
    [record] = [r for r in caplog.records if "NODE_ACCOUNT_BINDING_UNVERIFIED" in r.getMessage()]
    assert record.msg.extra["unread"] == unread


async def test_an_own_key_account_expects_no_claim():
    result = await NodeAccountCheck().run(_relay_ctx(_HostFiles("hotkey=5Account\nclaim=\n"), claim=None))
    assert result.passed


async def test_an_older_miner_is_not_checked():
    runner = _HostFiles("hotkey=5Victim\n")
    assert (await NodeAccountCheck().run(_relay_ctx(runner, hotkeys=None))).passed
    assert runner.commands == []


@pytest.mark.parametrize("rented", [True, False])
async def test_a_rented_node_on_another_accounts_machine_keeps_its_rental(rented, monkeypatch):
    """Rented: score 0 only (calculate_scores reads node_account_passed); the verified job is kept and the run goes on,
    so the renter's pod runs on and is billed. Free: the verified job is cleared and the run ends (Sergej, lio#2288)."""
    monkeypatch.setattr("services.task.checks.node_account.has_live_rental", lambda ctx: rented)
    result = await NodeAccountCheck().run(_relay_ctx(_HostFiles("hotkey=5Victim\nclaim=\n")))
    assert not result.passed and result.event.reason_code == "NODE_ACCOUNT_MISMATCH"
    if rented:
        assert (result.updates, result.fatal) == ({"node_account_passed": False}, False)
    else:
        assert (result.updates, result.fatal) == ({"node_account_passed": False, "clear_verified_job_info": True}, True)
    # an availability entry: the backend hides the node, so a partly rented one takes no rental on its free GPUs
    from services.task.availability import NODE_GPU_FAULT_CODES, availability_errors

    [entry] = availability_errors([result.event])
    assert (entry.reason_code, entry.reach_target) == ("NODE_ACCOUNT_MISMATCH", "node")
    assert entry.reason_code in NODE_GPU_FAULT_CODES  # never silenced as our own outage


@pytest.mark.parametrize("rented", [True, False])
async def test_a_failed_or_stalled_read_hides_the_node_until_a_cycle_reads_it(rented, monkeypatch):
    """A relay can stall the read: a read that does not succeed is refused like a mismatch, under its own reason."""
    from services.task.availability import NODE_GPU_FAULT_CODES, availability_errors

    monkeypatch.setattr("services.task.checks.node_account.has_live_rental", lambda ctx: rented)
    result = await NodeAccountCheck().run(_relay_ctx(_HostFiles("", success=False)))
    assert not result.passed and result.fatal is (not rented)
    assert result.event.event == "Node account ownership could not be verified at this address"
    [entry] = availability_errors([result.event])
    assert (entry.reason_code, entry.reach_target) == ("NODE_ACCOUNT_UNVERIFIED", "node")
    assert entry.reason_code not in NODE_GPU_FAULT_CODES  # a reach failure: silenced with our own outage, like one


async def test_another_machines_host_key_is_a_mismatch_even_when_the_account_read_stalls(monkeypatch):
    """The host-key comparison needs no read, so it runs first: a stalled read cannot turn this direct evidence into
    NODE_ACCOUNT_UNVERIFIED, which our own-outage silencing would drop (Sergej, lio#2288 at a81b1b9)."""
    from services.task.availability import NODE_GPU_FAULT_CODES, availability_errors

    monkeypatch.setattr("services.task.checks.node_account.has_live_rental", lambda ctx: True)
    monkeypatch.setattr("services.task.checks.node_account.presented_host_key", lambda ctx: "ssh-ed25519 AAAAother")
    runner = _HostFiles("", success=False)
    ctx = _relay_ctx(runner, address="node.example.test")
    ctx.executor.expected_ssh_host_key = "ssh-ed25519 AAAAregistered"
    result = await NodeAccountCheck().run(ctx)
    assert not result.passed and result.event.reason_code == "NODE_ACCOUNT_MISMATCH"
    [entry] = availability_errors([result.event])
    assert entry.reason_code in NODE_GPU_FAULT_CODES  # kept through an outage cycle, so a rented node stays hidden
    assert runner.commands == []


class _Ssh:
    """An asyncssh connection whose every command ends `outcome`: an exit status, None (no exit status) or a stall."""

    def __init__(self, outcome):
        self.outcome = outcome

    async def run(self, cmd, input=None):  # noqa: A002
        import asyncio
        from types import SimpleNamespace

        if self.outcome == "stall":
            await asyncio.sleep(3600)
        return SimpleNamespace(stdout="", stderr="", exit_status=self.outcome)


@pytest.mark.parametrize("outcome, read_error", [(1, "exit 1"), (None, "no_exit_status"), ("stall", "timeout")])
async def test_every_way_the_read_can_fail_is_refused(outcome, read_error, monkeypatch):
    """Through the real runner: a nonzero exit, a channel closed without an exit status and a timeout (after its
    retry) are all NODE_ACCOUNT_UNVERIFIED; the read that worked on an older executor passes (above)."""
    from services.task import runner as runner_module
    from services.task.checks import node_account as module

    monkeypatch.setattr(module, "READ_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(runner_module.asyncio, "sleep", _no_wait(runner_module.asyncio.sleep))
    monkeypatch.setattr(module, "has_live_rental", lambda ctx: False)
    result = await NodeAccountCheck().run(_relay_ctx(runner_module.SSHCommandRunner(_Ssh(outcome))))
    assert not result.passed and result.fatal
    assert result.event.reason_code == "NODE_ACCOUNT_UNVERIFIED"
    assert result.event.what_we_saw["read_error"] == read_error


def _no_wait(sleep):
    """The runner's back-off between attempts, skipped; the fake connection's own stall still sleeps."""

    async def fast(delay):
        await sleep(0 if delay < 60 else delay)

    return fast


class _DropsTheSession(_HostFiles):
    """The account read on a session that closes under it (a recreated executor container, or a relay)."""

    def __init__(self, ssh):
        super().__init__("", success=False)
        self.ssh = ssh

    async def run(self, cmd, **kwargs):
        self.ssh.is_closed.return_value = True
        return await super().run(cmd, **kwargs)


@pytest.mark.parametrize("rented", [True, False])
@pytest.mark.parametrize(
    "case, drops, reconnects, files, reasons, account_passed",
    [
        # the session dropped and the reconnect worked: the read runs again on the new session and stands
        ("reconnected, own machine", True, True, f"hotkey={POOL}\nclaim={CLAIM}\n", ["NODE_ACCOUNT_SKIPPED"], True),
        ("reconnected, mismatch", True, True, "hotkey=5Victim\nclaim=\n", ["NODE_ACCOUNT_MISMATCH"], False),
        # the reconnect failed: the run ends on the dead session, as for any other check
        (
            "reconnect failed",
            True,
            False,
            None,
            ["NODE_ACCOUNT_UNVERIFIED", "EXECUTOR_TRANSPORT_UNREACHABLE"],
            False,
        ),
        # a mismatch read on a session that stayed open: no reconnect
        ("confirmed mismatch", False, None, "hotkey=5Victim\nclaim=\n", ["NODE_ACCOUNT_MISMATCH"], False),
    ],
)
async def test_a_dropped_session_during_the_account_read_is_reconnected_through_the_pipeline(
    case, drops, reconnects, files, reasons, account_passed, rented, monkeypatch
):
    """Through Pipeline: a read that failed on a dropped session is retried, never taken as evidence; only the
    refusal of a free node clears its verified job (a rented one keeps it, its rental runs on and is billed)."""
    from unittest.mock import MagicMock

    from services.task import pipeline as pipeline_module
    from services.task.pipeline import Pipeline

    class _Sink:
        async def emit(self, event):
            pass

    def _session():
        ssh = MagicMock(spec=asyncssh.SSHClientConnection)
        ssh.is_closed.return_value = False
        return ssh

    monkeypatch.setattr("services.task.checks.node_account.has_live_rental", lambda ctx: rented)
    monkeypatch.setattr(pipeline_module, "SSHCommandRunner", lambda conn, max_retries=1: _HostFiles(files))
    calls = []

    async def reconnect():
        calls.append(1)
        if not reconnects:
            raise OSError("connection refused")
        return _session()

    first = _session()
    ctx = _relay_ctx(_DropsTheSession(first) if drops else _HostFiles(files)).model_copy(update={"ssh": first})
    _, events, end = await Pipeline([NodeAccountCheck()], _Sink(), reconnect=reconnect).run(ctx)

    assert [e.reason_code for e in events] == reasons, case
    assert len(calls) == (1 if drops else 0)
    assert end.node_account_passed is account_passed
    assert end.clear_verified_job_info is (not account_passed and not rented)


class _FindingThenTheSessionCloses:
    """A check that finished reading a finding off the node, cleared the verified job and found the session closed
    afterwards (a provider dropping it): not replayable, so the run ends on EXECUTOR_TRANSPORT_UNREACHABLE."""

    check_id = "test.validate.finding"
    fatal = True

    def __init__(self, template):
        self.template = template

    async def run(self, ctx):
        from services.task.messages import render_message
        from services.task.pipeline import CheckResult

        ctx.ssh.is_closed.return_value = True
        event = render_message(self.template, ctx=ctx, check_id=self.check_id)
        return CheckResult(passed=False, event=event, updates={"clear_verified_job_info": True}, fatal=True)


def _finding(name):
    from services.task.messages import BannedGpuMessages, NvmlDigestMessages, SpecChangeMessages

    return {
        "GPU_BANNED": BannedGpuMessages.GPU_BANNED,
        "NVML_DIGEST_MISMATCH": NvmlDigestMessages.DIGEST_MISMATCH,
        "SPEC_CHANGED": SpecChangeMessages.SPEC_CHANGED,
    }[name]


class _Shell:
    def __init__(self, *args, **kwargs):
        self.ssh_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.parametrize(
    "cause, case, sent_in_the_cycle, sent_at_the_end",
    [
        # the account read the session drop cut, reconnect refused: held for the cycle end
        ("NODE_ACCOUNT_UNVERIFIED", "withheld", False, False),
        ("NODE_ACCOUNT_UNVERIFIED", "stands", False, True),
        # a miner job cancelled at the wait: its result is judged on itself, as its verdict would have been
        ("NODE_ACCOUNT_UNVERIFIED", "job_timed_out", False, False),
        ("NODE_ACCOUNT_UNVERIFIED", "job_timed_out_outside_a_window", False, True),
        # a finished finding followed by a closed session, inside a window: its reset goes at once
        ("GPU_BANNED", "withheld", True, False),
        ("NVML_DIGEST_MISMATCH", "withheld", True, False),
        ("SPEC_CHANGED", "withheld", True, False),
    ],
)
async def test_a_reset_waits_for_the_rollout_decision_only_when_its_own_cause_is_a_read_that_did_not_finish(
    cause, case, sent_in_the_cycle, sent_at_the_end, monkeypatch
):
    """DAH-3405 + DAH-6539, 23 Nov: a recreated executor container dropped the account read and refused the
    reconnect; the run ended on EXECUTOR_TRANSPORT_UNREACHABLE, which the rollout window withheld at cycle end, but
    its NODE_ACCOUNT_UNVERIFIED reset had gone out inside the cycle and hid the node. Such a reset now waits for the
    cycle end, which drops it when the window withholds its result and sends it otherwise. DAH-6951: the reset's own
    cause decides, not the run's last event: a reset a finished finding set (GPU_BANNED, NVML_DIGEST_MISMATCH,
    SPEC_CHANGED) goes out at once even when the session was found closed after the check, inside a window.
    Through TaskService.create_task, as the wave runs it."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from core.validator import Validator
    from payload_models.payloads import MinerJobRequestPayload
    from services.executor_rollout import RolloutWindow
    from services.task import pipeline as pipeline_module
    from services.task.pipeline import Pipeline
    from services.task.service import TaskService

    class _Sink:
        async def emit(self, event):
            pass

    async def reconnect():
        raise OSError("connection refused")

    monkeypatch.setattr("services.task.checks.node_account.has_live_rental", lambda ctx: False)
    monkeypatch.setattr("services.task.service.settings.DRY_RUN", False)
    first = MagicMock(spec=asyncssh.SSHClientConnection)
    first.is_closed.return_value = False
    if cause == "NODE_ACCOUNT_UNVERIFIED":
        ctx, check = _relay_ctx(_DropsTheSession(first)), NodeAccountCheck()
    else:
        ctx, check = _relay_ctx(_HostFiles("")), _FindingThenTheSessionCloses(_finding(cause))
    ctx = ctx.model_copy(update={"ssh": first})
    monkeypatch.setattr(pipeline_module, "SSHCommandRunner", lambda conn, max_retries=1: _HostFiles(""))
    run = await Pipeline([check], _Sink(), reconnect=reconnect).run(ctx)
    assert not run[0] and run[1][-1].reason_code == "EXECUTOR_TRANSPORT_UNREACHABLE"

    # the wave's run, through TaskService.create_task
    redis = MagicMock(clear_verified_job_info=AsyncMock())
    service = TaskService.__new__(TaskService)
    service.redis_service = redis
    service.held_resets = {}
    service.ssh_service = MagicMock(decrypt_payload=MagicMock(return_value="key"))
    service.attestation_service = MagicMock(
        prepare_host_policy=AsyncMock(
            return_value=MagicMock(
                known_hosts=None, attestation_digest=None, tee_type=None, gpu_attestation_passed=None,
                attestation_passed=False,
            )
        )
    )
    service.pipeline_factory = MagicMock(build_context=AsyncMock(return_value=MagicMock(verified={})))
    service.pipeline_factory.build_pipeline.return_value.run = AsyncMock(return_value=run)
    miner = MinerJobRequestPayload(
        job_batch_id="batch-1", miner_hotkey=POOL, miner_coldkey="5Cold", miner_address="10.0.0.5", miner_port=8080,
    )
    with (
        patch("services.task.service.InteractiveShellService", _Shell),
        patch("services.task.service.PipelineFactory.takes_fast_path", return_value=False),
        patch("services.task.service.run_liumd_shadow", AsyncMock()),
    ):
        result = await service.create_task(
            miner_info=miner, executor_info=run[2].executor, keypair=MagicMock(ss58_address="5Val"),
            private_key="key", public_key="pub", encrypted_files=MagicMock(), rented_data=MagicMock(),
            default_docker_image_digests={}, hold_rollout_resets=True,
        )
    assert result.failure_reason_code == "EXECUTOR_TRANSPORT_UNREACHABLE"
    assert redis.clear_verified_job_info.await_count == (1 if sent_in_the_cycle else 0)
    assert service.held_resets == ({} if sent_in_the_cycle else {"batch-1": [result]})

    # the cycle end
    window = RolloutWindow(
        digest="sha256:" + "b" * 64, previous_digest="sha256:" + "a" * 64, started_at=None,
        opened_job_block=None if case in ("stands", "job_timed_out_outside_a_window") else 7500,
        cycles_seen=1, grace_cycles=2,
    )
    validator = Validator.__new__(Validator)
    validator.default_extra = {}
    validator.redis_service = redis
    validator.miner_service = SimpleNamespace(task_service=service, outage_tally=None)
    validator.rollout_tracker = MagicMock(record_withheld=AsyncMock())
    validator.fetch_executor_digest_or_none = AsyncMock(return_value=window.digest)
    validator.observe_executor_rollout = AsyncMock(return_value=window)
    # the timed-out miner job was cancelled: its results never reach the cycle end
    cycle = {} if case.startswith("job_timed_out") else {POOL: [result]}
    standing, withheld = await validator.withhold_verdicts_for_rollout(cycle, 7500, "batch-1", window)
    assert service.held_resets == {}
    assert [w.result for w in withheld] == ([result] if case == "withheld" else [])

    assert redis.clear_verified_job_info.await_count == (1 if sent_in_the_cycle or sent_at_the_end else 0)
    if redis.clear_verified_job_info.await_count:
        reset = redis.clear_verified_job_info.await_args.kwargs
        assert reset["executor_id"] == "node-1"
        assert reset["evidence"]["reason_code"] == cause


async def test_held_resets_are_decided_per_cycle():
    """DAH-6951: held resets are kept by job_batch_id; the cycle's own finally decides what its withhold did not
    reach, so nothing is carried into the next cycle (where an old reset could undo a newer pass) or lost."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from core.validator import Validator
    from services.executor_rollout import RolloutWindow

    def _held(batch):
        return SimpleNamespace(
            score=0, failure_reason_code="EXECUTOR_TRANSPORT_UNREACHABLE", executor_image_report=None, spec=None,
            executor_info=SimpleNamespace(uuid=f"node-{batch}"), held_verified_job_reset={"executor_id": batch},
        )

    validator = Validator.__new__(Validator)
    validator.default_extra = {}
    validator.redis_service = MagicMock(clear_verified_job_info=AsyncMock())
    held = {"batch-1": [_held("batch-1")], "batch-2": [_held("batch-2")]}
    validator.miner_service = SimpleNamespace(task_service=SimpleNamespace(held_resets=held))
    outside = RolloutWindow(
        digest=None, previous_digest=None, started_at=None, opened_job_block=None, cycles_seen=0, grace_cycles=2
    )

    await validator.send_held_verified_job_resets("batch-1", outside, 7500)

    validator.redis_service.clear_verified_job_info.assert_awaited_once_with(executor_id="batch-1")
    assert list(held) == ["batch-2"]
