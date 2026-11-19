"""DAH-5729: NodeAccountCheck over real SSH handshakes with two machines on two loopback addresses.

Machine A holds the host key the portal recorded when the account registered the node; machine B (another
account's) holds its own. Host keys come from the executor's real ssh_host_keys.sh, and machine A serves the key of
its executor container after the container was recreated. The validator's SSH session is a real asyncssh connection.
"""

import subprocess
from datetime import UTC, datetime, timedelta
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


NOTIFIED = {"NODE_ACCOUNT_UNVERIFIED": "2026-11-18T07:00:00Z"}  # the portal sent the per-node notice


def _relay_ctx(runner, hotkeys=("5Account", POOL), claim=CLAIM, address="203.0.113.7", claims=None, enforce=None):
    executor = ExecutorSSHInfo(
        uuid="node-1", address=address, port=8001, ssh_username="root", ssh_port=22, python_path="/usr/bin/python3",
        root_dir="/root", expected_node_hotkeys=list(hotkeys) if hotkeys else None, expected_account_claim=claim,
        expected_account_claims=claims, check_enforce_after=enforce,
    )
    return make_context(executor=executor, runner=runner, miner_hotkey=POOL)  # the validator reaches it via the pool hotkey


@pytest.mark.parametrize(
    "files, reason",
    [
        ("hotkey=5Victim\nclaim=\n", "NODE_ACCOUNT_MISMATCH"),  # another provider's own-key machine
        (f"hotkey={POOL}\nclaim=claim-of-another-account\n", "NODE_ACCOUNT_MISMATCH"),  # another account's pool node
        (f"hotkey={POOL}\nclaim={CLAIM}\n", "NODE_ACCOUNT_OK"),  # its own pool node: a pass, logged as one
        ("hotkey=5Account\nclaim=\n", "NODE_ACCOUNT_OK"),  # its own own-key node
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
    result = await NodeAccountCheck().run(_relay_ctx(_HostFiles("", success=False), enforce=NOTIFIED))
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
    runner = runner_module.SSHCommandRunner(_Ssh(outcome))
    result = await NodeAccountCheck().run(_relay_ctx(runner, enforce=NOTIFIED))
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
        ("reconnected, own machine", True, True, f"hotkey={POOL}\nclaim={CLAIM}\n", ["NODE_ACCOUNT_OK"], True),
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
    runner = _DropsTheSession(first) if drops else _HostFiles(files)
    ctx = _relay_ctx(runner, enforce=NOTIFIED).model_copy(update={"ssh": first})
    _, events, end = await Pipeline([NodeAccountCheck()], _Sink(), reconnect=reconnect).run(ctx)

    assert [e.reason_code for e in events] == reasons, case
    assert len(calls) == (1 if drops else 0)
    assert end.node_account_passed is account_passed
    assert end.clear_verified_job_info is (not account_passed and not rented)


# DAH-6539 follow-up (Mikhail's incident review, 19 Nov 2026)


@pytest.mark.parametrize(
    "files",
    [f"hotkey={POOL}\nclaim=claim-of-their-older-account\n", "hotkey=5TheirOtherHotkey\nclaim=\n"],
)
async def test_a_node_its_owner_set_up_under_another_of_their_accounts_passes(files):
    """itssb 17 Nov: a node installed under the owner's older account. The portal sends every claim and hotkey of the
    owner's accounts, so it passes with no re-setup and is not hidden."""
    ctx = _relay_ctx(
        _HostFiles(files),
        hotkeys=("5Account", POOL, "5TheirOtherHotkey"),
        claims=[CLAIM, "claim-of-their-older-account"],
    )
    result = await NodeAccountCheck().run(ctx)
    assert (result.passed, result.event.reason_code) == (True, "NODE_ACCOUNT_OK")


@pytest.mark.parametrize(
    "claim, passed, reason",
    [
        ("claim-of-their-custodied-account", True, "NODE_ACCOUNT_OK"),
        ("claim-of-a-stranger", False, "NODE_ACCOUNT_MISMATCH"),
        ("", False, "NODE_ACCOUNT_MISMATCH"),  # review: a pool node with its claim deleted proves nothing
    ],
)
async def test_a_pool_setup_on_an_own_key_listing_needs_an_owner_claim(claim, passed, reason):
    """Review (#5): the claim follows the machine's setup hotkey, not the key the listing is grouped under. On an
    own-key listing, a machine set up for the shared pool hotkey passes only with a claim of the owner's accounts:
    a linked custodied account's setup passes, any other pool node is refused."""
    executor = ExecutorSSHInfo(
        uuid="node-1", address="203.0.113.7", port=8001, ssh_username="root", ssh_port=22,
        python_path="/usr/bin/python3", root_dir="/root", expected_node_hotkeys=["5OwnKey"],
        expected_claims_by_hotkey={POOL: ["claim-of-their-custodied-account"]}, check_enforce_after={},
    )
    ctx = make_context(executor=executor, runner=_HostFiles(f"hotkey={POOL}\nclaim={claim}\n"), miner_hotkey="5OwnKey")

    result = await NodeAccountCheck().run(ctx)

    assert (result.passed, result.event.reason_code) == (passed, reason)


async def test_an_empty_claim_on_a_hotkey_the_listing_already_accepts_keeps_the_legacy_result():
    """Review: the map only tightens a hotkey it alone accepts; an older executor without a claim on the listing's own
    pool hotkey stays unverified, as before the map."""
    executor = ExecutorSSHInfo(
        uuid="node-1", address="203.0.113.7", port=8001, ssh_username="root", ssh_port=22,
        python_path="/usr/bin/python3", root_dir="/root", expected_node_hotkeys=[POOL],
        expected_account_claims=[CLAIM], expected_claims_by_hotkey={POOL: [CLAIM]}, check_enforce_after={},
    )
    ctx = make_context(executor=executor, runner=_HostFiles(f"hotkey={POOL}\nclaim=\n"), miner_hotkey=POOL)

    result = await NodeAccountCheck().run(ctx)

    assert (result.passed, result.event.reason_code) == (True, "NODE_ACCOUNT_SKIPPED")


@pytest.mark.parametrize(
    "until, passed",
    [
        (timedelta(minutes=5), True),  # the grant still runs: the partner's setup is the owner's
        (timedelta(seconds=-1), False),  # ended before the check, though the miner read the row before its end
        ("not a time", False),  # an unreadable end grants nothing beyond the listing
        (None, True),  # an older miner sends no end: the binding as sent
    ],
)
async def test_an_owner_wide_grant_counts_only_until_its_end_when_the_node_is_checked(until, passed):
    """Off-GitHub review on df0df43 (Critical 1): the miner can serve a cached snapshot read just before the grant's
    `owner_binding_until` and finish the upload after it. The end now travels with the node and is applied here, at
    check time, so a partner unlinked meanwhile stops passing and the listing account's own binding applies."""
    if isinstance(until, timedelta):
        until = (datetime.now(UTC) + until).isoformat()
    executor = ExecutorSSHInfo(
        uuid="node-1", address="203.0.113.7", port=8001, ssh_username="root", ssh_port=22,
        python_path="/usr/bin/python3", root_dir="/root", expected_node_hotkeys=["5Account", "5Partner"],
        check_enforce_after={}, owner_binding_until=until,
        listing_binding=None if until is None else {"expected_node_hotkeys": ["5Account"]},
    )
    ctx = make_context(executor=executor, runner=_HostFiles("hotkey=5Partner\nclaim=\n"), miner_hotkey="5Account")

    result = await NodeAccountCheck().run(ctx)

    assert (result.passed, result.event.reason_code) == (
        (True, "NODE_ACCOUNT_OK") if passed else (False, "NODE_ACCOUNT_MISMATCH")
    )


async def test_another_owners_setup_is_refused_at_once_without_a_notice():
    """The security case does not wait for the notice: no check_enforce_after, refused all the same."""
    ctx = _relay_ctx(
        _HostFiles(f"hotkey={POOL}\nclaim=claim-of-a-stranger\n"),
        claims=[CLAIM, "claim-of-their-older-account"],
    )
    result = await NodeAccountCheck().run(ctx)
    assert not result.passed and result.event.reason_code == "NODE_ACCOUNT_MISMATCH"
    assert result.event.what_we_saw["node_claim"] == "claim-of-a-stranger"


@pytest.mark.parametrize(
    "enforce", [{}, {"NODE_ACCOUNT_UNVERIFIED": "2999-01-01T00:00:00Z"}, {"x": "y"}]
)
async def test_a_failed_read_only_reports_until_the_provider_was_notified(enforce):
    """Not proof of another owner: before the portal's notice (or before its time) the run passes, the finding is no
    availability entry, and it is kept for the final event, from which the portal sends the notice."""
    from services.task.availability import availability_errors

    result = await NodeAccountCheck().run(
        _relay_ctx(_HostFiles("", success=False), enforce=enforce)
    )
    assert result.passed
    assert (result.event.reason_code, result.event.severity) == (
        "NODE_ACCOUNT_UNVERIFIED",
        "warning",
    )
    assert result.updates == {"reported_only": ("NODE_ACCOUNT_UNVERIFIED",)}
    assert availability_errors([result.event]) == []


class _Ends:
    """The check that ends a run: Finalize, or a rented node's tenant check, which halts before Finalize."""

    check_id = "executor.validate.ends"
    fatal = False

    def __init__(self, halt: bool, fails: bool = False):
        self.halt, self.fails = halt, fails

    async def run(self, ctx):
        from datetime import UTC, datetime

        from services.task.models import ValidationEvent
        from services.task.pipeline import CheckResult

        event = ValidationEvent(
            event="end", reason_code="ALREADY_RENTED" if self.halt else "OK", severity="info", impact="x",
            check_id=self.check_id, when=datetime.now(UTC),
        )
        return CheckResult(passed=not self.fails, event=event, halt=self.halt, fatal=self.fails or None)


@pytest.mark.parametrize("halt, fails", [(True, False), (False, False), (False, True)])
async def test_the_last_event_of_any_run_lists_the_findings_that_only_reported(halt, fails):
    """Review: a rented node's run halts before Finalize, and a later fatal check ends a run early; the finding still
    reaches the portal on the run's last event."""
    from services.task.pipeline import Pipeline

    class _Sink:
        async def emit(self, event):
            pass

    ctx = _relay_ctx(_HostFiles("", success=False), enforce={})
    _, events, _ = await Pipeline([NodeAccountCheck(), _Ends(halt, fails), _Ends(False)], _Sink()).run(ctx)

    assert events[-1].what_we_saw["reported_only"] == ["NODE_ACCOUNT_UNVERIFIED"]
    assert "reported_only" not in events[0].what_we_saw
    ctx = _relay_ctx(_HostFiles(f"hotkey={POOL}\nclaim={CLAIM}\n"), enforce={})
    _, events, _ = await Pipeline([NodeAccountCheck(), _Ends(halt, fails)], _Sink()).run(ctx)
    assert "reported_only" not in events[-1].what_we_saw


class _FingerprintDropout:
    """A rented node whose GPU set changed: the fingerprint fails without ending the run (the rented-pod checks still
    run for their reports), and _ended later moves this event last."""

    check_id = "gpu.validate.fingerprint"
    fatal = False

    async def run(self, ctx):
        from datetime import UTC, datetime

        from services.task.models import ValidationEvent
        from services.task.pipeline import CheckResult

        event = ValidationEvent(
            event="gpu", reason_code="GPU_MISSING", severity="error", impact="x", check_id=self.check_id,
            when=datetime.now(UTC),
        )
        return CheckResult(passed=False, event=event, fatal=False, updates={"gpu_fingerprint_passed": False})


async def test_a_rented_gpu_dropout_publishes_the_findings_that_only_reported():
    """Review: the halt event got the report, then _ended moved the fingerprint failure last; the published (last)
    event must carry it."""
    from services.task.pipeline import Pipeline

    class _Sink:
        async def emit(self, event):
            pass

    ctx = _relay_ctx(_HostFiles("", success=False), enforce={})
    passed, events, _ = await Pipeline([NodeAccountCheck(), _FingerprintDropout(), _Ends(True)], _Sink()).run(ctx)

    assert not passed and events[-1].reason_code == "GPU_MISSING"
    assert events[-1].what_we_saw["reported_only"] == ["NODE_ACCOUNT_UNVERIFIED"]


async def test_a_portal_that_sends_no_notice_map_keeps_the_failed_read_enforced(monkeypatch):
    """Validator first, portal later: no map at all means an older portal, so the stalled read is refused as before."""
    monkeypatch.setattr("services.task.checks.node_account.has_live_rental", lambda ctx: False)
    result = await NodeAccountCheck().run(_relay_ctx(_HostFiles("", success=False), enforce=None))
    assert not result.passed and result.event.reason_code == "NODE_ACCOUNT_UNVERIFIED"
