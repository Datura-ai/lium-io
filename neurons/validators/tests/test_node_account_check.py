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
# an idle node's refusal clears its verified job, marked so the backend applies the reset only while it stays unrented
HOST_KEY_PART = "executor.validate.node_account.host_key"
BINDING_PART = "executor.validate.node_account.account_binding"


def refused_idle(part):
    """A free node's refusal: the reset names the part that failed, the one a later pass must prove to lift a hold."""
    return {
        "node_account_passed": False,
        "clear_verified_job_info": True,
        "clear_verified_job_unrented_only": True,
        "clear_verified_job_evidence": {"check_id": part},
    }
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
    assert result.updates == refused_idle(HOST_KEY_PART)


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
        assert result.updates == refused_idle(BINDING_PART)
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
        assert (result.updates, result.fatal) == (refused_idle(BINDING_PART), True)
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
    if end.clear_verified_job_info:
        assert end.clear_verified_job_unrented_only is True


# DAH-6784: a pass lists the parts it proved; a part whose input is missing is not proven


@pytest.mark.parametrize(
    "files, hotkeys, claim, proven",
    [
        (f"hotkey={POOL}\nclaim={CLAIM}\n", ("5Account", POOL), CLAIM, (BINDING_PART,)),  # pool node, claims match
        ("hotkey=5Account\nclaim=\n", ("5Account", POOL), CLAIM, (BINDING_PART,)),  # an own-key node needs no claim
        ("hotkey=5Account\nclaim=\n", ("5Account", POOL), None, (BINDING_PART,)),  # nor an expected one
        ("hotkey=\nclaim=\n", ("5Account", POOL), CLAIM, None),  # no host files: nothing proven
        (f"hotkey={POOL}\nclaim=\n", ("5Account", POOL), CLAIM, None),  # the observed claim missing
        # the expected claim missing: a pool node's claim is compared with nothing, so nothing is proven
        (f"hotkey={POOL}\nclaim={CLAIM}\n", ("5Account", POOL), None, None),
        (f"hotkey={POOL}\nclaim=claim-of-another-account\n", ("5Account", POOL), None, None),
        (f"hotkey={POOL}\nclaim=\n", ("5Account", POOL), None, None),
        ("hotkey=5Account\n", None, CLAIM, None),  # an IP node naming no account: nothing to prove
    ],
)
async def test_only_a_proven_binding_is_listed(files, hotkeys, claim, proven):
    result = await NodeAccountCheck().run(_relay_ctx(_HostFiles(files), hotkeys=hotkeys, claim=claim))
    assert result.passed
    assert result.updates.get("idle_check_passed") == proven


async def test_a_matched_host_key_is_listed_as_that_part_only(recorded_machine):
    keys, recorded = recorded_machine
    result = await _check_against(keys["ed25519"], recorded)
    assert result.event.reason_code == "NODE_ACCOUNT_OK" and result.updates == {"idle_check_passed": (HOST_KEY_PART,)}


def _lifts(failed: str, passed: tuple[str, ...]) -> bool:
    """The backend's rule (lium-platform daos/executor.py _fault_checks_proven): a hold lifts once the check its reset
    named is among the ids a later pass listed."""
    return failed in passed


@pytest.mark.parametrize(
    "fail, later, lifts",
    [
        # a binding hold, then a cycle whose miner lists no hotkeys: only the host key is proven, the hold stays
        ({"hotkeys": ("5Account", POOL), "files": "hotkey=5Victim\nclaim=\n", "host_key": True}, {"hotkeys": None, "host_key": True}, False),
        # the converse: a host-key hold, then a cycle without the recorded key and a proven binding
        ({"host_key": "other"}, {"host_key": False, "files": f"hotkey={POOL}\nclaim={CLAIM}\n"}, False),
        # a pool node's claim mismatch, then a cycle whose miner omits the expected claim: the claim is never compared
        ({"files": f"hotkey={POOL}\nclaim=claim-of-another-account\n"}, {"files": f"hotkey={POOL}\nclaim=claim-of-another-account\n", "claim": None}, False),
        # the same part proven again lifts it
        ({"hotkeys": ("5Account", POOL), "files": "hotkey=5Victim\nclaim=\n", "host_key": True}, {"files": f"hotkey={POOL}\nclaim={CLAIM}\n", "host_key": True}, True),
        ({"host_key": "other"}, {"host_key": True}, True),
    ],
)
async def test_a_hold_lifts_only_on_proof_of_the_part_that_failed(fail, later, lifts, monkeypatch):
    from services.task.pipeline import updates_with_clear_verified_job_evidence

    monkeypatch.setattr("services.task.checks.node_account.has_live_rental", lambda ctx: False)

    async def cycle(hotkeys=("5Account", POOL), files="hotkey=\nclaim=\n", host_key=False, claim=CLAIM):
        monkeypatch.setattr(
            "services.task.checks.node_account.presented_host_key",
            lambda ctx: "ssh-ed25519 AAAAother" if host_key == "other" else "ssh-ed25519 AAAAregistered",
        )
        ctx = _relay_ctx(_HostFiles(files), hotkeys=hotkeys, claim=claim, address="node.example.test")
        ctx.executor.expected_ssh_host_key = "ssh-ed25519 AAAAregistered" if host_key else None
        result = await NodeAccountCheck().run(ctx)
        return updates_with_clear_verified_job_evidence(result, NodeAccountCheck.check_id)

    failed = (await cycle(**fail))["clear_verified_job_evidence"]["check_id"]
    assert _lifts(failed, (await cycle(**later)).get("idle_checks_passed", ())) is lifts


async def test_both_parts_proven_list_the_whole_check_too(monkeypatch):
    # a reset that named no check needs every idle-only check id, the bare one of this check among them
    monkeypatch.setattr("services.task.checks.node_account.presented_host_key", lambda ctx: "ssh-ed25519 AAAAregistered")
    ctx = _relay_ctx(_HostFiles(f"hotkey={POOL}\nclaim={CLAIM}\n"), address="node.example.test")
    ctx.executor.expected_ssh_host_key = "ssh-ed25519 AAAAregistered"
    result = await NodeAccountCheck().run(ctx)
    assert result.updates == {"idle_check_passed": (HOST_KEY_PART, BINDING_PART, "executor.validate.node_account")}


async def test_a_node_set_up_under_another_account_of_the_listing_owner_passes_by_the_portals_own_answer(monkeypatch):
    """DAH-6740: set up under the owner's account X, listed under their account Y. The relayed list names Y only; the
    owner map this validator read from the portal itself names X for Y, so the setup passes. A hotkey the portal does
    not name for Y is still another owner's machine."""
    from clients import validator_portal_api

    monkeypatch.setattr(validator_portal_api, "_owner_hotkeys", {})
    monkeypatch.setattr(validator_portal_api, "_owner_hotkeys_read_at", None)
    validator_portal_api.remember_owner_hotkeys({"5ListingY": ["5OwnerX"]})
    executor = ExecutorSSHInfo(
        uuid="node-1", address="203.0.113.7", port=8001, ssh_username="root", ssh_port=22,
        python_path="/usr/bin/python3", root_dir="/root", expected_node_hotkeys=["5ListingY"], expected_account_claim=None,
    )

    def ctx(files):
        return make_context(executor=executor, runner=_HostFiles(files), miner_hotkey="5ListingY")

    assert (await NodeAccountCheck().run(ctx("hotkey=5OwnerX\nclaim=\n"))).passed
    refused = await NodeAccountCheck().run(ctx("hotkey=5Victim\nclaim=\n"))
    assert refused.event.reason_code == "NODE_ACCOUNT_MISMATCH"
    # the portal answer aged out (outage): X is a stranger again
    monkeypatch.setattr(
        validator_portal_api, "_owner_hotkeys_read_at",
        validator_portal_api.time.monotonic() - validator_portal_api.OWNER_HOTKEYS_MAX_AGE_SECONDS - 1,
    )
    stale = await NodeAccountCheck().run(ctx("hotkey=5OwnerX\nclaim=\n"))
    assert stale.event.reason_code == "NODE_ACCOUNT_MISMATCH"
