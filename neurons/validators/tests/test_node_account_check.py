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
    assert result.updates == {"clear_verified_job_info": True}


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
