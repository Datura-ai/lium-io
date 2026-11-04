"""DAH-5729: NodeAccountCheck against real HTTP executors on two loopback addresses.

Address A answers for the account that registered the node; address B for another account. The SSH
session's peer address stands for the address the validator's pinned dial reached after DNS resolution.
"""

import pytest
import pytest_asyncio
from aiohttp import web
from datura.requests.miner_requests import ExecutorSSHInfo

from services.task.checks.node_account import NodeAccountCheck
from tests.helpers import make_context

IP_A, IP_B = "127.0.0.1", "127.0.0.2"
OWN_HOTKEY, OTHER_HOTKEY, POOL_HOTKEY = "5Own", "5Other", "5Pool"
OWN_CLAIM, OTHER_CLAIM = "a" * 32, "b" * 32

pytestmark = pytest.mark.asyncio


class _Ssh:
    def __init__(self, ip: str):
        self.ip = ip

    def get_extra_info(self, name):
        return (self.ip, 22) if name == "peername" else None


async def _executor_at(ip: str, identity: dict, port: int = 0):
    async def miner_identity(request):  # the executor's GET /miner-identity shape
        return web.json_response({**identity, "nonce": request.query["nonce"]})

    app = web.Application()
    app.router.add_get("/miner-identity", miner_identity)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, ip, port)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1]


@pytest_asyncio.fixture
async def two_nodes():
    """The same port on A and B (the node's registered API port), each answering for its own account."""
    runner_a, port = await _executor_at(IP_A, {"miner_hotkey": OWN_HOTKEY})
    runner_b, _ = await _executor_at(IP_B, {"miner_hotkey": OTHER_HOTKEY}, port)
    yield port
    await runner_a.cleanup()
    await runner_b.cleanup()


def _ctx(address: str, port: int, dialed: str, **expected):
    executor = ExecutorSSHInfo(
        uuid="node-1", address=address, port=port, ssh_username="root", ssh_port=22,
        python_path="/usr/bin/python3", root_dir="/root", **expected,
    )
    return make_context(executor=executor, ssh=_Ssh(dialed))


async def test_hostname_node_answering_for_its_own_account_passes(two_nodes):
    result = await NodeAccountCheck().run(_ctx("node.example.com", two_nodes, IP_A, expected_miner_hotkey=OWN_HOTKEY))
    assert result.passed


async def test_hostname_re_pointed_to_another_accounts_node_is_refused(two_nodes):
    result = await NodeAccountCheck().run(_ctx("node.example.com", two_nodes, IP_B, expected_miner_hotkey=OWN_HOTKEY))
    assert not result.passed
    assert result.event.reason_code == "NODE_ACCOUNT_MISMATCH"
    assert result.updates == {"clear_verified_job_info": True}


async def test_ip_addressed_node_is_unchanged(two_nodes):
    # registered at A, the dial reaches a node of another account: an IP node is never compared
    result = await NodeAccountCheck().run(_ctx(IP_A, two_nodes, IP_B, expected_miner_hotkey=OWN_HOTKEY))
    assert result.passed
    assert result.event.reason_code == "NODE_ACCOUNT_SKIPPED"


async def test_missing_expected_identity_is_not_failed(two_nodes):
    result = await NodeAccountCheck().run(_ctx("node.example.com", two_nodes, IP_B))
    assert result.passed
    assert result.event.reason_code == "NODE_ACCOUNT_SKIPPED"


async def test_unanswered_identity_fails_a_hostname_node(two_nodes):
    result = await NodeAccountCheck().run(_ctx("node.example.com", 1, IP_A, expected_miner_hotkey=OWN_HOTKEY))
    assert not result.passed
    assert result.event.reason_code == "NODE_ACCOUNT_UNREADABLE"


@pytest_asyncio.fixture
async def two_pool_nodes():
    """Both answer the shared pool hotkey; only the claim tells the accounts apart."""
    runner_a, port = await _executor_at(IP_A, {"miner_hotkey": POOL_HOTKEY, "account_claim": OWN_CLAIM})
    runner_b, _ = await _executor_at(IP_B, {"miner_hotkey": POOL_HOTKEY, "account_claim": OTHER_CLAIM}, port)
    yield port
    await runner_a.cleanup()
    await runner_b.cleanup()


async def test_pool_node_is_bound_by_the_account_claim(two_pool_nodes):
    expected = {"expected_miner_hotkey": POOL_HOTKEY, "expected_account_claim": OWN_CLAIM}
    own = await NodeAccountCheck().run(_ctx("node.example.com", two_pool_nodes, IP_A, **expected))
    other = await NodeAccountCheck().run(_ctx("node.example.com", two_pool_nodes, IP_B, **expected))
    assert own.passed
    assert not other.passed and other.event.reason_code == "NODE_ACCOUNT_MISMATCH"
