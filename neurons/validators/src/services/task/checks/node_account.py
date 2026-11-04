"""DAH-5729: a hostname-addressed central node is validated only for the account that registered it.

A central miner lists each node with the identity the portal registered it under (`expected_miner_hotkey`,
and `expected_account_claim` for a node on the shared pool hotkey). Each cycle this check asks the executor
at the address the validator's SSH session actually dialed (after DNS resolution, see node_host) which
identity it is set up for, and fails the node when the answer is another account's.

Scope: only a node registered by hostname, since an IP address cannot be re-pointed after registration.
No expected identity (a standalone or older miner): skipped and logged once per node, so the validator can
ship before the miner.
"""

from __future__ import annotations

import hmac
import secrets

import aiohttp
from yarl import URL

from core.utils import _m, get_extra_info, get_logger
from services.node_host import is_ip_literal

from ..messages import NodeAccountMessages as Msg, render_message
from ..pipeline import CheckResult, Context

logger = get_logger(__name__)

MINER_IDENTITY_PATH = "/miner-identity"
REQUEST_TIMEOUT_SECONDS = 10.0
_logged_without_expected: set[str] = set()


def _dialed_ip(ctx: Context) -> str | None:
    try:
        peer = ctx.ssh.get_extra_info("peername")
    except Exception:
        return None
    return peer[0] if isinstance(peer, tuple | list) and peer else None


async def read_node_identity(ip: str, port: int) -> dict | None:
    """The executor's /miner-identity answer at ip:port, or None when it gave no valid answer."""
    nonce = secrets.token_hex(16)
    url = URL.build(scheme="http", host=ip, port=port, path=MINER_IDENTITY_PATH, query={"nonce": nonce})
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)) as session:
            async with session.get(url, allow_redirects=False) as resp:
                if resp.status != 200:
                    return None
                answer = await resp.json(content_type=None)
    except Exception:
        return None
    if not isinstance(answer, dict) or answer.get("nonce") != nonce:
        return None
    return answer


def identity_matches(answer: dict, expected_hotkey: str, expected_claim: str | None) -> bool:
    if expected_claim:  # a pool-hotkey node: the claim is what ties it to one account
        claim = answer.get("account_claim")
        return isinstance(claim, str) and hmac.compare_digest(claim, expected_claim)
    return answer.get("miner_hotkey") == expected_hotkey


class NodeAccountCheck:
    check_id = "executor.validate.node_account"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        executor = ctx.executor
        what = {"executor_uuid": executor.uuid, "address": executor.address}
        if is_ip_literal(executor.address):
            return CheckResult(passed=True, event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what))
        if not executor.expected_miner_hotkey:
            if executor.uuid not in _logged_without_expected:
                _logged_without_expected.add(executor.uuid)
                logger.info(
                    _m(
                        "Hostname node listed without its registered identity; account check skipped",
                        extra=get_extra_info({**what, "miner_hotkey": ctx.miner_hotkey}),
                    )
                )
            return CheckResult(passed=True, event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what))

        ip = _dialed_ip(ctx)
        answer = await read_node_identity(ip, executor.port) if ip else None
        what["dialed_address"] = ip
        if answer is None:
            event = render_message(Msg.UNREADABLE, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=False, event=event)
        if not identity_matches(answer, executor.expected_miner_hotkey, executor.expected_account_claim):
            what["reported_miner_hotkey"] = answer.get("miner_hotkey")
            event = render_message(Msg.MISMATCH, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=False, event=event, updates={"clear_verified_job_info": True})
        return CheckResult(passed=True, event=render_message(Msg.MATCH, ctx=ctx, check_id=self.check_id, what=what))
