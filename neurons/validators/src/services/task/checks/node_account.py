"""DAH-5729: a hostname-addressed central node is validated only for the account that registered it.

A central miner lists each node with the SSH host key the portal recorded when the account proved it controls the
node (`expected_ssh_host_key`). Completing the SSH key exchange proves the machine the validator dialed (after DNS
resolution, see node_host) holds that host key's private half, so this check requires the key the server presented
in the handshake to be one of the recorded ones (one per line): a hostname re-pointed to another machine presents another key.

Scope: only a node registered by hostname, since an IP address cannot be re-pointed after registration.
No expected key (a standalone or older miner, or a portal that does not record it yet): skipped and logged once per
node, so the validator can ship before the miner.
"""

from __future__ import annotations

from core.utils import _m, get_extra_info, get_logger
from services.node_host import is_ip_literal

from ..messages import NodeAccountMessages as Msg, render_message
from ..pipeline import CheckResult, Context

logger = get_logger(__name__)

_logged_without_expected: set[str] = set()


def _key_body(openssh: str | None) -> str:
    """`<type> <base64>` of an OpenSSH public key line, without its comment."""
    return " ".join((openssh or "").split()[:2])


def presented_host_key(ctx: Context) -> str:
    try:
        key = ctx.ssh.get_server_host_key()
    except Exception:
        return ""
    return _key_body(key.export_public_key("openssh").decode()) if key is not None else ""


class NodeAccountCheck:
    check_id = "executor.validate.node_account"
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        executor = ctx.executor
        what = {"executor_uuid": executor.uuid, "address": executor.address}
        # the portal records every host key the node presented, one `<type> <base64>` per line
        expected = {_key_body(line) for line in (executor.expected_ssh_host_key or "").splitlines()} - {""}
        if is_ip_literal(executor.address) or not expected:
            if not is_ip_literal(executor.address) and executor.uuid not in _logged_without_expected:
                _logged_without_expected.add(executor.uuid)
                logger.info(
                    _m(
                        "Hostname node listed without its registered host key; account check skipped",
                        extra=get_extra_info({**what, "miner_hotkey": ctx.miner_hotkey}),
                    )
                )
            return CheckResult(passed=True, event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what))

        if presented_host_key(ctx) not in expected:
            event = render_message(Msg.MISMATCH, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=False, event=event, updates={"clear_verified_job_info": True})
        return CheckResult(passed=True, event=render_message(Msg.MATCH, ctx=ctx, check_id=self.check_id, what=what))
