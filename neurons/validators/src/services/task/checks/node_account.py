"""DAH-5729: a hostname-addressed central node is validated only for the account that registered it.

A central miner lists each node with the SSH host key the portal recorded when the account proved it controls the
node (`expected_ssh_host_key`). Completing the SSH key exchange proves the machine the validator dialed (after DNS
resolution, see node_host) holds that host key's private half, so this check requires the key the server presented
in the handshake to be one of the recorded ones (one per line): a hostname re-pointed to another machine presents another key.

Scope: only a node registered by hostname. No expected key (a standalone or older miner, or a portal that does not
record it yet): skipped and logged once per node, so the validator can ship before the miner.

DAH-6539: an IP:port can be re-pointed too, by a relay that forwards it to another account's machine (seen
11-16 Nov 2026: every executor also accepts the shared pool hotkey, so the victim installed our key).
So for every node the miner lists with the account's hotkeys, the check reads the hotkey and account claim the
machine itself is set up for (`/etc/liumd/miner_hotkey`, `/etc/liumd/account_claim`, written by the executor) with a
fixed command, and refuses a machine set up for another hotkey, or reporting another account's claim. The portal
requires both at registration (DAH-4375), so an honest node matches. An executor that predates the files, or a node
without a claim, is not refused for it. Until NODE_ACCOUNT_BINDING_ENFORCE is on, a mismatch is only logged.
"""

from __future__ import annotations

from core.config import settings
from core.utils import _m, get_extra_info, get_logger
from services.node_host import host_key_body, is_ip_literal, registered_host_keys

from ..messages import NodeAccountMessages as Msg, render_message
from ..pipeline import CheckResult, Context

logger = get_logger(__name__)

# fixed text, no node-supplied value; `$(...)` drops the trailing newline the executor writes
READ_NODE_ACCOUNT_COMMAND = (
    "printf 'hotkey=%s\\nclaim=%s\\n' \"$(cat /etc/liumd/miner_hotkey 2>/dev/null)\" "
    "\"$(cat /etc/liumd/account_claim 2>/dev/null)\""
)
READ_TIMEOUT_SECONDS = 15

_logged_without_expected: set[str] = set()


def presented_host_key(ctx: Context) -> str:
    try:
        key = ctx.ssh.get_server_host_key()
    except Exception:
        return ""
    return host_key_body(key.export_public_key("openssh").decode()) if key is not None else ""


class NodeAccountCheck:
    check_id = "executor.validate.node_account"
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        executor = ctx.executor
        what = {"executor_uuid": executor.uuid, "address": executor.address}
        if executor.expected_node_hotkeys and ctx.runner is not None:
            refused = await self._other_account(ctx, what)
            if refused is not None:
                return refused
        # the portal records every host key the node presented, one `<type> <base64>` per line
        expected = registered_host_keys(executor)
        if not expected:
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

    async def _other_account(self, ctx: Context, what: dict) -> CheckResult | None:
        """MISMATCH when the machine reached is set up for another account than the one listing it, else None."""
        executor = ctx.executor
        result = await ctx.runner.run(READ_NODE_ACCOUNT_COMMAND, timeout=READ_TIMEOUT_SECONDS, retryable=True)
        found = dict(line.split("=", 1) for line in (result.stdout or "").splitlines() if "=" in line)
        hotkey, claim = found.get("hotkey", "").strip(), found.get("claim", "").strip()
        what = {**what, "node_hotkey": hotkey, "expected_node_hotkeys": executor.expected_node_hotkeys}
        if not result.success or not hotkey:
            # an executor that predates the host files: nothing to compare yet
            logger.info(_m("Node account files unread; account binding not checked", extra=get_extra_info(what)))
            return None
        other_hotkey = hotkey not in executor.expected_node_hotkeys
        other_claim = bool(claim and executor.expected_account_claim and claim != executor.expected_account_claim)
        if not (other_hotkey or other_claim):
            return None
        what["claim_mismatch"] = other_claim
        if not settings.NODE_ACCOUNT_BINDING_ENFORCE:
            # shadow run first (Sergej, lio#2288): count would-be refusals in prod before any node is refused
            logger.warning(_m("NODE_ACCOUNT_BINDING_WOULD_REFUSE", extra=get_extra_info(what)))
            return None
        event = render_message(Msg.MISMATCH, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=False, event=event, updates={"clear_verified_job_info": True})
