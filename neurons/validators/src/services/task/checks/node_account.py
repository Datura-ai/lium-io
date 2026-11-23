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
fixed command. It refuses only on direct evidence of another account: a machine set up for a hotkey that is not one
of the account's, or a pool node reporting another account's claim. The portal requires both at registration
(DAH-4375), so an honest node matches. Missing files and an empty claim prove nothing either way (executors write the claim
file from this release on): the node passes and the read is logged (NODE_ACCOUNT_BINDING_UNVERIFIED). A read that
fails, times out or closes without an exit status (a relay can stall it) is refused as NODE_ACCOUNT_UNVERIFIED.
"""

from __future__ import annotations

from core.utils import _m, get_extra_info, get_logger
from services.node_host import host_key_body, is_ip_literal, registered_host_keys

from ..messages import NodeAccountMessages as Msg, render_message
from ..availability import ReachSource, ReachTarget
from ..pipeline import CheckResult, Context
from .banned_provider import has_live_rental

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
        # the portal records every host key the node presented, one `<type> <base64>` per line
        expected = registered_host_keys(executor)
        # compared first: it needs no read, so a stalled account read cannot hide this direct evidence
        if expected and presented_host_key(ctx) not in expected:
            return _refused(ctx, self.check_id, Msg.MISMATCH, what)
        if executor.expected_node_hotkeys and ctx.runner is not None:
            refused = await self._other_account(ctx, what)
            if refused is not None:
                return refused
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

        return CheckResult(passed=True, event=render_message(Msg.MATCH, ctx=ctx, check_id=self.check_id, what=what))

    async def _other_account(self, ctx: Context, what: dict) -> CheckResult | None:
        """MISMATCH when the machine reached is set up for another account than the one listing it, else None."""
        executor = ctx.executor
        result = await ctx.runner.run(READ_NODE_ACCOUNT_COMMAND, timeout=READ_TIMEOUT_SECONDS, retryable=True)
        found = dict(line.split("=", 1) for line in (result.stdout or "").splitlines() if "=" in line)
        hotkey, claim = found.get("hotkey", "").strip(), found.get("claim", "").strip()
        what = {**what, "node_hotkey": hotkey, "expected_node_hotkeys": executor.expected_node_hotkeys}
        expected_claim = executor.expected_account_claim
        if not result.success:
            # failed, timed out or closed without an exit status: a relay could stall it on purpose, so ownership is
            # not proven and the node is hidden until a cycle reads it
            what["read_error"] = result.error_type or f"exit {result.exit_code}"
            # DAH-6951: only a session that closed under the read (no exit status, a transport error, a signal) is
            # one a rollout's container restart explains; a read that timed out (a stall) never is
            closed = result.error_type != "timeout" and (result.error_type is not None or result.exit_code < 0)
            return _refused(
                ctx, self.check_id, Msg.BINDING_UNVERIFIED, what, evidence={"session_closed_under_read": True} if closed else None
            )
        if not hotkey:
            # the read worked and found no host files: an older executor, no evidence either way
            logger.info(_m("NODE_ACCOUNT_BINDING_UNVERIFIED", extra=get_extra_info({**what, "unread": "hotkey"})))
            return None
        if hotkey not in executor.expected_node_hotkeys:
            what["binding_mismatch"] = "hotkey"
        elif expected_claim and hotkey == ctx.miner_hotkey and claim and claim != expected_claim:
            # a node set up for the listing (pool) hotkey proves its account only by the claim; one set up for the
            # account's own hotkey needs none
            what["binding_mismatch"] = "claim"
        else:
            if expected_claim and hotkey == ctx.miner_hotkey and not claim:
                logger.info(_m("NODE_ACCOUNT_BINDING_UNVERIFIED", extra=get_extra_info({**what, "unread": "claim"})))
            return None
        return _refused(ctx, self.check_id, Msg.BINDING_MISMATCH, what)


def _refused(ctx: Context, check_id: str, template, what: dict, evidence: dict | None = None) -> CheckResult:
    """Like GPU_MODEL_MISMATCH: an availability entry, so the backend hides the node and refuses new rentals on its
    free GPUs too (the node stays active), and score 0. With a live rental the renter's pod runs on and is billed as
    usual: the verified job is kept and the run is not fatal, because clearing it resets the node inactive in the
    backend, which would end the rental. Without one the verified job is cleared and the run ends here."""
    what = {**what, "reach_source": str(ReachSource.VALIDATOR), "reach_target": str(ReachTarget.NODE)}
    event = render_message(template, ctx=ctx, check_id=check_id, what=what)
    if has_live_rental(ctx):
        return CheckResult(passed=False, event=event, updates={"node_account_passed": False}, fatal=False)
    return CheckResult(
        passed=False,
        event=event,
        updates={
            "node_account_passed": False,
            "clear_verified_job_info": True,
            **({"clear_verified_job_evidence": evidence} if evidence else {}),
        },
        fatal=True,
    )
