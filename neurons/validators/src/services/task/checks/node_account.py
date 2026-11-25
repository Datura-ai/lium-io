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
from clients.validator_portal_api import portal_owner_hotkeys
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

# DAH-6784: the two parts of this check, each its own id in a reset's evidence and in a pass's idle_checks_passed. The
# backend lifts a hold once a pass lists the id the reset named, so a hold one part left lifts only on proof of that part.
CHECK_ID = "executor.validate.node_account"
HOST_KEY_PART = f"{CHECK_ID}.host_key"
BINDING_PART = f"{CHECK_ID}.account_binding"

_logged_without_expected: set[str] = set()


def presented_host_key(ctx: Context) -> str:
    try:
        key = ctx.ssh.get_server_host_key()
    except Exception:
        return ""
    return host_key_body(key.export_public_key("openssh").decode()) if key is not None else ""


class NodeAccountCheck:
    check_id = CHECK_ID
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        executor = ctx.executor
        what = {"executor_uuid": executor.uuid, "address": executor.address}
        # the portal records every host key the node presented, one `<type> <base64>` per line
        expected = registered_host_keys(executor)
        # compared first: it needs no read, so a stalled account read cannot hide this direct evidence
        if expected and presented_host_key(ctx) not in expected:
            return _refused(ctx, self.check_id, Msg.MISMATCH, what, HOST_KEY_PART)
        binding_proven: bool | None = None  # None: the node names no account to bind it to
        if executor.expected_node_hotkeys:
            binding_proven = False
            if ctx.runner is not None:
                binding_outcome = await self._other_account(ctx, what)
                if isinstance(binding_outcome, CheckResult):
                    return binding_outcome
                binding_proven = binding_outcome
        # DAH-6784: the parts this pass proved; a part whose input is missing (no recorded host key, no hotkeys listed,
        # host files unread) is not proven, so a hold that part left stays. Both proven: the whole check too.
        proven = (HOST_KEY_PART,) * bool(expected) + (BINDING_PART,) * (binding_proven is True)
        full = {"idle_check_passed": proven + (CHECK_ID,) * (len(proven) == 2)} if proven else {}
        if not expected:
            if not is_ip_literal(executor.address) and executor.uuid not in _logged_without_expected:
                _logged_without_expected.add(executor.uuid)
                logger.info(
                    _m(
                        "Hostname node listed without its registered host key; account check skipped",
                        extra=get_extra_info({**what, "miner_hotkey": ctx.miner_hotkey}),
                    )
                )
            return CheckResult(
                passed=True, event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what), updates=full
            )

        return CheckResult(
            passed=True, event=render_message(Msg.MATCH, ctx=ctx, check_id=self.check_id, what=what), updates=full
        )

    async def _other_account(self, ctx: Context, what: dict) -> CheckResult | bool:
        """MISMATCH when the machine reached is set up for another account than the one listing it; else whether the
        binding was read and proven (False: an older executor without host files, or an unread claim)."""
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
            return _refused(ctx, self.check_id, Msg.BINDING_UNVERIFIED, what, BINDING_PART)
        if not hotkey:
            # the read worked and found no host files: an older executor, no evidence either way
            logger.info(_m("NODE_ACCOUNT_BINDING_UNVERIFIED", extra=get_extra_info({**what, "unread": "hotkey"})))
            return False
        if hotkey not in executor.expected_node_hotkeys and hotkey not in portal_owner_hotkeys(ctx.miner_hotkey):
            # DAH-6740: the listing owner's other accounts, as this validator read them from the portal itself; the
            # list a miner relays is never widened
            what["binding_mismatch"] = "hotkey"
        elif expected_claim and hotkey == ctx.miner_hotkey and claim and claim != expected_claim:
            # a node set up for the listing (pool) hotkey proves its account only by the claim; one set up for the
            # account's own hotkey needs none
            what["binding_mismatch"] = "claim"
        else:
            # a pool-key node is bound to this account only by a claim that matches the one expected: without either
            # side of that comparison nothing is proven (no refusal either)
            if hotkey == ctx.miner_hotkey and not (expected_claim and claim):
                unread = "claim" if expected_claim else "expected_claim"
                logger.info(_m("NODE_ACCOUNT_BINDING_UNVERIFIED", extra=get_extra_info({**what, "unread": unread})))
                return False
            return True
        return _refused(ctx, self.check_id, Msg.BINDING_MISMATCH, what, BINDING_PART)


def _refused(ctx: Context, check_id: str, template, what: dict, part: str) -> CheckResult:
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
            # the part that failed, so only a pass that proves that part lifts the hold this reset may leave
            "clear_verified_job_evidence": {"check_id": part},
            # has_live_rental reads a snapshot: a rental placed since must not be ended, so the backend applies this
            # reset only while no customer pod runs on the node, under its row lock
            "clear_verified_job_unrented_only": True,
        },
        fatal=True,
    )
