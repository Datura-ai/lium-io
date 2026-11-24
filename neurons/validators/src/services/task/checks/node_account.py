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

Follow-up of 19 Nov: the portal now sends the hotkeys and claims of every account of the listing account's owner, so a
node its owner set up under another of their own accounts passes without a re-setup. A machine set up for another
owner is still refused.
"""

from __future__ import annotations

from datetime import UTC, datetime

from datura.requests.miner_requests import ExecutorSSHInfo

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
BINDING_FIELDS = (
    "expected_node_hotkeys",
    "expected_account_claim",
    "expected_account_claims",
    "expected_claims_by_hotkey",
)

_logged_without_expected: set[str] = set()


def presented_host_key(ctx: Context) -> str:
    try:
        key = ctx.ssh.get_server_host_key()
    except Exception:
        return ""
    return host_key_body(key.export_public_key("openssh").decode()) if key is not None else ""


def fresh(until: str | None) -> bool:
    """Whether the portal read the miner sent this node from still vouches for it (`snapshot_valid_until`): False once
    the time passed or when it does not parse (a time without a zone included)."""
    try:
        return datetime.now(UTC) < datetime.fromisoformat(str(until))
    except (TypeError, ValueError):
        return False


def current_binding(executor: ExecutorSSHInfo) -> ExecutorSSHInfo:
    """The node as it may be bound now (DAH-6539): the owner-wide binding the miner sent (`owner_binding`) counts while
    the portal read it came from vouches for it (`snapshot_valid_until`), read here at check time because the miner can
    serve a cached snapshot and finish the upload after it; from then on, or when the time does not parse, the listing
    account's own binding (the expected_* fields as sent)."""
    if not isinstance(executor.owner_binding, dict) or not fresh(executor.snapshot_valid_until):
        return executor
    return executor.model_copy(update={field: executor.owner_binding.get(field) for field in BINDING_FIELDS})


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
        """MISMATCH when the machine reached is set up for another owner than the one listing it, else None.

        Accepted: a machine set up for one of the hotkeys and claims the portal sends for this node, which cover every
        account of the listing account's owner (same own-key coldkey or confirmed e-mail). A node the owner set up
        under another of their own accounts is theirs and passes with no re-setup; another owner's setup is refused."""
        result = await ctx.runner.run(READ_NODE_ACCOUNT_COMMAND, timeout=READ_TIMEOUT_SECONDS, retryable=True)
        # the binding as of the verdict, after the read: a read that outlasts the owner-wide grant's end is judged
        # against the listing account's own binding
        executor = current_binding(ctx.executor)
        if not executor.expected_node_hotkeys:
            return None  # a listing binding naming no hotkey: nothing to compare, as for a node sent without one
        found = dict(line.split("=", 1) for line in (result.stdout or "").splitlines() if "=" in line)
        hotkey, claim = found.get("hotkey", "").strip(), found.get("claim", "").strip()
        what = {**what, "node_hotkey": hotkey, "expected_node_hotkeys": executor.expected_node_hotkeys}
        if not result.success:
            # failed, timed out or closed without an exit status: a relay could stall it on purpose, so ownership is
            # not proven and the node is hidden until a cycle reads it
            what["read_error"] = result.error_type or f"exit {result.exit_code}"
            return _refused(ctx, self.check_id, Msg.BINDING_UNVERIFIED, what)
        if not hotkey:
            # the read worked and found no host files: an older executor, no evidence either way
            logger.info(_m("NODE_ACCOUNT_BINDING_UNVERIFIED", extra=get_extra_info({**what, "unread": "hotkey"})))
            return None
        claims = set(executor.expected_account_claims or ()) | ({executor.expected_account_claim} - {None, ""})
        by_hotkey = executor.expected_claims_by_hotkey or {}
        # an empty claim on a hotkey the listing already accepts keeps the legacy path below (an older executor)
        if hotkey in by_hotkey and (claim or hotkey not in executor.expected_node_hotkeys):
            # a shared (pool) hotkey of the owner's accounts that only this map accepts: every pool node can write
            # it, so only a matching claim proves the machine is this owner's; a missing claim proves nothing
            if claim in by_hotkey[hotkey]:
                return None
            what["binding_mismatch"] = "claim"
            what["node_claim"] = claim
        elif hotkey not in executor.expected_node_hotkeys and hotkey not in portal_owner_hotkeys(ctx.miner_hotkey):
            # DAH-6740: the listing owner's other accounts, as this validator read them from the portal itself; the
            # list a miner relays is never widened
            what["binding_mismatch"] = "hotkey"
        elif _pool_listed(executor, ctx, hotkey) and claim and claim not in claims:
            # a node set up for the listing (pool) hotkey proves its owner only by the claim; one set up for one of
            # the owner's own hotkeys needs none
            what["binding_mismatch"] = "claim"
            what["node_claim"] = claim
        else:
            if _pool_listed(executor, ctx, hotkey) and not claim:
                logger.info(_m("NODE_ACCOUNT_BINDING_UNVERIFIED", extra=get_extra_info({**what, "unread": "claim"})))
            return None
        return _refused(ctx, self.check_id, Msg.BINDING_MISMATCH, what)


def _pool_listed(executor: ExecutorSSHInfo, ctx: Context, hotkey: str) -> bool:
    """A machine set up for the hotkey the node is listed under, when that is a shared pool hotkey: only a claim proves
    its owner. The listing's own claim says so (the portal sends it only for a node listed under a pool hotkey), as
    does the listing hotkey being one of the owner's pool hotkeys; the owner-wide claims do not, an own-key listing gets
    them too and its own key settles ownership."""
    pools = executor.expected_claims_by_hotkey or {}
    return hotkey == ctx.miner_hotkey and (bool(executor.expected_account_claim) or hotkey in pools)


def _refused(ctx: Context, check_id: str, template, what: dict) -> CheckResult:
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
        updates={"node_account_passed": False, "clear_verified_job_info": True},
        fatal=True,
    )
