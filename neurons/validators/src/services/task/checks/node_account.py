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
fails, times out or closes without an exit status (a relay can stall it) is refused as NODE_ACCOUNT_UNVERIFIED once the
portal sent the provider the per-node notice, and only reported before that (checks/enforcement.py).

Follow-up of 19 Nov (Mikhail's incident review): the portal now sends the hotkeys and claims of every account of the
listing account's owner, so a node its owner set up under another of their own accounts passes without a re-setup.
A machine set up for another owner is still refused at once. A pass on an IP node logs NODE_ACCOUNT_OK, not SKIPPED.
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
from .enforcement import enforced, fresh, reported_only

logger = get_logger(__name__)

# fixed text, no node-supplied value; `$(...)` drops the trailing newline the executor writes. Only a file that is not
# there (cat's ENOENT, under the C locale) reads as empty (an older executor). Any other failure, in the lookup or the
# read, leaves the field empty and names it on the `unread` line. The hotkey and the claim are read by separate
# commands: a claim read that stalls cannot hide a hotkey that settles the result, and the claim is only read when it
# can change the verdict
_READ_FILE = (
    "r() { o=$(LC_ALL=C cat \"$1\" 2>&1) && { printf '%s' \"$o\"; return 0; }; "
    "case $o in *': No such file or directory') return 0;; esac; return 1; }; "
)
READ_NODE_HOTKEY_COMMAND = _READ_FILE + (
    "u=; h=$(r /etc/liumd/miner_hotkey) || { h=; u=hotkey; }; printf 'hotkey=%s\\nunread=%s\\n' \"$h\" \"$u\""
)
READ_NODE_CLAIM_COMMAND = _READ_FILE + (
    "u=; c=$(r /etc/liumd/account_claim) || { c=; u=claim; }; printf 'claim=%s\\nunread=%s\\n' \"$c\" \"$u\""
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
        bound = False
        if executor.expected_node_hotkeys and ctx.runner is not None:
            refused, bound = await self._other_account(ctx, what)
            if refused is not None:
                return refused
        if expected or bound:
            # a pass is logged as one: an IP node whose machine is set up for its account matched, it was not skipped
            return CheckResult(
                passed=True,
                event=render_message(Msg.MATCH, ctx=ctx, check_id=self.check_id, what=what),
            )
        if not is_ip_literal(executor.address) and executor.uuid not in _logged_without_expected:
            _logged_without_expected.add(executor.uuid)
            logger.info(
                _m(
                    "Hostname node listed without its registered host key; account check skipped",
                    extra=get_extra_info({**what, "miner_hotkey": ctx.miner_hotkey}),
                )
            )
        return CheckResult(
            passed=True,
            event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id, what=what),
        )

    async def _other_account(self, ctx: Context, what: dict) -> tuple[CheckResult | None, bool]:
        """(refusal or report, None when the machine may serve this listing; True when its files proved it does).

        Accepted: a machine set up for one of the hotkeys and claims the portal sends for this node, which cover every
        account of the listing account's owner (same own-key coldkey or confirmed e-mail). A node the owner set up
        under another of their own accounts is theirs and passes with no re-setup; only another owner's setup is
        refused, at once (the relay case). A read that fails proves nothing either way: it only reports until the
        provider was notified (checks/enforcement.py)."""
        started = datetime.now(UTC)  # the deadline is judged at the read's start, not when its failure is handled
        result = await ctx.runner.run(READ_NODE_HOTKEY_COMMAND, timeout=READ_TIMEOUT_SECONDS, retryable=True)
        # the binding as of the verdict, after the last read: a read that outlasts the owner-wide grant's end is judged
        # against the listing account's own binding
        executor = current_binding(ctx.executor)
        if not executor.expected_node_hotkeys:
            return None, False  # a listing binding naming no hotkey: nothing to compare, as for a node sent without one
        found = _fields(result)
        hotkey = found.get("hotkey", "").strip()
        what = {
            **what,
            "node_hotkey": hotkey,
            "expected_node_hotkeys": executor.expected_node_hotkeys,
        }
        if not result.success or "hotkey" in found["unread"]:
            # failed, timed out or closed without an exit status, or the hotkey file could not be read: ownership is
            # not proven
            failure = result.error_type or f"exit {result.exit_code}"
            what["read_error"] = "hotkey unreadable" if result.success else failure
            return self._unverified(ctx, what, started), False
        if not hotkey:
            # the read worked and found no host files: an older executor, no evidence either way
            logger.info(
                _m(
                    "NODE_ACCOUNT_BINDING_UNVERIFIED",
                    extra=get_extra_info({**what, "unread": "hotkey"}),
                )
            )
            return None, False
        by_hotkey = executor.expected_claims_by_hotkey or {}
        if (
            hotkey not in executor.expected_node_hotkeys
            and hotkey not in by_hotkey
            and hotkey not in portal_owner_hotkeys(ctx.miner_hotkey)
        ):
            # another owner's setup: the hotkey alone settles it, whatever the claim would read. DAH-6740: the listing
            # owner's other accounts, as this validator read them from the portal itself; the list a miner relays is
            # never widened
            what["binding_mismatch"] = "hotkey"
            return _refused(ctx, self.check_id, Msg.BINDING_MISMATCH, what), False
        if hotkey not in by_hotkey and not (_pool_listed(executor, ctx, hotkey)):
            return None, True  # one of the owner's own hotkeys: no claim can change that
        result = await ctx.runner.run(READ_NODE_CLAIM_COMMAND, timeout=READ_TIMEOUT_SECONDS, retryable=True)
        executor = current_binding(ctx.executor)
        if not executor.expected_node_hotkeys:
            return None, False
        found = _fields(result)
        claim = found.get("claim", "").strip()
        claim_error = None if result.success else result.error_type or f"exit {result.exit_code}"
        unread = found["unread"] | ({"claim"} if claim_error else set())
        what["expected_node_hotkeys"] = executor.expected_node_hotkeys
        claims = set(executor.expected_account_claims or ()) | ({executor.expected_account_claim} - {None, ""})
        by_hotkey = executor.expected_claims_by_hotkey or {}
        if "claim" in unread:
            # decided by the hotkey alone where it settles the result: another owner's setup is refused, one of the
            # owner's own hotkeys passes. A setup the claim decides (a shared pool hotkey) is unverified.
            if hotkey not in executor.expected_node_hotkeys and hotkey not in by_hotkey:
                what["binding_mismatch"] = "hotkey"
                return _refused(ctx, self.check_id, Msg.BINDING_MISMATCH, what), False
            if hotkey in by_hotkey or (_pool_listed(executor, ctx, hotkey)):
                what["read_error"] = claim_error or "claim unreadable"
                return self._unverified(ctx, what, started), False
            return None, True
        # an empty claim on a hotkey the listing already accepts keeps the legacy path below (an older executor)
        if hotkey in by_hotkey and (claim or hotkey not in executor.expected_node_hotkeys):
            # a shared (pool) hotkey of the owner's accounts that only this map accepts: every pool node can write
            # it, so only a matching claim proves the machine is this owner's; a missing claim proves nothing
            if claim in by_hotkey[hotkey]:
                return None, True
            what["binding_mismatch"] = "claim"
            what["node_claim"] = claim
        elif hotkey not in executor.expected_node_hotkeys:
            what["binding_mismatch"] = "hotkey"
        elif _pool_listed(executor, ctx, hotkey) and claim and claim not in claims:
            # a node set up for the listing (pool) hotkey proves its owner only by the claim; one set up for one of
            # the owner's own hotkeys needs none. The claim is the HMAC the node publishes at /miner-identity.
            what["binding_mismatch"] = "claim"
            what["node_claim"] = claim
        else:
            if _pool_listed(executor, ctx, hotkey) and not claim:
                logger.info(
                    _m(
                        "NODE_ACCOUNT_BINDING_UNVERIFIED",
                        extra=get_extra_info({**what, "unread": "claim"}),
                    )
                )
                return None, False
            return None, True
        return _refused(ctx, self.check_id, Msg.BINDING_MISMATCH, what), False

    def _unverified(self, ctx: Context, what: dict, started: datetime) -> CheckResult:
        """Ownership not proven: refused once the provider was notified (checks/enforcement.py), reported before."""
        if enforced(ctx, Msg.BINDING_UNVERIFIED.reason, started):
            return _refused(ctx, self.check_id, Msg.BINDING_UNVERIFIED, what)
        what = {**what, "enforced": False}
        # the run is only reported, but nothing proved this machine the account's: no repair, cleanup or rule
        # writes to it unless the host key it presented is one the portal registered for the node. Fail closed: a
        # machine that gets no writes gets no new rental either, which would run without the rental egress rules
        if registered_host_keys(ctx.executor):
            return reported_only(
                ctx, render_message(Msg.UNVERIFIED_REPORTED, ctx=ctx, check_id=self.check_id, what=what)
            )
        what = {**what, "reach_source": str(ReachSource.VALIDATOR), "reach_target": str(ReachTarget.NODE)}
        result = reported_only(
            ctx, render_message(Msg.UNVERIFIED_REPORTED_NOT_RENTABLE, ctx=ctx, check_id=self.check_id, what=what)
        )
        result.updates["host_writes_deferred"] = True
        return result


def _pool_listed(executor: ExecutorSSHInfo, ctx: Context, hotkey: str) -> bool:
    """A machine set up for the hotkey the node is listed under, when that is a shared pool hotkey: only a claim proves
    its owner. The listing's own claim says so (the portal sends it only for a node listed under a pool hotkey), as
    does the listing hotkey being one of the owner's pool hotkeys; the owner-wide claims do not, an own-key listing gets
    them too and its own key settles ownership (lio#2349 stand)."""
    pools = executor.expected_claims_by_hotkey or {}
    return hotkey == ctx.miner_hotkey and (bool(executor.expected_account_claim) or hotkey in pools)


def _fields(result) -> dict:
    """The `key=value` lines of a read, with `unread` as the set of fields it could not read."""
    found = dict(line.split("=", 1) for line in (result.stdout or "").splitlines() if "=" in line)
    return {**found, "unread": set(found.get("unread", "").split())}


def _refused(ctx: Context, check_id: str, template, what: dict) -> CheckResult:
    """Like GPU_MODEL_MISMATCH: an availability entry, so the backend hides the node and refuses new rentals on its
    free GPUs too (the node stays active), and score 0. With a live rental the renter's pod runs on and is billed as
    usual: the verified job is kept and the run is not fatal, because clearing it resets the node inactive in the
    backend, which would end the rental. Without one the verified job is cleared and the run ends here."""
    what = {**what, "reach_source": str(ReachSource.VALIDATOR), "reach_target": str(ReachTarget.NODE)}
    event = render_message(template, ctx=ctx, check_id=check_id, what=what)
    if has_live_rental(ctx):
        return CheckResult(
            passed=False, event=event, updates={"node_account_passed": False, "host_writes_deferred": True}, fatal=False
        )
    return CheckResult(
        passed=False,
        event=event,
        updates={"node_account_passed": False, "clear_verified_job_info": True, "host_writes_deferred": True},
        fatal=True,
    )
