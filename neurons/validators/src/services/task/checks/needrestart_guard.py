from __future__ import annotations

import shlex

from ..messages import NeedrestartGuardMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

GUARDED = ("present", "no_needrestart")
WRITE_TIMEOUT_SECONDS = 30

# the exact file nvidia_docker_sysbox_setup.sh's keep_needrestart_off_the_container_runtime writes
NEEDRESTART_GUARD_FILE = "/etc/needrestart/conf.d/lium-container-runtime.conf"
NEEDRESTART_GUARD_TEXT = """# Lium: never restart the container runtime under running pods; restart it in a maintenance window.
$nrconf{override_rc}->{qr(^sysbox)} = 0;
$nrconf{override_rc}->{qr(^docker)} = 0;
$nrconf{override_rc}->{qr(^containerd)} = 0;
$nrconf{override_rc}->{qr(^nvidia-fabricmanager)} = 0;
$nrconf{override_rc}->{qr(^nvidia-persistenced)} = 0;"""


def needrestart_guard_inner(root: str = "") -> str:
    """Runs in the host mount namespace; root prefixes every path (tests only)."""
    f = shlex.quote(root + NEEDRESTART_GUARD_FILE)
    return (
        f"{{ [ -e {shlex.quote(root + '/usr/sbin/needrestart')} ] || "
        f"[ -f {shlex.quote(root + '/etc/needrestart/needrestart.conf')} ]; }} "
        "|| { echo needrestart_guard_skipped no_needrestart; exit 0; }; "
        f"c={shlex.quote(NEEDRESTART_GUARD_TEXT)}; "
        f'[ "$(cat {f} 2>/dev/null)" = "$c" ] && exit 0; '
        # tmp + mv: needrestart never evals a half-written file (it reads conf.d/*.conf only)
        f"{{ mkdir -p \"$(dirname {f})\" && printf '%s\\n' \"$c\" > {f}.tmp && mv {f}.tmp {f}; }} 2>/dev/null "
        "&& echo needrestart_guard_set host || echo needrestart_guard_failed host write_denied"
    )


def needrestart_guard_script() -> str:
    return (
        "if command -v nsenter >/dev/null 2>&1; then nsenter -t 1 -m -- sh -c "
        + shlex.quote(needrestart_guard_inner())
        + " 2>/dev/null || echo needrestart_guard_skipped nsenter_failed; "
        "else echo needrestart_guard_skipped no_nsenter; fi"
    )


async def write_needrestart_guard(ssh) -> str:
    """"set", "skipped <reason>" or "failed <reason>". Never raises."""
    try:
        res = await ssh.run(needrestart_guard_script(), timeout=WRITE_TIMEOUT_SECONDS, check=False)
    except Exception as exc:  # noqa: BLE001
        return f"failed ssh {exc!r}"[:200]
    out = (getattr(res, "stdout", "") or "").strip()
    if not out:
        return "unchanged"
    verb, _, reason = out.splitlines()[-1].partition(" ")
    return {"needrestart_guard_set": "set"}.get(verb, f"{verb.removeprefix('needrestart_guard_')} {reason}".strip())


class NeedrestartGuardCheck:
    """Warn when the host's apt (needrestart) may restart the container runtime under running pods.

    Reads ``specs.needrestart_guard`` from the scrape, which looks for the
    /etc/needrestart/conf.d/lium-container-runtime.conf the setup script writes since 16 Nov 2026 (DAH-6384).
    Hosts set up earlier have needrestart and no guard, and an unattended apt upgrade there restarted sysbox-fs under
    running pods (dead /proc). Where the guard is not read as present, this writes the setup script's exact file
    (a file write, nothing restarted; skipped under dry run) and warns only when the host is still unguarded. It runs
    right after the scrape, before any fatal check, so every node that is scraped gets it, idle or rented. Non-fatal, before the rented short-circuit so rented
    hosts, where a runtime restart breaks live pods, are reported too. The score is not changed. The warning is
    read from the validator log and its providers are told directly, as with PidHostRequiredCheck's rented warning;
    it is not published to the backend.
    """

    check_id = "executor.validate.needrestart_guard"
    replays_after_session_drop = True
    fatal = False

    def __init__(self, write: bool = True):
        self.write = write

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs or {}
        guard = specs.get("needrestart_guard")
        what = {"needrestart_guard": guard, "unguarded": specs.get("needrestart_guard_unguarded")}
        if specs.get("needrestart_guard_reason"):
            what["reason"] = specs["needrestart_guard_reason"]
        if not isinstance(guard, str):
            what = {"scrape_error": specs.get("needrestart_guard_scrape_error")}
        if guard not in GUARDED and self.write:
            what["write"] = await write_needrestart_guard(ctx.ssh)
            if what["write"] == "set":
                return CheckResult(passed=True, event=render_message(Msg.WRITTEN, ctx=ctx, check_id=self.check_id, what=what))
        if not isinstance(guard, str):
            return CheckResult(passed=True, event=render_message(Msg.UNKNOWN, ctx=ctx, check_id=self.check_id, what=what))
        if guard not in GUARDED:
            return CheckResult(passed=True, event=render_message(Msg.MISSING, ctx=ctx, check_id=self.check_id, what=what))
        return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))
