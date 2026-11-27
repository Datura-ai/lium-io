from __future__ import annotations

import re
from functools import cache
from pathlib import Path

from core.config import settings

from ..messages import SysboxLayerResetMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

INSTALL_TIMEOUT_SECONDS = 120
SETUP_SCRIPT_NAME = "nvidia_docker_sysbox_setup.sh"
# the image copy (Dockerfile, build context executor_setup), then the repo's own (tests, local runs)
SETUP_SCRIPT_PATHS = (
    Path("/app/executor_setup") / SETUP_SCRIPT_NAME,
    Path(__file__).resolve().parents[5] / "executor" / SETUP_SCRIPT_NAME,
)
# runs the bundled setup script, fed on stdin, in the host mount namespace (as the needrestart guard writes its file)
HOST_COMMAND = (
    "if command -v nsenter >/dev/null 2>&1; then nsenter -t 1 -m -- bash -s -- {option} 2>/dev/null "
    "|| echo layer_reset_failed nsenter_failed; else echo layer_reset_skipped no_nsenter; fi"
)
INSTALL_COMMAND = HOST_COMMAND.format(option="--install-layer-reset")
REMOVE_COMMAND = HOST_COMMAND.format(option="--remove-layer-reset")


@cache
def setup_script() -> tuple[str, int] | None:
    """The setup script this validator ships and its LAYER_RESET_VERSION; None when the image has no usable copy."""
    for path in SETUP_SCRIPT_PATHS:
        try:
            text = path.read_text()
        except OSError:
            continue
        match = re.search(r"^LAYER_RESET_VERSION=(\d+)$", text, re.M)
        if match:
            return text, int(match.group(1))
    return None


def host_may_be_rented(ctx: Context) -> bool:
    """Any pod on this executor's rented list, stopped ones too (a stopped pod is a live rental), or no list this cycle.
    Only the removal waits for a cycle that shows the host idle; the install runs on rented hosts too, as it never
    starts, stops or restarts anything (tests/test_sysbox_setup_layer_reset.py)."""
    rented_data = ctx.state.rented_data
    listed = rented_data.executors.get(ctx.executor.uuid) if rented_data is not None else None
    return rented_data is None or bool(listed and listed.pods)


def install_wanted(mode: str, miner_hotkey: str) -> bool:
    allowlist = {h.strip() for h in settings.SYSBOX_LAYER_RESET_INSTALL_HOTKEYS.split(",") if h.strip()}
    return mode == "all" or (mode == "allowlist" and miner_hotkey in allowlist)


async def run_setup_option(ssh, script: str, command: str) -> str:
    """"installed <version>", "removed", "skipped <reason>" or "failed <reason>". Never raises."""
    try:
        res = await ssh.run(command, input=script, timeout=INSTALL_TIMEOUT_SECONDS, check=False)
    except Exception as exc:  # noqa: BLE001
        return f"failed ssh {exc!r}"[:200]
    lines = (getattr(res, "stdout", "") or "").strip().splitlines()
    if not lines:
        return "failed no_answer"
    verb, _, reason = lines[-1].partition(" ")
    # the host writes this line (a failed install adds its step and stderr): printable ASCII only, bounded
    return re.sub(r"[^\x20-\x7e]", "?", f"{verb.removeprefix('layer_reset_')} {reason}".strip())[:200]


class SysboxLayerResetCheck:
    """Install the boot-time layer reset on Sysbox hosts that lack it or run an old one (DAH-6804).

    After a power loss Sysbox never shifts a pod's writable layer back, so the pod restarts with its files owned by
    host uid 0 and sudo and apt break inside it. lium-sysbox-layer-reset.service shifts it back at boot, before Docker
    (lium-io #2276), but only the setup script installs it and providers rarely re-run that. The scrape reports the
    installed version (``specs.sysbox_layer_reset_version``: 0 none, 1 before the version line, N); below the
    shipped setup script's LAYER_RESET_VERSION this runs that script's ``--install-layer-reset`` on the host, which
    installs the unit and Docker's drop-in only: it never runs the reset, never restarts Docker or Sysbox and never
    stops, restarts or re-creates a container. It installs on rented hosts too: they are the ones a power loss hurts,
    and the reset only acts at the next boot, before Docker starts the pods.
    SYSBOX_LAYER_RESET_INSTALL rolls it out: off, allowlist, then all; remove (DAH-7052) runs the script's
    ``--remove-layer-reset`` instead on every host that has the unit, at an idle cycle (host_may_be_rented). Skipped under dry run. Non-fatal, right after
    the scrape. The score is not changed.
    """

    check_id = "executor.validate.sysbox_layer_reset"
    replays_after_session_drop = True
    fatal = False

    def __init__(self, write: bool = True):
        self.write = write

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs or {}
        installed = specs.get("sysbox_layer_reset_version")
        shipped = setup_script()
        mode = settings.SYSBOX_LAYER_RESET_INSTALL
        what: dict = {"installed": installed, "current": shipped and shipped[1], "mode": mode}
        if not isinstance(installed, int) or isinstance(installed, bool) or shipped is None:
            # not a Sysbox host (no key), the read failed, or this image ships no setup script
            what["scrape_error"] = specs.get("sysbox_layer_reset_scrape_error")
            return CheckResult(passed=True, event=render_message(Msg.UNKNOWN, ctx=ctx, check_id=self.check_id, what=what))
        if mode == "remove":
            if installed == 0:
                return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))
            if not self.write or host_may_be_rented(ctx):
                event = render_message(Msg.REMOVE_PENDING, ctx=ctx, check_id=self.check_id, what=what)
                return CheckResult(passed=True, event=event)
            what["remove"] = await run_setup_option(ctx.ssh, shipped[0], REMOVE_COMMAND)
            template = Msg.REMOVED if what["remove"] == "removed" else Msg.REMOVE_FAILED
            return CheckResult(passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what))
        if installed >= shipped[1]:
            return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))
        if not (self.write and install_wanted(mode, ctx.miner_hotkey)):
            return CheckResult(passed=True, event=render_message(Msg.OUTDATED, ctx=ctx, check_id=self.check_id, what=what))
        what["install"] = await run_setup_option(ctx.ssh, shipped[0], INSTALL_COMMAND)
        template = Msg.INSTALLED if what["install"] == f"installed {shipped[1]}" else Msg.INSTALL_FAILED
        return CheckResult(passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what))
