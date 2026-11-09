from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import asyncssh

from core.docker_utils import DockerCommand
from protocol.vc_protocol.validator_requests import PodSshResult
from services.redis_service import IN_CONTAINER_SSH_BOOTSTRAP_PATH

from ..messages import PodSshdRestartMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rented_gpu_drop import POD_STATUS_RUNNING
from .rented_pod_ssh import tcp_connect_fault

_BOOTSTRAP_SCRIPT = Path(__file__).resolve().parents[2] / "assets" / "sshd_bootstrap.sh"
PROBE_TIMEOUT_SECONDS = 5
RESTART_TIMEOUT_SECONDS = 120
# exit codes of the in-pod gate, outside the bootstrap's own 0/1
NO_WATCHDOG_RECORD, WATCHDOG_ALIVE, RENTER_LOGGED_IN = 10, 11, 12
# the gate, in the pod: sshd was brought up by the create's bootstrap (its watchdog pidfile), the watchdog that
# would restart it is gone, and the renter never logged in (the watchdog's marker, or a wtmp record)
_GATE = (
    f"p=$(cat /run/sshd-watchdog.pid 2>/dev/null) || exit {NO_WATCHDOG_RECORD}; "
    f'case "$(tr "\\0" " " < /proc/$p/cmdline 2>/dev/null)" in *--watchdog-loop*) exit {WATCHDOG_ALIVE};; esac; '
    f"[ ! -e /var/lib/lium/ssh-login-seen ] || exit {RENTER_LOGGED_IN}; "
    f"[ ! -s /var/log/wtmp ] || exit {RENTER_LOGGED_IN}; "
    f"cat > {IN_CONTAINER_SSH_BOOTSTRAP_PATH} && sh {IN_CONTAINER_SSH_BOOTSTRAP_PATH} --grace 0"
)


def restart_command(name: str) -> str:
    """One host command: re-run the create's sshd bootstrap in the pod, behind the gate above."""
    script = base64.b64encode(_BOOTSTRAP_SCRIPT.read_bytes()).decode()
    return f"printf %s {script} | base64 -d | " + DockerCommand.exec_command(name, _GATE)


class PodSshdRestartCheck:
    """Bring back the sshd of a rented pod the renter has never logged in to.

    sshd and its watchdog are children of the create's one-time `docker exec`: a container that restarts, or a
    main process that takes them down, leaves the pod running and billed with SSH refused. For each running
    rented pod whose SSH port sends no banner, the bootstrap is re-run inside it, but only when its watchdog is
    gone and no login was ever seen there. A pod the renter has logged in to is theirs to run: it is left alone,
    as is a pod whose image starts its own sshd. Never fails or scores; billing is not touched.
    """

    check_id = "host.repair.pod_sshd_restart"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        rented_data = ctx.state.rented_data
        executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        if executor is None or executor.miner_hotkey != ctx.miner_hotkey:
            return self._event(ctx, Msg.OK, {"pods": {}})
        manual = {str(uuid).lower() for uuid in (rented_data.manual_rental_executors or {})}
        if str(ctx.executor.uuid).lower() in manual:
            return self._event(ctx, Msg.OK, {"pods": {}})  # the renter holds the host at root
        pods = [p for p in executor.pods if p.ssh_port and p.status in (None, POD_STATUS_RUNNING)]
        probes = await asyncio.gather(
            *(tcp_connect_fault(executor.executor_ip_address, p.ssh_port, PROBE_TIMEOUT_SECONDS) for p in pods)
        )
        outcomes: dict[str, str] = {}
        for pod, probe in zip(pods, probes):
            # a timeout is the network as often as the pod
            if probe.result in (PodSshResult.REFUSED, PodSshResult.NO_BANNER):
                outcomes[pod.pod_id] = await self._restart(ctx, pod.docker_container_name)
        template = Msg.RESTARTED if "restarted" in outcomes.values() else Msg.OK
        return self._event(ctx, template, {"pods": outcomes})

    async def _restart(self, ctx: Context, name: str) -> str:
        try:
            res = await asyncio.wait_for(
                ctx.ssh.run(restart_command(name), check=False), timeout=RESTART_TIMEOUT_SECONDS
            )
        except (asyncssh.Error, OSError, TimeoutError) as exc:
            return f"error: {exc!r}"
        return {
            0: "restarted",
            NO_WATCHDOG_RECORD: "not bootstrapped by lium",
            WATCHDOG_ALIVE: "watchdog running",
            RENTER_LOGGED_IN: "renter logged in",
        }.get(res.exit_status, f"failed: exit {res.exit_status}")

    def _event(self, ctx: Context, template, what: dict) -> CheckResult:
        what = {"executor_uuid": ctx.executor.uuid, **what}
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )
