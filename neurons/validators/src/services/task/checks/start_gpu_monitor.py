from __future__ import annotations

from typing import Any, Dict
import shlex
import uuid

from services import executor_payload as payload

from ..messages import StartGpuMonitorMessages as Msg, render_message
from ..pipeline import CheckResult, Context


# the monitor an executor image of before DAH-6681 ran from its own source tree: never trusted, stopped on sight
LEGACY_MONITOR = "/root/app/src/gpus_utility.py"
SAME_MOUNT_NS = payload.SAME_MOUNT_NS
# the start time (clock ticks after boot) of process $pid: with the PID it names one process, never a later reuse of it
_START_TIME = "$(sed 's/.*) //' /proc/$pid/stat 2>/dev/null | cut -d' ' -f20)"


class StartGPUMonitorCheck:
    """Ensure `gpus_utility.py` is running on the executor, where it keeps the Docker template
    images synced and pruned.

    The name is historical: the script also used to push GPU telemetry to the backend, but that
    path was disabled on both ends long ago and deleted in DAH-2419. Runtime GPU utilisation now
    reaches the backend through the machine-spec scrape instead. The metrics-push arguments below
    are still sent so that executors running an older image (the miner picks the image version)
    keep starting — the script accepts and ignores them, and they go once the fleet has rolled.

    The script and the interpreter come from the validator (services.executor_payload), never from the
    executor's reported root_dir/python_path.
    """

    check_id = "prep.start_gpu_monitor"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        runner = ctx.runner
        executor = ctx.executor

        validator_keypair = ctx.config.validator_keypair
        compute_rest_app_url = ctx.config.compute_rest_app_url

        if not compute_rest_app_url:
            event = render_message(
                Msg.CONFIG_MISSING,
                ctx=ctx,
                check_id=self.check_id,
                what={"compute_rest_app_url": compute_rest_app_url},
            )
            return CheckResult(passed=False, event=event)

        script_path = f"{payload.service_dir_glob('monitor')}*/{payload.GPU_MONITOR[0]}"
        program_id = str(uuid.uuid4())
        command_args: Dict[str, Any] = {
            "program_id": program_id,
            "signature": f"0x{validator_keypair.sign(program_id.encode()).hex()}",
            "executor_id": executor.uuid,
            "validator_hotkey": validator_keypair.ss58_address,
            "compute_rest_app_url": compute_rest_app_url,
        }
        args_string = " ".join([f"--{k} {shlex.quote(str(v))}" for k, v in command_args.items()])

        # one command under an executor-local lock, so two overlapping checks never both start a monitor or remove
        # each other's. A stop returns only once no matching process is left: TERM, KILL after 5 s, and the command
        # fails after 10 s or on any pkill/pgrep error, so nothing is replaced while an old monitor lives ("[p]" keeps
        # the pattern from matching the shell that runs it).
        def stop(pattern: str) -> str:
            match = f"{SAME_MOUNT_NS} -f -- {shlex.quote(pattern)}"
            return (
                f'pkill {match}; [ "$?" -le 1 ] || exit 1; i=0; '
                f'while :; do pgrep {match} > /dev/null; r=$?; [ "$r" = 1 ] && break; [ "$r" = 0 ] || exit 1; '
                f'i=$((i + 1)); [ "$i" -ge 100 ] && exit 1; [ "$i" = 50 ] && pkill -KILL {match}; sleep 0.1; done'
            )

        # held while a check looks for the monitor and starts one (outside the monitor directories' glob)
        lock = f"{payload.SERVICE_DIR_PARENT}/.lium-validator-monitor-lock"
        legacy = "[p]ython.*" + LEGACY_MONITOR.replace(".", "\\.")
        dirs = shlex.quote(payload.service_dir_glob("monitor")) + "*"
        sums = shlex.quote(payload.digests(payload.GPU_MONITOR))
        ours = "[p]ython.*" + (payload.service_dir_glob("monitor") + "[^/]*/" + payload.GPU_MONITOR[0]).replace(".", "\\.")
        command = (
            f"exec 9> {shlex.quote(lock)} && flock -w 30 9 || exit 1; "
            # a monitor running the executor's own copy is always stopped, so only the validator's runs
            f"{stop(legacy)}; "
            # reused only when it is the process this validator started (the PID and start time it recorded in the
            # monitor's root-only directory), in the executor's mount namespace, and that directory holds the current
            # payload; a monitor of an earlier build or any other process is replaced
            f'for d in {dirs}; do read pid start < "$d/.run" 2>/dev/null || continue; '
            f'case "$pid" in ""|*[!0-9]*) continue;; esac; [ "{_START_TIME}" = "$start" ] || continue; '
            '[ "$(readlink /proc/$pid/ns/mnt)" = "$(readlink /proc/$$/ns/mnt)" ] || continue; '
            f'printf %s {sums} | (cd "$d" && sha256sum -c --status -) && {{ echo "RUNNING $pid"; exit 0; }}; done; '
            f"{stop(ours)}; rm -rf {dirs}; "
            f"{payload.install('monitor', payload.GPU_MONITOR)}; "
            f"nohup {payload.run(payload.GPU_MONITOR)} {args_string} > /dev/null 2>&1 9>&- & "
            f'pid=$!; printf "%s %s\\n" "$pid" "{_START_TIME}" > "$d/.run" || exit 1; '
            # the child must still be up once its interpreter has loaded the script
            'sleep 1; kill -0 "$pid" 2>/dev/null || exit 1; echo "STARTED $pid"'
        )
        start_res = await runner.run(command, timeout=60, retryable=False)
        outcome = start_res.stdout.split()[:1] if start_res.success else []

        if outcome == ["RUNNING"]:
            event = render_message(
                Msg.ALREADY_RUNNING,
                ctx=ctx,
                check_id=self.check_id,
                what={"script_path": script_path, "python_path": payload.EXECUTOR_PYTHON},
            )
            return CheckResult(passed=True, event=event)

        if outcome == ["STARTED"]:
            event = render_message(
                Msg.STARTED,
                ctx=ctx,
                check_id=self.check_id,
                what={"script_path": script_path, "python_path": payload.EXECUTOR_PYTHON, "command_id": start_res.command_id},
            )
            return CheckResult(passed=True, event=event)

        event = render_message(
            Msg.START_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "exit_code": start_res.exit_code,
                "error_type": start_res.error_type,
                "stderr": start_res.stderr[-400:],
            },
        )
        return CheckResult(passed=False, event=event)
