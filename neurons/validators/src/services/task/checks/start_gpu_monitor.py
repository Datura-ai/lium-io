from __future__ import annotations

from typing import Any, Dict
import shlex
import uuid

from services import executor_payload as payload

from ..messages import StartGpuMonitorMessages as Msg, render_message
from ..pipeline import CheckResult, Context


# the monitor an executor image of before DAH-6681 ran from its own source tree: never trusted, stopped on sight
LEGACY_MONITOR = "/root/app/src/gpus_utility.py"


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
        # a monitor running the executor's own copy is stopped, so only the validator's runs ("[p]" keeps the
        # pattern from matching the shell that runs this command)
        legacy = "[p]ython.*" + LEGACY_MONITOR.replace(".", "\\.")
        await runner.run(f"pkill -f -- {shlex.quote(legacy)}; true", timeout=10, retryable=False)
        # a monitor already started by the validator. Detection only: nothing is executed.
        pattern = f"python.*{payload.service_dir_glob('monitor')}[^/]*/{payload.GPU_MONITOR[0]}"
        check_cmd = f"ps auxww | grep -E {shlex.quote(pattern)} | grep -v grep"
        check_res = await runner.run(check_cmd, timeout=10, retryable=False)
        processes = [line for line in check_res.stdout.splitlines() if "grep" not in line]

        if processes:
            event = render_message(
                Msg.ALREADY_RUNNING,
                ctx=ctx,
                check_id=self.check_id,
                what={"script_path": script_path, "python_path": payload.EXECUTOR_PYTHON},
            )
            return CheckResult(passed=True, event=event)

        # await runner.run("pip install aiohttp click pynvml psutil", timeout=30, retryable=True)

        program_id = str(uuid.uuid4())
        command_args: Dict[str, Any] = {
            "program_id": program_id,
            "signature": f"0x{validator_keypair.sign(program_id.encode()).hex()}",
            "executor_id": executor.uuid,
            "validator_hotkey": validator_keypair.ss58_address,
            "compute_rest_app_url": compute_rest_app_url,
        }

        args_string = " ".join([f"--{k} {shlex.quote(str(v))}" for k, v in command_args.items()])
        # no monitor runs (checked above): the directories of earlier ones go before a fresh one is written
        start_cmd = (
            f"rm -rf {shlex.quote(payload.service_dir_glob('monitor'))}*; "
            f"{payload.install('monitor', payload.GPU_MONITOR)}; "
            f"nohup {payload.run(payload.GPU_MONITOR)} {args_string} > /dev/null 2>&1 &"
        )
        start_res = await runner.run(start_cmd, timeout=50, retryable=False)

        if start_res.success:
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
