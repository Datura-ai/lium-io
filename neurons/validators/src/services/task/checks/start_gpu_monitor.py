from __future__ import annotations

from typing import Any, Dict
import uuid

from ..messages import StartGpuMonitorMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class StartGPUMonitorCheck:
    """Ensure the GPU monitor agent is running so validators receive live telemetry.

    This mirrors legacy logic that boots `gpus_utility.py` on the executor. Without it we
    lose visibility into runtime utilisation, so the remainder of the pipeline may produce
    misleading scores when GPUs are already busy.
    """

    check_id = "prep.start_gpu_monitor"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        runner = ctx.runner
        executor = ctx.executor

        validator_keypair = ctx.config.validator_keypair
        compute_rest_app_url = ctx.config.compute_rest_app_url
        script_relative_path = ctx.config.gpu_monitor_script_relative

        if not compute_rest_app_url:
            event = render_message(
                Msg.CONFIG_MISSING,
                ctx=ctx,
                check_id=self.check_id,
                what={"compute_rest_app_url": compute_rest_app_url},
            )
            return CheckResult(passed=False, event=event)

        script_path = f"{executor.root_dir.rstrip('/')}/{script_relative_path.lstrip('/')}"

        check_cmd = f'ps aux | grep "python.*{script_path}" | grep -v grep'
        check_res = await runner.run(check_cmd, timeout=10, retryable=False)
        processes = [line for line in check_res.stdout.splitlines() if "grep" not in line]

        if processes:
            event = render_message(
                Msg.ALREADY_RUNNING,
                ctx=ctx,
                check_id=self.check_id,
                what={"script_path": script_path},
            )
            return CheckResult(passed=True, event=event)

        await runner.run("pip install aiohttp click pynvml psutil", timeout=30, retryable=True)

        program_id = str(uuid.uuid4())
        command_args: Dict[str, Any] = {
            "program_id": program_id,
            "signature": f"0x{validator_keypair.sign(program_id.encode()).hex()}",
            "executor_id": executor.uuid,
            "validator_hotkey": validator_keypair.ss58_address,
            "compute_rest_app_url": compute_rest_app_url,
        }

        args_string = " ".join([f"--{k} {v}" for k, v in command_args.items()])
        start_cmd = (
            f"nohup {executor.python_path} {script_path} {args_string} > /dev/null 2>&1 &"
        )
        start_res = await runner.run(start_cmd, timeout=50, retryable=False)

        if start_res.success:
            event = render_message(
                Msg.STARTED,
                ctx=ctx,
                check_id=self.check_id,
                what={"script_path": script_path, "command_id": start_res.command_id},
            )
            return CheckResult(passed=True, event=event)

        event = render_message(
            Msg.START_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            what={"exit_code": start_res.exit_code, "stderr": start_res.stderr[-400:]},
        )
        return CheckResult(passed=False, event=event)
