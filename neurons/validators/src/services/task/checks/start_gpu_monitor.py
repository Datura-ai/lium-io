from __future__ import annotations

from typing import Any, Dict
import uuid

from bittensor import Keypair

from ..models import build_msg
from ..pipeline import CheckResult, Context


class StartGPUMonitorCheck:
    """Ensure the GPU monitor agent is running so validators receive live telemetry.

    This mirrors legacy logic that boots `gpus_utility.py` on the executor. Without it we
    lose visibility into runtime utilisation, so the remainder of the pipeline may produce
    misleading scores when GPUs are already busy.
    """

    check_id = "prep.start_gpu_monitor"
    fatal = False

    def __init__(
        self,
        *,
        validator_keypair: Keypair,
        compute_rest_app_url: str | None,
        script_relative_path: str = "src/gpus_utility.py",
    ):
        self.validator_keypair = validator_keypair
        self.compute_rest_app_url = compute_rest_app_url
        self.script_relative_path = script_relative_path

    async def run(self, ctx: Context) -> CheckResult:
        runner = ctx.runner
        executor = ctx.executor

        script_path = f"{executor.root_dir}/{self.script_relative_path}"

        check_cmd = f'ps aux | grep "python.*{script_path}" | grep -v grep'
        check_res = await runner.run(check_cmd, timeout=10, retryable=False)

        if check_res.stdout.strip():
            event = build_msg(
                event="GPU monitor already running",
                reason="MONITOR_RUNNING",
                severity="info",
                category="prep",
                impact="Proceed",
                what={"script_path": script_path},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event)

        await runner.run("pip install aiohttp click pynvml psutil", timeout=30, retryable=True)

        program_id = str(uuid.uuid4())
        command_args: Dict[str, Any] = {
            "program_id": program_id,
            "signature": f"0x{self.validator_keypair.sign(program_id.encode()).hex()}",
            "executor_id": executor.uuid,
            "validator_hotkey": self.validator_keypair.ss58_address,
            "compute_rest_app_url": self.compute_rest_app_url,
        }

        args_string = " ".join([f"--{k} {v}" for k, v in command_args.items()])
        start_cmd = (
            f"nohup {executor.python_path} {script_path} {args_string} > /dev/null 2>&1 &"
        )
        start_res = await runner.run(start_cmd, timeout=50, retryable=False)

        if start_res.success:
            event = build_msg(
                event="GPU monitor started",
                reason="MONITOR_STARTED",
                severity="info",
                category="prep",
                impact="Proceed with monitoring enabled",
                what={"script_path": script_path, "command_id": start_res.command_id},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event)

        event = build_msg(
            event="Failed to start GPU monitor",
            reason="MONITOR_START_FAILED",
            severity="warning",
            category="prep",
            impact="Validation continues without real-time GPU monitoring",
            remediation="Check Python installation and script permissions on executor",
            what={"exit_code": start_res.exit_code, "stderr": start_res.stderr[-400:]},
            check_id=self.check_id,
            ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
        )
        return CheckResult(passed=False, event=event)
