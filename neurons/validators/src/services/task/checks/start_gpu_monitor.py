from __future__ import annotations

from typing import Any, Dict

from datura.requests.miner_requests import ExecutorSSHInfo

from ..models import build_msg
from ..pipeline import CheckResult, Context


class StartGPUMonitorCheck:
    check_id = "prep.start_gpu_monitor"
    fatal = False

    def __init__(
        self,
        *,
        script_path: str,
        command_args: Dict[str, Any],
        executor_info: ExecutorSSHInfo,
    ):
        self.script_path = script_path
        self.command_args = command_args
        self.executor_info = executor_info

    async def run(self, ctx: Context) -> CheckResult:
        runner = ctx.runner

        check_cmd = f'ps aux | grep "python.*{self.script_path}" | grep -v grep'
        check_res = await runner.run(check_cmd, timeout=10, retryable=False)

        if check_res.stdout.strip():
            event = build_msg(
                event="GPU monitor already running",
                reason="MONITOR_RUNNING",
                severity="info",
                category="prep",
                impact="Proceed",
                what={"script_path": self.script_path},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event)

        await runner.run("pip install aiohttp click pynvml psutil", timeout=30, retryable=True)

        args_string = " ".join([f"--{k} {v}" for k, v in self.command_args.items()])
        start_cmd = (
            f"nohup {self.executor_info.python_path} {self.script_path} {args_string} > /dev/null 2>&1 &"
        )
        start_res = await runner.run(start_cmd, timeout=50, retryable=False)

        if start_res.success:
            event = build_msg(
                event="GPU monitor started",
                reason="MONITOR_STARTED",
                severity="info",
                category="prep",
                impact="Proceed with monitoring enabled",
                what={"script_path": self.script_path, "command_id": start_res.command_id},
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
