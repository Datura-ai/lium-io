from __future__ import annotations

import json
from typing import Callable

from ..models import build_msg
from ..pipeline import CheckResult, Context
from ..runner import SSHCommandRunner


class MachineSpecScrapeCheck:
    check_id = "gpu.scrape.machine_spec"
    fatal = True

    def __init__(
        self,
        *,
        script_filename: str,
        decrypt_func: Callable[[str, str], str],
        deobfuscate_func: Callable[[dict], dict],
        timeout: int = 300,
    ):
        self.script_filename = script_filename
        self.decrypt = decrypt_func
        self.deobfuscate = deobfuscate_func
        self.timeout = timeout

    async def run(self, ctx: Context) -> CheckResult:
        runner: SSHCommandRunner = ctx.runner

        if not ctx.remote_dir:
            event = build_msg(
                event="Remote directory not set",
                reason="MISSING_REMOTE_DIR",
                severity="error",
                category="prep",
                impact="Cannot locate scrape script",
                remediation="Internal error - UploadFilesCheck must run before MachineSpecScrapeCheck",
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        script_path = f"{ctx.remote_dir}/{self.script_filename}"

        res = await runner.run(f"chmod +x {script_path} && {script_path}", timeout=self.timeout, retryable=False)

        if not res.success or not res.stdout.strip():
            event = build_msg(
                event="Machine specs scrape failed",
                reason="SCRAPE_FAILED",
                severity="error",
                category="env",
                impact="Validation halted — GPU unverified",
                remediation=(
                    f"Ensure the scrape script exists and is executable:\n"
                    f"  chmod +x {script_path}\n"
                    f"Check stderr and environment (Python deps) on the executor."
                ),
                what={
                    "command_id": res.command_id,
                    "exit_code": res.exit_code,
                    "duration_ms": res.duration_ms,
                    "stderr_tail": res.stderr[-400:],
                },
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        try:
            line = res.stdout.splitlines()[0].strip()
            if not ctx.encrypt_key:
                raise ValueError("Missing encrypt_key in context")

            decrypted = self.decrypt(ctx.encrypt_key, line)
            raw = json.loads(decrypted)
            specs = self.deobfuscate(raw)

            gpu_info = specs.get("gpu", {}) or {}
            gpu_count = gpu_info.get("count", 0) or 0
            gpu_details = gpu_info.get("details", []) or []
            gpu_model = None
            if gpu_count > 0 and gpu_details:
                gpu_model = gpu_details[0].get("name")

            gpu_model_count = f"{gpu_model}:{gpu_count}" if gpu_model is not None else None
            gpu_uuids = ",".join(detail.get("uuid", "") for detail in gpu_details if detail.get("uuid"))

            event = build_msg(
                event="Machine specs scraped",
                reason="SCRAPE_OK",
                severity="info",
                category="env",
                impact="Proceed",
                what={
                    "gpu_count": gpu_count,
                    "gpu_model": gpu_model,
                },
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            updates = {
                "specs": specs,
                "gpu_model": gpu_model,
                "gpu_count": gpu_count,
                "gpu_details": gpu_details,
                "gpu_processes": specs.get("gpu_processes", []) or [],
                "sysbox_runtime": specs.get("sysbox_runtime", False) or False,
            }
            if gpu_model_count:
                updates["gpu_model_count"] = gpu_model_count
            if gpu_uuids:
                updates["gpu_uuids"] = gpu_uuids
            return CheckResult(passed=True, event=event, updates=updates)

        except Exception as exc:
            event = build_msg(
                event="Machine specs parse/decrypt failed",
                reason="SCRAPE_PARSE_FAILED",
                severity="error",
                category="env",
                impact="Validation halted — GPU unverified",
                remediation="Confirm encryption key, payload, and repo versions on both validator and executor.",
                what={"exception": str(exc)[:300], "stdout_head": res.stdout[:200]},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)
