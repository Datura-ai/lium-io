from __future__ import annotations

from typing import Callable

from ..models import build_msg
from ..pipeline import CheckResult, Context


class UploadFilesCheck:
    check_id = "prep.upload_validation_files"
    fatal = True

    def __init__(
        self,
        *,
        local_dir: str,
        executor_root: str,
        generate_random_name: Callable[[], str],
    ):
        self.local_dir = local_dir
        self.executor_root = executor_root
        self.generate_random_name = generate_random_name

    async def run(self, ctx: Context) -> CheckResult:
        random_name = self.generate_random_name()
        remote_dir = f"{self.executor_root}/{random_name}"

        try:
            async with ctx.ssh.start_sftp_client() as sftp:
                await sftp.put(self.local_dir, remote_dir, recurse=True)

            event = build_msg(
                event="Validation files uploaded",
                reason="UPLOAD_OK",
                severity="info",
                category="prep",
                impact="Proceed to validation",
                what={"remote_dir": remote_dir, "local_dir": self.local_dir},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=True, event=event, updates={"remote_dir": remote_dir})
        except Exception as exc:
            event = build_msg(
                event="Failed to upload validation files",
                reason="UPLOAD_FAILED",
                severity="error",
                category="prep",
                impact="Validation halted",
                remediation="Check network connectivity, disk space on executor, and SSH permissions",
                what={"error": str(exc)[:200]},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)
