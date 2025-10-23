from __future__ import annotations

import uuid
from dataclasses import replace

from ..models import build_msg
from ..pipeline import CheckResult, Context


class UploadFilesCheck:
    """Push encrypted validation assets to the executor before any remote commands run.

    Legacy `create_task_old` failed immediately if staging failed; keeping this guard up
    front guarantees all later checks operate on freshly uploaded scripts and secrets.
    """

    check_id = "prep.upload_validation_files"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        local_dir = ctx.state.upload_local_dir
        executor_root = ctx.config.executor_root

        if not local_dir or not executor_root:
            event = build_msg(
                event="Upload configuration missing",
                reason="UPLOAD_CONFIG_MISSING",
                severity="error",
                category="prep",
                impact="Validation halted",
                remediation="Validator bug: missing upload metadata in context",
                what={
                    "local_dir": local_dir,
                    "executor_root": executor_root,
                },
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        random_name = uuid.uuid4().hex
        remote_dir = f"{executor_root.rstrip('/')}/{random_name}"

        try:
            async with ctx.ssh.start_sftp_client() as sftp:
                await sftp.put(local_dir, remote_dir, recurse=True)

            event = build_msg(
                event="Validation files uploaded",
                reason="UPLOAD_OK",
                severity="info",
                category="prep",
                impact="Proceed to validation",
                what={"remote_dir": remote_dir, "local_dir": local_dir},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            updated_state = replace(
                ctx.state,
                upload_remote_dir=remote_dir,
                remote_dir=remote_dir,
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={"state": updated_state},
            )
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
