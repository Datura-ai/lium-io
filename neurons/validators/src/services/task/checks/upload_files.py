from __future__ import annotations

import uuid
from dataclasses import replace

from ..messages import UploadFilesMessages as Msg, render_message
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
            event = render_message(
                Msg.CONFIG_MISSING,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "local_dir": local_dir,
                    "executor_root": executor_root,
                },
            )
            return CheckResult(passed=False, event=event)

        random_name = uuid.uuid4().hex
        remote_dir = f"{executor_root.rstrip('/')}/{random_name}"

        try:
            async with ctx.ssh.start_sftp_client() as sftp:
                await sftp.put(local_dir, remote_dir, recurse=True)

            event = render_message(
                Msg.UPLOAD_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={"remote_dir": remote_dir, "local_dir": local_dir},
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
            event = render_message(
                Msg.UPLOAD_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={"error": str(exc)[:200]},
            )
            return CheckResult(passed=False, event=event)
