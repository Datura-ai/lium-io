from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace

from ..messages import UploadFilesMessages as Msg, render_message
from ..pipeline import CheckResult, Context


class UploadFilesCheck:
    """Push encrypted validation assets to the executor before any remote commands run.

    This check uploads the encrypted validation scripts and secrets to the executor before
    any remote commands run, ensuring all later checks operate on freshly uploaded files.
    """

    check_id = "prep.upload_validation_files"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        local_dir = ctx.state.upload_local_dir
        executor_root = ctx.config.executor_root
        DEFAULT_TIMEOUT = 300

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
            async with asyncio.timeout(DEFAULT_TIMEOUT):
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
        except asyncio.TimeoutError:
            event = render_message(
                Msg.UPLOAD_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={"error": f"Upload timed out after {DEFAULT_TIMEOUT} seconds"},
            )
            return CheckResult(passed=False, event=event)
        except Exception as exc:
            event = render_message(
                Msg.UPLOAD_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={"error": str(exc)[:200]},
            )
            return CheckResult(passed=False, event=event)
