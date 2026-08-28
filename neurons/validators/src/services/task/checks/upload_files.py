from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace

from ..messages import UploadFilesMessages as Msg, render_message
from ..pipeline import CheckResult, Context

UPLOAD_TIMEOUT = 300
MAX_RETRIES = 2


class UploadFailed(Exception):
    """The validation assets never reached the executor."""


async def upload_validation_files(ctx: Context, *, attempts: int = MAX_RETRIES) -> str:
    # copy the local validation assets into a fresh random directory on the executor
    local_dir = ctx.state.upload_local_dir
    executor_root = ctx.config.executor_root
    if not local_dir or not executor_root:
        raise UploadFailed(f"missing upload config: local_dir={local_dir}, executor_root={executor_root}")

    remote_dir = f"{executor_root.rstrip('/')}/{uuid.uuid4().hex}"
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            async with asyncio.timeout(UPLOAD_TIMEOUT):
                async with ctx.ssh.start_sftp_client() as sftp:
                    await sftp.put(local_dir, remote_dir, recurse=True)
            return remote_dir
        except asyncio.TimeoutError:
            # A retry would only re-spend the same 300 s against the same slow uplink.
            raise UploadFailed(f"Upload timed out after {UPLOAD_TIMEOUT} seconds")
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                await asyncio.sleep(1.0 * attempt)

    raise UploadFailed(str(last_exc)[:200])


class UploadFilesCheck:
    """Push encrypted validation assets to the executor before any remote commands run.

    This check uploads the encrypted validation scripts and secrets to the executor before
    any remote commands run, ensuring all later checks operate on freshly uploaded files.
    """

    check_id = "prep.upload_validation_files"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        # DAH-2794: with the source in hand there is nothing to upload — the scrape travels
        # down stdin instead. Whether this executor's interpreter can actually run it is not
        # guessed here: MachineSpecScrapeCheck tries, and uploads the binary itself if it cannot.
        if ctx.config.machine_scrape_source:
            event = render_message(
                Msg.UPLOAD_SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"python_path": ctx.executor.python_path},
            )
            return CheckResult(passed=True, event=event)

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

        try:
            remote_dir = await upload_validation_files(ctx)
        except UploadFailed as exc:
            event = render_message(
                Msg.UPLOAD_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={"error": str(exc)},
            )
            return CheckResult(passed=False, event=event)

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
        return CheckResult(passed=True, event=event, updates={"state": updated_state})
