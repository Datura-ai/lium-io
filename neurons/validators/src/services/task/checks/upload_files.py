from __future__ import annotations

import asyncio
import shlex
import uuid
from dataclasses import replace

from ..messages import UploadFilesMessages as Msg, render_message
from ..pipeline import CheckResult, Context

PROBE_TIMEOUT = 30


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

        # DAH-2794: the scrape can travel down stdin instead of being uploaded, but only to an
        # executor whose interpreter can actually run it. The miner picks the executor image, so
        # that is a per-node question and it is asked here, not assumed from the flag.
        if ctx.config.machine_scrape_source and await self._can_run_source(ctx):
            event = render_message(
                Msg.UPLOAD_SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                what={"python_path": ctx.executor.python_path},
            )
            return CheckResult(
                passed=True,
                event=event,
                updates={"state": replace(ctx.state, scrape_over_stdin=True)},
            )

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

        MAX_RETRIES = 2
        last_exc: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
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
                last_exc = exc
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1.0 * attempt)

        event = render_message(
            Msg.UPLOAD_FAILED,
            ctx=ctx,
            check_id=self.check_id,
            what={"error": str(last_exc)[:200]},
        )
        return CheckResult(passed=False, event=event)

    async def _can_run_source(self, ctx: Context) -> bool:
        # Cheap, per-cycle: one round trip, no bytes. Names exactly what the scrape imports —
        # an image predating either dependency cannot run the source and must get the binary.
        python_path = ctx.executor.python_path
        if not python_path:
            return False

        probe = f"{shlex.quote(python_path)} -I -c 'import psutil, cryptography.fernet'"
        res = await ctx.runner.run(probe, timeout=PROBE_TIMEOUT, retryable=False)
        return res.success
