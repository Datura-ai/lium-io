from __future__ import annotations

from ..messages import DiskHealthMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context


class DiskHealthCheck:
    """Report whether the disk that holds the containers still takes writes (DAH-2928).

    Pure-data: reads ``specs.disk_health`` as MachineSpecScrapeCheck left it, so it runs right
    after the GPU spec checks and before the rented short-circuit, for rented and idle executors
    alike. Observe-only, non-fatal: a docker root that is mounted read-only or refuses writes with
    EROFS/EIO/ENOSPC/EDQUOT cannot start a container, and the check says so with a warning event,
    but the score is not changed - a false reading here would zero rented and idle executors
    fleet-wide, so the reading is proven on live executors first.
    """

    check_id = "executor.validate.disk_health"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        specs = ctx.state.specs or {}
        health = specs.get("disk_health")
        if not isinstance(health, dict):
            # a validator scrape that predates the probe, or the probe itself failed: unknown, not bad
            event = render_message(
                Msg.UNKNOWN,
                ctx=ctx,
                check_id=self.check_id,
                what={"scrape_error": specs.get("disk_health_scrape_error")},
            )
            return CheckResult(passed=True, event=event)

        read_only_mounts = health.get("read_only_mounts") or []
        if read_only_mounts or health.get("write_probe") == "failed":
            event = render_message(
                Msg.NOT_WRITABLE,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "docker_root_dir": health.get("docker_root_dir"),
                    "read_only_mounts": read_only_mounts,
                    "write_probe": health.get("write_probe"),
                    "write_probe_error": health.get("write_probe_error"),
                },
            )
            # Non-fatal and passed: the event is the warning; nothing downstream reads a verdict.
            return CheckResult(passed=True, event=event)

        event = render_message(
            Msg.OK,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "docker_root_dir": health.get("docker_root_dir"),
                "write_probe": health.get("write_probe"),
            },
        )
        return CheckResult(passed=True, event=event)
