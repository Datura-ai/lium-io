from __future__ import annotations

from typing import Any

from ..messages import DiskHealthMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context


def disk_error_summary(health: dict[str, Any]) -> dict[str, Any]:
    """The readings that say something is wrong, and nothing else - what the event carries."""
    summary: dict[str, Any] = {}
    if health.get("kernel_io_errors"):
        summary["kernel_io_errors"] = health["kernel_io_errors"]
        summary["kernel_io_error_lines"] = health.get("kernel_io_error_lines") or []
    if health.get("block_io_errors"):
        summary["block_io_errors"] = health["block_io_errors"]
    if health.get("nvme_states"):
        summary["nvme_states"] = health["nvme_states"]
    smart = health.get("smart")
    if isinstance(smart, dict):
        failed = {device: verdict for device, verdict in smart.items() if verdict != "PASSED"}
        if failed:
            summary["smart"] = failed
    return summary


class DiskHealthCheck:
    """Gate on the disk that holds the containers still taking writes (DAH-2928).

    Pure-data: reads ``specs.disk_health`` as MachineSpecScrapeCheck left it, so it runs right
    after the GPU spec checks and before the rented short-circuit, for rented and idle executors
    alike. The scrape's write probe on the docker root is the one reading acted on: a docker root
    that is mounted read-only or refuses writes with EROFS/EIO cannot start a container, and the
    executor is scored zero until it can. Kernel I/O errors, sysfs error counters, NVMe controller
    state and SMART verdicts are reported (they travel to the backend in specs) but do not fail the
    check on their own - a USB stick's errors and a dying NVMe look the same in a count.
    """

    check_id = "executor.validate.disk_health"
    fatal = True

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
                    **disk_error_summary(health),
                },
            )
            return CheckResult(passed=False, event=event)

        summary = disk_error_summary(health)
        if summary:
            event = render_message(
                Msg.ERRORS_REPORTED,
                ctx=ctx,
                check_id=self.check_id,
                what={"docker_root_dir": health.get("docker_root_dir"), **summary},
            )
            return CheckResult(passed=True, event=event)

        event = render_message(
            Msg.OK,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "docker_root_dir": health.get("docker_root_dir"),
                "write_probe": health.get("write_probe"),
                "smart": health.get("smart"),
            },
        )
        return CheckResult(passed=True, event=event)
