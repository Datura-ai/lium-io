from __future__ import annotations

import asyncssh

from core.config import settings

from ..messages import DockerCpuMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .cpu_truth import advertised_cpu_count
from .rental_owner import rental_of_this_node

# `docker info` is instant on a live daemon; a wedged one lands in the unknown path, never a failure.
DOCKER_CPU_READ_TIMEOUT_SECONDS = 15


class DockerCpuCheck:
    """Fail an unrented executor whose Docker daemon sees fewer CPUs than the machine reports.

    A daemon limited to fewer CPUs (CPU affinity or a cgroup on the Docker service) cannot start a
    container with the CPUs the node is listed with, so its rentals fail. Compares the scraped
    CPU(s) count with `docker info` NCPU. With DOCKER_CPU_MISMATCH_ENFORCEMENT_ENABLED off, a
    mismatch is logged only. Rented nodes are warned and keep their score. An unreadable count passes.
    """

    check_id = "executor.validate.docker_cpu"
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        host = advertised_cpu_count(ctx)
        try:
            docker = await self._read_docker_ncpu(ctx.ssh)
        except (asyncssh.Error, OSError):
            docker = None
        rented = rental_of_this_node(ctx)
        is_rented = rented is not None and len(rented.pods) > 0
        what = {"host_cpus": host, "docker_cpus": docker, "is_rented": is_rented}
        if host is None or docker is None or docker >= host:
            return CheckResult(passed=True, event=render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what=what))
        remediation = f"Docker sees {docker} of {host} CPUs; remove the CPU limit on the Docker service."
        if is_rented or not settings.DOCKER_CPU_MISMATCH_ENFORCEMENT_ENABLED:
            event = render_message(
                Msg.WARNING, ctx=ctx, check_id=self.check_id, what=what, remediation=remediation
            )
            return CheckResult(passed=True, event=event)
        event = render_message(Msg.MISMATCH, ctx=ctx, check_id=self.check_id, what=what, remediation=remediation)
        return CheckResult(passed=False, event=event)

    async def _read_docker_ncpu(self, ssh) -> int | None:
        res = await ssh.run("/usr/bin/docker info --format '{{.NCPU}}'", timeout=DOCKER_CPU_READ_TIMEOUT_SECONDS)
        if getattr(res, "exit_status", 1) != 0:
            return None
        try:
            count = int((getattr(res, "stdout", "") or "").strip())
        except ValueError:
            return None
        return count if count > 0 else None
