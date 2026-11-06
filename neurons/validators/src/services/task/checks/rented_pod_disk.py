"""DAH-5951: a rented pod cannot fill the Docker disk of a host that has no per-container disk limit.

On a host whose Docker storage cannot enforce ``--storage-opt size`` (``specs.storage_limit_supported`` not
True), a pod's writable layer can grow until the host's Docker disk is full, and then every pod on the host
fails. Such a host also gets the pod's volume from Docker's ``local`` driver, on the same disk and with no
size either. Each validation this check reads the Docker disk's free space and what each rented pod holds on
it (writable layer plus local volumes, read with ``du``, never changed). When the disk is almost full it
pauses (``docker pause``) each rented pod holding a large share of the disk: its data and volume are kept.
Once the disk has room again it resumes them. A host with the limit is never touched.
"""

from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass

import asyncssh

from ..messages import RentedPodDiskMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rented_pod_resources import running_rented_container_names

READ_TIMEOUT_SECONDS = 120  # `docker inspect --size` walks each writable layer
ACT_TIMEOUT_SECONDS = 60
# Pause below 5% free; resume at 10% free or more, so one cycle's noise does not flip a pod back and forth.
PAUSE_BELOW_FREE_FRACTION = 0.05
RESUME_AT_FREE_FRACTION = 0.10
# Only a pod that itself holds at least this share of the disk is paused: a disk the host filled
# itself never stops a renter.
MIN_LAYER_FRACTION = 0.10


@dataclass(frozen=True)
class PodLayer:
    container: str
    status: str
    size_rw: int
    volume_bytes: int

    @property
    def used(self) -> int:
        return self.size_rw + self.volume_bytes


def read_command(names: list[str]) -> str:
    quoted = " ".join(shlex.quote(n) for n in names)
    return (
        "r=$(/usr/bin/docker info -f '{{.DockerRootDir}}') && "
        "df -P -B1 \"$r\" | awk 'NR==2{print \"DF\", $2, $4}'; "
        "/usr/bin/docker inspect --size --format "
        "'{{.Name}} {{.State.Status}} {{.SizeRw}}{{range .Mounts}}{{if eq .Driver \"local\"}} {{.Source}}{{end}}{{end}}' "
        + quoted
        + " | while read -r n s w rest; do v=0; for d in $rest; do "
        "b=$(du -sxB1 \"$d\" 2>/dev/null | awk '{print $1}'); v=$((v + ${b:-0})); done; "
        'echo "$n $s $w $v"; done'
    )


def parse_reading(stdout: str) -> tuple[int | None, int | None, list[PodLayer]]:
    """(disk size, free bytes, pods) from read_command's output; None where a figure is missing."""
    size = free = None
    pods = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "DF" and parts[1].isdigit() and parts[2].isdigit():
            size, free = int(parts[1]), int(parts[2])
        elif len(parts) == 4 and parts[2].isdigit() and parts[3].isdigit():
            pods.append(PodLayer(parts[0].lstrip("/"), parts[1], int(parts[2]), int(parts[3])))
    return size, free, pods


def plan(size: int, free: int, pods: list[PodLayer]) -> tuple[list[str], list[str]]:
    """(containers to pause, containers to resume)."""
    if free < PAUSE_BELOW_FREE_FRACTION * size:
        return [
            p.container
            for p in pods
            if p.status == "running" and p.used >= MIN_LAYER_FRACTION * size
        ], []
    if free >= RESUME_AT_FREE_FRACTION * size:
        return [], [p.container for p in pods if p.status == "paused"]
    return [], []


class RentedPodDiskCheck:
    """Pause a rented pod that is filling a no-quota host's Docker disk; resume it once there is room.

    Non-fatal and never changes the score. Never fails on an inability to measure.
    """

    check_id = "host.validate.rented_pod_disk"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if (ctx.state.specs or {}).get("storage_limit_supported") is True:
            return self._event(ctx, Msg.LIMITED)
        names = running_rented_container_names(ctx)
        if not names:
            return self._event(ctx, Msg.NOT_RENTED)
        try:
            res = await ctx.ssh.run(read_command(names), timeout=READ_TIMEOUT_SECONDS)
            size, free, pods = parse_reading(str(getattr(res, "stdout", "") or ""))
            if not size or free is None:
                return self._event(ctx, Msg.UNKNOWN, reason="docker disk not readable")
            to_pause, to_resume = plan(size, free, pods)
            what = {
                "disk_bytes": size,
                "free_bytes": free,
                "pods": [asdict(p) for p in pods],
            }
            if to_pause:
                await ctx.ssh.run(
                    "/usr/bin/docker pause " + " ".join(shlex.quote(n) for n in to_pause),
                    timeout=ACT_TIMEOUT_SECONDS,
                )
                return self._event(ctx, Msg.PAUSED, paused=to_pause, **what)
            if to_resume:
                await ctx.ssh.run(
                    "/usr/bin/docker unpause " + " ".join(shlex.quote(n) for n in to_resume),
                    timeout=ACT_TIMEOUT_SECONDS,
                )
                return self._event(ctx, Msg.RESUMED, resumed=to_resume, **what)
            return self._event(ctx, Msg.OK, **what)
        except (asyncssh.Error, OSError) as exc:
            return self._event(ctx, Msg.UNKNOWN, reason="ssh transport error", error=repr(exc))

    def _event(self, ctx: Context, template, **what) -> CheckResult:
        what = {"executor_uuid": ctx.executor.uuid, **what}
        return CheckResult(
            passed=True, event=render_message(template, ctx=ctx, check_id=self.check_id, what=what)
        )
