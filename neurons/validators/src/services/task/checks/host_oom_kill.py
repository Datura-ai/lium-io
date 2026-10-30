from __future__ import annotations

import asyncio
import re

from services.const import LIUM_INFRA_CONTAINER_PREFIXES, RENTAL_CONTAINER_PREFIXES

from ..messages import HostOomKillMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

# A kill older than this no longer explains the node's state.
WINDOW_SECONDS = 24 * 3600
SSH_TIMEOUT_SECONDS = 10
# The executor container is privileged (CAP_SYSLOG), so its dmesg is the host kernel's ring buffer.
# dmesg exit status is lost in the pipe: a failed read prints UNREADABLE so it never reads as quiet.
UNREADABLE = "DMESG_UNREADABLE"
READ_COMMAND = (
    f"cat /proc/uptime; {{ dmesg 2>/dev/null || echo {UNREADABLE}; }}"
    f" | grep -E 'oom-kill:|Killed process|{UNREADABLE}' | tail -n 40"
)

_LINE_TIME = re.compile(r"^\[\s*(\d+(?:\.\d+)?)\]\s*")
# cgroup v2 + systemd: /system.slice/docker-<id>.scope; cgroupfs: /docker/<id>
_DOCKER_ID = re.compile(r"docker[-/]([0-9a-f]{64})")
_KILLED = re.compile(r"Killed process (\d+) \(.*?\).*?anon-rss:(\d+)kB")
_SAFE = re.compile(r"[^\w.:-]")


def parse_global_oom_kills(stdout: str) -> list[dict]:
    """Host-wide (CONSTRAINT_NONE) kernel OOM kills in `READ_COMMAND` output, oldest first, with their age.

    A kill inside a container's own memory limit (CONSTRAINT_MEMCG) is the workload's business and is left out.
    Raises ValueError when the kernel log or the uptime could not be read, so it is never reported as quiet.
    """
    lines = stdout.splitlines()
    if UNREADABLE in lines:
        raise ValueError("dmesg unreadable")
    try:
        uptime = float(lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError("uptime unreadable") from exc
    rss_by_pid = {m.group(1): int(m.group(2)) for line in lines if (m := _KILLED.search(line))}
    kills = []
    for line in lines[1:]:
        stamp = _LINE_TIME.match(line)
        _, marker, body = line.partition("oom-kill:")
        if not (stamp and marker):
            continue
        fields = dict(part.split("=", 1) for part in body.strip().split(",") if "=" in part)
        if fields.get("constraint") != "CONSTRAINT_NONE":
            continue
        container = _DOCKER_ID.search(fields.get("task_memcg", ""))
        pid = fields.get("pid", "")
        kills.append(
            {
                "seconds_ago": max(0, int(uptime - float(stamp.group(1)))),
                "task": _SAFE.sub("", fields.get("task", ""))[:15],
                "container_id": container.group(1) if container else None,
                "anon_rss_kb": rss_by_pid.get(pid),
            }
        )
    return kills


def summarize(kill: dict, container_name: str | None) -> dict:
    """What the provider is shown. A renter's pod is named only by its container; its process name stays out."""
    container_id = kill["container_id"]
    name = _SAFE.sub("", (container_name or "").lstrip("/"))[:80] or None
    if container_id is None:
        kind = "host_process"
    elif name is None:
        kind = "removed_container"
    elif name.startswith(("pod_", "filler_")):
        kind = "rental_pod"
    elif name.startswith(RENTAL_CONTAINER_PREFIXES + LIUM_INFRA_CONTAINER_PREFIXES):
        kind = "lium_container"
    else:
        kind = "other_container"
    rss = kill["anon_rss_kb"]
    return {
        "kind": kind,
        "container_id": container_id[:12] if container_id else None,
        "container_name": name,
        "task": None if kind in ("rental_pod", "removed_container") else kill["task"],
        "memory_gib": round(rss / 2**20, 1) if rss is not None else None,
        "minutes_ago": kill["seconds_ago"] // 60,
    }


class HostOomKillCheck:
    """Name what the host kernel's out-of-memory killer killed in the last day (ticket-0903).

    A host-wide OOM kill takes the biggest memory holder: a renter's pod, a Lium container or a provider's own
    process. The summary rides in the context of every later event of the cycle (as `executor_image_status`
    does), so the portal can tell the provider which one it was. Observation only: never fails or halts.
    """

    check_id = "executor.observe.host_oom_kill"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        what: dict = {"executor_uuid": ctx.executor.uuid}
        try:
            result = await self._ssh(ctx, READ_COMMAND)
            kills = [
                k
                for k in parse_global_oom_kills(result.stdout or "")
                if k["seconds_ago"] <= WINDOW_SECONDS
            ]
            if not kills:
                return CheckResult(
                    passed=True,
                    event=render_message(Msg.NONE, ctx=ctx, check_id=self.check_id, what=what),
                )
            kill = kills[-1]
            name = None
            if kill["container_id"]:
                inspected = await self._ssh(
                    ctx, f"/usr/bin/docker inspect --format '{{{{.Name}}}}' {kill['container_id']}"
                )
                name = (inspected.stdout or "").strip() if inspected.exit_status == 0 else None
        except Exception as exc:  # noqa: BLE001 - observation only, never fatal; a timeout lands here too
            what["error"] = repr(exc)[:200]
            return CheckResult(
                passed=True,
                event=render_message(Msg.UNREAD, ctx=ctx, check_id=self.check_id, what=what),
            )
        summary = summarize(kill, name)
        return CheckResult(
            passed=True,
            event=render_message(
                Msg.KILLED, ctx=ctx, check_id=self.check_id, what={**what, **summary}
            ),
            updates={"default_extra": {**ctx.default_extra, "host_oom_kill": summary}},
        )

    @staticmethod
    async def _ssh(ctx: Context, command: str):
        return await asyncio.wait_for(
            ctx.ssh.run(command, check=False), timeout=SSH_TIMEOUT_SECONDS
        )
