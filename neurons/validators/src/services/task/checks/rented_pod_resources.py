from __future__ import annotations

import json
import math
import shlex

import asyncssh
from services.const import RENTAL_CPU_COUNT_LABEL

from core.config import settings

from ..messages import RentedPodResourcesMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .cpu_truth import _count_cpu_list, advertised_cpu_count
from .rented_gpu_drop import POD_STATUS_RUNNING

READ_TIMEOUT_SECONDS = 30
_INSPECT_FORMAT = (
    '{"name":{{json .Name}},"cpuset":{{json .HostConfig.CpusetCpus}},'
    '"nano":{{json .HostConfig.NanoCpus}},"oom":{{json .HostConfig.OomScoreAdj}},'
    '"pid":{{json .State.Pid}},"labels":{{json .Config.Labels}}}'
)
# The failing findings; oom_priority_raised is recorded only, a root renter can raise it themselves.
SCORED_FINDINGS = ("cpuset_pinned", "cpu_quota_reduced")


class RentedPodResourcesCheck:
    """Read each running rented pod's HostConfig and init oom_score_adj and flag a host that
    `docker update`d the renter below what the validator deployed.

    The validator never sets a cpuset and records the --cpus it deployed in an immutable label
    (RENTAL_CPU_COUNT_LABEL). Non-fatal: shadow always passes, and only
    RENTED_POD_RESOURCES_ENFORCEMENT_ENABLED lets a cpuset/quota finding fail and zero the score.
    Never fails on an inability to measure.
    """

    check_id = "host.validate.rented_pod_resources"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        if not settings.RENTED_POD_RESOURCES_CHECK_ENABLED:
            return CheckResult(
                passed=True, event=render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            )

        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        if rented_executor is not None and rented_executor.miner_hotkey != ctx.miner_hotkey:
            rented_executor = None
        names = [
            pod.docker_container_name
            for pod in (rented_executor.pods if rented_executor else [])
            if pod.status in (None, POD_STATUS_RUNNING)
        ]
        if not names:
            return CheckResult(
                passed=True, event=render_message(Msg.NOT_RENTED, ctx=ctx, check_id=self.check_id)
            )

        try:
            pods = await self._inspect(ctx.ssh, names)
            oom = await self._read_oom(ctx.ssh, [p["pid"] for p in pods])
        except (asyncssh.Error, OSError) as exc:
            return self._unknown(ctx, reason="ssh transport error", details={"error": repr(exc)})
        if not pods:
            return self._unknown(
                ctx, reason="no rented container readable", details={"pods": names}
            )

        host_cpus = advertised_cpu_count(ctx)
        outcomes = [judge_pod(pod, oom.get(pod["pid"]), host_cpus) for pod in pods]
        what = {"executor_uuid": ctx.executor.uuid, "pods": outcomes}
        if any(f in SCORED_FINDINGS for o in outcomes for f in o["findings"]):
            enforce = settings.RENTED_POD_RESOURCES_ENFORCEMENT_ENABLED
            event = render_message(
                Msg.RESOURCES_REDUCED,
                ctx=ctx,
                check_id=self.check_id,
                severity=None if enforce else "warning",
                impact=None if enforce else "Shadow observation only: score was NOT changed",
                what=what,
            )
            updates = {"rented_pod_resources_passed": False} if enforce else {}
            return CheckResult(passed=not enforce, event=event, updates=updates)

        event = render_message(Msg.RESOURCES_OK, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)

    async def _inspect(self, ssh, names: list[str]) -> list[dict]:
        # docker inspect exits 1 when one name is missing but still prints the others
        res = await ssh.run(
            f"/usr/bin/docker inspect --format {shlex.quote(_INSPECT_FORMAT)} "
            + " ".join(shlex.quote(n) for n in names),
            timeout=READ_TIMEOUT_SECONDS,
        )
        pods = []
        for line in (getattr(res, "stdout", "") or "").splitlines():
            try:
                pod = json.loads(line)
            except ValueError:
                continue
            if isinstance(pod, dict) and isinstance(pod.get("pid"), int):
                pods.append(pod)
        return pods

    async def _read_oom(self, ssh, pids: list[int]) -> dict[int, int]:
        pids = [p for p in pids if p > 0]
        if not pids:
            return {}
        res = await ssh.run(
            f"for p in {' '.join(map(str, pids))}; do "
            'echo "$p $(cat /proc/$p/oom_score_adj 2>/dev/null)"; done',
            timeout=READ_TIMEOUT_SECONDS,
        )
        oom = {}
        for line in (getattr(res, "stdout", "") or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
                oom[int(parts[0])] = int(parts[1])
        return oom

    def _unknown(self, ctx: Context, *, reason: str, details: dict | None = None) -> CheckResult:
        event = render_message(
            Msg.RESOURCES_UNKNOWN,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid, "reason": reason, **(details or {})},
        )
        return CheckResult(passed=True, event=event)


def judge_pod(pod: dict, init_oom: int | None, host_cpus: int | None) -> dict:
    labels = pod.get("labels") or {}
    try:
        deployed = int(labels.get(RENTAL_CPU_COUNT_LABEL))
    except (TypeError, ValueError):
        deployed = None
    nano = pod.get("nano") if isinstance(pod.get("nano"), int) else 0
    cpuset = pod.get("cpuset") or ""
    # Entitlement: the deployed --cpus, else the current quota (a pod created before the label),
    # else the whole host (no --cpus, e.g. CVM). None left: any cpuset is foreign, we never set one.
    entitled = deployed or (math.ceil(nano / 1e9) if nano else None) or host_cpus
    cpuset_cpus = _count_cpu_list(cpuset)
    findings = []
    if cpuset_cpus is not None and (entitled is None or cpuset_cpus < entitled):
        findings.append("cpuset_pinned")
    # NanoCpus 0 means no limit, never a cut
    if deployed and 0 < nano < deployed * 1_000_000_000:
        findings.append("cpu_quota_reduced")
    deployed_oom = pod.get("oom")
    if init_oom is not None and isinstance(deployed_oom, int) and init_oom > deployed_oom:
        findings.append("oom_priority_raised")
    return {
        "container": str(pod.get("name") or "").lstrip("/"),
        "cpuset": cpuset,
        "cpuset_cpus": cpuset_cpus,
        "nano_cpus": nano,
        "deployed_cpus": deployed,
        "oom_score_adj_deployed": deployed_oom,
        "oom_score_adj_init": init_oom,
        "findings": findings,
    }
