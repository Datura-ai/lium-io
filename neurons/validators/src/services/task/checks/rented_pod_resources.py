from __future__ import annotations

import json
import math
import shlex
from dataclasses import asdict, dataclass

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
    '"nano":{{json .HostConfig.NanoCpus}},"quota":{{json .HostConfig.CpuQuota}},'
    '"period":{{json .HostConfig.CpuPeriod}},"oom":{{json .HostConfig.OomScoreAdj}},'
    '"pid":{{json .State.Pid}},"labels":{{json .Config.Labels}}}'
)
# The failing findings; oom_priority_raised is recorded only, a root renter can raise it themselves.
SCORED_FINDINGS = ("cpuset_pinned", "cpu_quota_reduced")
# Owner (19 Oct): a host may keep a little CPU back, but the rental must keep at least this share of
# its advertised cores, in both the cpuset and the CPU quota.
MIN_RENTAL_CPU_SHARE = 0.95
_DEFAULT_CPU_PERIOD_US = 100_000
# Per init PID: oom_score_adj, then (cgroup v2) the effective cpuset and every cpu.max from the pod's
# cgroup up to the root, so a host that writes the cgroup directly (not via docker update) is seen too.
_INIT_READ_SCRIPT = (
    "for p in {pids}; do "
    "o=$(cat /proc/$p/oom_score_adj 2>/dev/null); cs=; cm=; "
    "g=$(awk -F: '$1==\"0\"{{print $3}}' /proc/$p/cgroup 2>/dev/null); "
    'if [ -n "$g" ] && [ -f /sys/fs/cgroup/cgroup.controllers ]; then '
    'd="/sys/fs/cgroup$g"; cs=$(cat "$d/cpuset.cpus.effective" 2>/dev/null); '
    'while [ "$d" != /sys/fs/cgroup ] && [ -n "$d" ]; do '
    'm=$(cat "$d/cpu.max" 2>/dev/null) && cm="$cm;$m"; d="${{d%/*}}"; done; fi; '
    'echo "$p|$o|$cs|$cm"; done'
)


@dataclass(frozen=True)
class InspectedPod:
    """What `docker inspect` recorded for one rented container, plus its init PID's live reads."""

    name: str
    cpuset: str
    nano_cpus: int
    cpu_quota: int
    cpu_period: int
    oom_score_adj_deployed: int | None
    pid: int
    labels: dict
    init_oom_score_adj: int | None = None
    cgroup_cpuset: str = ""
    cgroup_quota_cpus: float | None = None


@dataclass(frozen=True)
class InitReading:
    oom_score_adj: int | None
    cgroup_cpuset: str
    cgroup_quota_cpus: float | None


@dataclass(frozen=True)
class PodVerdict:
    container: str
    cpuset: str
    cpuset_cpus: int | None
    nano_cpus: int
    cpu_quota: int
    cpu_period: int
    quota_cpus: float | None
    cgroup_cpuset_cpus: int | None
    cgroup_quota_cpus: float | None
    deployed_cpus: int | None
    min_cpus: float | None
    oom_score_adj_deployed: int | None
    oom_score_adj_init: int | None
    findings: list[str]


class RentedPodResourcesCheck:
    """Read each running rented pod's HostConfig and its init PID's oom_score_adj and effective cgroup
    cpuset/cpu.max, and flag a host that cut the renter below MIN_RENTAL_CPU_SHARE of its advertised
    cores (cpuset or quota), by `docker update` or by writing the cgroup directly.

    The validator never sets a cpuset and records the --cpus it deployed in an immutable label
    (RENTAL_CPU_COUNT_LABEL). Non-fatal: a cpuset/quota finding fails the check and zeroes the score.
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
            init_reads_by_pid = await self._read_init(ctx.ssh, [p.pid for p in pods])
        except (asyncssh.Error, OSError) as exc:
            return self._unknown(ctx, reason="ssh transport error", details={"error": repr(exc)})
        if not pods:
            return self._unknown(
                ctx, reason="no rented container readable", details={"pods": names}
            )

        host_cpus = advertised_cpu_count(ctx)
        outcomes = [
            judge_pod(_with_init(pod, init_reads_by_pid.get(pod.pid)), host_cpus) for pod in pods
        ]
        what = {"executor_uuid": ctx.executor.uuid, "pods": [asdict(o) for o in outcomes]}
        if any(f in SCORED_FINDINGS for o in outcomes for f in o.findings):
            event = render_message(Msg.RESOURCES_REDUCED, ctx=ctx, check_id=self.check_id, what=what)
            return CheckResult(passed=False, event=event, updates={"rented_pod_resources_passed": False})

        event = render_message(Msg.RESOURCES_OK, ctx=ctx, check_id=self.check_id, what=what)
        return CheckResult(passed=True, event=event)

    async def _inspect(self, ssh, names: list[str]) -> list[InspectedPod]:
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
                pods.append(_inspected(pod))
        return pods

    async def _read_init(self, ssh, pids: list[int]) -> dict[int, InitReading]:
        pids = [p for p in pids if p > 0]
        if not pids:
            return {}
        res = await ssh.run(
            _INIT_READ_SCRIPT.format(pids=" ".join(map(str, pids))), timeout=READ_TIMEOUT_SECONDS
        )
        reads = {}
        for line in (getattr(res, "stdout", "") or "").splitlines():
            parts = line.strip().split("|")
            if len(parts) != 4 or not parts[0].isdigit():
                continue
            oom = int(parts[1]) if parts[1].lstrip("-").isdigit() else None
            reads[int(parts[0])] = InitReading(oom, parts[2].strip(), _cpu_max_cpus(parts[3]))
        return reads

    def _unknown(self, ctx: Context, *, reason: str, details: dict | None = None) -> CheckResult:
        event = render_message(
            Msg.RESOURCES_UNKNOWN,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid, "reason": reason, **(details or {})},
        )
        return CheckResult(passed=True, event=event)


def _int(value) -> int | None:
    return value if isinstance(value, int) else None


def _inspected(raw: dict) -> InspectedPod:
    period = _int(raw.get("period")) or 0
    return InspectedPod(
        name=str(raw.get("name") or "").lstrip("/"),
        cpuset=raw.get("cpuset") or "",
        nano_cpus=_int(raw.get("nano")) or 0,
        cpu_quota=_int(raw.get("quota")) or 0,
        cpu_period=period if period > 0 else 0,
        oom_score_adj_deployed=_int(raw.get("oom")),
        pid=raw["pid"],
        labels=raw.get("labels") or {},
    )


def _with_init(pod: InspectedPod, read: InitReading | None) -> InspectedPod:
    if read is None:
        return pod
    return InspectedPod(
        **{
            **asdict(pod),
            "init_oom_score_adj": read.oom_score_adj,
            "cgroup_cpuset": read.cgroup_cpuset,
            "cgroup_quota_cpus": read.cgroup_quota_cpus,
        }
    )


def _cpu_max_cpus(chain: str) -> float | None:
    """The tightest `cpu.max` ("<quota> <period>" or "max <period>") on the pod's cgroup path, in CPUs."""
    limits = []
    for entry in chain.split(";"):
        parts = entry.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) > 0:
            limits.append(int(parts[0]) / int(parts[1]))
    return min(limits) if limits else None


def judge_pod(pod: InspectedPod, host_cpus: int | None) -> PodVerdict:
    try:
        deployed_cpus = int(pod.labels.get(RENTAL_CPU_COUNT_LABEL))
    except (TypeError, ValueError):
        deployed_cpus = None
    nano, quota, period = pod.nano_cpus, pod.cpu_quota, pod.cpu_period
    # CPUs the quota allows: NanoCpus (--cpus) or CpuQuota/CpuPeriod (--cpu-quota); 0 or -1 = unlimited
    limits = [nano / 1e9] if nano > 0 else []
    if quota > 0:
        limits.append(quota / (period or _DEFAULT_CPU_PERIOD_US))
    quota_cpus = min(limits) if limits else None
    cpuset = pod.cpuset
    # Advertised cores: the deployed --cpus, else the current quota (a pod created before the label),
    # else the whole host (no --cpus, e.g. CVM). None left: any cpuset is foreign, we never set one.
    entitled = deployed_cpus or (math.ceil(quota_cpus) if quota_cpus else None) or host_cpus
    floor = MIN_RENTAL_CPU_SHARE * entitled if entitled else None
    cpuset_cpus = _count_cpu_list(cpuset)
    # the kernel's view: a host can narrow the cgroup itself, leaving HostConfig untouched
    cgroup_cpuset_cpus = _count_cpu_list(pod.cgroup_cpuset) if pod.cgroup_cpuset else None
    below = floor - 1e-9 if floor is not None else None
    findings = []
    if (cpuset_cpus is not None and (below is None or cpuset_cpus < below)) or (
        below is not None and cgroup_cpuset_cpus is not None and cgroup_cpuset_cpus < below
    ):
        findings.append("cpuset_pinned")
    if below is not None and any(
        q is not None and q < below for q in (quota_cpus, pod.cgroup_quota_cpus)
    ):
        findings.append("cpu_quota_reduced")
    init_oom = pod.init_oom_score_adj
    if (
        init_oom is not None
        and pod.oom_score_adj_deployed is not None
        and init_oom > pod.oom_score_adj_deployed
    ):
        findings.append("oom_priority_raised")
    return PodVerdict(
        container=pod.name,
        cpuset=cpuset,
        cpuset_cpus=cpuset_cpus,
        nano_cpus=nano,
        cpu_quota=quota,
        cpu_period=period,
        quota_cpus=quota_cpus,
        cgroup_cpuset_cpus=cgroup_cpuset_cpus,
        cgroup_quota_cpus=pod.cgroup_quota_cpus,
        deployed_cpus=deployed_cpus,
        min_cpus=round(floor, 2) if floor else None,
        oom_score_adj_deployed=pod.oom_score_adj_deployed,
        oom_score_adj_init=init_oom,
        findings=findings,
    )
