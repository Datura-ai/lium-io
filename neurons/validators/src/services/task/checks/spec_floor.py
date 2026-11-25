from __future__ import annotations

import re

from incentive.ram_vram import insufficient_ram, ram_and_vram_gb
from services.redis_service import GPU_ANCHOR_KEY

from ..messages import SpecFloorMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .banned_provider import has_live_rental

# Far below every healthy listed shape; a node under them reports specs no real GPU host has.
MIN_RAM_GB_PER_GPU = 4
KIB_PER_GB = 1024 * 1024
_GPU_UUID = re.compile(
    r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
_ZERO_UUID = "GPU-00000000-0000-0000-0000-000000000000"


def _int(value) -> int:
    # Untrusted JSON: Infinity raises OverflowError, NaN ValueError; both read as invalid (0).
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def spec_floor_findings(specs: dict) -> list[str]:
    cpu, ram, gpu = (specs.get(k) or {} for k in ("cpu", "ram", "gpu"))
    gpu_count = _int(gpu.get("count"))
    uuids = [str(d.get("uuid") or "") for d in gpu.get("details") or []]
    findings = []
    if _int(cpu.get("count")) <= 0:
        findings.append("cpu_count")
    if gpu_count and _int(ram.get("total")) < MIN_RAM_GB_PER_GPU * KIB_PER_GB * gpu_count:
        findings.append("ram_per_gpu")
    if gpu_count != len(uuids):
        findings.append("gpu_count_vs_cards")
    # One card listed twice is a known scrape glitch on healthy nodes; the kernel-count check
    # already fails a node whose distinct card set is short.
    if any(not _GPU_UUID.match(u) or u.upper() == _ZERO_UUID for u in uuids):
        findings.append("gpu_uuid_format")
    return findings


def new_node_ram_shortfall(ctx: Context) -> dict | None:
    """DAH-2621: a node never verified before (no GPU anchor) and not rented, whose RAM is below the idle-pay
    RAM/VRAM rule, is not onboarded. A node that passed before keeps the idle-pay gate only. Unreadable or missing
    RAM/VRAM fails open, as the idle-pay gate does."""
    if (ctx.verified or {}).get(GPU_ANCHOR_KEY) or has_live_rental(ctx):
        return None
    try:
        ram_gb, vram_gb = ram_and_vram_gb(ctx.state.specs or {})
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if not (vram_gb > 0 and ram_gb > 0):
        return None
    shortfall = insufficient_ram(ram_gb, vram_gb)
    return shortfall.model_dump() if shortfall else None


class SpecFloorCheck:
    """Hard floors on the scraped specs: CPU cores, RAM per GPU, GPU count against the listed cards, GPU UUID
    format, and for a new node RAM of at least half its GPU VRAM. Pure data, before the rented short-circuit.
    Fatal on an idle node; under a live rental the run goes on to the rented-pod checks and the score is zeroed
    through spec_floor_passed."""

    check_id = "host.validate.spec_floor"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        findings = spec_floor_findings(ctx.state.specs or {})
        what: dict = {}
        remediation = None
        if ram_shortfall := new_node_ram_shortfall(ctx):
            findings.append("ram_below_half_vram")
            what = ram_shortfall
            remediation = Msg.REJECTED.remediation + Msg.RAM_BELOW_HALF_VRAM_REMEDIATION.format(**ram_shortfall)
        if not findings:
            event = render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what={})
            return CheckResult(passed=True, event=event)
        event = render_message(
            Msg.REJECTED,
            ctx=ctx,
            check_id=self.check_id,
            what={"findings": findings, **what},
            remediation=remediation,
        )
        if not has_live_rental(ctx):
            return CheckResult(passed=False, event=event)
        return CheckResult(
            passed=False, event=event, updates={"spec_floor_passed": False}, fatal=False
        )
