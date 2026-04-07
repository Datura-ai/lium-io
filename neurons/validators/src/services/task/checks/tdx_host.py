from __future__ import annotations

import re
from dataclasses import replace

from ..messages import TdxHostMessages as Msg, render_message
from ..pipeline import CheckResult, Context

# CPU model name patterns for TDX-capable Intel processors.
# Source: Intel TDX Enabling Guide — supported CPU families at silicon level,
# independent of BIOS enablement or kernel module state.
#
#  • 4th/5th Gen Xeon Scalable (Sapphire + Emerald Rapids):
#      Platinum 8400-8999, Gold 6400-6999 series
#  • Intel Xeon 6 (Granite Rapids P-core + Sierra Forest E-core):
#      "Xeon 6NNNx" naming convention
# Separator pattern that tolerates the "(R)" trademark token between words.
# Real lscpu output uses "Xeon(R) Platinum" rather than "Xeon Platinum".
_SEP = r"(?:\s*\(R\))?\s+"

_TDX_CAPABLE_PATTERNS: list[re.Pattern] = [
    # Intel Xeon 6 series (Granite Rapids P-core + Sierra Forest E-core),
    # e.g. "Intel(R) Xeon(R) 6980P", "Intel(R) Xeon(R) 6766E".
    re.compile(r"Xeon" + _SEP + r"6\d{3}", re.IGNORECASE),
    # 4th/5th Gen Xeon Scalable — Platinum 8400+,
    # e.g. "Intel(R) Xeon(R) Platinum 8592+", "Intel(R) Xeon(R) Platinum 8490H".
    re.compile(r"Xeon" + _SEP + r"Platinum" + _SEP + r"8[4-9]\d{2}", re.IGNORECASE),
    # 4th/5th Gen Xeon Scalable — Gold 6400+,
    # e.g. "Intel(R) Xeon(R) Gold 6442Y", "Intel(R) Xeon(R) Gold 6554S".
    re.compile(r"Xeon" + _SEP + r"Gold" + _SEP + r"6[4-9]\d{2}", re.IGNORECASE),
    # 4th/5th Gen Xeon W, e.g. "Intel(R) Xeon(R) w9-3496X".
    re.compile(r"Xeon" + _SEP + r"W[579]-[23][4-9]\d{2}", re.IGNORECASE),
]


class TdxHostCheck:
    """Determine whether the executor's CPU silicon supports Intel TDX.

    Reads the CPU model already collected by MachineSpecScrapeCheck from
    specs["cpu"]["model"] — no additional SSH commands are issued.

    This is a silicon-level hardware property: it is True if the CPU belongs
    to a TDX-capable generation (4th/5th Gen Xeon Scalable or Intel Xeon 6),
    regardless of how BIOS or the kernel is configured.

    Always returns passed=True and records tdx_host_supported in specs.
    """

    check_id = "host.detect.tdx"
    fatal = False

    @staticmethod
    def is_tdx_capable(cpu_model: str) -> bool:
        """Return True if the CPU model name matches a known TDX-capable Intel processor."""
        return any(p.search(cpu_model) for p in _TDX_CAPABLE_PATTERNS)

    async def run(self, ctx: Context) -> CheckResult:
        cpu_info = ctx.state.specs.get("cpu") or {}
        cpu_model = cpu_info.get("model") or ""

        tdx_supported = self.is_tdx_capable(cpu_model)

        updated_specs = {**ctx.state.specs, "tdx_host_supported": tdx_supported}
        updated_state = replace(ctx.state, specs=updated_specs)

        msg_template = Msg.TDX_SUPPORTED if tdx_supported else Msg.TDX_NOT_SUPPORTED
        event = render_message(
            msg_template,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "cpu_model": cpu_model,
                "tdx_host_supported": tdx_supported,
            },
        )
        return CheckResult(passed=True, event=event, updates={"state": updated_state})
