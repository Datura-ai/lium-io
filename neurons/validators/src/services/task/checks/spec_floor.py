from __future__ import annotations

import re

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
# the scrape hashes the docker, nvidia-smi and libnvidia-ml files as "<md5>:<sha256>"
_FILE_CHECKSUM = re.compile(r"^[0-9a-f]{32}:[0-9a-f]{64}$")
# md5 and sha256 of zero bytes: none of those files is empty on a real host
_EMPTY_INPUT_DIGESTS = {
    "d41d8cd98f00b204e9800998ecf8427e",
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


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
    checksums = specs.get("md5_checksums")
    # absent as a whole, the NVML digest check fails the node on its missing libnvidia-ml digest
    if checksums is not None and not _file_checksums_plausible(checksums):
        findings.append("file_checksums")
    return findings


def _file_checksums_plausible(checksums) -> bool:
    if not isinstance(checksums, dict):
        return False
    for name in ("docker", "nvidia_smi", "libnvidia_ml"):
        value = checksums.get(name)
        if not isinstance(value, str) or not _FILE_CHECKSUM.match(value):
            return False
        if _EMPTY_INPUT_DIGESTS & set(value.split(":")):
            return False
    return True


class SpecFloorCheck:
    """Hard floors on the scraped specs: CPU cores, RAM per GPU, GPU count against the listed cards, GPU UUID
    format, the file checksums' shape (none of an empty file). Pure data, before the rented short-circuit. Fatal on an idle node; under a live rental the run goes on
    to the rented-pod checks and the score is zeroed through spec_floor_passed."""

    check_id = "host.validate.spec_floor"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        findings = spec_floor_findings(ctx.state.specs or {})
        if not findings:
            event = render_message(Msg.OK, ctx=ctx, check_id=self.check_id, what={})
            return CheckResult(passed=True, event=event)
        event = render_message(Msg.REJECTED, ctx=ctx, check_id=self.check_id, what={"findings": findings})
        if not has_live_rental(ctx):
            return CheckResult(passed=False, event=event)
        return CheckResult(
            passed=False, event=event, updates={"spec_floor_passed": False}, fatal=False
        )
