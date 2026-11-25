"""DAH-2621 RAM/VRAM rule, shared by the idle-pay gate (rental_price) and onboarding (SpecFloorCheck)."""

from __future__ import annotations

from pydantic import BaseModel

# A machine whose system RAM is below this share of its summed GPU VRAM cannot stage a model into its GPUs.
# Half, not all of it: stock 8x B200/B300 boards carry 0.9x and real offenders sit far below (DAH-2621 fleet data).
MIN_RAM_TO_VRAM_RATE = 0.5
# MemTotal reports a few percent less than the installed RAM (24 GB installed reads 23.5 GB)
RAM_MEMTOTAL_TOLERANCE = 1.05


class InsufficientRam(BaseModel):
    """Measured totals of a machine whose system RAM is below the required share of its GPU VRAM."""

    vram_gb: float
    ram_gb: float
    rate: float  # required RAM-to-VRAM share the machine failed to clear
    required_ram_gb: float  # the RAM reading that clears it


def ram_and_vram_gb(spec: dict) -> tuple[float, float]:
    """(system RAM, summed GPU VRAM) in GB from a scrape; raises on an unreadable one."""
    gpu_details = (spec.get("gpu") or {}).get("details") or []
    vram_gb = sum(float(gpu.get("capacity") or 0) for gpu in gpu_details) / 1024  # capacity is MB
    ram_gb = float((spec.get("ram") or {}).get("total") or 0) / 1024**2  # MemTotal, kB
    return ram_gb, vram_gb


def insufficient_ram(ram_gb: float, vram_gb: float) -> InsufficientRam | None:
    """None when the RAM clears the rule; callers handle unmeasured (zero) readings first."""
    if ram_gb * RAM_MEMTOTAL_TOLERANCE >= vram_gb * MIN_RAM_TO_VRAM_RATE:
        return None
    required_ram_gb = round(vram_gb * MIN_RAM_TO_VRAM_RATE / RAM_MEMTOTAL_TOLERANCE, 1)
    if required_ram_gb * RAM_MEMTOTAL_TOLERANCE < vram_gb * MIN_RAM_TO_VRAM_RATE:
        # nearest rounding went below the floor: name the next reading the check above accepts
        required_ram_gb = round(required_ram_gb + 0.1, 1)
    # rounded only for the report: the check above must use the raw readings
    return InsufficientRam(
        vram_gb=round(vram_gb, 1),
        ram_gb=round(ram_gb, 1),
        rate=MIN_RAM_TO_VRAM_RATE,
        required_ram_gb=required_ram_gb,
    )
