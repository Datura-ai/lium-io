"""Validator-side GPU model+VRAM pre-check.

Runs before libdmcompverify.so generateChallenge to fail fast on
inconsistent specs. Raises a typed GpuPrecheckError on rejection;
returns None on pass. The .so check remains authoritative for
everything else.
"""
from __future__ import annotations

import logging
from typing import Optional

from .gpu_spec_table import (
    KNOWN_UNRANGED,
    get_expected_vram_range,
    normalize_gpu_model,
)

logger = logging.getLogger(__name__)


class GpuPrecheckError(Exception):
    """Base class for pre-check rejections."""


class UnsupportedGpuModelError(GpuPrecheckError):
    """Normalized model name has no entry in GPU_VRAM_RANGES or KNOWN_UNRANGED."""


class VramRangeMismatchError(GpuPrecheckError):
    """gpu_capacity_mb is outside the expected range for the normalized model."""


class MissingGpuFieldError(GpuPrecheckError):
    """Required machine_info field is missing or non-positive."""


def precheck_gpu_spec(gpu_model: Optional[str], gpu_capacity_mb: int) -> None:
    """Validate GPU model ↔ VRAM consistency.

    Raises:
        MissingGpuFieldError: gpu_model empty or gpu_capacity_mb <= 0.
        UnsupportedGpuModelError: normalized model not in GPU_VRAM_RANGES.
        VramRangeMismatchError: VRAM outside the expected range.

    Returns:
        None on pass (including models in KNOWN_UNRANGED).
    """
    raw = gpu_model or ""
    normalized = normalize_gpu_model(raw)

    if not raw or not isinstance(gpu_capacity_mb, int) or gpu_capacity_mb <= 0:
        raise MissingGpuFieldError(
            f"gpu_model or gpu_capacity_mb missing/zero "
            f"(raw={raw!r}, capacity={gpu_capacity_mb})"
        )

    if normalized in KNOWN_UNRANGED:
        logger.debug("precheck: %r in KNOWN_UNRANGED; passthrough", normalized)
        return

    rng = get_expected_vram_range(normalized)
    if rng is None:
        raise UnsupportedGpuModelError(
            f"no VRAM range for normalized model {normalized!r} (raw={raw!r})"
        )

    vmin, vmax = rng
    if not (vmin <= gpu_capacity_mb <= vmax):
        raise VramRangeMismatchError(
            f"gpu_capacity_mb={gpu_capacity_mb} outside [{vmin},{vmax}] for {normalized!r}"
        )
