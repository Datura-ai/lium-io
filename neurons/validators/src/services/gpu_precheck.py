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
    get_expected_vram_windows,
    normalize_gpu_model,
)

logger = logging.getLogger(__name__)


class GpuPrecheckError(Exception):
    """Base class for pre-check rejections."""


class UnsupportedGpuModelError(GpuPrecheckError):
    """Normalized model name has no entry in GPU_VRAM_RANGES or KNOWN_UNRANGED."""


class VramRangeMismatchError(GpuPrecheckError):
    """gpu_capacity_mb falls in none of the expected windows for the normalized model."""


class MissingGpuFieldError(GpuPrecheckError):
    """Required machine_info field is missing or non-positive."""


def precheck_gpu_spec(gpu_model: Optional[str], gpu_capacity_mb: int) -> None:
    """Validate GPU model ↔ VRAM consistency.

    Raises:
        MissingGpuFieldError: gpu_model empty or gpu_capacity_mb <= 0.
        UnsupportedGpuModelError: normalized model not in GPU_VRAM_RANGES.
        VramRangeMismatchError: VRAM falls in none of the expected windows.

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

    windows = get_expected_vram_windows(normalized)
    if windows is None:
        raise UnsupportedGpuModelError(
            f"no VRAM windows for normalized model {normalized!r} (raw={raw!r})"
        )

    if not any(vmin <= gpu_capacity_mb <= vmax for vmin, vmax in windows):
        raise VramRangeMismatchError(
            f"gpu_capacity_mb={gpu_capacity_mb} in none of {windows} for {normalized!r}"
        )
