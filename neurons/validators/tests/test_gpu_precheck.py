"""Unit + parity tests for services.gpu_precheck and services.gpu_spec_table."""
from __future__ import annotations

import pytest

from services import gpu_spec_table
from services.const import GPU_MODEL_RATES
from services.gpu_precheck import (
    GpuPrecheckError,
    MissingGpuFieldError,
    UnsupportedGpuModelError,
    VramRangeMismatchError,
    precheck_gpu_spec,
)


# --- Happy path --------------------------------------------------------------
def test_happy_path_returns_none():
    assert precheck_gpu_spec("NVIDIA H100 80GB HBM3", 81920) is None


def test_normalization_then_pass():
    # Tesla V100-SXM2-32GB normalizes to "NVIDIA Tesla V100 Tensor Core GPU";
    # range covers 16/32GB variants.
    assert precheck_gpu_spec("Tesla V100-SXM2-32GB", 32768) is None


# --- Missing field ----------------------------------------------------------
@pytest.mark.parametrize("model,mb", [
    ("", 81920),
    (None, 81920),
    ("NVIDIA H100 80GB HBM3", 0),
    ("NVIDIA H100 80GB HBM3", -1),
])
def test_missing_field_raises(model, mb):
    with pytest.raises(MissingGpuFieldError):
        precheck_gpu_spec(model, mb)


# --- Unknown model ----------------------------------------------------------
def test_unknown_model_raises():
    with pytest.raises(UnsupportedGpuModelError):
        precheck_gpu_spec("NVIDIA Foo Bar", 12288)


# --- VRAM range checks ------------------------------------------------------
@pytest.mark.parametrize("model,mb,expected_ok", [
    # H100 80GB HBM3 range: (73728, 86016)
    ("NVIDIA H100 80GB HBM3", 73728, True),   # vmin inclusive
    ("NVIDIA H100 80GB HBM3", 73727, False),  # just below vmin
    ("NVIDIA H100 80GB HBM3", 86016, True),   # vmax inclusive
    ("NVIDIA H100 80GB HBM3", 86017, False),  # just above vmax
    ("NVIDIA H100 80GB HBM3", 81559, True),   # prod-observed
    # RTX 4090 range: (22118, 25805)
    ("NVIDIA GeForce RTX 4090", 22118, True), # vmin inclusive
    ("NVIDIA GeForce RTX 4090", 22117, False),# just below vmin
    ("NVIDIA GeForce RTX 4090", 23028, True), # prod-observed low end
    ("NVIDIA GeForce RTX 4090", 25805, True), # vmax inclusive
    # Wildly off should still fail
    ("NVIDIA H100 80GB HBM3", 24064, False),
])
def test_vram_boundaries(model, mb, expected_ok):
    if expected_ok:
        assert precheck_gpu_spec(model, mb) is None
    else:
        with pytest.raises(VramRangeMismatchError):
            precheck_gpu_spec(model, mb)


# --- Exception hierarchy ----------------------------------------------------
def test_typed_errors_inherit_base():
    """All typed exceptions are catchable as GpuPrecheckError."""
    with pytest.raises(GpuPrecheckError):
        precheck_gpu_spec("", 0)
    with pytest.raises(GpuPrecheckError):
        precheck_gpu_spec("Unknown Foo", 12288)
    with pytest.raises(GpuPrecheckError):
        precheck_gpu_spec("NVIDIA H100 80GB HBM3", 1)


# --- KNOWN_UNRANGED ---------------------------------------------------------
def test_known_unranged_passthrough(monkeypatch):
    import services.gpu_precheck as precheck_mod
    monkeypatch.setattr(precheck_mod, "KNOWN_UNRANGED", {"NVIDIA Test Unranged"})
    # Even with a wildly off VRAM, KNOWN_UNRANGED passes through.
    assert precheck_gpu_spec("NVIDIA Test Unranged", 1) is None


# --- Normalization helpers --------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("Tesla V100-SXM2-32GB", "NVIDIA Tesla V100 Tensor Core GPU"),
    ("Tesla V100-PCIE-16GB", "NVIDIA Tesla V100 Tensor Core GPU"),
    ("Tesla H100 80GB HBM3", "NVIDIA H100 80GB HBM3"),
    ("NVIDIA T4", "NVIDIA T4 Tensor Core GPU"),
    ("NVIDIA A10", "NVIDIA A10 Tensor Core GPU"),
])
def test_normalization_lookup(raw, expected):
    assert gpu_spec_table.normalize_gpu_model(raw) == expected


def test_normalize_strips_whitespace():
    assert gpu_spec_table.normalize_gpu_model("  NVIDIA H100 80GB HBM3  ") == "NVIDIA H100 80GB HBM3"


def test_normalize_handles_none_and_empty():
    assert gpu_spec_table.normalize_gpu_model(None) == ""
    assert gpu_spec_table.normalize_gpu_model("") == ""


def test_normalize_passthrough_unmapped():
    assert gpu_spec_table.normalize_gpu_model("Future GPU XYZ") == "Future GPU XYZ"


# --- CI parity --------------------------------------------------------------
def test_gpu_model_rates_parity():
    """Every active key in const.GPU_MODEL_RATES MUST be covered by either
    GPU_VRAM_RANGES or KNOWN_UNRANGED.
    """
    rates_keys = {k for k in GPU_MODEL_RATES.keys() if k is not None}
    covered = set(gpu_spec_table.GPU_VRAM_RANGES.keys()) | gpu_spec_table.KNOWN_UNRANGED
    missing = sorted(rates_keys - covered)
    assert not missing, (
        f"GPU_MODEL_RATES has {len(missing)} active key(s) with no GPU_VRAM_RANGES "
        f"or KNOWN_UNRANGED entry: {missing}"
    )


def test_gpu_vram_ranges_well_formed():
    """Every range tuple is (int, int) with vmin <= vmax and both positive."""
    for model, rng in gpu_spec_table.GPU_VRAM_RANGES.items():
        assert isinstance(rng, tuple) and len(rng) == 2, f"{model}: not a 2-tuple"
        vmin, vmax = rng
        assert isinstance(vmin, int) and isinstance(vmax, int), f"{model}: non-int bounds"
        assert vmin > 0 and vmax > 0, f"{model}: non-positive bounds"
        assert vmin <= vmax, f"{model}: vmin {vmin} > vmax {vmax}"
