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
    GPU_VRAM_SIZES_MB or KNOWN_UNRANGED.
    """
    rates_keys = {k for k in GPU_MODEL_RATES.keys() if k is not None}
    covered = set(gpu_spec_table.GPU_VRAM_SIZES_MB.keys()) | gpu_spec_table.KNOWN_UNRANGED
    missing = sorted(rates_keys - covered)
    assert not missing, (
        f"GPU_MODEL_RATES has {len(missing)} active key(s) with no GPU_VRAM_SIZES_MB "
        f"or KNOWN_UNRANGED entry: {missing}"
    )


def test_gpu_vram_sizes_well_formed():
    """Every model maps to a non-empty list of positive, strictly-increasing
    nominal MB sizes (one per real SKU)."""
    for model, sizes in gpu_spec_table.GPU_VRAM_SIZES_MB.items():
        assert isinstance(sizes, list) and sizes, f"{model}: not a non-empty list"
        prev = None
        for size in sizes:
            assert isinstance(size, int), f"{model}: non-int size {size!r}"
            assert size > 0, f"{model}: non-positive size {size}"
            if prev is not None:
                assert size > prev, f"{model}: sizes not strictly increasing at {size}"
            prev = size


def test_computed_windows_well_formed():
    """Windows computed from the sizes table are non-empty lists of (int, int)
    with vmin <= vmax, both positive, and sorted/non-overlapping."""
    for model in gpu_spec_table.GPU_VRAM_SIZES_MB:
        windows = gpu_spec_table.get_expected_vram_windows(model)
        assert isinstance(windows, list) and windows, f"{model}: not a non-empty list"
        prev_max = None
        for rng in windows:
            assert isinstance(rng, tuple) and len(rng) == 2, f"{model}: window not a 2-tuple"
            vmin, vmax = rng
            assert isinstance(vmin, int) and isinstance(vmax, int), f"{model}: non-int bounds"
            assert vmin > 0 and vmax > 0, f"{model}: non-positive bounds"
            assert vmin <= vmax, f"{model}: vmin {vmin} > vmax {vmax}"
            if prev_max is not None:
                assert vmin > prev_max, f"{model}: windows overlap/unsorted at {rng}"
            prev_max = vmax


# --- Multi-variant: impostor rejection in the inter-variant gap -------------
@pytest.mark.parametrize("model,mb", [
    # Tesla V100 16/32 GB: a 24 GB reading (e.g. RTX 4090) must NOT pass.
    ("NVIDIA Tesla V100 Tensor Core GPU", 24564),
    # RTX 4060 Ti 8/16 GB: a 12 GB reading must NOT pass.
    ("NVIDIA GeForce RTX 4060 Ti", 12030),
    # Tesla M40 12/24 GB: 16 GB and 20 GB readings must NOT pass.
    ("NVIDIA Tesla M40", 16376),
    ("NVIDIA Tesla M40", 20480),
    # RTX A2000 6/12 GB: an 8 GB reading must NOT pass.
    ("NVIDIA RTX A2000", 8192),
    # B200 192 GB only: the ~48 GB MIG/outlier and an 80 GB H100 must NOT pass.
    ("NVIDIA B200", 49140),
    ("NVIDIA B200", 81559),
])
def test_multivariant_intermediate_rejected(model, mb):
    with pytest.raises(VramRangeMismatchError):
        precheck_gpu_spec(model, mb)


@pytest.mark.parametrize("model,mb", [
    # Each real variant of a multi-variant card must pass.
    ("NVIDIA Tesla V100 Tensor Core GPU", 16384),   # 16 GB
    ("NVIDIA Tesla V100 Tensor Core GPU", 32768),   # 32 GB
    ("NVIDIA GeForce RTX 4060 Ti", 8192),           # 8 GB
    ("NVIDIA GeForce RTX 4060 Ti", 16376),          # 16 GB
    ("NVIDIA Tesla M40", 12288),                    # 12 GB
    ("NVIDIA Tesla M40", 24564),                    # 24 GB
    ("NVIDIA RTX PRO 5000 Blackwell", 49140),       # 48 GB
    ("NVIDIA RTX PRO 5000 Blackwell", 73728),       # 72 GB (previously false-rejected)
    ("NVIDIA B200", 183359),                        # 192 GB
])
def test_multivariant_real_variants_pass(model, mb):
    assert precheck_gpu_spec(model, mb) is None


# --- DAH-2160: format-only refactor guards ----------------------------------
# The table now stores nominal MB sizes; windows are derived by
# vram_window_for_size. These tests prove the refactor changes only the storage
# FORMAT, not any pre-check value or range.

@pytest.mark.parametrize("nominal_mb,expected", [
    (81920, (73728, 86016)),    # 80 GB -> H100 window
    (8192, (7373, 8602)),       # 8 GB
    (16384, (14746, 17203)),    # 16 GB
    (196608, (176947, 206438)), # 192 GB -> B200 window
    (11264, (10138, 11827)),    # 11 GB -> RTX 2080 Ti window
])
def test_vram_window_for_size(nominal_mb, expected):
    assert gpu_spec_table.vram_window_for_size(nominal_mb) == expected


def test_vram_window_uses_named_ratios():
    # The ratios are the calibration contract; lock their values.
    assert gpu_spec_table.VRAM_FLOOR_RATIO == 0.90
    assert gpu_spec_table.VRAM_CEIL_RATIO == 1.05


def test_get_expected_vram_windows_from_sizes():
    # Single-SKU and multi-variant models both derive windows from their sizes.
    assert gpu_spec_table.get_expected_vram_windows("NVIDIA H100 80GB HBM3") == [(73728, 86016)]
    assert gpu_spec_table.get_expected_vram_windows("NVIDIA GeForce RTX 4060 Ti") == [
        (7373, 8602), (14746, 17203),
    ]
    # Unknown model -> None (unchanged contract).
    assert gpu_spec_table.get_expected_vram_windows("NVIDIA Foo Bar") is None


# Frozen snapshot of the (vram_mb_min, vram_mb_max) windows BEFORE the DAH-2160
# refactor — copied VERBATIM from the pre-refactor GPU_VRAM_RANGES dict (the
# hand-stored tuples), NOT recomputed from the new sizes table. This makes the
# test an independent regression guard: the computed windows MUST match these
# exact values for every model, proving no pre-check value or range changed.
_EXPECTED_WINDOWS: dict[str, list[tuple[int, int]]] = {
    "NVIDIA B300 SXM6 AC": [(265421, 309658)],
    "NVIDIA B200": [(176947, 206438)],
    "NVIDIA H200": [(129946, 151603)],
    "NVIDIA H200 NVL": [(129946, 151603)],
    "NVIDIA H100 80GB HBM3": [(73728, 86016)],
    "NVIDIA H100 NVL": [(86630, 101069)],
    "NVIDIA H100 PCIe": [(73728, 86016)],
    "NVIDIA H800 80GB HBM3": [(73728, 86016)],
    "NVIDIA H800 NVL": [(86630, 101069)],
    "NVIDIA H800 PCIe": [(73728, 86016)],
    "NVIDIA GeForce RTX 5090": [(29491, 34406)],
    "NVIDIA GeForce RTX 5080": [(14746, 17203)],
    "NVIDIA GeForce RTX 5070 Ti": [(14746, 17203)],
    "NVIDIA GeForce RTX 5070": [(11059, 12902)],
    "NVIDIA GeForce RTX 5060 Ti": [(7373, 8602), (14746, 17203)],
    "NVIDIA GeForce RTX 5060": [(7373, 8602)],
    "NVIDIA GeForce RTX 4090": [(22118, 25805)],
    "NVIDIA GeForce RTX 4090 D": [(22118, 25805)],
    "NVIDIA GeForce RTX 4080 SUPER": [(14746, 17203)],
    "NVIDIA GeForce RTX 4080": [(14746, 17203)],
    "NVIDIA GeForce RTX 4070 Ti SUPER": [(14746, 17203)],
    "NVIDIA GeForce RTX 4070 Ti": [(11059, 12902)],
    "NVIDIA GeForce RTX 4070 SUPER": [(11059, 12902)],
    "NVIDIA GeForce RTX 4070": [(11059, 12902)],
    "NVIDIA GeForce RTX 4060 Ti": [(7373, 8602), (14746, 17203)],
    "NVIDIA GeForce RTX 4060": [(7373, 8602)],
    "NVIDIA RTX PRO 4000 Blackwell": [(22118, 25805)],
    "NVIDIA RTX PRO 5000 Blackwell": [(44237, 51610), (66355, 77414)],
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": [(88474, 103219)],
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": [(88474, 103219)],
    "NVIDIA RTX 4500 Ada Generation": [(22118, 25805)],
    "NVIDIA RTX 5000 Ada Generation": [(29491, 34406)],
    "NVIDIA RTX 5880 Ada Generation": [(44237, 51610)],
    "NVIDIA RTX 6000 Ada Generation": [(44237, 51610)],
    "NVIDIA L4": [(22118, 25805)],
    "NVIDIA L40S": [(44237, 51610)],
    "NVIDIA L40": [(44237, 51610)],
    "NVIDIA A100 80GB PCIe": [(73728, 86016)],
    "NVIDIA A100-SXM4-80GB": [(73728, 86016)],
    "NVIDIA A10 Tensor Core GPU": [(22118, 25805)],
    "NVIDIA RTX A6000": [(44237, 51610)],
    "NVIDIA RTX A5000": [(22118, 25805)],
    "NVIDIA RTX A4500": [(18432, 21504)],
    "NVIDIA RTX A4000": [(14746, 17203)],
    "NVIDIA RTX A2000": [(5530, 6451), (11059, 12902)],
    "NVIDIA T4 Tensor Core GPU": [(14746, 17203)],
    "NVIDIA Tesla V100 Tensor Core GPU": [(14746, 17203), (29491, 34406)],
    "NVIDIA Quadro RTX 8000": [(44237, 51610)],
    "NVIDIA Quadro RTX 6000": [(22118, 25805)],
    "NVIDIA Quadro RTX 5000": [(14746, 17203)],
    "NVIDIA TITAN V": [(11059, 12902)],
    "NVIDIA TITAN RTX": [(22118, 25805)],
    "NVIDIA GeForce RTX 3090 Ti": [(22118, 25805)],
    "NVIDIA GeForce RTX 3090": [(22118, 25805)],
    "NVIDIA GeForce RTX 3080 Ti": [(11059, 12902)],
    "NVIDIA GeForce RTX 3080": [(9216, 10752), (11059, 12902)],
    "NVIDIA GeForce RTX 3070 Ti": [(7373, 8602)],
    "NVIDIA GeForce RTX 3070": [(7373, 8602)],
    "NVIDIA GeForce RTX 3060 Ti": [(7373, 8602)],
    "NVIDIA GeForce RTX 3060": [(7373, 8602), (11059, 12902)],
    "NVIDIA GeForce RTX 3060 Laptop GPU": [(5530, 6451)],
    "NVIDIA GeForce RTX 3050": [(5530, 6451), (7373, 8602)],
    "NVIDIA GeForce RTX 2080 Ti": [(10138, 11827)],
    "NVIDIA GeForce RTX 2080 SUPER": [(7373, 8602)],
    "NVIDIA GeForce RTX 2070 SUPER": [(7373, 8602)],
    "NVIDIA GeForce RTX 2060 SUPER": [(7373, 8602)],
    "NVIDIA GeForce RTX 2060": [(5530, 6451), (11059, 12902)],
    "NVIDIA GeForce GTX 1660 Ti": [(5530, 6451)],
    "NVIDIA GeForce GTX 1660 SUPER": [(5530, 6451)],
    "NVIDIA GeForce GTX 1660": [(5530, 6451)],
    "NVIDIA Tesla P100": [(11059, 12902), (14746, 17203)],
    "NVIDIA Tesla P40": [(22118, 25805)],
    "NVIDIA Quadro P4000": [(7373, 8602)],
    "NVIDIA TITAN Xp": [(11059, 12902)],
    "NVIDIA GeForce GTX 1080 Ti": [(10138, 11827)],
    "NVIDIA GeForce GTX 1080": [(7373, 8602)],
    "NVIDIA GeForce GTX 1070 Ti": [(7373, 8602)],
    "NVIDIA GeForce GTX 1070": [(7373, 8602)],
    "NVIDIA GeForce GTX 1060": [(2765, 3226), (5530, 6451)],
    "NVIDIA Tesla M40": [(11059, 12902), (22118, 25805)],
}


def test_computed_windows_match_pre_refactor_snapshot():
    """Every model's computed windows MUST equal the pre-refactor hand-stored
    windows — the refactor changes storage format only, never a pre-check range."""
    # No model added or dropped relative to the frozen snapshot.
    assert set(gpu_spec_table.GPU_VRAM_SIZES_MB) == set(_EXPECTED_WINDOWS)
    for model, expected in _EXPECTED_WINDOWS.items():
        assert gpu_spec_table.get_expected_vram_windows(model) == expected, model
