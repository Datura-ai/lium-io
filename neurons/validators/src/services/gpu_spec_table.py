"""GPU model -> VRAM size table for validator-side pre-check.

# SYNC: This table mirrors compile-time data inside libdmcompverify.so.
# libdmcompverify is LIEF-obfuscated; its internal VRAM table cannot be
# extracted programmatically, so parity is enforced via a CI test against
# const.GPU_MODEL_RATES (see tests/test_gpu_precheck.py::test_gpu_model_rates_parity).
# Whenever a model is added to GPU_MODEL_RATES, it MUST be added here (with
# one or more nominal VRAM sizes) or to KNOWN_UNRANGED (intentional passthrough).

Source of truth:
    - Each model maps to a LIST of nominal VRAM sizes in MB — one entry per real
      shipping VRAM SKU. Single-SKU cards have a 1-element list; multi-variant
      cards have one entry per variant (e.g. RTX 3060 8/12 GB -> [8192, 12288]).
    - The acceptance (vram_mb_min, vram_mb_max) windows are NOT stored. They are
      computed on demand by `vram_window_for_size` as nominal × [0.90, 1.05].
      Storing only the size removes the duplicated derived bounds and makes the
      nominal size the single source of truth.

Window calibration (applied by vram_window_for_size):
    - Per-window bounds are nominal × [VRAM_FLOOR_RATIO, VRAM_CEIL_RATIO].
    - The 0.90 floor exists because NVML's `.total` is NOT marketing capacity —
      it is `physical_VRAM - vendor_reserved_regions - ECC_overhead - driver_firmware
      reservation`. In production we observed NVML reporting up to ~6.3% below the
      marketing nominal (e.g. RTX A4000 reports 15352 MB on a 16 GB card; RTX 4090
      reports 23028 MB on a 24 GB card; L40 reports 46068 MB on a 48 GB card).
      The 0.90 floor gives ~10% headroom to absorb this reservation safely.
    - The 1.05 ceiling absorbs upward vendor drift (e.g. H100 80GB reports
      81559 MB, slightly above 80×1024).
    - DISCRETE windows (one per nominal size, not one widened span) are deliberate:
      a single span from the smallest variant's floor to the largest variant's
      ceiling would accept every intermediate capacity that corresponds to no real
      SKU — e.g. a 24 GB card masquerading as a 16/32 GB V100, or an 80 GB H100
      masquerading as a 192 GB B200. Per-variant windows reject those intermediate
      impostors while the matmul check downstream remains authoritative.
    - B200: only the 192 GB size is listed. Prod data showed 99.75% of B200
      entries at 183359 MB (correct for 192 GB) and 0.25% at 49140 MB (a ~48 GB
      MIG slice or mislabel). The ~48 GB outlier is intentionally NOT given a
      size — it must fail precheck rather than widen acceptance into the
      64/80/94/96/141 GB range that real impostors would land in.

Calibration source: prod executor_gpu.capacity samples (lium_db); see PR #1043
follow-up calibration query.

Pre-check policy:
    - Models present here -> VRAM window-checked (windows computed from size).
    - Models in KNOWN_UNRANGED -> passthrough with code='unranged'.
    - Models in neither -> raises UnsupportedGpuModelError.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

# Per-window bounds are nominal × [VRAM_FLOOR_RATIO, VRAM_CEIL_RATIO].
# See the module docstring for why 0.90 / 1.05 (NVML under-reporting + vendor drift).
# These ratios encode the libdmcompverify.so contract and must stay in sync with it.
VRAM_FLOOR_RATIO = 0.90
VRAM_CEIL_RATIO = 1.05

# --- Nominal VRAM sizes (canonical model -> [vram_mb, ...]) -------------------
# Annotations:
#   `observed: X` notes the prod NVML-reported value(s) we've seen for that
#   model (most-common first). Used to verify the computed window covers reality.
#   Multi-variant cards list one nominal size per real SKU (flagged `# multi-variant`).
GPU_VRAM_SIZES_MB: Dict[str, List[int]] = {
    # Blackwell data-center
    "NVIDIA B300 SXM6 AC":                                [294912],        # 288 GB; observed: 275040
    "NVIDIA B200":                                        [196608],        # 192 GB; observed: 183359 (49140 ~48GB outlier rejected)
    # Hopper data-center
    "NVIDIA H200":                                        [144384],        # 141 GB; observed: 143771
    "NVIDIA H200 NVL":                                    [144384],        # 141 GB; observed: 143771
    "NVIDIA H100 80GB HBM3":                              [81920],         # 80 GB; observed: 81559
    "NVIDIA H100 NVL":                                    [96256],         # 94 GB; observed: 95830
    "NVIDIA H100 PCIe":                                   [81920],         # 80 GB; observed: 81559
    "NVIDIA H800 80GB HBM3":                              [81920],         # 80 GB
    "NVIDIA H800 NVL":                                    [96256],         # 94 GB
    "NVIDIA H800 PCIe":                                   [81920],         # 80 GB
    # Blackwell consumer
    "NVIDIA GeForce RTX 5090":                            [32768],         # 32 GB; observed: 32607
    "NVIDIA GeForce RTX 5080":                            [16384],         # 16 GB
    "NVIDIA GeForce RTX 5070 Ti":                         [16384],         # 16 GB
    "NVIDIA GeForce RTX 5070":                            [12288],         # 12 GB
    "NVIDIA GeForce RTX 5060 Ti":                         [8192, 16384],   # 8/16 GB multi-variant
    "NVIDIA GeForce RTX 5060":                            [8192],          # 8 GB
    # Ada consumer
    "NVIDIA GeForce RTX 4090":                            [24576],         # 24 GB; observed: 24564 (99.6%), 23028 (0.4%)
    "NVIDIA GeForce RTX 4090 D":                          [24576],         # 24 GB
    "NVIDIA GeForce RTX 4080 SUPER":                      [16384],         # 16 GB
    "NVIDIA GeForce RTX 4080":                            [16384],         # 16 GB
    "NVIDIA GeForce RTX 4070 Ti SUPER":                   [16384],         # 16 GB
    "NVIDIA GeForce RTX 4070 Ti":                         [12288],         # 12 GB
    "NVIDIA GeForce RTX 4070 SUPER":                      [12288],         # 12 GB
    "NVIDIA GeForce RTX 4070":                            [12288],         # 12 GB
    "NVIDIA GeForce RTX 4060 Ti":                         [8192, 16384],   # 8/16 GB multi-variant
    "NVIDIA GeForce RTX 4060":                            [8192],          # 8 GB
    # Blackwell pro / Ada pro / Quadro-class
    "NVIDIA RTX PRO 2000 Blackwell":                      [16384],         # 16 GB; observed: 16311
    "NVIDIA RTX PRO 4000 Blackwell":                      [24576],         # 24 GB
    "NVIDIA RTX PRO 4500 Blackwell":                      [32768],         # 32 GB
    "NVIDIA RTX PRO 4500 Blackwell Server Edition":       [32768],         # 32 GB
    "NVIDIA RTX PRO 5000 Blackwell":                      [49152, 73728],  # 48/72 GB multi-variant
    "NVIDIA RTX PRO 6000 Blackwell Server Edition":       [98304],         # 96 GB; observed: 97887
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition":  [98304],         # 96 GB; observed: 97887
    "NVIDIA RTX PRO 6000D Blackwell Workstation Edition": [86016],         # 84 GB; China SKU, 448-bit bus
    "NVIDIA RTX 6000D":                                   [86016],         # 84 GB; China SKU, 448-bit bus
    "NVIDIA RTX 4500 Ada Generation":                     [24576],         # 24 GB; observed: 23034
    "NVIDIA RTX 5000 Ada Generation":                     [32768],         # 32 GB
    "NVIDIA RTX 5880 Ada Generation":                     [49152],         # 48 GB
    "NVIDIA RTX 6000 Ada Generation":                     [49152],         # 48 GB; observed: 49140 (92%), 46068 (8%)
    # Lovelace data-center
    "NVIDIA L4":                                          [24576],         # 24 GB; observed: 23034
    "NVIDIA L40S":                                        [49152],         # 48 GB; observed: 46068 (91%), 49140 (9%)
    "NVIDIA L40":                                         [49152],         # 48 GB; observed: 46068 (75%), 49140 (25%)
    # Ampere data-center
    "NVIDIA A100 80GB PCIe":                              [81920],         # 80 GB; observed: 81920
    "NVIDIA A100-SXM4-80GB":                              [81920],         # 80 GB; observed: 81920
    "NVIDIA A800 80GB PCIe":                              [81920],         # 80 GB
    "NVIDIA A10 Tensor Core GPU":                         [24576],         # 24 GB
    # Only 8 GB is a factory SKU; 40/64 GB come from the community driver unlock.
    "NVIDIA CMP 170HX":                                   [8192, 40960, 65536], # 8/40/64 GB multi-variant
    # Ampere pro
    "NVIDIA RTX A6000":                                   [49152],         # 48 GB; observed: 49140
    "NVIDIA RTX A5000":                                   [24576],         # 24 GB; observed: 24564
    "NVIDIA RTX A4500":                                   [20480],         # 20 GB
    "NVIDIA RTX A4000":                                   [16384],         # 16 GB; observed: 16376 (71%), 15352 (29%)
    "NVIDIA RTX A2000":                                   [6144, 12288],   # 6/12 GB multi-variant
    # Turing data-center / pro
    "NVIDIA T4 Tensor Core GPU":                          [16384],         # 16 GB
    "NVIDIA Tesla V100 Tensor Core GPU":                  [16384, 32768],  # 16/32 GB multi-variant
    "NVIDIA Quadro RTX 8000":                             [49152],         # 48 GB
    "NVIDIA Quadro RTX 6000":                             [24576],         # 24 GB
    "NVIDIA Quadro RTX 5000":                             [16384],         # 16 GB
    "NVIDIA TITAN V":                                     [12288],         # 12 GB
    "NVIDIA TITAN RTX":                                   [24576],         # 24 GB
    # Ampere consumer
    "NVIDIA GeForce RTX 3090 Ti":                         [24576],         # 24 GB
    "NVIDIA GeForce RTX 3090":                            [24576],         # 24 GB; observed: 24576
    "NVIDIA GeForce RTX 3080 Ti":                         [12288],         # 12 GB
    "NVIDIA GeForce RTX 3080":                            [10240, 12288],  # 10/12 GB multi-variant
    "NVIDIA GeForce RTX 3070 Ti":                         [8192],          # 8 GB; observed: 8192
    "NVIDIA GeForce RTX 3070":                            [8192],          # 8 GB
    "NVIDIA GeForce RTX 3060 Ti":                         [8192],          # 8 GB
    "NVIDIA GeForce RTX 3060":                            [8192, 12288],   # 8/12 GB multi-variant
    "NVIDIA GeForce RTX 3060 Laptop GPU":                 [6144],          # 6 GB
    "NVIDIA GeForce RTX 3050":                            [6144, 8192],    # 6/8 GB multi-variant
    # Turing consumer
    "NVIDIA GeForce RTX 2080 Ti":                         [11264],         # 11 GB
    "NVIDIA GeForce RTX 2080 SUPER":                      [8192],          # 8 GB
    "NVIDIA GeForce RTX 2070 SUPER":                      [8192],          # 8 GB
    "NVIDIA GeForce RTX 2060 SUPER":                      [8192],          # 8 GB
    "NVIDIA GeForce RTX 2060":                            [6144, 12288],   # 6/12 GB multi-variant
    "NVIDIA GeForce GTX 1660 Ti":                         [6144],          # 6 GB
    "NVIDIA GeForce GTX 1660 SUPER":                      [6144],          # 6 GB
    "NVIDIA GeForce GTX 1660":                            [6144],          # 6 GB
    # Pascal
    "NVIDIA Tesla P100":                                  [12288, 16384],  # 12/16 GB multi-variant
    "NVIDIA Tesla P40":                                   [24576],         # 24 GB
    "NVIDIA Quadro P4000":                                [8192],          # 8 GB
    "NVIDIA TITAN Xp":                                    [12288],         # 12 GB
    "NVIDIA GeForce GTX 1080 Ti":                         [11264],         # 11 GB
    "NVIDIA GeForce GTX 1080":                            [8192],          # 8 GB
    "NVIDIA GeForce GTX 1070 Ti":                         [8192],          # 8 GB
    "NVIDIA GeForce GTX 1070":                            [8192],          # 8 GB
    "NVIDIA GeForce GTX 1060":                            [3072, 6144],    # 3/6 GB multi-variant (rare 5GB SKU not covered)
    # Maxwell
    "NVIDIA Tesla M40":                                   [12288, 24576],  # 12/24 GB multi-variant
}

# --- Intentionally unranged models (passthrough) -----------------------------
# Models that are in GPU_MODEL_RATES but for which we intentionally do not
# enforce a VRAM range (e.g., legacy data, awaiting spec confirmation).
# Pre-check returns None for these.
KNOWN_UNRANGED: Set[str] = set()


# --- Normalization (CUDA deviceProp.name -> canonical key) -------------------
# Lookup table, NOT regex suffix-strip. CUDA can report names like
# "Tesla V100-SXM2-32GB" or "Tesla H100" that don't match GPU_MODEL_RATES
# keys directly. Add entries as new mismatches are observed in production
# logs. Unmapped names pass through unchanged (and will likely produce
# UnsupportedGpuModelError until added here).
NORMALIZATION_MAP: Dict[str, str] = {
    # Tesla V100 family — CUDA reports "Tesla V100-..." in some driver versions
    "Tesla V100-SXM2-16GB":                 "NVIDIA Tesla V100 Tensor Core GPU",
    "Tesla V100-SXM2-32GB":                 "NVIDIA Tesla V100 Tensor Core GPU",
    "Tesla V100-PCIE-16GB":                 "NVIDIA Tesla V100 Tensor Core GPU",
    "Tesla V100-PCIE-32GB":                 "NVIDIA Tesla V100 Tensor Core GPU",
    # H100 — some driver/OS combos report without the "NVIDIA " prefix
    "Tesla H100 80GB HBM3":                 "NVIDIA H100 80GB HBM3",
    # T4 / A10 — CUDA sometimes drops the "Tensor Core GPU" suffix
    "Tesla T4":                             "NVIDIA T4 Tensor Core GPU",
    "NVIDIA T4":                            "NVIDIA T4 Tensor Core GPU",
    "NVIDIA A10":                           "NVIDIA A10 Tensor Core GPU",
    # Add observed CUDA names here as production logs surface them.
}


def normalize_gpu_model(name: Optional[str]) -> str:
    """Map a CUDA-reported GPU name to a canonical GPU_MODEL_RATES key.

    Returns:
        - "" if `name` is empty/None.
        - The mapped canonical name if `name.strip()` is in NORMALIZATION_MAP.
        - Otherwise `name.strip()` unchanged (caller decides what to do).
    """
    if not name:
        return ""
    stripped = name.strip()
    return NORMALIZATION_MAP.get(stripped, stripped)


def vram_window_for_size(nominal_mb: int) -> Tuple[int, int]:
    """Compute the (vram_mb_min, vram_mb_max) acceptance window for one nominal size.

    Bounds are nominal × [VRAM_FLOOR_RATIO, VRAM_CEIL_RATIO], rounded to int. This
    is the single place min/max VRAM is derived; the table stores only the size.
    """
    return (round(nominal_mb * VRAM_FLOOR_RATIO), round(nominal_mb * VRAM_CEIL_RATIO))


def get_expected_vram_windows(canonical_model: str) -> Optional[List[Tuple[int, int]]]:
    """Return the list of (vram_mb_min, vram_mb_max) windows for `canonical_model`.

    Windows are computed from the model's nominal VRAM size(s) via
    `vram_window_for_size`. Returns None if the model is unknown. A reading passes
    the pre-check if it falls within ANY returned window.
    """
    sizes = GPU_VRAM_SIZES_MB.get(canonical_model)
    if sizes is None:
        return None
    return [vram_window_for_size(size_mb) for size_mb in sizes]
