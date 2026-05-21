"""GPU model -> VRAM range table for validator-side pre-check.

# SYNC: This table mirrors compile-time data inside libdmcompverify.so.
# libdmcompverify is LIEF-obfuscated; its internal VRAM table cannot be
# extracted programmatically, so parity is enforced via a CI test against
# const.GPU_MODEL_RATES (see tests/test_gpu_precheck.py::test_gpu_model_rates_parity).
# Whenever a model is added to GPU_MODEL_RATES, it MUST be added here (with
# a VRAM range) or to KNOWN_UNRANGED (intentional passthrough).

VRAM range calibration:
    - Bounds are nominal × [0.90, 1.05].
    - The 0.90 floor exists because NVML's `.total` is NOT marketing capacity —
      it is `physical_VRAM - vendor_reserved_regions - ECC_overhead - driver_firmware
      reservation`. In production we observed NVML reporting up to ~6.3% below the
      marketing nominal (e.g. RTX A4000 reports 15352 MB on a 16 GB card; RTX 4090
      reports 23028 MB on a 24 GB card; L40 reports 46068 MB on a 48 GB card).
      The 0.90 floor gives ~10% headroom to absorb this reservation safely.
    - The 1.05 ceiling absorbs upward vendor drift (e.g. H100 80GB reports
      81559 MB, slightly above 80×1024).
    - Multi-variant cards (same model name across different VRAM SKUs, e.g.
      RTX 3060 8/12GB) use widened ranges covering all variants, flagged
      `# multi-variant`.
    - B200 special-case: prod data shows 99.75% of B200 entries at 183359 MB
      (correct for 192 GB), but 0.25% report 49140 MB (likely MIG slice or
      mislabeled). Lower bound widened to 47000 to avoid false-rejection of
      the rare outlier; matmul check downstream is authoritative for any
      cryptographic substitution.

Calibration source: prod executor_gpu.capacity samples (lium_db); see PR #1043
follow-up calibration query.

Pre-check policy:
    - Models present here -> VRAM range-checked.
    - Models in KNOWN_UNRANGED -> passthrough with code='unranged'.
    - Models in neither -> raises UnsupportedGpuModelError.
"""
from __future__ import annotations
from typing import Dict, Optional, Set, Tuple

# --- VRAM ranges (canonical model -> (vram_mb_min, vram_mb_max)) -------------
# Annotations:
#   `observed: X` notes the prod NVML-reported value(s) we've seen for that
#   model (most-common first). Used to verify the chosen range covers reality.
GPU_VRAM_RANGES: Dict[str, Tuple[int, int]] = {
    # Blackwell data-center
    "NVIDIA B300 SXM6 AC":                                  (265421, 309658),  # 288 GB; observed: 275040
    "NVIDIA B200":                                          (47000, 206438),   # 192 GB; observed: 183359 (99.75%), 49140 (0.25% MIG/outlier)
    # Hopper data-center
    "NVIDIA H200":                                          (129946, 151603),  # 141 GB; observed: 143771
    "NVIDIA H200 NVL":                                      (129946, 151603),  # 141 GB; observed: 143771
    "NVIDIA H100 80GB HBM3":                                (73728, 86016),    # 80 GB; observed: 81559
    "NVIDIA H100 NVL":                                      (86630, 101069),   # 94 GB; observed: 95830
    "NVIDIA H100 PCIe":                                     (73728, 86016),    # 80 GB; observed: 81559
    "NVIDIA H800 80GB HBM3":                                (73728, 86016),    # 80 GB
    "NVIDIA H800 NVL":                                      (86630, 101069),   # 94 GB
    "NVIDIA H800 PCIe":                                     (73728, 86016),    # 80 GB
    # Blackwell consumer
    "NVIDIA GeForce RTX 5090":                              (29491, 34406),    # 32 GB; observed: 32607
    "NVIDIA GeForce RTX 5080":                              (14746, 17203),    # 16 GB
    "NVIDIA GeForce RTX 5070 Ti":                           (14746, 17203),    # 16 GB
    "NVIDIA GeForce RTX 5070":                              (11059, 12902),    # 12 GB
    "NVIDIA GeForce RTX 5060 Ti":                           (7373, 17203),     # 8/16 GB multi-variant
    "NVIDIA GeForce RTX 5060":                              (7373, 8602),      # 8 GB
    # Ada consumer
    "NVIDIA GeForce RTX 4090":                              (22118, 25805),    # 24 GB; observed: 24564 (99.6%), 23028 (0.4%)
    "NVIDIA GeForce RTX 4090 D":                            (22118, 25805),    # 24 GB
    "NVIDIA GeForce RTX 4080 SUPER":                        (14746, 17203),    # 16 GB
    "NVIDIA GeForce RTX 4080":                              (14746, 17203),    # 16 GB
    "NVIDIA GeForce RTX 4070 Ti SUPER":                     (14746, 17203),    # 16 GB
    "NVIDIA GeForce RTX 4070 Ti":                           (11059, 12902),    # 12 GB
    "NVIDIA GeForce RTX 4070 SUPER":                        (11059, 12902),    # 12 GB
    "NVIDIA GeForce RTX 4070":                              (11059, 12902),    # 12 GB
    "NVIDIA GeForce RTX 4060 Ti":                           (7373, 17203),     # 8/16 GB multi-variant
    "NVIDIA GeForce RTX 4060":                              (7373, 8602),      # 8 GB
    # Blackwell pro / Ada pro / Quadro-class
    "NVIDIA RTX PRO 4000 Blackwell":                        (22118, 25805),    # 24 GB
    "NVIDIA RTX PRO 5000 Blackwell":                        (44237, 51610),    # 48 GB
    "NVIDIA RTX PRO 6000 Blackwell Server Edition":         (88474, 103219),   # 96 GB; observed: 97887
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition":    (88474, 103219),   # 96 GB; observed: 97887
    "NVIDIA RTX 5000 Ada Generation":                       (29491, 34406),    # 32 GB
    "NVIDIA RTX 5880 Ada Generation":                       (44237, 51610),    # 48 GB
    "NVIDIA RTX 6000 Ada Generation":                       (44237, 51610),    # 48 GB; observed: 49140 (92%), 46068 (8%)
    # Lovelace data-center
    "NVIDIA L4":                                            (22118, 25805),    # 24 GB; observed: 23034
    "NVIDIA L40S":                                          (44237, 51610),    # 48 GB; observed: 46068 (91%), 49140 (9%)
    "NVIDIA L40":                                           (44237, 51610),    # 48 GB; observed: 46068 (75%), 49140 (25%)
    # Ampere data-center
    "NVIDIA A100 80GB PCIe":                                (73728, 86016),    # 80 GB; observed: 81920
    "NVIDIA A100-SXM4-80GB":                                (73728, 86016),    # 80 GB; observed: 81920
    "NVIDIA A10 Tensor Core GPU":                           (22118, 25805),    # 24 GB
    # Ampere pro
    "NVIDIA RTX A6000":                                     (44237, 51610),    # 48 GB; observed: 49140
    "NVIDIA RTX A5000":                                     (22118, 25805),    # 24 GB; observed: 24564
    "NVIDIA RTX A4500":                                     (18432, 21504),    # 20 GB
    "NVIDIA RTX A4000":                                     (14746, 17203),    # 16 GB; observed: 16376 (71%), 15352 (29%)
    "NVIDIA RTX A2000":                                     (5530, 12902),     # 6/12 GB multi-variant
    # Turing data-center / pro
    "NVIDIA T4 Tensor Core GPU":                            (14746, 17203),    # 16 GB
    "NVIDIA Tesla V100 Tensor Core GPU":                    (14746, 34406),    # 16/32 GB multi-variant
    "NVIDIA Quadro RTX 8000":                               (44237, 51610),    # 48 GB
    "NVIDIA Quadro RTX 6000":                               (22118, 25805),    # 24 GB
    "NVIDIA Quadro RTX 5000":                               (14746, 17203),    # 16 GB
    "NVIDIA TITAN V":                                       (11059, 12902),    # 12 GB
    "NVIDIA TITAN RTX":                                     (22118, 25805),    # 24 GB
    # Ampere consumer
    "NVIDIA GeForce RTX 3090 Ti":                           (22118, 25805),    # 24 GB
    "NVIDIA GeForce RTX 3090":                              (22118, 25805),    # 24 GB; observed: 24576
    "NVIDIA GeForce RTX 3080 Ti":                           (11059, 12902),    # 12 GB
    "NVIDIA GeForce RTX 3080":                              (9216, 12902),     # 10/12 GB multi-variant
    "NVIDIA GeForce RTX 3070 Ti":                           (7373, 8602),      # 8 GB; observed: 8192
    "NVIDIA GeForce RTX 3070":                              (7373, 8602),      # 8 GB
    "NVIDIA GeForce RTX 3060 Ti":                           (7373, 8602),      # 8 GB
    "NVIDIA GeForce RTX 3060":                              (7373, 12902),     # 8/12 GB multi-variant
    "NVIDIA GeForce RTX 3060 Laptop GPU":                   (5530, 6451),      # 6 GB
    "NVIDIA GeForce RTX 3050":                              (5530, 8602),      # 6/8 GB multi-variant
    # Turing consumer
    "NVIDIA GeForce RTX 2080 Ti":                           (10138, 11827),    # 11 GB
    "NVIDIA GeForce RTX 2080 SUPER":                        (7373, 8602),      # 8 GB
    "NVIDIA GeForce RTX 2070 SUPER":                        (7373, 8602),      # 8 GB
    "NVIDIA GeForce RTX 2060 SUPER":                        (7373, 8602),      # 8 GB
    "NVIDIA GeForce RTX 2060":                              (5530, 12902),     # 6/12 GB multi-variant
    "NVIDIA GeForce GTX 1660 Ti":                           (5530, 6451),      # 6 GB
    "NVIDIA GeForce GTX 1660 SUPER":                        (5530, 6451),      # 6 GB
    "NVIDIA GeForce GTX 1660":                              (5530, 6451),      # 6 GB
    # Pascal
    "NVIDIA Tesla P100":                                    (11059, 17203),    # 12/16 GB multi-variant
    "NVIDIA Tesla P40":                                     (22118, 25805),    # 24 GB
    "NVIDIA Quadro P4000":                                  (7373, 8602),      # 8 GB
    "NVIDIA TITAN Xp":                                      (11059, 12902),    # 12 GB
    "NVIDIA GeForce GTX 1080 Ti":                           (10138, 11827),    # 11 GB
    "NVIDIA GeForce GTX 1080":                              (7373, 8602),      # 8 GB
    "NVIDIA GeForce GTX 1070 Ti":                           (7373, 8602),      # 8 GB
    "NVIDIA GeForce GTX 1070":                              (7373, 8602),      # 8 GB
    "NVIDIA GeForce GTX 1060":                              (2765, 6451),      # 3/6 GB multi-variant
    # Maxwell
    "NVIDIA Tesla M40":                                     (11059, 25805),    # 12/24 GB multi-variant
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


def get_expected_vram_range(canonical_model: str) -> Optional[Tuple[int, int]]:
    """Return (vram_mb_min, vram_mb_max) for `canonical_model`, or None if unknown."""
    return GPU_VRAM_RANGES.get(canonical_model)
