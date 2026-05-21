"""GPU model -> VRAM range table for validator-side pre-check.

# SYNC: This table mirrors compile-time data inside libdmcompverify.so.
# libdmcompverify is LIEF-obfuscated; its internal VRAM table cannot be
# extracted programmatically, so parity is enforced via a CI test against
# const.GPU_MODEL_RATES (see tests/test_gpu_precheck.py::test_gpu_model_rates_parity).
# Whenever a model is added to GPU_MODEL_RATES, it MUST be added here (with
# a VRAM range) or to KNOWN_UNRANGED (intentional passthrough).

VRAM range sourcing:
    - Nominal capacities sourced from NVIDIA published specifications as of 2026-05-21.
    - Each range is computed at ±5% of nominal MB to absorb vendor-reported drift
      (e.g., CUDA reporting 80,062 MB for an 80 GB HBM3 H100).
    - For models that ship in multiple VRAM variants under the SAME GPU_MODEL_RATES key
      (e.g., RTX 3060 6GB and 12GB), the range is widened to cover 0.95 * min_variant
      to 1.05 * max_variant. These wide-range entries are flagged with a `# multi-variant`
      comment.

Pre-check policy:
    - Models present here -> VRAM range-checked.
    - Models in KNOWN_UNRANGED -> passthrough with code='unranged'.
    - Models in neither -> code='unknown_model' (warn) / UnsupportedGpuModelError (strict).
"""
from __future__ import annotations
from typing import Dict, Optional, Set, Tuple

# --- VRAM ranges (canonical model -> (vram_mb_min, vram_mb_max)) -------------
GPU_VRAM_RANGES: Dict[str, Tuple[int, int]] = {
    # Blackwell data-center
    "NVIDIA B300 SXM6 AC":                                  (280166, 309658),  # 288 GB HBM3e
    "NVIDIA B200":                                          (186778, 206438),  # 192 GB HBM3e
    # Hopper data-center
    "NVIDIA H200":                                          (137165, 151603),  # 141 GB HBM3e
    "NVIDIA H200 NVL":                                      (137165, 151603),  # 141 GB HBM3e
    "NVIDIA H100 80GB HBM3":                                (77824, 86016),    # 80 GB
    "NVIDIA H100 NVL":                                      (91443, 101069),   # 94 GB HBM3
    "NVIDIA H100 PCIe":                                     (77824, 86016),    # 80 GB
    "NVIDIA H800 80GB HBM3":                                (77824, 86016),    # 80 GB
    "NVIDIA H800 NVL":                                      (91443, 101069),   # 94 GB
    "NVIDIA H800 PCIe":                                     (77824, 86016),    # 80 GB
    # Blackwell consumer
    "NVIDIA GeForce RTX 5090":                              (31130, 34406),    # 32 GB GDDR7
    "NVIDIA GeForce RTX 5080":                              (15565, 17203),    # 16 GB
    "NVIDIA GeForce RTX 5070 Ti":                           (15565, 17203),    # 16 GB
    "NVIDIA GeForce RTX 5070":                              (11674, 12902),    # 12 GB
    "NVIDIA GeForce RTX 5060 Ti":                           (7782, 17203),     # 8/16 GB multi-variant
    "NVIDIA GeForce RTX 5060":                              (7782, 8602),      # 8 GB
    # Ada consumer
    "NVIDIA GeForce RTX 4090":                              (23347, 25805),    # 24 GB
    "NVIDIA GeForce RTX 4090 D":                            (23347, 25805),    # 24 GB
    "NVIDIA GeForce RTX 4080 SUPER":                        (15565, 17203),    # 16 GB
    "NVIDIA GeForce RTX 4080":                              (15565, 17203),    # 16 GB
    "NVIDIA GeForce RTX 4070 Ti SUPER":                     (15565, 17203),    # 16 GB
    "NVIDIA GeForce RTX 4070 Ti":                           (11674, 12902),    # 12 GB
    "NVIDIA GeForce RTX 4070 SUPER":                        (11674, 12902),    # 12 GB
    "NVIDIA GeForce RTX 4070":                              (11674, 12902),    # 12 GB
    "NVIDIA GeForce RTX 4060 Ti":                           (7782, 17203),     # 8/16 GB multi-variant
    "NVIDIA GeForce RTX 4060":                              (7782, 8602),      # 8 GB
    # Blackwell pro / Ada pro / Quadro-class
    "NVIDIA RTX PRO 4000 Blackwell":                        (23347, 25805),    # 24 GB
    "NVIDIA RTX PRO 5000 Blackwell":                        (46694, 51610),    # 48 GB
    "NVIDIA RTX PRO 6000 Blackwell Server Edition":         (93389, 103219),   # 96 GB
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition":    (93389, 103219),   # 96 GB
    "NVIDIA RTX 5000 Ada Generation":                       (31130, 34406),    # 32 GB
    "NVIDIA RTX 5880 Ada Generation":                       (46694, 51610),    # 48 GB
    "NVIDIA RTX 6000 Ada Generation":                       (46694, 51610),    # 48 GB
    # Lovelace data-center
    "NVIDIA L4":                                            (23347, 25805),    # 24 GB
    "NVIDIA L40S":                                          (46694, 51610),    # 48 GB
    "NVIDIA L40":                                           (46694, 51610),    # 48 GB
    # Ampere data-center
    "NVIDIA A100 80GB PCIe":                                (77824, 86016),    # 80 GB
    "NVIDIA A100-SXM4-80GB":                                (77824, 86016),    # 80 GB
    "NVIDIA A10 Tensor Core GPU":                           (23347, 25805),    # 24 GB
    # Ampere pro
    "NVIDIA RTX A6000":                                     (46694, 51610),    # 48 GB
    "NVIDIA RTX A5000":                                     (23347, 25805),    # 24 GB
    "NVIDIA RTX A4500":                                     (19456, 21504),    # 20 GB
    "NVIDIA RTX A4000":                                     (15565, 17203),    # 16 GB
    "NVIDIA RTX A2000":                                     (5837, 12902),     # 6/12 GB multi-variant
    # Turing data-center / pro
    "NVIDIA T4 Tensor Core GPU":                            (15565, 17203),    # 16 GB
    "NVIDIA Tesla V100 Tensor Core GPU":                    (15565, 34406),    # 16/32 GB multi-variant
    "NVIDIA Quadro RTX 8000":                               (46694, 51610),    # 48 GB
    "NVIDIA Quadro RTX 6000":                               (23347, 25805),    # 24 GB
    "NVIDIA Quadro RTX 5000":                               (15565, 17203),    # 16 GB
    "NVIDIA TITAN V":                                       (11674, 12902),    # 12 GB
    "NVIDIA TITAN RTX":                                     (23347, 25805),    # 24 GB
    # Ampere consumer
    "NVIDIA GeForce RTX 3090 Ti":                           (23347, 25805),    # 24 GB
    "NVIDIA GeForce RTX 3090":                              (23347, 25805),    # 24 GB
    "NVIDIA GeForce RTX 3080 Ti":                           (11674, 12902),    # 12 GB
    "NVIDIA GeForce RTX 3080":                              (9728, 12902),     # 10/12 GB multi-variant
    "NVIDIA GeForce RTX 3070 Ti":                           (7782, 8602),      # 8 GB
    "NVIDIA GeForce RTX 3070":                              (7782, 8602),      # 8 GB
    "NVIDIA GeForce RTX 3060 Ti":                           (7782, 8602),      # 8 GB
    "NVIDIA GeForce RTX 3060":                              (7782, 12902),     # 8/12 GB multi-variant
    "NVIDIA GeForce RTX 3060 Laptop GPU":                   (5837, 6451),      # 6 GB
    "NVIDIA GeForce RTX 3050":                              (5837, 8602),      # 6/8 GB multi-variant
    # Turing consumer
    "NVIDIA GeForce RTX 2080 Ti":                           (10700, 11827),    # 11 GB
    "NVIDIA GeForce RTX 2080 SUPER":                        (7782, 8602),      # 8 GB
    "NVIDIA GeForce RTX 2070 SUPER":                        (7782, 8602),      # 8 GB
    "NVIDIA GeForce RTX 2060 SUPER":                        (7782, 8602),      # 8 GB
    "NVIDIA GeForce RTX 2060":                              (5837, 12902),     # 6/12 GB multi-variant
    "NVIDIA GeForce GTX 1660 Ti":                           (5837, 6451),      # 6 GB
    "NVIDIA GeForce GTX 1660 SUPER":                        (5837, 6451),      # 6 GB
    "NVIDIA GeForce GTX 1660":                              (5837, 6451),      # 6 GB
    # Pascal
    "NVIDIA Tesla P100":                                    (11674, 17203),    # 12/16 GB multi-variant
    "NVIDIA Tesla P40":                                     (23347, 25805),    # 24 GB
    "NVIDIA Quadro P4000":                                  (7782, 8602),      # 8 GB
    "NVIDIA TITAN Xp":                                      (11674, 12902),    # 12 GB
    "NVIDIA GeForce GTX 1080 Ti":                           (10700, 11827),    # 11 GB
    "NVIDIA GeForce GTX 1080":                              (7782, 8602),      # 8 GB
    "NVIDIA GeForce GTX 1070 Ti":                           (7782, 8602),      # 8 GB
    "NVIDIA GeForce GTX 1070":                              (7782, 8602),      # 8 GB
    "NVIDIA GeForce GTX 1060":                              (2918, 6451),      # 3/6 GB multi-variant
    # Maxwell
    "NVIDIA Tesla M40":                                     (11674, 25805),    # 12/24 GB multi-variant
}

# --- Intentionally unranged models (passthrough) -----------------------------
# Models that are in GPU_MODEL_RATES but for which we intentionally do not
# enforce a VRAM range (e.g., legacy data, awaiting spec confirmation).
# Pre-check returns ok=True with code='unranged' for these.
KNOWN_UNRANGED: Set[str] = set()


# --- Normalization (CUDA deviceProp.name -> canonical key) -------------------
# Lookup table, NOT regex suffix-strip. CUDA can report names like
# "Tesla V100-SXM2-32GB" or "Tesla H100" that don't match GPU_MODEL_RATES
# keys directly. Add entries as new mismatches are observed in production
# logs. Unmapped names pass through unchanged (and will likely produce
# code='unknown_model' until added here).
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
