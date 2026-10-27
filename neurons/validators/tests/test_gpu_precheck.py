"""Unit + parity tests for services.gpu_precheck and services.gpu_spec_table."""
from __future__ import annotations

import pytest

from incentive import config as incentive_config
from services import const, gpu_spec_table
from services.const import GPU_MODEL_RATES


# --- Happy path --------------------------------------------------------------


# --- Missing field ----------------------------------------------------------


# --- Unknown model ----------------------------------------------------------


# --- VRAM range checks ------------------------------------------------------


# --- Exception hierarchy ----------------------------------------------------


# --- KNOWN_UNRANGED ---------------------------------------------------------


# --- Normalization helpers --------------------------------------------------


# --- CI parity --------------------------------------------------------------
B300_AC, B300_PC = "NVIDIA B300 SXM6 AC", "NVIDIA B300 SXM6 PC"


@pytest.mark.parametrize("module", [incentive_config, const, gpu_spec_table], ids=lambda m: m.__name__)
def test_every_table_naming_the_b300_ac_card_carries_the_pc_alias_at_the_same_value(module):
    # the PC name is the AC card's alias, derived from the AC entry; a table added later that names AC joins by existing
    tables = {name: t for name, t in vars(module).items() if isinstance(t, dict) and B300_AC in t}
    assert tables, module.__name__
    for name, table in tables.items():
        assert table.get(B300_PC) == table[B300_AC], f"{module.__name__}.{name}"


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


# --- Multi-variant: impostor rejection in the inter-variant gap -------------


# --- DAH-2160: format-only refactor guards ----------------------------------
# The table now stores nominal MB sizes; windows are derived by
# vram_window_for_size. These tests prove the refactor changes only the storage
# FORMAT, not any pre-check value or range.


# Frozen snapshot of the (vram_mb_min, vram_mb_max) windows BEFORE the DAH-2160
# refactor — copied VERBATIM from the pre-refactor GPU_VRAM_RANGES dict (the
# hand-stored tuples), NOT recomputed from the new sizes table. This makes the
# test an independent regression guard: the computed windows MUST match these
# exact values for every model, proving no pre-check value or range changed.
_EXPECTED_WINDOWS: dict[str, list[tuple[int, int]]] = {
    "NVIDIA B300 SXM6 AC": [(265421, 309658)],
    "NVIDIA B300 SXM6 PC": [(265421, 309658)],
    "NVIDIA GB300": [(265421, 309658)],  # the B300 size
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
    "NVIDIA GeForce RTX 5090 D": [(29491, 34406)],
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
    "NVIDIA RTX PRO 2000 Blackwell": [(14746, 17203)],
    "NVIDIA RTX PRO 4000 Blackwell": [(22118, 25805)],
    "NVIDIA RTX PRO 4500 Blackwell": [(29491, 34406)],
    "NVIDIA RTX PRO 4500 Blackwell Server Edition": [(29491, 34406)],
    "NVIDIA RTX PRO 5000 Blackwell": [(44237, 51610), (66355, 77414)],
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": [(88474, 103219)],
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": [(88474, 103219)],
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition": [(88474, 103219)],
    "NVIDIA RTX PRO 6000D Blackwell Workstation Edition": [(77414, 90317)],
    "NVIDIA RTX 6000D": [(77414, 90317)],
    "NVIDIA RTX 4500 Ada Generation": [(22118, 25805)],
    "NVIDIA RTX 5000 Ada Generation": [(29491, 34406)],
    "NVIDIA RTX 5880 Ada Generation": [(44237, 51610)],
    "NVIDIA RTX 6000 Ada Generation": [(44237, 51610)],
    "NVIDIA L4": [(22118, 25805)],
    "NVIDIA L40S": [(44237, 51610)],
    "NVIDIA L40": [(44237, 51610)],
    "NVIDIA A100 80GB PCIe": [(73728, 86016)],
    "NVIDIA A100-SXM4-80GB": [(73728, 86016)],
    "NVIDIA A800 80GB PCIe": [(73728, 86016)],
    "NVIDIA A10 Tensor Core GPU": [(22118, 25805)],
    "NVIDIA CMP 170HX": [(7373, 8602), (36864, 43008), (58982, 68813)],
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


