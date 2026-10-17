"""Unit + parity tests for services.gpu_precheck and services.gpu_spec_table."""
from __future__ import annotations

import pytest

from incentive import config as incentive_config
from services import const, gpu_spec_table
from services.const import GPU_MODEL_RATES


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
