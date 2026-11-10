"""GB300 is priced like B300 for now, until it has its own market data.

GB300 is a different card, so it is its own base model with its own idle tier at the B300 caps (1x: 4, 8x: 32); an
idle GB300 never fills the B300 tier. Its idle rate and score rate are the B300 AC entries; its VRAM sizes are the B300 size plus the DGX Station's 252 GB.
"""


import pytest
from incentive.config import RENTAL_PRICES_PER_HOUR
from services import const, gpu_spec_table

B300 = "NVIDIA B300 SXM6 AC"
GB300 = "NVIDIA GB300"


@pytest.mark.parametrize(
    "table",
    [RENTAL_PRICES_PER_HOUR, const.GPU_MODEL_RATES],
    ids=["RENTAL_PRICES_PER_HOUR", "GPU_MODEL_RATES"],
)
def test_gb300_carries_the_b300_value(table) -> None:
    assert table[GB300] == table[B300]


def test_gb300_vram_covers_the_b300_size_and_the_dgx_station_252_gb() -> None:
    sizes = gpu_spec_table.GPU_VRAM_SIZES_MB[GB300]
    assert sizes == [252 * 1024, *gpu_spec_table.GPU_VRAM_SIZES_MB[B300]]
    windows = gpu_spec_table.get_expected_vram_windows(GB300)
    # a 252 GB card reads a few percent under nominal, like the B300's 275040 of 294912
    assert any(low <= 248000 <= high for low, high in windows)
    assert any(low <= 275040 <= high for low, high in windows)
