"""GB300 is priced like B300 for now, until it has its own market data.

GB300 is a different card, so it is its own base model with its own idle tier at the B300 caps (1x: 4, 8x: 32); an
idle GB300 never fills the B300 tier. Its idle rate, score rate and VRAM size are the B300 AC entries.
"""


import pytest
from incentive.config import RENTAL_PRICES_PER_HOUR
from services import const, gpu_spec_table

B300 = "NVIDIA B300 SXM6 AC"
GB300 = "NVIDIA GB300"


@pytest.mark.parametrize(
    "table",
    [RENTAL_PRICES_PER_HOUR, const.GPU_MODEL_RATES, gpu_spec_table.GPU_VRAM_SIZES_MB],
    ids=["RENTAL_PRICES_PER_HOUR", "GPU_MODEL_RATES", "GPU_VRAM_SIZES_MB"],
)
def test_gb300_carries_the_b300_value(table) -> None:
    assert table[GB300] == table[B300]


