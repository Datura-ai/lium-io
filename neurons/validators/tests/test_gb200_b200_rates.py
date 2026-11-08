"""DAH-6008: GB200 is priced like B200 for now, until it has its own market data.

GB200 is a different card, so it is its own base model with its own idle tier at the B200 caps; an idle GB200 never
fills the B200 tier. Its idle rate, score rate and VRAM size are the B200 entries. A GB200 NVL72 compute tray is a
4-GPU host, so it reaches the 1x idle tier through GPU splitting.
"""

import pytest
from incentive.config import BASE_GPU_MAP, MAX_UNRENTED_GPUS_BY_TYPE, RENTAL_PRICES_PER_HOUR
from services import const, gpu_spec_table
from services.gpu_reference_table import pick_reference

B200 = "NVIDIA B200"
GB200 = "NVIDIA GB200"


@pytest.mark.parametrize(
    "table",
    [RENTAL_PRICES_PER_HOUR, const.GPU_MODEL_RATES, gpu_spec_table.GPU_VRAM_SIZES_MB],
    ids=["RENTAL_PRICES_PER_HOUR", "GPU_MODEL_RATES", "GPU_VRAM_SIZES_MB"],
)
def test_gb200_carries_the_b200_value(table) -> None:
    assert table[GB200] == table[B200]


def test_gb200_is_its_own_base_model_at_the_b200_caps() -> None:
    assert BASE_GPU_MAP[GB200] == "GB200"
    assert MAX_UNRENTED_GPUS_BY_TYPE["GB200"] == MAX_UNRENTED_GPUS_BY_TYPE["B200"]
    assert MAX_UNRENTED_GPUS_BY_TYPE["GB200"] is not MAX_UNRENTED_GPUS_BY_TYPE["B200"]


def test_gb200_is_accepted_by_name() -> None:
    assert gpu_spec_table.reference_gpu_model(GB200, 189471) == GB200


def test_an_unknown_192gb_card_still_references_the_b200() -> None:
    assert pick_reference(183359) == B200
