"""DAH-6008: GB200 is priced like B200 for now, until it has its own market data.

GB200 is its own base model with its own idle tier at the B200 caps, so an idle GB200 never fills the B200 tier.
A GB200 NVL72 compute tray is a 4-GPU host: no idle tier pays 4 cards, so it is paid only through GPU splitting.
"""

import pytest
from services import gpu_spec_table
from services.gpu_reference_table import pick_reference

from tests.test_idle_slot_caps_by_tier import _run_with_production_config
from tests.test_rental_price_incentive_flow import _make_pcc_job

B200 = "NVIDIA B200"
GB200 = "NVIDIA GB200"


@pytest.mark.asyncio
async def test_an_idle_4x_gb200_is_paid_at_the_b200_1x_rate_only_when_split_to_one_card():
    jobs = {
        "unsplit": [_make_pcc_job("gb200-4x", GB200, 4)],
        "split": [_make_pcc_job("gb200-4x-split", GB200, 4, supports_gpu_splitting=True, gpu_splitting_min_count=1)],
        "b200": [_make_pcc_job("b200-1x", B200, 1)],
    }

    incentive = await _run_with_production_config(jobs)

    unsplit, split, b200 = jobs["unsplit"][0], jobs["split"][0], jobs["b200"][0]
    assert unsplit.effective_rate == 0.0
    assert split.count_bucket == 1
    assert split.effective_rate == pytest.approx(b200.effective_rate)
    assert split.effective_rate > 0
    assert incentive.unrented_count_by_bucket[("GB200", 1)] == 4
    assert incentive.unrented_count_by_bucket[("B200", 1)] == 1


def test_gb200_is_accepted_by_name() -> None:
    assert gpu_spec_table.reference_gpu_model(GB200, 189471) == GB200


def test_an_unknown_192gb_card_still_references_the_b200() -> None:
    assert pick_reference(183359) == B200
