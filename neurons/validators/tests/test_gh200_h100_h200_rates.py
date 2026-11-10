"""DAH-6147: GH200 is its own model, the 96 GB part at the H100 SXM rate and the 144 GB part at the H200 rate.

Both share the base model "GH200" with its own idle tier at the H100 caps, so an idle GH200 never fills an H100 or
H200 tier. The name is self-reported: off a Grace (aarch64) host it is scored as the card its VRAM references.
"""

import pytest
from incentive.config import BASE_GPU_MAP, RENTAL_PRICES_PER_HOUR
from services.const import GPU_MODEL_RATES
from services.gpu_reference_table import pick_reference

from services import gpu_spec_table
from tests.test_idle_slot_caps_by_tier import _run_with_production_config
from tests.test_rental_price_incentive_flow import _make_pcc_job

GH200_96, GH200_144 = "NVIDIA GH200 480GB", "NVIDIA GH200 144G HBM3e"
H100, H200 = "NVIDIA H100 80GB HBM3", "NVIDIA H200"


@pytest.mark.parametrize("gh200, peer", [(GH200_96, H100), (GH200_144, H200)])
def test_each_gh200_is_paid_like_its_hopper_peer(gh200, peer):
    assert GPU_MODEL_RATES[gh200] == GPU_MODEL_RATES[peer]
    assert RENTAL_PRICES_PER_HOUR[gh200] == RENTAL_PRICES_PER_HOUR[peer]
    assert BASE_GPU_MAP[gh200] == "GH200"
    assert gpu_spec_table.reference_gpu_model(gh200, 97871) == gh200


@pytest.mark.asyncio
async def test_an_idle_gh200_fills_its_own_tier_not_the_h100s():
    jobs = {
        "gh200": [_make_pcc_job("gh200-1x", GH200_96, 1)],
        "h100": [_make_pcc_job("h100-1x", H100, 1)],
    }

    incentive = await _run_with_production_config(jobs)

    assert jobs["gh200"][0].effective_rate == pytest.approx(jobs["h100"][0].effective_rate)
    assert jobs["gh200"][0].effective_rate > 0
    assert incentive.unrented_count_by_bucket[("GH200", 1)] == 1
    assert incentive.unrented_count_by_bucket[("H100", 1)] == 1


@pytest.mark.parametrize(
    "arch, expected",
    [("aarch64", GH200_96), ("x86_64", "NVIDIA H800 NVL"), (None, "NVIDIA H800 NVL")],
)
def test_a_gh200_name_counts_only_on_a_grace_host(arch, expected):
    # a 94 GB H100/H800 NVL calling itself GH200 on an x86 host keeps what its VRAM referenced before GH200 was listed
    specs = {"machine_arch": arch} if arch else {}
    details = [{"name": GH200_96, "capacity": 95830}, {"name": H100, "capacity": 81559}]

    listed = gpu_spec_table.list_gh200_off_grace_as_reference(specs, details)

    assert [gpu["name"] for gpu in listed] == [expected, H100]


def test_unknown_96gb_and_144gb_cards_still_reference_the_cheapest_card_of_that_size():
    assert pick_reference(95830) == "NVIDIA H800 NVL"
    assert pick_reference(143771) == "NVIDIA H200 NVL"
