"""DAH-6147: GH200 is its own model, the 96 GB part at the H100 SXM rate and the 144 GB part at the H200 rate.

Both share the base model "GH200" with its own idle tier at the H100 caps, so an idle GH200 never fills an H100 or
H200 tier. The name is self-reported: off a Grace (aarch64) host it is scored as the card its VRAM references.
"""

import pytest
from services.gpu_reference_table import pick_reference

from services import gpu_spec_table
from tests.test_idle_slot_caps_by_tier import _run_with_production_config
from tests.test_rental_price_incentive_flow import _make_pcc_job

GH200_96, GH200_144 = "NVIDIA GH200 480GB", "NVIDIA GH200 144G HBM3e"
H100, H200 = "NVIDIA H100 80GB HBM3", "NVIDIA H200"


@pytest.mark.parametrize("gh200", [GH200_96, GH200_144])
def test_a_reported_gh200_is_scored_as_itself(gh200):
    assert gpu_spec_table.reference_gpu_model(gh200, 97871) == gh200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gh200, peer, peer_base", [(GH200_96, H100, "H100"), (GH200_144, H200, "H200")]
)
async def test_an_idle_gh200_is_paid_like_its_peer_from_its_own_tier(gh200, peer, peer_base):
    jobs = {
        "gh200": [_make_pcc_job("gh200-1x", gh200, 1)],
        "peer": [_make_pcc_job("peer-1x", peer, 1)],
    }

    incentive = await _run_with_production_config(jobs)

    assert jobs["gh200"][0].effective_rate == pytest.approx(jobs["peer"][0].effective_rate)
    assert jobs["gh200"][0].effective_rate > 0
    assert incentive.unrented_count_by_bucket[("GH200", 1)] == 1
    assert incentive.unrented_count_by_bucket[(peer_base, 1)] == 1


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
