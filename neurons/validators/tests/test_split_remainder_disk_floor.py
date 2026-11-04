"""DAH-4961: the free remainder of a partially rented split node earns idle pay only for the free splits
SplitDiskCheck found disk room for (`specs.split_disk`), as only those are listed; same pattern as the
per-split port rule (DAH-3698)."""

import pytest
from incentive.miner_incentive_log import ZeroIncentiveReason
from services.const import FRESH_SIZING_HEADROOM_GB
from services.task.checks.disk_space import SplitDiskCheck
from tests.test_disk_space_check import _split_state
from tests.test_split_remainder_port_floor import MINER_HOTKEY, _build_incentive, _make_job, _reason_codes, _score_split

from core.config import settings

DISK = ZeroIncentiveReason.DISK_LIMITED_REMAINDER.value
PORT = ZeroIncentiveReason.PORT_LIMITED_REMAINDER.value


def _sizing(listable: int, free_splits: int = 8) -> dict:
    # 1-GPU splits: the listable GPUs are the listable splits
    sizing = {"free_split_count": free_splits, "listable_split_count": listable, "listable_gpu_count": listable}
    return {"split_disk": {**sizing, "required_free_gb_per_split": 6.2}}


@pytest.mark.asyncio
@pytest.mark.parametrize(("listable", "paid_gpus"), [(8, 8), (5, 5), (0, 0)])
async def test_remainder_is_paid_one_split_per_split_with_disk_room(listable, paid_gpus):
    # 9 GPUs in 1-GPU splits, 1 rented, a free port for every split: only the disk limits the pay
    job, incentive = await _score_split(8, gpu_count=9, rented=1, spec_extra=_sizing(listable))

    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == (paid_gpus or None)
    assert job.gpu_count == 9  # the merged row still reports the whole node
    assert (job.incentive_idle > 0) is (paid_gpus > 0)
    if paid_gpus == 8:
        assert DISK not in _reason_codes(job)
    else:
        assert _reason_codes(job) == [DISK]
        assert job.zero_incentive_reasons[0].context["paid_split_count"] == listable
        assert job.zero_incentive_reasons[0].context["gpu_count"] == 8 - paid_gpus


@pytest.mark.asyncio
async def test_no_sizing_keeps_the_idle_pay():
    # an unsplit node or an older validator publishes no split_disk: no change
    job, incentive = await _score_split(8, gpu_count=9, rented=1)

    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == 8
    assert DISK not in _reason_codes(job)


@pytest.mark.asyncio
async def test_a_gpu_both_rules_leave_unpaid_is_counted_once():
    # 7 free ports pay 7 splits, disk room for 5: 5 paid, 3 unpaid, and the merged row keeps 9 GPUs
    job, incentive = await _score_split(7, gpu_count=9, rented=1, spec_extra=_sizing(5))

    assert incentive.unrented_count_by_bucket.get(("H200", 1)) == 5
    assert job.gpu_count == 9
    assert _reason_codes(job) == [PORT, DISK]


@pytest.mark.asyncio
@pytest.mark.parametrize("room_for_splits", [0, 1, 2, 3])
async def test_scoring_pays_exactly_the_gpus_the_listing_offers(context_factory, room_for_splits):
    # 8 GPUs in 2-GPU splits, 1 rented: 7 free = 3 splits + 1 leftover GPU riding with the last listable split
    free_gb = settings.MIN_FREE_DISK_GB / 4 * room_for_splits + FRESH_SIZING_HEADROOM_GB
    ctx = context_factory(state=_split_state(free_gb, pods=(1,)), rented=True)
    sizing = (await SplitDiskCheck().run(ctx)).updates["state"].specs["split_disk"]

    job = _make_job(available_port_count=8, gpu_count=8, rented_gpu_count=1, min_count=2)
    job.spec["split_disk"] = sizing
    incentive = _build_incentive(job)
    incentive._expand_partially_rented_split_results()
    _, remainder = incentive.job_results[MINER_HOTKEY]
    limited = incentive._disk_limited_remainder(remainder)

    assert sizing["listable_gpu_count"] == [0, 3, 5, 7][room_for_splits]
    assert remainder.gpu_count - (limited.unpaid_gpu_count if limited else 0) == sizing["listable_gpu_count"]
