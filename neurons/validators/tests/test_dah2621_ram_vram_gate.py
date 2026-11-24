"""DAH-2621: an idle node with too little RAM for its GPU VRAM earns no unrented incentive."""

import pytest

from tests.test_every_zero_incentive_reason import _codes, _idle_full_8x, _incentive

GIB_KB = 1024**2


def _node(gpus: int, vram_gb_per_gpu: float, ram_gb: float):
    spec = {
        "hard_disk": {"total": 4000 * GIB_KB},
        "gpu": {"details": [{"capacity": vram_gb_per_gpu * 1024}] * gpus},
        "ram": {"total": ram_gb * GIB_KB},
    }
    return _idle_full_8x(gpu_model="NVIDIA H200", gpu_count=gpus, spec=spec)


@pytest.mark.asyncio
async def test_a_node_with_11_gb_ram_and_a_64_gb_card_earns_no_idle_pay():
    result = await _incentive().calculate_executor_score(_node(1, 64, 11.2))

    assert "insufficient_ram_for_vram" in _codes(result)
    assert result.eligible_for_rental_share is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gpus", "vram", "ram"),
    [
        # 24 GB installed on a 48 GB card: MemTotal reads under half, the tolerance keeps it
        (1, 48, 23.5),
        (4, 141, 1500),  # well above
        (2, 32, 31.2),  # 31.2 GB for 64 GB summed VRAM, 0.49x read: the tolerance keeps it
        (2, 32, 32.0),  # exactly half
        # 15,608 MiB clears 16 GB with the tolerance; rounded to 15.2 GB first, it would not
        (1, 32, 15608 / 1024),
        # stock 8x flagship at 0.93x keeps idle pay: the reason the floor is half, not all
        (8, 180, 1339.2),
    ],
)
async def test_a_node_with_enough_ram_keeps_idle_pay(gpus, vram, ram):
    result = await _incentive().calculate_executor_score(_node(gpus, vram, ram))

    assert "insufficient_ram_for_vram" not in _codes(result)


@pytest.mark.asyncio
async def test_a_scrape_without_ram_fails_open_and_says_so(caplog):
    node = _node(1, 64, 0)
    node.spec.pop("ram")

    with caplog.at_level("WARNING"):
        result = await _incentive().calculate_executor_score(node)

    assert "insufficient_ram_for_vram" not in _codes(result)
    assert "Cannot measure system RAM against GPU VRAM" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("vram", [32, 48, 80, 141, 21])
async def test_the_ram_the_rejection_names_earns_idle_pay(vram):
    # 32 GB VRAM: 16/1.05 = 15.238 rounded to nearest named 15.2 GB, which the gate still rejected
    measured = _incentive()._insufficient_ram(_node(1, vram, 8))

    result = await _incentive().calculate_executor_score(_node(1, vram, measured.required_ram_gb))

    assert "insufficient_ram_for_vram" not in _codes(result)
