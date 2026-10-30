from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG

SERVER = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
WORKSTATION = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
MAX_Q = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"


def test_rtx_pro_6000_editions_are_anchored_at_parity():
    """Fails when a price edit re-opens the gap between the two editions (0.86 vs 1.0 before
    DAH-3230): the two are the same card for a renter, so this table anchors both at the
    Workstation price. The backend's MACHINE_PRICES follows later, in lium-platform#250."""
    prices = DEFAULT_SHARED_CONFIG.machine_prices

    assert prices[SERVER] == prices[WORKSTATION]
    # the anchor moved the Server Edition up to the Workstation, not the Workstation down
    assert prices[WORKSTATION] > prices["NVIDIA RTX 6000 Ada Generation"]


def test_b300_sxm6_pc_matches_the_ac_card():
    # provider-observed on real hardware, 21 Sep 2026 (nvidia-smi: name NVIDIA B300 SXM6 PC, 275040 MiB, all 8 GPUs);
    # not in NVIDIA's public chip list — listed as the AC card's alias, derived from the AC entry
    cfg = DEFAULT_SHARED_CONFIG
    assert cfg.machine_prices["NVIDIA B300 SXM6 PC"] == cfg.machine_prices["NVIDIA B300 SXM6 AC"]
    assert cfg.required_deposit_amount["NVIDIA B300 SXM6 PC"] == cfg.required_deposit_amount["NVIDIA B300 SXM6 AC"]


def test_rtx_5090_d_is_its_own_class_below_the_5090():
    # DAH-3289: the China SKU gets its own price class, as the 4090 D does, not the 5090's price
    prices = DEFAULT_SHARED_CONFIG.machine_prices
    assert prices["NVIDIA GeForce RTX 5090 D"] < prices["NVIDIA GeForce RTX 5090"]
    assert "NVIDIA GeForce RTX 5090 D" in DEFAULT_SHARED_CONFIG.required_deposit_amount


def test_rtx_pro_6000_max_q_matches_the_workstation():
    # DAH-3453: the 300 W Max-Q variant of the same 96 GB card, priced at Workstation parity
    cfg = DEFAULT_SHARED_CONFIG
    assert cfg.machine_prices[MAX_Q] == cfg.machine_prices[WORKSTATION]
    assert cfg.required_deposit_amount[MAX_Q] == cfg.required_deposit_amount[WORKSTATION]
