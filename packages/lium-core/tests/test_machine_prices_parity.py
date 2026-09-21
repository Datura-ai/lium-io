from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG

SERVER = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
WORKSTATION = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"


def test_rtx_pro_6000_editions_are_anchored_at_parity():
    """Fails when a price edit re-opens the gap between the two editions (0.86 vs 1.0 before
    DAH-3230): the two are the same card for a renter, so this table anchors both at the
    Workstation price. The backend's MACHINE_PRICES follows later, in lium-platform#250."""
    prices = DEFAULT_SHARED_CONFIG.machine_prices

    assert prices[SERVER] == prices[WORKSTATION]
    # the anchor moved the Server Edition up to the Workstation, not the Workstation down
    assert prices[WORKSTATION] > prices["NVIDIA RTX 6000 Ada Generation"]


def test_b300_sxm6_pc_matches_the_ac_card():
    # Host nvidia-smi 21 Sep 2026: name NVIDIA B300 SXM6 PC, 275040 MiB. Same class as the AC card.
    cfg = DEFAULT_SHARED_CONFIG
    assert cfg.machine_prices["NVIDIA B300 SXM6 PC"] == cfg.machine_prices["NVIDIA B300 SXM6 AC"]
    assert cfg.required_deposit_amount["NVIDIA B300 SXM6 PC"] == cfg.required_deposit_amount["NVIDIA B300 SXM6 AC"]
