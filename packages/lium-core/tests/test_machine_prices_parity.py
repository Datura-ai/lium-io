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
