from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG


def test_rtx_pro_6000_editions_are_anchored_at_parity():
    # DAH-3230: the two editions are the same card for a renter; the validator's unrented incentive
    # and the backend's MACHINE_PRICES anchor both at the Workstation price.
    prices = DEFAULT_SHARED_CONFIG.machine_prices

    server = prices["NVIDIA RTX PRO 6000 Blackwell Server Edition"]
    workstation = prices["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"]
    assert server == workstation == 1.0
