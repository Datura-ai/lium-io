"""Suite-wide setup for the tester: the one boundary the stack does not have is the chain.

`services.ioc` constructs every validator service at import; `CollateralContractService` asks `SubtensorClient` for
its singleton, whose constructor dials the network in `BITTENSOR_NETWORK` and `exit()`s when the hotkey is not
registered there. The e2e validator is registered nowhere, so the client is kept but never connected — the same
state a production validator is in during a chain outage (`_subtensor is None`; every reader either degrades or
fails the way it does then). Nothing else is patched: miner, executor, SSH, docker, redis and the check pipeline
are the real code.
"""

import pytest


def _stub_out_the_chain() -> None:
    import clients.subtensor_client as sc

    if getattr(sc.SubtensorClient, "_e2e_no_chain", False):
        return
    sc.SubtensorClient.initialize_subtensor = lambda self: None  # never dial, never exit()
    sc.SubtensorClient.set_subtensor = lambda self: None
    sc.SubtensorClient._e2e_no_chain = True


_stub_out_the_chain()


@pytest.fixture(scope="session")
def no_chain():
    return True
