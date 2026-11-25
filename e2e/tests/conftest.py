"""Suite-wide setup for the tester: the one boundary the stack does not have is the chain.

`services.ioc` constructs every validator service at import; any of them that asks `SubtensorClient` for its
singleton gets a constructor that dials `BITTENSOR_CHAIN_ENDPOINT` (else the `BITTENSOR_NETWORK` node) and `exit()`s when the hotkey is not
registered there. The e2e validator is registered nowhere, so the client is kept but never connected — the same
state a production validator is in during a chain outage (`_subtensor is None`; every reader either degrades or
fails the way it does then). Nothing else is patched: miner, executor, SSH, docker, redis and the check pipeline
are the real code.
"""

import asyncio

# One event loop for the whole tester process: services.ioc builds the validator's clients (redis, aiohttp) on the
# loop that is current at import, and asyncio.run() would hand every test a fresh loop those clients are not bound
# to ("attached to a different loop"). Suites call lib.run(coro), which uses this loop.
LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)


def _stub_out_the_chain() -> None:
    import clients.subtensor_client as sc

    if getattr(sc.SubtensorClient, "_e2e_no_chain", False):
        return
    sc.SubtensorClient.initialize_subtensor = lambda self: None  # never dial, never exit()
    sc.SubtensorClient.set_subtensor = lambda self: None
    sc.SubtensorClient._e2e_no_chain = True


_stub_out_the_chain()


def _admit_the_seeded_executor() -> None:
    """The validator dials only public node addresses (services.node_host, DAH-6782). The stack's executor and miner sit
    on the compose network at E2E_EXECUTOR_IP and E2E_MINER_IP (private addresses by design), so exactly those two are
    let through, for this tester process only; every other private, loopback or link-local literal is still refused as
    in production."""
    import os

    import services.node_host as node_host

    if getattr(node_host, "_e2e_admitted", None):
        return
    seeded, check = {os.environ["E2E_EXECUTOR_IP"], os.environ["E2E_MINER_IP"]}, node_host._check_literal

    def check_literal(host: str, port: int) -> None:
        if host.strip("[]") not in seeded:
            check(host, port)

    node_host._check_literal = check_literal
    node_host._e2e_admitted = seeded


_admit_the_seeded_executor()
