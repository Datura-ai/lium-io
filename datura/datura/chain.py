"""The chain endpoints a neuron dials, in order, and the shape its chain client comes back in.

The validator (`bittensor.Subtensor`) and the central miner (`bittensor.AsyncSubtensor`) dial an
ORDERED list: our own endpoints first (`BITTENSOR_CHAIN_ENDPOINTS`, comma-separated, else the single
`BITTENSOR_CHAIN_ENDPOINT`), the public `BITTENSOR_NETWORK` node always last. On a connect or read
failure the client moves to the next entry and logs `Subtensor endpoint switched from=… to=…`; on
the next sync cycle it goes back to the first one, so a proxy outage never leaves a neuron without a
chain client and a recovered proxy is picked up again without a restart (taiberium, lium-io#1393).
"""

from typing import Generic, NamedTuple, TypeVar

SubtensorT = TypeVar("SubtensorT")

# Which setting chose the endpoint: `BITTENSOR_CHAIN_ENDPOINTS[<i>]`, `BITTENSOR_CHAIN_ENDPOINT`
# or `BITTENSOR_NETWORK`, with ` (own endpoint failed)` appended when it was reached by a switch.
EndpointSource = str

PUBLIC_NODE_SOURCE = "BITTENSOR_NETWORK"
SWITCHED_SUFFIX = " (own endpoint failed)"


class ChainEndpoint(NamedTuple):
    value: str  # a ws:// URL or a network name, the `network=` argument of the Subtensor constructor
    source: EndpointSource


class ChainConnection(NamedTuple, Generic[SubtensorT]):
    subtensor: SubtensorT
    endpoint_source: EndpointSource


def chain_endpoint_candidates(
    *, chain_endpoints: str | None, chain_endpoint: str | None, network: str
) -> list[ChainEndpoint]:
    """The ordered dial list for these settings.

    `chain_endpoints` (comma-separated) wins over the single `chain_endpoint`; blanks are dropped and
    a duplicate of the public network name is not listed twice. Providers set neither, so their list
    is just the network name, as before.
    """
    own: list[ChainEndpoint] = []
    if chain_endpoints and chain_endpoints.strip():
        for index, raw in enumerate(chain_endpoints.split(",")):
            value = raw.strip()
            if value and value != network and value not in {e.value for e in own}:
                own.append(ChainEndpoint(value, f"BITTENSOR_CHAIN_ENDPOINTS[{index}]"))
    elif chain_endpoint and chain_endpoint.strip() and chain_endpoint.strip() != network:
        own.append(ChainEndpoint(chain_endpoint.strip(), "BITTENSOR_CHAIN_ENDPOINT"))
    return own + [ChainEndpoint(network, PUBLIC_NODE_SOURCE)]


class EndpointCursor:
    """Which entry of the dial list a client is on. `advance()` moves to the next entry (wrapping to
    the first when the last one fails too) and reports the switch; `reset()` returns to the first
    entry for the next sync cycle."""

    def __init__(self, candidates: list[ChainEndpoint]):
        if not candidates:
            raise ValueError("chain endpoint list is empty")
        self.candidates = candidates
        self.index = 0

    @property
    def current(self) -> ChainEndpoint:
        return self.candidates[self.index]

    @property
    def on_first(self) -> bool:
        return self.index == 0

    def advance(self) -> tuple[ChainEndpoint, ChainEndpoint]:
        previous = self.current
        self.index = (self.index + 1) % len(self.candidates)
        return previous, self.current

    def reset(self) -> None:
        self.index = 0

    def source_label(self) -> EndpointSource:
        """The `endpoint_source` for the current entry: plain on the first entry, marked when a switch
        brought the client here."""
        source = self.current.source
        return source if self.on_first else f"{source}{SWITCHED_SUFFIX}"
