"""The chain endpoints a neuron dials, in order, and the shape its chain client comes back in.

The validator (`bittensor.Subtensor`) and the central miner (`bittensor.AsyncSubtensor`) dial an
ORDERED list: our own endpoints first (`BITTENSOR_CHAIN_ENDPOINTS`, comma-separated, else the single
`BITTENSOR_CHAIN_ENDPOINT`), the public `BITTENSOR_NETWORK` node always last. On a connect or read
failure the client moves to the next entry and logs `Subtensor endpoint switched from=… to=…`. The
failed entry is not dialled again for `retry_after_seconds` (`BITTENSOR_CHAIN_ENDPOINT_RETRY_AFTER_SECONDS`,
5 minutes by default), so a dead proxy does not stall every sync cycle with a new dial; after that
window the next sync cycle goes back to it. A proxy outage never leaves a neuron without a chain
client, and a recovered proxy is picked up again without a restart (taiberium, lium-io#1393).
"""

from time import monotonic
from typing import Generic, NamedTuple, TypeVar

SubtensorT = TypeVar("SubtensorT")

# Which setting chose the endpoint: `BITTENSOR_CHAIN_ENDPOINTS[<i>]`, `BITTENSOR_CHAIN_ENDPOINT`
# or `BITTENSOR_NETWORK`, with ` (own endpoint failed)` appended when it was reached by a switch.
EndpointSource = str

PUBLIC_NODE_SOURCE = "BITTENSOR_NETWORK"
DEFAULT_ENDPOINT_RETRY_AFTER_SECONDS = 300
SWITCHED_SUFFIX = " (own endpoint failed)"

# A connect or read on the websocket / substrate client, not Redis, the portal, or the database.
_CHAIN_ERROR_TYPES = (TimeoutError, ConnectionError)
_CHAIN_ERROR_MODULES = (
    "websocket",
    "websockets",
    "substrateinterface",
    "async_substrate_interface",
    "scalecodec",
)


def is_chain_error(error: BaseException) -> bool:
    """True when `error` came from the chain client (connect, websocket, RPC).

    A database error, a Redis miss, or a portal HTTP failure is False, so a healthy
    proxy is not abandoned for a local fault.
    """
    if isinstance(error, _CHAIN_ERROR_TYPES):
        return True
    module = type(error).__module__ or ""
    return any(module == prefix or module.startswith(prefix + ".") for prefix in _CHAIN_ERROR_MODULES)


class ChainEndpoint(NamedTuple):
    value: str  # a ws:// URL or a network name, the `network=` argument of the Subtensor constructor
    source: EndpointSource


class EndpointAdvance(NamedTuple):
    """What `EndpointCursor.advance()` did: the entry that failed and the one the cursor is on now."""

    previous: ChainEndpoint
    current: ChainEndpoint


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
    """Which entry of the dial list a client is on. `advance()` marks the current entry as failed
    and moves to the next entry that is not resting (wrapping to the first when the last one fails
    too). A failed entry rests for `retry_after_seconds`. `move_to_first_ready_endpoint()` moves to
    the first entry that is not resting, at the start of a sync cycle."""

    def __init__(self, candidates: list[ChainEndpoint], *, retry_after_seconds: float):
        if not candidates:
            raise ValueError("chain endpoint list is empty")
        self.candidates = candidates
        self.index = 0
        self.retry_after_seconds = retry_after_seconds
        self._resting_until: dict[int, float] = {}

    @property
    def current(self) -> ChainEndpoint:
        return self.candidates[self.index]

    @property
    def on_first(self) -> bool:
        return self.index == 0

    def advance(self) -> EndpointAdvance:
        previous = self.current
        self._resting_until[self.index] = monotonic() + self.retry_after_seconds
        next_index = (self.index + 1) % len(self.candidates)
        ready = self._first_ready_index(start=next_index)
        # every entry is resting: take the next one anyway, never stop dialling
        self.index = next_index if ready is None else ready
        return EndpointAdvance(previous, self.current)

    def move_to_first_ready_endpoint(self) -> bool:
        """Move to the first entry that is not resting. True when the cursor moved."""
        ready = self._first_ready_index(start=0)
        if ready is None or ready == self.index:
            return False
        self.index = ready
        return True

    def _first_ready_index(self, *, start: int) -> int | None:
        now = monotonic()
        count = len(self.candidates)
        for step in range(count):
            index = (start + step) % count
            if self._resting_until.get(index, 0.0) <= now:
                return index
        return None

    def source_label(self) -> EndpointSource:
        """The `endpoint_source` for the current entry: plain on the first entry, marked when a switch
        brought the client here."""
        source = self.current.source
        return source if self.on_first else f"{source}{SWITCHED_SUFFIX}"
