"""The shape a neuron's chain client comes back in: the client and which setting chose its endpoint.

The validator (`bittensor.Subtensor`) and the central miner (`bittensor.AsyncSubtensor`) both dial our own
chain endpoint first and fall back to the public network node; their `_connect_subtensor` return this
instead of a bare tuple so the label is a named field, not a positional string.
"""

from typing import Generic, Literal, NamedTuple, TypeVar

SubtensorT = TypeVar("SubtensorT")

EndpointSource = Literal[
    "BITTENSOR_CHAIN_ENDPOINT",
    "BITTENSOR_NETWORK",
    "BITTENSOR_NETWORK (own endpoint failed)",
]


class ChainConnection(NamedTuple, Generic[SubtensorT]):
    subtensor: SubtensorT
    endpoint_source: EndpointSource
