# datura

The protocol package shared by the three neurons in this repository. It has no
dependencies of its own and is installed from the working tree by
`neurons/validators`, `neurons/executor` and `neurons/miners`
(`datura @ file:///…/datura` in each `pyproject.toml`), so a change here is
picked up by all three on their next `pdm install`.

What lives here:

- `datura/requests/` — the WebSocket message types the validator and the miner
  exchange, as pydantic models: `validator_requests.py` (what the validator
  sends: authentication, SSH-key submit/remove, pod-log requests) and
  `miner_requests.py` (what the miner answers: accept/decline job, executor SSH
  info, generic error). `base.py` holds `BaseRequest`, which parses an incoming
  JSON frame into the right subclass by its `message_type`.
- `datura/consumers/base.py` — `BaseConsumer`, the FastAPI WebSocket
  receive/send loop both sides build their consumers on.
- `datura/errors/protocol.py` — the protocol-level exceptions.

Adding a message type means adding a model in the right `*_requests.py` and a
member to its `RequestType` enum; the validator and the miner must ship the
change together, since neither side tolerates an unknown type.
