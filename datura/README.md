# datura

The protocol package shared by the three neurons in this repository. It has no
dependencies of its own and is installed from the working tree by
`neurons/validators`, `neurons/executor` and `neurons/miners`
(`datura @ file:///…/datura` in each `pyproject.toml`), so a change here is
picked up by all three on their next `pdm install`.

What lives here:

- `datura/requests/` — pydantic models shared by the neurons. The miner's
  `POST /executors` accepts `SimpleValidatorRequest` and returns `ExecutorListResponse`.
  SSH-key submission, SSH-key removal and pod-log requests authenticate with signed HTTP
  headers built from `AuthenticationPayload`; their successful responses are
  `AcceptSSHKeyRequest`, `SSHKeyRemoved` and `PodLogsResponse`, and a refused request
  answers `FailedRequest`.
- `datura/errors/protocol.py` — the protocol-level exceptions.

Adding a message type means adding a model in the right `*_requests.py` and a
member to its `RequestType` enum; the validator and the miner must ship the
change together, since neither side tolerates an unknown type.
