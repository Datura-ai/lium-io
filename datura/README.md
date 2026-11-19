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
  `AcceptSSHKeyRequest`, `SSHKeyRemoved` and `PodLogsResponse`. The handlers return
  `FailedRequest` for operation errors they catch. A missing required auth header, or an
  `X-Timestamp` that is not an integer, is refused by FastAPI's header validation with HTTP 422
  and a validation-error list in `detail`. The other refusals carry a string `detail`: HTTP 400 for
  a timestamp before 2000-01-01 after normalization (values above 1e12 are read as milliseconds),
  HTTP 401 for a timestamp outside the allowed window after normalization or an invalid or malformed signature, and
  HTTP 403 for an unregistered validator hotkey. On a single-hotkey miner an `X-Miner-Hotkey` other than
  this miner's own hotkey also gets 403. A central miner serves many portal hotkeys, so any
  `X-Miner-Hotkey` is accepted, and a request body that names a different miner than the header gets 403.
- `datura/errors/protocol.py` — the protocol-level exceptions.

Adding a message type means adding a model in the right `*_requests.py` and a
member to its `RequestType` enum; the validator and the miner must ship the
change together, since neither side tolerates an unknown type.
