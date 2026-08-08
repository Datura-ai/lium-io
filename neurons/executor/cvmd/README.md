# cvmd — CVM host daemon

The control-plane daemon that runs on a CVM host. It exposes a signed HTTPS API with two
authorized clients holding disjoint scopes, and it holds the node's state machine.

**DAH-2575 ships the skeleton only.** The mutating endpoints are registered with full auth and
scope enforcement but return `501` — the CVM operations behind them land in DAH-2576. That split
is deliberate: the scope matrix is testable end-to-end today, so 2576 fills in handlers rather
than also wiring auth.

## API

| Route | Auth | Now |
|---|---|---|
| `GET /health` | none | `200 {"version": ...}` |
| `GET /v1/state` | either key | the persisted state document |
| `POST /v1/cvm` | validator key for `kind=validation`, platform key for `kind=renter` | `501` |
| `DELETE /v1/cvm` | platform key | `501` |

A scope violation on an otherwise valid request is `403`. Everything else that fails auth is
`401`. A validly signed body that is not usable is `422`, never a scope bypass.

## Signing a request

Four headers: `X-Cvmd-Hotkey`, `X-Cvmd-Timestamp` (unix ns), `X-Cvmd-Nonce` (≥16 random bytes,
hex), `X-Cvmd-Signature` (hex, `0x` optional). The signature is over:

```
blob = sha256(
    b"cvmd-v1\x00"                       # domain separator
  | lp(method) | lp(request_target)      # request_target = path + query, exactly as received
  | lp(body_bytes)
  | lp(timestamp_ascii) | lp(nonce_ascii)
)
    where lp(x) = uint32_be(len(x)) | x
```

`src/cvmd/auth/blob.py` is the authoritative definition; `tests/fixtures/golden_vector.json` is
the reference vector for client implementations. Two deviations from architecture doc §03 are
recorded in that module's docstring — method and target are signed, and fields are
length-prefixed.

## Replay protection

A request is accepted only if it is inside the freshness window, above the startup floor, and its
`(hotkey, nonce)` pair is unseen. The nonce is fsynced **before** the request reaches a handler.

The startup floor is read once at startup and never advances during a process lifetime. That is
the whole design — a floor that advanced per request would be a strict monotonic timestamp, which
rejects the second of any two concurrent requests and every client retry. `tests/test_replay.py`
asserts both halves: replays are refused, and concurrent out-of-order requests are not.

## bittensor version

cvmd pins `bittensor==11.0.2`, deliberately independent of the executor's `9.0.0`. CVM hosts run
the Ubuntu 25.10/26.04 system Python (3.13/3.14), which 9.x does not support. The split is safe
because a 9.x signature verifies under 11.x — `tests/test_golden_vector.py` pins that with a
fixture signed under a real 9.10.1 venv. Do not "fix" the mismatch by downgrading.

Two v11 API breaks the executor's middleware predates: `bittensor.Keypair` moved to
`bittensor.sp_core.Keypair`, and `verify()` is strictly bytes-typed.

## Development

```bash
pdm use -f python3.13
pdm install
pdm run pytest
```

## Packaging

```bash
./packaging/build.sh
```

Produces `dist/cvmd-<version>.tar.gz` and prints its sha256 — the value the DAH-2544 Ansible role
takes as `lium_cvmd_package_sha256`. The tarball carries the wheel, a hash-pinned
`requirements.lock`, the unit file, a default config, and `install.sh`.
