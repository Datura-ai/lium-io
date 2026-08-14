# `attest-agent`

The renter CVM's trust-check agent (DAH-2579). It answers one question — "prove what you
are, right now, for this challenge" — and it can do nothing else.

## Why it ships in the compose, not in the image

FR-E6 puts one small Lium agent inside a renter CVM. Baking it into the OS image would
have been simpler and is the wrong trade twice over:

- **Updating it would mint a new image hash.** Every host would need the new image staged
  and the new hash approved before a single agent could be patched — so a one-line fix to
  the agent becomes a fleet-wide image rollout.
- **The customer could not see it.** Shipped in the compose *they* submitted, the agent is
  a service they can read, alongside a `read_only` filesystem and no socket. In the image
  it would be something running in their CVM that they have to take on trust.

The cost is that the agent is inside the compose measurement, so changing it changes the
`compose_hash`. That is the correct direction: a changed agent *should* be a changed
measurement, because it is part of what the CVM is.

## What a trust check gets

```
POST /v1/attest  {"nonce": "<64 hex>"}
GET  /health
```

Those two routes are the entire surface, and a test asserts it. There is no exec, no log
endpoint, no path to the workload — the injected compose block grants no docker socket and
no volume the customer did not declare.

The answer carries a fresh TDX quote and NVIDIA CC evidence, both bound to the caller's
nonce:

```
report_data = sha256("LIUM_RENTER_ATTEST_TLS_V1\0" ‖ tls_spki_der ‖ gpu_uuid_digest) ‖ nonce
              |_______________________ 32 bytes ______________________________|  |_ 32 _|
                                 identity                                         freshness
```

**The tag is versioned and NUL-terminated** so these 32 bytes state their purpose. Without
it the digest is just "a hash over a TLS key", indistinguishable from any other use of the
same key material, and a later recipe would be the same bytes read a second way instead of
a new construction a verifier can accept or refuse by name.

**The TLS key is in there because the quote and the channel have to be the same thing.**
The validator's only way into a renter CVM is this agent's TLS endpoint. A quote that did
not name the key proves some TDX guest exists somewhere; an attacker who can terminate TLS
relays a genuine quote from a CVM it does not own, and the verifier cannot tell. The key is
generated *inside* the guest on first boot, so its private half exists only in encrypted
CVM memory and on the CVM's encrypted disk. It is hashed as its SubjectPublicKeyInfo DER —
the bytes a client sees on the wire, since that is all a verifier holds.

**The GPU UUIDs are in there because of FR-G6** — so the GPU claim travels inside the
hardware's signature instead of beside it, where the host could rewrite it or move it onto
another quote.

**What that does not do is authenticate the GPUs.** NVML UUIDs are read by the guest and
vouched for by nobody. Whether those GPUs exist, are genuine, are in CC mode, and are not
also answering for another node is decided on the per-GPU `ueid` claims in the NVIDIA
evidence returned beside the quote — the one GPU identifier here a node cannot choose. The
validator counts ueids, and refuses a response with a missing or repeated one.

## Idempotent, not amnesiac

Asked twice with the same nonce, the agent returns the identical answer and takes no second
quote. It never produces two different quotes for one challenge.

Refusing a repeat outright would break a verifier's retry after a network timeout — and it
would not add anything, because **rejecting a *reused* nonce is the verifier's half**: only
the party that issued a nonce knows whether it has seen it before. `TestTheVerifiersHalf`
in the test suite pins that half next to the agent whose answers it judges.

The ledger is bounded, because it is keyed on caller-supplied input and an unbounded map
would be memory the caller controls.

## Degrading honestly

A CVM with no GPUs, or a build without the NVIDIA verifier stack, returns `null` GPU
evidence **with a stated reason** — never an empty list. "This CVM holds no GPUs" and "this
agent could not tell" lead to opposite decisions on the validator's side, so they do not
share an encoding.

When evidence is produced it is returned as the payload NRAS accepts —
`{"nonce", "evidence_list", "arch"}` — not a bare evidence list, which NRAS will not judge.
The collection flow mirrors the executor's own collector, single-GPU branch included.

The dstack SDK and the NVIDIA stack are not declared in `pyproject.toml`: both exist only
inside a CVM and both drag fragile native trees, so declaring them would make the agent
uninstallable anywhere else — including in the environment that tests what it does when
they are absent. The Dockerfile adds them; `evidence.py` imports them at the point of use.

## Running the tests

```bash
cd neurons/executor/attest-agent
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python fastapi uvicorn pydantic cryptography pytest httpx
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

The quote and the GPU evidence are the seam — there is no TDX and no CC-capable GPU in a
test process. Everything around them is real, including the EC key. What the suite checks
is the part hardware cannot: that the bytes the agent *says* it bound are the bytes it
asked the hardware to sign.
