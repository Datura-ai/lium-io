# CVM local E2E harness (DAH-2633)

One command that drives the whole renter path and the validation-CVM lifecycle against a
real TDX host, without deploying anything: local code from both repos, one remote cvmd,
an SSH tunnel in between.

```bash
neurons/validators/.venv/bin/python tools/cvm-local-e2e/run.py \
    --host ubuntu@<cvm-host> \
    --ssh-key ~/.ssh/<key> \
    --backend-src ../lium-io-backend/apps/server/src \
    --agent-image "ghcr.io/datura-ai/lium-attest-agent@sha256:<digest>"
```

A clean run walks: idle host → renter order (compose derived and signed by the backend's
own modules) → measured launch (the host's `compose_hash` must equal the derived one) →
scope proofs (the validator key can neither create nor destroy a renter CVM) → verified
teardown with the switch window observed as `SWITCHING` (the facts DAH-2630's grace
decides on) → validator-signed validation-CVM launch from the host's signed catalog
(DAH-2629's switch back) → final teardown, host left exactly as found.

## What it deliberately does and does not touch

- **Never binds port 32000.** The tunnel uses high local ports, and the run aborts before
  launching anything if the remote cvmd's configured port range could allocate 32000 —
  that port belongs to the production subnet on shared hosts.
- **Only an idle host.** If `/v1/state` shows any CVM, the run aborts rather than
  competing with whatever owns it.
- **Restores the host.** Unless `--keep-validation-cvm` is passed, the run ends with a
  verified teardown and asserts no disks and no staged composes remain.

## Keys

`--platform-uri` (default `//Bob`, renter scope) and `--validator-uri` (default
`//Alice`, validation scope) are the dev keys of a **test** cvmd that binds
`127.0.0.1` and is reached only through this tunnel. They must never appear in a
production `authorized_clients.json`. Point these at real keys when running against a
host whose daemon lists them.

## Full-stack mode (through the backend API)

The direct mode above exercises the real derivation, signing, measurement, scope, and
lifecycle code. To drive the same flow through the **backend API and the validator
relay** (DAH-2628's order path end to end), run the local stack and let it do the
talking:

1. Bring up the backend locally (postgres + redis + `apps/server`) with:
   - `CVM_RENTAL_ENABLED=true`
   - `CVM_ATTEST_AGENT_IMAGE=<digest-pinned image>`
   - `CVMD_URL_OVERRIDE=https://127.0.0.1:18443` (this tunnel)
   - a seeded artifact catalog (os image, QEMU, validation compose rows) whose
     entries match what the remote host has staged
2. Run a local validator connected to that backend, with
   `CVMD_URL_OVERRIDE=https://127.0.0.1:18443` so its relay and lifecycle client reach
   the same tunnel.
3. Place the order through the API and read access back:

   ```bash
   curl -X POST localhost:8100/executors/<uuid>/rent-cvm -d '{"compose": "..."}'
   curl localhost:8100/pods/<pod_id>/cvm
   ```

The tunnel, the safety guards, and the host expectations are identical in both modes —
the stack mode simply replaces this script's in-process calls with the deployed wiring.
