# Publishing the attest-agent image (DAH-2632)

The attest-agent ships **through the measured customer compose**, injected by the backend
(`services/cvm_compose.py`) as an image reference pinned **by digest**. That digest is
folded into every derived renter `compose_hash`, which is what the host measures and the
validator verifies. Three consequences follow, and they are the whole of this document:

1. **Only a digest may be deployed.** `CVM_ATTEST_AGENT_IMAGE` must be of the form
   `ghcr.io/datura-ai/lium-attest-agent@sha256:<64 hex>`. A tag is refused by review, not
   by code: a mutable tag would mean the approved compose bytes stay approved while the
   image behind them changes.

2. **A new digest is a catalog rollout, never a swap.** Changing the env var changes the
   compose hash of every rental derived after it — but every rental derived *before* it
   still measures as the old hash, and validators verify rentals against what was sold.
   So a re-publish rolls out as:

   1. Run the `Publish: attest-agent image` workflow (or push an `attest-agent-v*` tag).
      The job summary prints the exact `CVM_ATTEST_AGENT_IMAGE` line to deploy.
   2. Set the new value in the target environment's config and restart the backend.
      New orders derive with the new digest from that moment.
   3. Leave the old digest's rentals alone: they re-attest against the compose hash the
      backend stored for their order (`pod.cvm_info.expectations`), so they stay valid
      until they end. There is nothing to migrate.

3. **The published image must never be a personal namespace.** The au11 hardware
   acceptance used a personal test namespace; that reference must not appear in staging
   or production config. The org image is `ghcr.io/datura-ai/lium-attest-agent`,
   published by CI from this directory with `GITHUB_TOKEN` — reproducible from the
   commit named in the job summary.

## Verifying a build before pinning it

The golden loop that matters: the backend's derive with the pinned digest must produce a
compose whose hash the host measures identically. That property is pinned by tests on
both sides (`lium-io-backend: test_dah2580_renter_provisioning.py` derives with a
digest-pinned image; `cvmd: tests/test_renter_launch.py` measures what dstack writes), so
a green suite on both repos plus the digest from the job summary is the whole check.

To inspect what a digest actually resolves to:

```bash
docker buildx imagetools inspect ghcr.io/datura-ai/lium-attest-agent@sha256:<digest>
```
