# Publishing the guest-SSH image (DAH-2684)

The guest-SSH service ships the same way the attest-agent does — **through the measured
customer compose**, injected by the backend (`services/cvm_compose.py`) as an image
reference pinned **by digest**. That digest is folded into every derived renter
`compose_hash`, which the host measures and the validator verifies. The rules are the
attest-agent's rules (`../attest-agent/PUBLISHING.md`); restated here so nobody has to
guess whether they apply:

1. **Only a digest may be deployed.** `CVM_SSH_IMAGE` must be of the form
   `ghcr.io/datura-ai/lium-cvm-ssh@sha256:<64 hex>`. A tag would let the sshd that
   authenticates renters change underneath an approved measurement.

2. **A new digest is a catalog rollout, never a swap.** Rentals derived before the change
   keep measuring as their old hash and stay valid until they end (`pod.cvm_info.expectations`
   is per order). To re-publish:

   1. Run the `Publish: cvm-ssh image` workflow (or push a `cvm-ssh-v*` tag). The job
      summary prints the exact `CVM_SSH_IMAGE` line to deploy.
   2. Set it in the target environment's config and restart the backend. New orders derive
      with the new digest from that moment.
   3. Leave existing rentals alone; there is nothing to migrate.

3. **Never a personal namespace in staging or production config.** The org image is
   `ghcr.io/datura-ai/lium-cvm-ssh`, built by CI from this directory with `GITHUB_TOKEN`
   and reproducible from the commit in the job summary.

## Before pinning a build

`bash tests/test_guest_ssh.sh` against the built image is the mechanism check (see
README). The golden loop — the backend's derive with the pinned digest producing a compose
whose hash the host measures identically — is pinned by the same tests as the agent's
(`lium-io-backend: test_dah2580_renter_provisioning.py`, `cvmd: tests/test_renter_launch.py`).

To see what a digest resolves to:

```bash
docker buildx imagetools inspect ghcr.io/datura-ai/lium-cvm-ssh@sha256:<digest>
```
