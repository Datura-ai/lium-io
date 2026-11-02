# Installing `liumd` in the executor image

## Current: PR artifact

The image vendors the `liumd` binary built from a lium-platform PR artifact:

```
neurons/executor/liumd/liumd-amd64
neurons/executor/liumd/liumd-arm64
neurons/executor/liumd/SOURCE.json
neurons/executor/liumd/provenance-amd64.json
neurons/executor/liumd/provenance-arm64.json
```

This is not a signed release. `SOURCE.json` records the source PR, source head and merge commits,
the workflow run, each target's embedded build sha and its sha256. The Dockerfile installs the
matching binary at `/usr/local/bin/liumd`, and `verify_liumd.py` checks the target, build sha and
sha256 during the image build.

This lets the executor and validator changes proceed without waiting for a `liumd-v*` tag. A
signed release replaces it when one exists.

The binary is standalone: it embeds VerifyX, matmul and inspector. Do not install `liumd.sh` over
`/usr/local/bin/liumd`; the client runs that path directly. No `children.json`, no Python wrapper
and no `/usr/local/lib/liumd` tree are needed.

## Signed release (later)

The signed release path verifies provenance instead of trusting a PR artifact:

1. The companion's trusted CI builds `liumd` from a full commit hash and publishes the artifact
   with a detached signature made by a release key held in that CI. The signed statement names
   both the commit and the artifact's SHA-256, so a statement cannot vouch for another release's
   binary.
2. The Dockerfile pins the release public key and the commit hash, downloads the artifact, and
   verifies the signature, that the signed statement names that commit, and that the downloaded
   file's SHA-256 equals the one in the statement, before the file is installed or run. A failed
   check fails the build.
3. Only then does the Dockerfile install the binary at `/usr/local/bin/liumd` and run
   `liumd version`.

The binary is not committed to lium-platform. The PR artifact above is the bridge until the first
signed release.

## To do when the signed release lands

- Replace the vendored binaries and `SOURCE.json` with the artifact download and signature check.
- Remove the `/etc/liumd` writer from `run.sh`; liumd reads the executor container env through
  Docker and ignores those files. The old Python path still uses them, so remove this with the
  liumd cutover, not before.
- Restore `neurons/validators/tests/test_liumd_exec_e2e.py` from this branch's history
  (`4929da826:neurons/validators/tests/test_liumd_exec_e2e.py`); it runs the real binary and
  skips when it is absent.
