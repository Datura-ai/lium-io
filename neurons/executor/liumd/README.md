# Installing `liumd` in the executor image

Follows lium-io#1494, which adds the host files and the validator's shadow but ships no binary.

## Why the binary is not committed

A committed binary with an adjacent SHA-256 only proves the bytes did not change. It does not show
which source revision they were built from, and the image build would execute them as root.

## Source access

The `liumd` source lives in a repository this image build cannot read, so the image cannot be built
from a pinned source commit in a multi-stage build. The install therefore verifies provenance:

1. The companion's trusted CI builds `liumd` from a full commit hash and publishes the artifact
   with a detached signature made by a release key held in that CI.
2. The Dockerfile pins the release public key and the commit hash, downloads the artifact, and
   verifies the signature and that the signed statement names that commit, before the file is
   installed or run. A failed check fails the build.
3. Only then does the Dockerfile run the existing install steps: `install` the binary, `liumd.sh`
   as `/usr/local/bin/liumd`, the `liumd_host_files.py children` manifest and `liumd version`.

## To do before the install lands

- Create the release key and the signing step in the companion's CI (none exists yet).
- Restore `neurons/validators/tests/test_liumd_exec_e2e.py` from lium-io#1494 history
  (`4929da826:neurons/validators/tests/test_liumd_exec_e2e.py`); it runs the real binary and
  skips when it is absent.
