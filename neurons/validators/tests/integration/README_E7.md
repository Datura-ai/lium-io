# E7 — local differential gate for the docker-service refactor (DAH-2382)

E7 is the differential harness from the characterization oracle (§E row E7).
It drives the live docker-service layers through a **real local docker daemon**
and pins the comparable surface (run spec, daemon-side container config, redis
call-order, delete boundary) to a golden snapshot. It is the only check that
catches mock-vs-reality drift and side-effect reordering that unit goldens
cannot see.

Decision **D-B** (2026-07-19): no docker-capable CI runner is budgeted, so E7
runs as a **mandatory LOCAL gate before every extraction PR** (PR-2..PR-12),
with the staging canary as the final backstop. Default CI collects this module
and skips it (same env gate as `test_rental_docker_sdk_local.py`).

## Prerequisites

- A local docker daemon (Docker Desktop is fine) — the test pulls `alpine:3.20`
  and creates/removes one container plus one named volume, all `e7`-named and
  cleaned up afterwards.
- docker-py reads `DOCKER_HOST`, not the docker CLI context. If the test skips
  with "local Docker daemon unavailable" while `docker info` works (Docker
  Desktop on macOS without the default-socket option), run with
  `DOCKER_HOST=unix://$HOME/.docker/run/docker.sock make e7`.
- The validator venv (`pdm install` in `neurons/validators/`).
- No GPU needed: PR-1a scope drives the GPU-less run-spec layer (see the module
  docstring of `test_docker_service_differential.py` for the exact scope).

## The one command

```bash
cd neurons/validators
make e7          # run the gate (byte-equal against the golden snapshot)
make e7-record   # re-record the golden after an INTENDED behavior change
```

Equivalent raw invocation:

```bash
RUN_RENTAL_DOCKER_SDK_INTEGRATION=1 pdm run pytest \
    tests/integration/test_docker_service_differential.py -q
```

## What extraction-PR authors must do (D-B, honour-system gate)

1. Run `make e7` locally against the branch **before pushing**.
2. Record the run in the PR description, e.g.:
   `E7: ran locally, 1 passed @ <commit sha>`.
3. If the snapshot drifted: either the extraction changed behavior (fix it), or
   the change is intended — then re-record with `make e7-record` and call the
   snapshot diff out in the PR as an explicit behavior change.

## PR-2+ evolution: old-vs-new

In PR-1a there is one implementation, so the harness asserts self-consistency
against the pinned golden. From PR-2 on, the same probe runs against the
pre-refactor facade (whose surface is the pinned golden) **and** the extracted
package — any drift between the two fails the gate. PR-2 also extends the probe
from the current narrow layer (run-spec build + live create retry loop + delete
boundary) to the full `create_container`/`delete_container` facade over the
`local_ssh_docker_endpoint` fixture, which populates the redis call-order
channel already present in the snapshot schema.
