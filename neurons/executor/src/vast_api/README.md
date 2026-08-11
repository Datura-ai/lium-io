# vast_api — Vast.ai side manager for a Lium executor box

HTTP API that manages the Vast.ai marketplace side of a machine that already runs the
Lium executor: builds and runs the `vast-uns` nested-dockerd container, enrolls kaalia,
registers the machine, and handles list/unlist/price/status/delete. It executes the
proven SUCCESS-PATH recipe idempotently (setup ladder G0→G2; listing is a separate,
explicit pricing call).

**Target home: the executor app.** `router` is an executor-mountable FastAPI router; in
the product it mounts into the executor and rides the existing backend→executor auth.
For the experiment the same module runs standalone in a sidecar container ("shell") on
the same box, created *through executor-1's own docker socket* so the causal chain
platform → executor-1 → vast_api → vast-uns is honored:

```
docker exec executor-executor-1 docker run -d --name vast-manager \
  --privileged --pid host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v vast-api-runs:/var/lib/vast-api/runs \
  -v /path/to/vast_host_key:/run/secrets/vast_host_key:ro \
  -p 8151:8151 -e VAST_API_TOKEN=<token> \
  <vast-shell image>
```

Build the shell image from `neurons/executor/Dockerfile.vast-shell`. Bearer token is
mandatory on every route, `/healthz` included.

## Env vars (all overridable, defaults in `config.py`)

| var | default | meaning |
|---|---|---|
| `VAST_API_TOKEN` | — (required) | bearer token for every route |
| `VAST_ACCOUNT_KEY_FILE` | `/run/secrets/vast_host_key` | Vast host api key file (never logged) |
| `EXECUTOR_CONTAINER_NAME` | `executor-executor-1` | G0 gate: must be running |
| `VAST_UNS_NAME` | `vast-uns` | nested-dockerd container name |
| `VAST_UNS_IMAGE_TAG` | `vast-uns-kaalia:img` | image built from `assets/` |
| `STATE_DIR_HOST` | `/var/lib/vast-uns-state` | kaalia identity, survives container recreate |
| `DMI_BIN_HOST` | `/var/lib/vast-dmi.bin` | host DMI dump fed to the dmidecode shim |
| `DATA_ROOT_IMG` | `/ephemeral/vast-dockerd.img` | loop-XFS image for the nested data-root |
| `DATA_ROOT_MOUNT` | `/mnt/vast-dockerd` | mountpoint bound to nested `/var/lib/docker` |
| `DATA_ROOT_SIZE_GB` | `400` | data-root size |
| `PORT_RANGE_START` / `PORT_RANGE_END` | `40000` / `40300` | kaalia rental port range |
| `RUNS_DIR` | `/var/lib/vast-api/runs` | run docs (JSON files) |
| `LISTEN_PORT` | `8151` | shell listen port (outside the rental range) |
