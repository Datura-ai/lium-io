# vast_api — Vast.ai side manager for a Lium executor box

Local Vast operations for a machine that already runs the Lium executor: builds and runs
the `vast-uns` nested-dockerd container, enrolls kaalia, registers the machine
(setup ladder G0→G2, the proven SUCCESS-PATH recipe, idempotent), reports local status
and tears the install down. Mounted into the executor app (`attach_vast_api` in
`wiring.py`), so the routes ride the executor's own port and `MinerMiddleware` auth:
every non-GET body carries a `MinerAuthPayload` signature; GET routes are open.

**No account key on the box** (plan-key-split): market operations
(list/unlist/price/self-test) and everything else needing the Vast account key live in
the backend. The box receives only a per-setup `machine_key` inside the signed
`POST /vast/setup` body — backend-minted, rotates on every mint.

Routes: `GET /healthz` · `POST /vast/setup` · `GET /vast/runs`, `GET /vast/runs/{id}` ·
`GET /vast/status` (local sections only) · `DELETE /vast` (local teardown; the backend
deletes the Vast account record first).

## Env vars (all overridable, defaults in `config.py`)

| var | default | meaning |
|---|---|---|
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

The trusted signer set for `/vast/*` additionally includes `VAST_ADMIN_HOTKEY`
(executor `core/config.py`) — the backend's dedicated keypair; it opens nothing outside
`/vast/*`.
