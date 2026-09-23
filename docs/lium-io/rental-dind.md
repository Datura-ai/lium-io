# Docker-in-Docker on rented pods (DAH-3796)

What the validator adds to a sysbox customer rental so Docker inside the pod works, and what that
costs. Code: `neurons/validators/src/services/rental_dind.py`, wired in `docker_service.py` and
`rental_docker_sdk.py`. Everything applies when a container is created (a new pod, a reboot or an
edit); a running container is never changed.

## Settings (validator env, all off by default)

| Setting | Effect |
|---|---|
| `RENTAL_DIND_ADDRESS_POOLS_ENABLED` | `default-address-pools` from `RENTAL_DIND_ADDRESS_POOLS` (default `10.200.0.0/14` in /24s, 1024 networks) is merged into the pod's `/etc/docker/daemon.json` before its first start. Without it the inner dockerd stops at 29 networks. |
| `RENTAL_DIND_PERSISTENT_STORE_ENABLED` | A per-pod volume `volume_<pod>_docker` at `/var/lib/docker`: inner images, containers and volumes survive a reboot or an edit. Plain (unencrypted) pods only, unless the next setting is on too. |
| `RENTAL_DIND_PERSISTENT_STORE_ENCRYPTED_PODS_ENABLED` | Also give encrypted pods the store volume. Read the plaintext note below first. |
| `RENTAL_DIND_WORKSPACE_VOLUME_ENABLED` | Encrypted pods get a per-pod volume `volume_<pod>_workspace` at `/workspace`, a path whose bind mounts work in inner containers (the gocryptfs `/root` cannot be bind-mounted under sysbox). |

## Encrypted pods: these volumes are plaintext

An encrypted pod's own volume holds only ciphertext; the plaintext exists only inside the running
pod (gocryptfs at `/root`). The two volumes above are ordinary local volumes on the host disk:

- `volume_<pod>_workspace` holds whatever the renter writes to `/workspace`, in plaintext, until the
  pod is deleted. Before this setting, `/workspace` was on the pod's rootfs, also plaintext on the
  host, but it disappeared at every reboot or edit.
- `volume_<pod>_docker`, if enabled for encrypted pods, holds the inner images, containers and
  their volumes in plaintext until the pod is deleted.

A renter who needs everything encrypted at rest keeps their data under `/root` and uses
`/workspace` only for what inner containers must bind-mount.

## The address pools

The pools must not shadow an address the pod itself uses. The parser rejects any overlap with this
fixed list (`RESERVED_POD_RANGES`):

- `172.16.0.0/12`: the host daemon's bridge and pools (Docker's defaults, the sysbox installer's
  `172.20/16` bip and `172.25/16` pool, `neurons/executor/daemon.json`'s `172.24/16` and `172.31/16`),
  so the pod's own address on `lium-rentals`.
- `192.168.0.0/16`: Docker's other default pool.
- `10.42.0.0/24`: a cluster pod's WireGuard overlay.

At create, the pools are also checked against the subnets of the host's `lium-rentals` network as
the host daemon reports them. A provider whose daemon hands out addresses inside the pools gets
Docker's own pools in the pod, and the validator logs `Inner Docker daemon address pools` with
`outcome=skipped_pod_network_overlap: …`.

## The store across dockerd versions

After a pod starts, the validator writes the pod's `dockerd --version` line to
`/var/lib/docker/.lium-dockerd-version` in its store. Before the next create on that store, it
reads the marker with a helper container and runs the new image's `dockerd --version` under the
pod's runtime (no network, no mounts). If the new dockerd is older than the one that wrote the
store (an edit to a template with an older Docker), the store is emptied first, because an older
dockerd may not start on a newer store. Logged as `Inner Docker store version` with `outcome`
`no_marker`, `kept`, `reset_on_downgrade`, `reset_failed: …` or `failed: …`; on a failure the
store is kept as it is.

If an edit fails after the reset, the restored container also finds an empty store.

## Removal

Every path that removes a pod's volume removes both companion volumes, whatever the settings are
now: delete, the pre-create sweep, failed-create cleanup, the stale-container cleanup, the rental
probe's shell fallback, and the vloopback volume sweep. Companions whose pod volume is already gone
and that no container references are swept by the vloopback sweep at create and by the periodic
stale-container cleanup, which skips pods the backend still lists on the executor.
