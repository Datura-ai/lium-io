# dm-crypt pod volumes (runbook)

An encrypted pod volume can be **gocryptfs** (inside the pod, the default) or **dm-crypt** (LUKS2 on the host). dm-crypt is much faster on small files and metadata. The backend asks for it per pod (`ContainerCreateRequest.volume_encryption_method = "dm-crypt"`, protocol 1.19.0). Stage 2: the backend asks only for internal pods on nodes it allowlists. Renters cannot choose it yet, and existing gocryptfs pods never change.

## How a volume is built

- **Backing file:** `<DockerRootDir>/lium-crypt/<pod_id>.img`, a loop device with direct I/O.
  - Sparse for a full-node rental, preallocated otherwise (the same rule as vloopback volumes).
- **Encryption:** LUKS2 `aes-xts-plain64`, 512-bit key, 4096-byte sectors, pbkdf2 with 1000 iterations.
  - The key is HKDF(`VOLUME_MASTER_SECRET`, pod_id) with info `lium-dmcrypt/v1`.
- **Mapping:** opened as `/dev/mapper/lium-<pod_id>` with no read/write workqueues (stored in the header), formatted ext4 with the host's e2fsprogs.
- **Docker volume:** the pod gets a Docker `local` volume (`type=ext4`, `device=/dev/mapper/lium-<pod_id>`), mounted at its workspace path. No `/dev/fuse`, no image label.
- **Where it runs:** the validator runs every step over SSH in the executor, inside the host's mount namespace (`nsenter -t 1 -m`).
- **The key** goes only on the SSH channel's stdin. It never appears in argv, the environment or the volume options.

## Host requirements

The scrape reports `dmcrypt_volume: true` when the host has `cryptsetup` (with `--perf-no_*_workqueue`), `dmsetup`, `losetup`, `udevadm`, `mkfs.ext4`, `fallocate`, `/dev/loop-control` and the `dm_crypt` module. The validator checks again at create time. A host that fails the check gets gocryptfs.

On Ubuntu: `apt-get install cryptsetup-bin`.

## Lifecycle

- **Create:** the loop file, LUKS2, open, `udevadm settle`, a check that the host device node exists, mkfs, then the Docker volume. If the create fails at any point, the validator removes the Docker volume, the mapping and the file.
- **Stop:** after the container stops, the validator closes the mapping and detaches the loop device; the file stays. A stopped pod is not decrypted on the host. Closing an already closed mapping does nothing. If the close fails, the stop is reported failed.
- **Start after a stop or a host reboot:** a stop closes the pod's mapping and a reboot closes every mapping. dockerd's own restart then fails at the mount ("failed to mount local volume … /dev/mapper/lium-…"), so the pod never runs on an empty directory. When the validator sees that error on a start, it opens the mapping and starts the pod again. This covers backend start, GPU reset, edit undo and reboot recovery in the rented-machine check.
- **Edit/reboot from the backend:** the backend sends the pod's stored method. The validator opens the existing volume before `docker run`.
- **Delete:** after the Docker volume is removed, the validator closes the mapping, detaches the loop device and removes the file. A mapping that is still mounted refuses to close and keeps its file.
- **Backup/restore:** the backend sends `volume_encrypted=false` for a dm-crypt pod, so the plain-volume path reads it. The mapping must be open, which it is only while the pod runs.
- **Disk accounting:** the fresh sizing adds the backing files' apparent size to the declared vloopback sizes.

## Check a node

- **Check the mappings:** `nsenter -t 1 -m dmsetup ls | grep lium-` lists the open mappings. `ls -ls <DockerRootDir>/lium-crypt` lists the backing files.
- **Look for leaks:** a file whose pod no longer exists is a leak. Remove it with the validator's delete command for that pod: close, `losetup -d`, `rm`.

## Turn it off

Empty the backend allowlist (`DMCRYPT_VOLUME_EXECUTOR_IDS`). New pods then get gocryptfs again. Keep the open, start and delete code paths while any dm-crypt pod exists. The data survives without them, but the pod cannot start until its mapping is opened.

## Not covered yet

- Online resize
- Hosts without udev
- Opt-in for renters
