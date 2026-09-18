# CVM host setup — dstack-nvidia-0.5.11

Provider guide for taking a TDX host from bare metal to an attested Lium CVM executor. The quick start in [../README.md](../README.md) assumes the host is already prepared; this document covers that preparation. Almost everything here is **one-time per host** — the per-executor work is one `.env` file and two `lium-cvm.sh` commands.

| Phase | Frequency | Hands-on time |
|---|---|---|
| §1–§4 host bring-up (BIOS, kernel, QEMU, key provider) | once per host | ~1–2 h (dominated by BIOS + QEMU build) |
| §5 executor deployment | once per executor | ~15 min |
| §6 upgrades | per release | ~10 min |
| §6.1 key-provider rebuild | only when a release changes `key-provider/` | ~15 min plus the CVM recreation |

## 1. Hardware and firmware

- **CPU**: Intel Xeon with TDX **and** SGX — Sapphire Rapids (XCC/MCC SKUs only), Emerald Rapids, or Granite Rapids. SGX is not optional: the sealing-key provider runs in an SGX enclave.
- **GPU**: NVIDIA Confidential-Computing capable — Hopper (H100/H200) or Blackwell (B200/B300). Hopper: CC mode for a single GPU per CVM, Protected PCIe (GPUs **and** NVSwitches, `CVM_GPUS=all` — a listed set leaves the NVSwitches on the host) for a whole 8× HGX board. Blackwell: CC mode only. Commands and checks: [Enable GPU Confidential Computing mode](https://docs.lium.io/providers/nodes/cvm#4-enable-gpu-confidential-computing-mode). Bring-up scripts written against the older single-GPU guidance, such as `cvm-host-phase2.sh`, set Protected PCIe back to off — keep Protected PCIe nodes out of them and re-query the mode after any run.
- **BIOS**: latest vendor BIOS; enable TDX, SGX, VT-x/VT-d. After boot, `/dev/sgx_enclave` and `/dev/sgx_provision` must exist.

## 2. Kernel and boot parameters

Any kernel with KVM TDX support works: Canonical's Intel-optimized kernel (`6.14.0-1009-intel` tested) or mainline **≥ 6.16** (KVM TDX merged upstream; 6.17 is what the full attestation chain was validated on).

`GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub` must include:

```
kvm_intel.tdx=on intel_iommu=on iommu=pt nohibernate \
vfio-pci.ids=<your GPU/NVSwitch PCI IDs, e.g. 10de:2335> \
vfio_iommu_type1.dma_entry_limit=16777216
```

- `vfio_iommu_type1.dma_entry_limit=16777216` is **required**, not tuning. The launcher passes GPUs through via the legacy type1 VFIO container (see §3), and TDX shared↔private page conversions exhaust the default 65 535 DMA mappings — QEMU dies mid-boot with `VFIO_MAP_DMA failed: No space left on device`. To apply without a reboot: `echo 16777216 | sudo tee /sys/module/vfio_iommu_type1/parameters/dma_entry_limit` (runtime-only; the GRUB entry is what persists).
- `vfio-pci.ids=` must cover the GPUs **and** any NVSwitches being passed through.

Then `sudo update-grub && sudo reboot`, and verify:

```bash
cat /sys/module/kvm_intel/parameters/tdx     # Y
lspci -nnk -s <gpu-addr>                      # Kernel driver in use: vfio-pci
```

## 3. QEMU — the dstack 9.2.1 build (required for attestation)

**Why this specific QEMU:** the validator's dstack-verifier reconstructs RTMR0 using an ACPI oracle built from the dstack QEMU 9.2.1 tree. RTMR0 only reproduces when the guest actually runs under that same QEMU. A host on distro QEMU ≥ 10.x can **never** pass — the oracle cannot forward-emulate newer ACPI. Once attestation enforcement is enabled on validators, a wrong host QEMU means failed attestation and a zero score.

**How often:** install **once per host**. Rebuild only when a release note says the verifier's QEMU changed — never per executor, per reboot, or per runner release. (A prebuilt tarball per supported Ubuntu release is planned; until then, build from source.)

```bash
sudo apt-get install -y --no-install-recommends build-essential git ninja-build pkg-config \
    python3-pip python3-venv libglib2.0-dev libpixman-1-dev libslirp-dev flex bison

git clone https://github.com/kvinwang/qemu-tdx.git --depth 1 \
    --branch dstack-qemu-9.2.1 --single-branch qemu-tdx-src
cd qemu-tdx-src
git fetch --depth 1 origin dbcec07c0854bf873d346a09e87e4c993ccf2633
git checkout dbcec07c0854bf873d346a09e87e4c993ccf2633   # pin: the exact tree the verifier oracle is built from

mkdir build && cd build
../configure --prefix=/opt/qemu-dstack --target-list=x86_64-softmmu \
    --disable-werror --enable-kvm --enable-slirp
make -j"$(nproc)"
sudo make install

/opt/qemu-dstack/bin/qemu-system-x86_64 --version   # QEMU emulator version 9.2.1
```

This is a plain build — do **not** add the `DUMP_ACPI_TABLES` define (that variant is only for the verifier's oracle binary and cannot run VMs).

Point the launcher at it via `/etc/dstack/client.conf`:

```ini
[qemu]
path = /opt/qemu-dstack/bin/qemu-system-x86_64
```

(`dstack.py` merges `/etc/dstack/client.conf`, `~/.config/dstack/client.conf`, and any `.dstack/client.conf` at or above the working directory, later files overriding earlier ones. `/etc/dstack/client.conf` is recommended because `lium-cvm.sh` runs under sudo.)

Optionally symlink it so `lium-cvm.sh check` finds a QEMU on PATH:

```bash
sudo ln -s /opt/qemu-dstack/bin/qemu-system-x86_64 /usr/local/bin/qemu-system-x86_64
```

The launcher handles everything else automatically: it stamps `qemu_version` and `ovmf_variant` into the measured vm_config, and uses the legacy type1 VFIO backend for QEMU < 10 (this build's iommufd binding fails with EINVAL on modern kernels — expected, handled).

## 4. Key provider and PCCS

The sealing-key provider must run on the host before any CVM boots (`lium-cvm.sh run` auto-starts it, but starting manually first surfaces build errors early). Start it through the upgrade guard. `key-provider/docker-compose.yaml` has no `build:` section, so `docker compose build` or `docker compose up` in that directory builds nothing; the build definition is `key-provider/docker-compose.build.yaml`, and only the guard passes it to compose:

```bash
sudo ./cvm_upgrade_guard.sh start                     # builds on a host with no CVM disk, else starts the pinned image
cd key-provider && docker compose logs -f gramine-sealing-key-provider   # watch first start
curl -k https://localhost:3443                        # endpoint reachable
```

Two containers: `aesmd` (SGX architectural enclaves, host network) and `gramine-sealing-key-provider` (`127.0.0.1:3443`). DCAP collateral comes from the PCCS configured in [`key-provider/sgx_default_qcnl.conf`](../key-provider/sgx_default_qcnl.conf) — the default is Phala's public PCCS and works out of the box; point `pccs_url` at your own PCCS if you run one.

**Why the guard.** The key provider's enclave measurement (MRENCLAVE) seals the key every CVM on the host derives its encrypted data-disk key from. A rebuilt image has a new MRENCLAVE, so every existing CVM data disk on the host, stopped CVMs included, becomes unreadable at its next boot. `cvm_upgrade_guard.sh` records the image id of the key provider in `/var/lib/lium-cvm/key-provider.image` on first start, tags it `lium-key-provider:pinned-<id>`, and every later `start` (also the one inside `lium-cvm.sh run`) runs `docker compose up -d --no-build` on exactly that image. If the pinned image is missing while CVM disks exist, `start` fails with recovery steps and builds nothing. The guard also refuses to run (exit 1) if `docker-compose.yaml` gains a `build:` section again, or if a file compose loads on its own (`docker-compose.yml`, `docker-compose.override.yaml`, `compose.yaml` and their variants) sits in `key-provider/`, because either lets a hand-run `docker compose build` rebuild the enclave. The guard's own compose calls name `docker-compose.yaml` with `-f`, so such a file or a `COMPOSE_FILE` variable never changes what the guard runs. `lium-cvm.sh new`, `lium-cvm.sh run` and the guard share one host lock (`/var/lock/lium-cvm.lock`), so a CVM cannot fetch a key between the guard's inventory check and an image switch. A host that predates the guard adopts the image its `dstack-key-provider` container runs on the first `start`.

## 5. Per-executor deployment

1. Configure:

   ```bash
   cp .env.example .env
   ```

   Required beyond the obvious (`MINER_HOTKEY_SS58_ADDRESS`, ports, `CVM_VCPUS/MEMORY/DISK`, `CVM_GPUS`):

   - `ENABLE_TDX_ATTESTATION=true`
   - `EXECUTOR_RUNNER_IMAGE_DIGEST=sha256:<64-hex>` — copy from the "CVM attestation" section of the release notes of the tag you checked out. While the notes of your tag do not carry that section yet (no released tag has it before the first release after this change), `python3 scripts/compose_hash.py --release-notes` prints the same section for the checkout: the digest and the expected compose hash. `lium-cvm.sh new` pins this digest into the measured compose (the attested trust boundary) and refuses to run without it.

2. Create, check the measurement, boot (the OS image downloads once per host, then is reused):

   ```bash
   sudo ./lium-cvm.sh new my-executor
   python3 scripts/compose_hash.py --check run/vms/my-executor/shared/app-compose.json
   sudo ./lium-cvm.sh run my-executor
   ```

   `--check` prints the expected compose hash (the one in the release-notes section) and the hash of the file `new` wrote, and exits 1 when they differ, so there is nothing to compare by eye. `sha256sum run/vms/my-executor/shared/app-compose.json` prints the same actual hash on its own. A mismatch means the digest or a measured file differs from the release and the validator does not know the hash. What that costs depends on the validator's `ENABLE_ATTESTATION_WHITELIST`: on, the CVM scores zero; off (the default, and production today), the validator accepts the unknown hash and scores the node as before. Do not run a CVM that does not match the release either way: fix `.env` or `git checkout` the release tag, `sudo rm -rf run/vms/my-executor`, and run `new` again.

   Everything attestation-related inside the guest — sysbox force-install, digest-pinned runner, quote generation — is baked into the measured compose; there is nothing to configure in the guest.

3. Verify: the executor API answers on `EXTERNAL_PORT` and SSH banners on `SSH_PORT` within ~3–5 minutes of boot. `./lium-cvm.sh list` shows the VM.

4. Register the executor with your miner exactly like a non-CVM executor. Attestation runs automatically when the validator connects.

**Do not modify** `app/docker-compose*.yml`, `app/pre_launch_script.sh`, or `app/init_script.sh`: they are measured into the compose hash the validator whitelists — any local change makes attestation fail.

## 6. Upgrades

Per release:

```bash
sudo ./lium-cvm.sh stop my-executor
git pull                                   # the release tag
# update EXECUTOR_RUNNER_IMAGE_DIGEST in .env from the release notes ("CVM attestation";
# no section yet → python3 scripts/compose_hash.py --release-notes prints it for the checkout)
sudo rm -rf run/vms/my-executor            # see warning below
sudo ./lium-cvm.sh new my-executor
python3 scripts/compose_hash.py --check run/vms/my-executor/shared/app-compose.json   # exit 1 = not the release's compose
sudo ./lium-cvm.sh run my-executor
```

> **Warning — data disk.** The CVM's encrypted data disk key derives from the launch measurements. Any upgrade that changes measurements (OS image, host QEMU, OVMF, compose content) makes the existing `hda.img` undecryptable: the guest reboot-loops with `Failed to open encrypted data disk` even though the launcher reports success. Drain rentals before upgrading and recreate the VM directory. The new compose hash is whitelisted by Datura at release time — nothing to do on your side, but the compose must be exactly as released.

The host QEMU from §3 is **not** touched by executor releases; leave it alone unless a release note says otherwise.

### 6.1 Key-provider upgrade (only when a release says so)

A release that changes `key-provider/` (its `Cargo.lock`, Dockerfile or upstream commit) needs a rebuilt key provider. The rebuild changes MRENCLAVE and with it the disk key of **every** CVM on the host, so it is refused while any CVM disk exists. The guard lists the disks and never deletes one:

```bash
sudo ./lium-cvm.sh inventory                         # every CVM disk on this host, all checkouts, stopped CVMs included
# drain rentals, then per CVM whose data is no longer needed:
sudo ./lium-cvm.sh stop my-executor
sudo rm -rf run/vms/my-executor                      # by hand; the guard prints this exact line per running or stopped disk
sudo ./cvm_upgrade_guard.sh upgrade                  # refused with exit 3 while any disk remains, exit 4 while the inventory is incomplete
cd key-provider && docker compose logs gramine-sealing-key-provider | grep -m1 mr_enclave   # the new MRENCLAVE
sudo ./lium-cvm.sh new my-executor && sudo ./lium-cvm.sh run my-executor
```

`upgrade` keeps the previous image as `lium-key-provider:pre-upgrade-<timestamp>` and pins the new image id. It builds with `docker compose -f docker-compose.yaml -f docker-compose.build.yaml build`; that is the only compose line that builds, and it is not a step for you to run by hand. `sudo ./cvm_upgrade_guard.sh upgrade --dry-run` runs the inventory and reports without changing anything. The inventory covers this checkout's `run/vms`, every VM root a `lium-cvm.sh new` or `run` on this host ever registered in `/var/lib/lium-cvm/vm-dirs`, and a sweep of `/home /root /opt /srv /mnt /data` plus every filesystem mounted below them for any other `hda.img`, with or without a `vm-manifest.json` beside it (`LIUM_CVM_SWEEP_ROOTS` widens it). A disk that is not running and whose manifest is gone is listed as `orphan` and blocks the upgrade like any other; the guard prints no `rm` line for it, because nothing proves the CVM stack made that directory, so check it by hand. A registered VM root that is missing (an unmounted disk looks like a removed checkout) or a directory the sweep cannot read makes the inventory incomplete, and the upgrade is refused until it is mounted, readable, or removed from `vm-dirs`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| QEMU exits during guest boot: `VFIO_MAP_DMA failed: No space left on device` | default `dma_entry_limit` (65 535) exhausted by TDX page conversions | §2 — set `vfio_iommu_type1.dma_entry_limit=16777216` (runtime echo + GRUB persist) |
| Guest reboot-loops; console shows `Failed to open encrypted data disk` | measurements changed under an existing `hda.img` (QEMU/image/compose changed), or the key provider was rebuilt outside `cvm_upgrade_guard.sh` | if a `lium-key-provider:pre-upgrade-*` or `pinned-*` tag holds the old image, `docker tag <that> lium-key-provider:local`, write its id to `/var/lib/lium-cvm/key-provider.image` and `sudo ./cvm_upgrade_guard.sh start`; otherwise recreate the VM (`stop`, remove `run/vms/<name>`, `new`, `run`) |
| `cvm_upgrade_guard.sh upgrade` exits 3 | CVM disks exist on the host (the list names each `hda.img`, stopped CVMs included) | §6.1 — remove each listed disk by hand once its data is not needed; the guard never deletes |
| `cvm_upgrade_guard.sh upgrade` exits 4 | a registered VM root is missing (unmounted disk, removed checkout) or a registered root or sweep directory is not readable | mount the disk or fix the permission, or remove the stale line from `/var/lib/lium-cvm/vm-dirs`, then retry |
| `cvm_upgrade_guard.sh` exits 1 with `declares a 'build:' section` or `<file> exists` | someone put a `build:` back into `key-provider/docker-compose.yaml`, or another compose file (`docker-compose.yml`, `docker-compose.override.yaml`, `compose.yaml`) sits in `key-provider/`; either lets a hand-run `docker compose build` rebuild the enclave past the guard | remove the `build:` section or the extra file (the build definition is `docker-compose.build.yaml`) and run the command again |
| `cvm_upgrade_guard.sh start` or `lium-cvm.sh run` exits 6 | the pinned key-provider image is not on the host, the container came up on another image, or there is no pin and nothing to adopt while CVM disks exist | with CVM disks: restore the exact image (`docker load` from your backup, or `docker tag <id> lium-key-provider:local`); nothing is built. With no CVM disk: `sudo ./cvm_upgrade_guard.sh upgrade` rebuilds and pins |
| `cvm_upgrade_guard.sh` exits 1: "docker is not reachable" or "flock (util-linux) is not installed" | the Docker daemon is down, the caller is not root, or util-linux is missing | start Docker, run with `sudo`, or install `util-linux` |
| `lium-cvm.sh new`/`run` exit 5: "holds /var/lock/lium-cvm.lock" | another `lium-cvm.sh` or an upgrade is running | let it finish, then retry (`LIUM_CVM_LOCK_WAIT` sets the wait, default 60 s) |
| Attestation fails on RTMR0 / quote not whitelisted, image and compose correct | host running distro QEMU instead of the dstack 9.2.1 build | §3 — install the fork and set `client.conf`; confirm with `--version` |
| QEMU launch error mentioning `iommufd` / `VFIO_DEVICE_BIND_IOMMUFD` EINVAL | launcher predates the QEMU-version-gated VFIO backend | update to the current release — `dstack.py` now selects type1 automatically for QEMU < 10 |
| Validation fails `CHECK_SYSBOX_COMPATIBILITY` inside the guest | compose not the released one (pre-launch sysbox force-install missing or altered) | redeploy with the unmodified released compose (§5) |
| `lium-cvm.sh check` reports QEMU missing although §3 is done | `check` looks for `qemu-system-x86_64` on PATH; the launcher itself uses `client.conf` | add the §3 symlink, or ignore this one check line |
