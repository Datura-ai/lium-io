# CVM host setup — dstack-nvidia-0.5.11

Provider guide for taking a TDX host from bare metal to an attested Lium CVM executor. The quick start in [../README.md](../README.md) assumes the host is already prepared; this document covers that preparation. Almost everything here is **one-time per host**, and §1–4 is now a single command — the per-executor work is one `.env` file and two `lium-cvm.sh` commands.

| Phase | Frequency | Hands-on time |
|---|---|---|
| §1–4 host bring-up — one command | once per host | ~1–2 h, mostly unattended (BIOS by hand, then the QEMU build) |
| §5 executor deployment | once per executor | ~15 min |
| §6 upgrades | per release | ~10 min |

## 1–4. Host bring-up — automated

One command takes a fresh Ubuntu **25.10** or **26.04 LTS** TDX host from bare OS
to CVM-ready:

```bash
cd neurons/executor/ansible
sudo ./bootstrap.sh
```

It is safe to re-run. Every run converges from wherever the host already is, and
it refuses anything that would damage a CVM that already exists.

See [../../ansible/README.md](../../ansible/README.md) for the flags, the
variables, how to read the report, and what a re-run will and will not touch.

### What it does

- **Kernel and boot** — `dma_entry_limit`, the vfio modules, and the GRUB
  parameters (`kvm_intel.tdx=on`, `intel_iommu=on`, `iommu=pt`, `nohibernate`,
  `vfio-pci.ids=`). It appends to `GRUB_CMDLINE_LINUX_DEFAULT` and never
  overwrites your existing `GRUB_CMDLINE_LINUX`.
- **GPUs** — installs nvtrust, turns confidential-compute mode on where it is
  off, and binds every GPU *and NVSwitch* to `vfio-pci`.
- **QEMU** — builds the pinned dstack 9.2.1 fork and writes
  `/etc/dstack/client.conf`. This is the long step, 10–40 minutes.
- **Docker** — Docker Engine from Docker's own repository.
- **SGX** — the Intel DCAP packages, PCCS, platform registration, TD quote
  plumbing, and the sealing-key provider.
- **Repository** — the lium-io checkout and the CVM OS image.
- **Report** — `/var/lib/lium-cvm/verify-report.json`, with a stable id, a
  status and a remediation for every check.

It may need **one reboot** to pick up the new kernel parameters. It asks first,
then continues by itself afterwards — exactly once.

### What it does NOT do

**BIOS settings.** Firmware cannot be changed from inside the OS. If one of
these is off, the run stops and names it:

| Setting | Where |
|---|---|
| Intel **TDX** (and TME / TME-MT) | CPU security menu |
| Intel **SGX**, PRMRR size ≥ 64M | CPU security menu; an SGX Factory Reset may be required |
| **VT-d** / Virtualization Technology for Directed I/O | chipset / IO menu |

Your kernel must also be a Canonical `intel` kernel or mainline ≥ 6.16.

Also not automated, by design: the per-executor `.env` (§5 below), datacenter
port ACLs, and running or supervising the CVM itself.

### If a run refuses

The playbook will not change anything measured while an encrypted CVM data disk
exists on the host — doing so makes that disk permanently undecryptable. On such
a host it automatically switches to a maintenance profile, converges everything
safe, and prints the recovery procedure. §6 below is the deliberate upgrade path.

### Hardware

| | |
|---|---|
| CPU | Intel Xeon with TDX (Emerald Rapids or newer) |
| GPU | NVIDIA Hopper-class with confidential compute — H100 / H200 |
| Disk | ≥ 40 GB free: ~10 GB for the QEMU build, ~30 GB for the OS image and data disk |
| OS | Ubuntu 25.10 or 26.04 LTS |


## 5. Per-executor deployment

1. Configure:

   ```bash
   cp .env.example .env
   ```

   Required beyond the obvious (`MINER_HOTKEY_SS58_ADDRESS`, ports, `CVM_VCPUS/MEMORY/DISK`, `CVM_GPUS`):

   - `ENABLE_TDX_ATTESTATION=true`
   - `EXECUTOR_RUNNER_IMAGE_DIGEST=sha256:<64-hex>` — copy from the release notes. `lium-cvm.sh new` pins this digest into the measured compose (the attested trust boundary) and refuses to run without it.

2. Create and boot (the OS image downloads once per host, then is reused):

   ```bash
   sudo ./lium-cvm.sh new my-executor
   sudo ./lium-cvm.sh run my-executor
   ```

   Everything attestation-related inside the guest — sysbox force-install, digest-pinned runner, quote generation — is baked into the measured compose; there is nothing to configure in the guest.

3. Verify: the executor API answers on `EXTERNAL_PORT` and SSH banners on `SSH_PORT` within ~3–5 minutes of boot. `./lium-cvm.sh list` shows the VM.

4. Register the executor with your miner exactly like a non-CVM executor. Attestation runs automatically when the validator connects.

**Do not modify** `app/docker-compose*.yml`, `app/pre_launch_script.sh`, or `app/init_script.sh`: they are measured into the compose hash the validator whitelists — any local change makes attestation fail.

## 6. Upgrades

Per release:

```bash
sudo ./lium-cvm.sh stop my-executor
git pull                                   # the release tag
# update EXECUTOR_RUNNER_IMAGE_DIGEST in .env from the release notes
sudo rm -rf run/vms/my-executor            # see warning below
sudo ./lium-cvm.sh new my-executor
sudo ./lium-cvm.sh run my-executor
```

> **Warning — data disk.** The CVM's encrypted data disk key derives from the launch measurements. Any upgrade that changes measurements (OS image, host QEMU, OVMF, compose content) makes the existing `hda.img` undecryptable: the guest reboot-loops with `Failed to open encrypted data disk` even though the launcher reports success. Drain rentals before upgrading and recreate the VM directory. The new compose hash is whitelisted by Datura at release time — nothing to do on your side, but the compose must be exactly as released.

The host QEMU from §3 is **not** touched by executor releases; leave it alone unless a release note says otherwise.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| QEMU exits during guest boot: `VFIO_MAP_DMA failed: No space left on device` | default `dma_entry_limit` (65 535) exhausted by TDX page conversions | §2 — set `vfio_iommu_type1.dma_entry_limit=16777216` (runtime echo + GRUB persist) |
| Guest reboot-loops; console shows `Failed to open encrypted data disk` | measurements changed under an existing `hda.img` (QEMU/image/compose changed) | recreate the VM (`stop`, remove `run/vms/<name>`, `new`, `run`) — data on the old disk is unrecoverable by design |
| Attestation fails on RTMR0 / quote not whitelisted, image and compose correct | host running distro QEMU instead of the dstack 9.2.1 build | §3 — install the fork and set `client.conf`; confirm with `--version` |
| QEMU launch error mentioning `iommufd` / `VFIO_DEVICE_BIND_IOMMUFD` EINVAL | launcher predates the QEMU-version-gated VFIO backend | update to the current release — `dstack.py` now selects type1 automatically for QEMU < 10 |
| Validation fails `CHECK_SYSBOX_COMPATIBILITY` inside the guest | compose not the released one (pre-launch sysbox force-install missing or altered) | redeploy with the unmodified released compose (§5) |
| `lium-cvm.sh check` reports QEMU missing although §3 is done | `check` looks for `qemu-system-x86_64` on PATH; the launcher itself uses `client.conf` | add the §3 symlink, or ignore this one check line |
