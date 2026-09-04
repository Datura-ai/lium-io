# CVM host setup — dstack-nvidia-0.5.11

Provider guide for taking a TDX host from bare metal to an attested Lium CVM executor. The quick start in [../README.md](../README.md) assumes the host is already prepared; this document covers that preparation. Almost everything here is **one-time per host** — the per-executor work is one `.env` file and two `lium-cvm.sh` commands.

| Phase | Frequency | Hands-on time |
|---|---|---|
| §1–§4 host bring-up (BIOS, kernel, QEMU, key provider) | once per host | ~1–2 h (dominated by BIOS + QEMU build) |
| §5 executor deployment | once per executor | ~15 min |
| §6 upgrades | per release | ~10 min |

## 1. Hardware and firmware

- **CPU**: Intel Xeon with TDX **and** SGX — Sapphire Rapids (XCC/MCC SKUs only), Emerald Rapids, or Granite Rapids. SGX is not optional: the sealing-key provider runs in an SGX enclave.
- **GPU**: NVIDIA Confidential-Computing capable — Hopper (H100/H200) or Blackwell (B200/B300). Hopper: CC mode for a single GPU per CVM, Protected PCIe (GPUs **and** NVSwitches, `CVM_GPUS=all` — a listed set leaves the NVSwitches on the host) for a whole 8× HGX board. Blackwell: CC mode only. Commands and checks: [Enable GPU Confidential Computing mode](https://docs.lium.io/providers/nodes/cvm#4-enable-gpu-confidential-computing-mode). Bring-up scripts written against the older single-GPU guidance, such as `cvm-host-phase2.sh`, set Protected PCIe back to off — keep Protected PCIe nodes out of them and re-query the mode after any run.
- **BIOS**: latest vendor BIOS; enable TDX, SGX, VT-x/VT-d, and **SGX Auto MP Registration** (§4). After boot, `/dev/sgx_enclave` and `/dev/sgx_provision` must exist.

### TDX module — Intel's minimum TCB

A TDX module below Intel's published minimum is refused by **every** attester (our key provider, our validator, Phala's own KMS): DCAP verification of the quote answers `No matching TCB level found`. For Granite Rapids (FMSPC `00A06D080000`) the minimum is **TDX module 2.0.08, build 882, minor SVN 5**; current is 2.0.18 (build 1013).

| tag | build | date | minor SVN |
|---|---|---|---|
| 2.0.02 | 786 | 2024-08-14 | 3 — below the minimum |
| 2.0.08 | 882 | 2025-04-02 | 5 — Intel minimum |
| 2.0.18 | 1013 | 2026-02-11 | 9 — latest |

Sapphire and Emerald Rapids run the 1.5 series, whose build numbers are not comparable with the table; `lium-cvm.sh check` only reports those and the quote's `tee_tcb_svn` is the proof.

Check on a 6.x kernel:

```bash
sudo dmesg | grep 'TDX module'   # ... build_date 20250402, build_num 882
```

Kernel 7.0 does not print that line and there is no `/sys/firmware/tdx`. There the proof is the quote: `tee_tcb_svn` starts `09 03 …` after updating to 2.0.18, and the key provider's DCAP verification stops saying `No matching TCB level found`. `sudo dmesg | grep virt/tdx` must still end in `module initialized`.

**Updating without a vendor BIOS.** Intel publishes the signed module at [intel/confidential-computing.tdx.tdx-module](https://github.com/intel/confidential-computing.tdx.tdx-module/releases); NVIDIA's *Deployment Guide for Confidential Computing* gives this procedure verbatim (and names `TDX_MODULE_2.0.08` for Granite Rapids), the alternative being an OEM BIOS:

```bash
tar -xvzf intel_tdx_module.tar.gz
sudo mkdir -p /boot/efi/EFI/TDX/
sudo cp TDX-Module/intel_tdx_module.so /boot/efi/EFI/TDX/TDX-SEAM.so
sudo cp TDX-Module/intel_tdx_module.so.sigstruct /boot/efi/EFI/TDX/TDX-SEAM.so.sigstruct
sudo reboot
```

The BIOS's TDX loader picks the module up from the ESP at boot. Only Intel-signed modules load, so a wrong file is inert — the built-in module keeps loading and nothing is lost. Verified working on an AIVRES KR9288-X3 (AMI 03.04.01) on 2026-09-03; whether a given board keeps this Intel reference-BIOS hook is what the reboot tells you.

### NUMA — possible vs online nodes

```bash
cat /sys/devices/system/node/possible   # e.g. 0-19
cat /sys/devices/system/node/online     # e.g. 0-1
```

Firmware that advertises more *possible* than *online* nodes (seen on a Xeon 6767P: 20 possible, 2 online) breaks the pinned Gramine 1.5 key provider: it sizes its NUMA arrays from `possible`, reads `node<i>/distance` expecting that many numbers, and `load_enclave` dies with `EINVAL` before ever opening `/dev/sgx_enclave` (Gramine issue #1452, fixed in 1.6). Hardware-dependent, kernel- and OS-independent — reprovisioning the OS changes nothing. Until the image is re-pinned (DAH-2860), rebuild the key provider on `gramineproject/gramine:1.9-jammy` (§4).

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

**GCC 15 (Ubuntu 26.04):** the bundled `dtc` subproject does not inherit `--disable-werror` and the build stops at `subprojects/dtc/libfdt/fdt_overlay.c:461: error: … -Werror=discarded-qualifiers`. Keep the subproject out of the build instead of patching it — the x86_64 target does not need libfdt:

```bash
../configure --prefix=/opt/qemu-dstack --target-list=x86_64-softmmu \
    --disable-werror --enable-kvm --enable-slirp --disable-fdt
```

Or `--enable-fdt=system` with `libfdt-dev` installed, to use the distro dtc. Verified against the pinned tree: `x86_64-softmmu.mak` declares no `TARGET_NEED_FDT`, and `configure` maps these flags to `-Dfdt=disabled` / `-Dfdt=system` (`meson.build` only builds `subprojects/dtc` for `-Dfdt=internal`). Not verified by an actual GCC 15 build — the host that hit this patched `fdt_overlay.c` by hand and got a working 9.2.1.

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

The sealing-key provider must run on the host before any CVM boots (`lium-cvm.sh run` auto-starts it, but starting manually first surfaces build errors early):

```bash
cd key-provider && docker compose up --build -d
docker compose logs -f gramine-sealing-key-provider   # watch first start
curl -k https://localhost:3443                        # endpoint reachable
```

Two containers: `aesmd` (SGX architectural enclaves, host network) and `gramine-sealing-key-provider` (`127.0.0.1:3443`).

**Register the platform with Intel.** DCAP collateral comes from the PCCS configured in [`key-provider/sgx_default_qcnl.conf`](../key-provider/sgx_default_qcnl.conf), which defaults to Phala's public PCCS. That default is not enough on its own: no PCCS can serve PCK certificates for a platform that Intel does not know about, and the key provider then fails with AESM error 44 → `load_enclave … EPERM`. Multi-package platforms need registration. Two steps, once per host:

1. Enable **SGX Auto MP Registration** in BIOS, or register manually against a PCCS you run with an Intel PCS API key:

   ```bash
   PCKIDRetrievalTool -url https://<your-pccs>:8081 -use_secure_cert false
   ```

2. Point `pccs_url` in `key-provider/sgx_default_qcnl.conf` at that PCCS. `aesmd` runs with `network_mode: host`, so a PCCS on the same box is `https://127.0.0.1:8081/sgx/certification/v4/`.

**Gramine version.** The image is pinned to Gramine 1.5, which fails to load the enclave on hosts whose firmware over-declares NUMA nodes (§1). On such a host rebuild it on `gramineproject/gramine:1.9-jammy` (upstream `MoeMahhouk/gramine-sealing-key-provider` already carries the 1.9 base) and install `libsgx-dcap-default-qpl`. The validator does not pin the key provider's `mr_enclave`, so a rebuild changes no whitelist.

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
# update EXECUTOR_RUNNER_IMAGE_DIGEST in .env — copy it from the release notes, never
# from an older doc or chat message: the digest is measured, so a wrong one costs a
# full redeploy (measurements change, the data disk is recreated)
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
| Key-provider log: `No matching TCB level found` while verifying the CVM's quote | TDX module below Intel's minimum (2.0 series on Granite Rapids: build < 882; the 1.5 series on Sapphire/Emerald Rapids has its own floor) | §1 — drop Intel's signed module into `/boot/efi/EFI/TDX/` and reboot, or get a vendor BIOS |
| `gramine-sealing-key-provider` dies with `load_enclave … EINVAL` before touching `/dev/sgx_enclave` | firmware declares more possible than online NUMA nodes; Gramine 1.5 bug | §4 — rebuild the key provider on `gramine:1.9` (DAH-2860) |
| AESM error 44 / `load_enclave … EPERM` | platform not registered with Intel, or a PCCS that has no PCK certs for it | §4 — SGX Auto MP Registration / `PCKIDRetrievalTool`, then point `pccs_url` at that PCCS |
| Executor answers the validator with `401 Invalid validator signature` | `EXECUTOR_RUNNER_IMAGE_DIGEST` pins the July staging build `sha256:f85b948b6cb280423b17e34ea28c0f98139243ecea32cd42106f90d19dc619f1` (a dev validator hotkey is baked into it) | §5/§6 — redeploy with the runner digest from the release notes (`sha256:8c07d3a91f8900bd3f0e19025fb7a2c32f82577385550187b28c7769060d18b7` as of 2026-08-19) |
| QEMU prints `TDX doesn't support requested feature: CPUID.07H_01H:EDX.avx10 [bit 19]` | `-cpu host` advertises AVX10, the TDX module masks it | harmless — CPUID bits are not measured into the RTMRs; the VM boots |
| `lium-cvm.sh check` reports QEMU missing although §3 is done | no `[qemu] path` in a `client.conf` `check` can read (it reads `/etc/dstack`, `~/.config/dstack` and `.dstack` next to the script) | §3 — set `/etc/dstack/client.conf`, or add the symlink |
