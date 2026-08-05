# `verify`

Writes `/var/lib/lium-cvm/verify-report.json` — the single machine-readable
output contract. Humans read it, `bootstrap.sh` gates its exit code on it, and
cvmd's startup checks will consume the same stable ids.

Validated against `schemas/verify-report.schema.json` (JSON Schema draft
2020-12). **Every id below has a matching `### <id>` heading in this file, and
`tests/test_verify_schema.sh` asserts that set is exactly equal to the schema's
`enum`.** An id you cannot look up is an id nobody can act on.

## Status vocabulary

| Status | Meaning |
|---|---|
| `OK` | Passed. |
| `WARN` | Worth reading; does not block. |
| `MUST_FIX` | Blocks. `bootstrap.sh` exits non-zero. |
| `UNKNOWN` | Could not be read **at all**. Never silently a pass, never silently a failure. |
| `SKIPPED` | Deliberately not run, with the reason in `observed`. |

## Everything is recomputed here

There is no cross-play handoff of preflight's results. `verify` recomputes each
check from `lium.*`, so the report is correct whether preflight ran, was skipped
by a tag selection, or ran non-fatally in maintenance mode.

That is also why there is **no `preflight.*` id namespace**. The six
preflight-derived checks use their existing ids — `os.supported`,
`cpu.tdx_enabled`, `iommu.vt_d`, `sgx.devices_present`, `boot.grub_present`,
`gpu.cc_capable_present`. Inventing parallel ones would add ids that are in
neither the schema nor this file, and break the invariant above.

## The reboot-budget reset

This role owns it. When the run reaches `verify` with a clean kernel command
line — no drift and nothing missing — `converge_reboots` is reset to 0.

That is the only automatic reset, and it belongs here because reaching `verify`
with a clean command line is precisely the definition of "the converge worked".
A genuine reboot loop can never satisfy it.

---

# Check ids

## Platform gates

### os.supported
The release is one of `lium_os_matrix`. Ubuntu 25.10 and 26.04 LTS only.
Blocker — older releases lack the TDX kernel support this stack needs.

### cpu.tdx_enabled
`/sys/module/kvm_intel/parameters/tdx` reads `Y`. Blocker. Fixed in BIOS, not
here — the remediation names the exact menu.

### iommu.vt_d
`/sys/class/iommu/` is non-empty. Blocker: without an IOMMU no device can be
passed through to a TD guest.

### sgx.devices_present
Both `/dev/sgx_enclave` and `/dev/sgx_provision` exist. Blocker. BIOS again,
plus a PRMRR size of 64M or more.

### boot.grub_present
`/boot/grub/grub.cfg` exists **and** `update-grub` is on `PATH`. Blocker,
because everything the `kernel` role does assumes GRUB. A systemd-boot or
vendor-bootloader host fails here **by name**, early, rather than being
discovered after two wasted reboots.

### gpu.cc_capable_present
At least one device matching `lium_cc_gpu_pci_ids`. **Two distinct outcomes:**
`UNKNOWN` when `lspci` itself is missing, `MUST_FIX` when the tool ran and found
nothing. Conflating them would let a missing tool masquerade as missing
hardware. The remediation enumerates the ids that were searched.

## SGX chain

### sgx.registration_pck_certs
The `pck_cert` row count in `/opt/intel/sgx-dcap-pccs/pckcache.db` is above zero.

**This is the only trustworthy registration signal.**
`/var/log/mpa_registration.log` reported *"registration is completed
successfully"* on the broken host **and** the healthy one, and
`platforms_registered = 0` is normal on a healthy host. Neither is usable.

`SKIPPED` on the degraded path (no `lium_intel_api_key`), with the remediation
naming the remote PCCS that collateral comes from instead.

### sgx.pccs_active
`systemctl is-active pccs`. `SKIPPED` on the degraded path. A PCCS that dies
with `ERR_MODULE_NOT_FOUND: config` needs `npm install` in
`/opt/intel/sgx-dcap-pccs` — the interactive postinst wizard normally does that,
and it fails under apt.

### sgx.qgsd_vsock_4050
`ss -a --vsock` shows a listener on 4050. Blocker. `/etc/qgs.conf` ships
`#port = 4050` **commented out**, so qgsd serves a unix socket, reports
`active`, and never answers the guest — which then reboots at ~160 s with
`vsock failure: Connection reset by peer (os error 104)`.

### sgx.qpl_secure_cert_false
`use_secure_cert` is `false` in `/etc/sgx_default_qcnl.conf`. Blocker. Intel
ships it `true`, which rejects a local self-signed PCCS certificate:
`[QCNL] CURL error: (60)`, `[QPL] ... 0xb033`, then the guest reboots. It must
be false in **both** QPL configs — the system one and the key provider's.

### sgx.keyprovider_production
The container logs `Running in PRODUCTION mode` with no `error 44` and no failed
`load_enclave`. **This is the end-to-end proof that the whole SGX chain works** —
registration, PCCS, QPL and qgsd all have to be right for the enclave to get
there.

### net.intel_api_reachable
A bounded TCP connect to `api.trustedservices.intel.com:443`. `WARN` only.
Egress filtering is common in provider datacenters, and an unreachable Intel API
turns in-band registration into an opaque hang.

## GPUs

### gpu.cc_mode_all_on
Every GPU reports CC mode on. `UNKNOWN` if any query failed — a mode we could
not read is not a mode we may flip. CC mode is **off by default** on a fresh
host, and `ENABLE_GPU_ATTESTATION=false` does **not** make it optional.

### gpu.vfio_bound_all
Every GPU **and NVSwitch** is bound to `vfio-pci`. Binding the GPUs alone gets
you a guest that boots and then cannot see NVLink.

### gpu.device_inventory
Informational count of GPUs and NVSwitches. Recorded so a device that
disappears between runs is visible — a hot-removed GPU is exactly what makes
`vfio-pci.ids=` try to narrow.

## Kernel

### kernel.dma_entry_limit
The **running** value of `dma_entry_limit` is at least `lium_dma_entry_limit`.
The shipped default of 65535 kills a TD guest mid-boot with
`VFIO_MAP_DMA failed: No space left on device`.

### kernel.dma_persisted
`/etc/modprobe.d/vfio-dma.conf` exists. The runtime write applies immediately
but does not survive a reboot, so both halves are checked separately.

### kernel.cmdline_tokens
Every required token resolves on `/proc/cmdline`, **last occurrence wins per
key**. This is the check whose OK state resets the reboot budget.

### kernel.grub_persisted
Every required token resolves in the merged GRUB configuration, so the next boot
will still have them.

### kernel.grub_conflicting_tokens
`WARN` when a required key appears more than once with **different** values.
Ours win by ordering, but relying on that forever is fragile. A duplicate with
the same value is not a conflict — the drop-in restates the full required set
every run by design.

## QEMU

### qemu.version_9_2_1
The installed QEMU is the pinned dstack build. A host running distro QEMU fails
attestation on RTMR0 with everything else looking correct.

### qemu.client_conf_path
`/etc/dstack/client.conf` points at our binary. Read directly — **never** via
`lium-cvm.sh check`, which looks on `PATH` while the launcher uses `client.conf`
and therefore reports a false MISSING on a correctly configured host.

### qemu.binary_sha256
The live binary still matches the fingerprint recorded at install time. This is
the fault-injection check: a QEMU that changed underneath a running CVM means
RTMR0 no longer matches and any existing `hda.img` is undecryptable.

## Platform and repository

### docker.installed
Docker responds to `--version`.

### repo.present
The `dstacktee` tree exists at `lium_repo_path`.

### repo.app_pristine
`git status --porcelain neurons/executor/dstacktee/app` is empty. **Blocker.**
Everything under `app/` is hashed into the compose the validator whitelists, so
any local edit produces a hash nobody has whitelisted and every attestation
fails.

### repo.ref_behind
`WARN` when the guard state is not `CLEAN` and the checkout was therefore left
where it was. Moving the ref rewrites the measured surface, and doing that as a
side effect of a maintenance run is not acceptable — upgrading a host that has a
CVM is a deliberate procedure.

### repo.os_image_present
A `digest.txt` exists under `run/images/`.

### repo.runner_digest_set
`EXECUTOR_RUNNER_IMAGE_DIGEST` is set in `dstacktee/.env`. **Presence only —
never a comparison against a whitelist.** A checkout's whitelist can be *behind*
the deployed validator, and on one occasion the whitelisted digest was the
broken one. Reporting a mismatch here would tempt someone into downgrading.

## Network

### net.local_listeners
Informational count from `ss -tlnp`.

### net.acl_probe_required
**Always `WARN`, always emitted.** `ss` proves a port is LISTENING, never that it
is REACHABLE. Two hosts silently blackholed 8001/2200/19001 while permitting 22
and 30000–32767, and from the host itself everything looked perfect.

The remediation points at an off-site **read** test
(`exec 3<>/dev/tcp/HOST/PORT && head -c 2 <&3`) with a **calibrated** vantage —
many networks forge SYN-ACKs for every destination, which makes a plain connect
test worthless. Fixing datacenter ACLs is out of scope for this playbook.

## Guard and stubs

### guard.host_state
The derived CVM state — `CLEAN`, `LIVE`, `DORMANT`, `FOREIGN`, `ZOMBIE` or
`UNKNOWN`. Its `remediation` carries the fact-composed recovery procedure, so
the report itself explains which roles were withheld and why.

### cvmd.stub_status
Always `SKIPPED`. cvmd ships as a variable interface only; the implementation
lands in DAH-2575.

### catalog.stub_status
Always `SKIPPED`. Same shape; the implementation lands in DAH-2576 / DAH-2578.
