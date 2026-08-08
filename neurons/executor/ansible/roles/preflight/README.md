# `preflight`

Hard gates. No BIOS automation — this role asserts, names, and stops.

Failures are **collected and reported together**, so a provider learns every
missing knob in one pass instead of one reboot at a time. That is the whole
design: providers run this unattended, once, with no Lium access to their host,
so the failure message *is* the support channel.

## The gates

| id | Passes when | Severity |
|---|---|---|
| `os.supported` | the release is in `lium_os_matrix` | blocker |
| `host.arch` | `x86_64` | blocker |
| `host.sudo` | running as root | blocker |
| `cpu.tdx_enabled` | `/sys/module/kvm_intel/parameters/tdx` is `Y` | blocker |
| `sgx.devices_present` | both `/dev/sgx_enclave` and `/dev/sgx_provision` | blocker |
| `iommu.vt_d` | `/sys/class/iommu/` is non-empty | blocker |
| `gpu.cc_capable_present` | a device matching `lium_cc_gpu_pci_ids` | blocker |
| `boot.grub_present` | `grub.cfg` exists **and** `update-grub` is on `PATH` | blocker |
| `disk.build_space` | ≥ `lium_min_build_space_gb` free | blocker |
| `disk.image_space` | ≥ `lium_min_image_space_gb` free | blocker |

## Two outcomes for the GPU gate

`lspci` missing and *no CC-capable device* are **different failures with
different messages**. Conflating them would let a missing tool masquerade as
missing hardware — and would let a CI grep pass for entirely the wrong reason.

The device-absent message enumerates the ids that were searched, so a provider
with an unlisted card learns that it is unsupported rather than that it is
broken.

## `lium_preflight_fatal`

`true` (the default) aborts the run — the day-zero behaviour.

`false`, which the maintenance profile sets, records every failed id and lets the
play continue to `verify`. A host that lost a BIOS setting after a firmware
update then gets a **report naming it**, instead of an abort and no report.

## Why this role is never `tags: always`

With `always` it would run under `--tags verify`, hard-fail by design on any host
with a knob off, and abort the play **before** `verify` emitted its report —
killing the read-only audit mode this playbook explicitly supports.

## Why each check is its own task

`when:` is evaluated as a real condition. A templated `{{ expr }}` stored in a
data structure renders to the **string** `"False"`, which is truthy, and the
whole aggregate would silently pass.
