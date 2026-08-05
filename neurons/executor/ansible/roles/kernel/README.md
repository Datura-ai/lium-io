# `kernel`

The kernel command line, the vfio bindings, and the one reboot this playbook may
need. **Destructive** — withheld by the maintenance profile.

## The ordering guarantee the safety rests on

`/etc/grub.d/10_linux` emits the default menu entry as:

```
linux ... ${GRUB_CMDLINE_LINUX} ${GRUB_CMDLINE_LINUX_DEFAULT}
```

`GRUB_CMDLINE_LINUX` first, `GRUB_CMDLINE_LINUX_DEFAULT` **last**. The kernel
resolves duplicate parameters **last occurrence wins, per key**.

Everything below follows from those two facts:

- The drop-in `/etc/default/grub.d/99-lium-cvm.cfg` **appends to
  `GRUB_CMDLINE_LINUX_DEFAULT`** and **never assigns `GRUB_CMDLINE_LINUX`**.
  Ubuntu sources `/etc/default/grub` first and `/etc/default/grub.d/*` after, so
  `"${GRUB_CMDLINE_LINUX_DEFAULT:-} ..."` preserves whatever the provider had.
- Because our tokens land last, they win — even against a stale copy of the same
  key that the provider already had.
- Therefore **every assertion resolves a key to its winning value**, never bare
  set membership. On a host with a stale
  `vfio_iommu_type1.dma_entry_limit=65535`, set membership would see our
  `16777216` present and report success while which value the kernel honours
  depended entirely on ordering.

A stale duplicate with a *different* value additionally raises the
`kernel.grub_conflicting_tokens` WARN naming it, so the provider can clean it up
rather than rely on the ordering forever. A duplicate with the *same* value is
not a conflict — the drop-in restates the full required set every run, so
re-stating something the provider already had is expected.

### Why the drop-in carries the full set, not just what is missing

Writing only the currently-missing tokens is not idempotent. The drop-in is part
of the configuration that is analysed, so on run 2 nothing would be missing, the
file would render empty, the tokens would disappear — and on run 3 they would be
missing again. Restating the full set is stable by construction.

### Why "found only in GRUB_CMDLINE_LINUX" is computed separately

Once the drop-in has appended the required set to `_DEFAULT`, every token is in
both lines and the **au2 shape** — arguments in `GRUB_CMDLINE_LINUX` with
`_DEFAULT` empty — is masked by our own fix. So that check runs its own analysis
with `--exclude-dropin`, asking what the *provider* has. Otherwise it would
report the shape on the first run and silently stop reporting it afterwards.

## `vfio-pci.ids=` may only widen

A hot-removed or wedged device disappears from `lspci`. The freshly discovered
set gets shorter, and writing the shorter list **unbinds that GPU on the next
boot**. The role therefore asserts the discovered set is a **superset** of every
pre-existing `vfio-pci.ids=` (from `GRUB_CMDLINE_LINUX`,
`GRUB_CMDLINE_LINUX_DEFAULT`, and `/proc/cmdline`). On violation it refuses,
names the ids that would be dropped, and hints:

```bash
echo 1 | sudo tee /sys/bus/pci/rescan
```

### …but only over ids this playbook manages

The comparison runs against the **managed universe** —
`lium_cc_gpu_pci_ids ∪ lium_nvswitch_pci_ids` — lower-cased on both sides,
because `vfio-pci` parses its ids with `%x` and `10DE:2335` is a configuration a
provider can and does type.

Everything else in a provider's `vfio-pci.ids=` is **carried forward unchanged**:
an SR-IOV NIC, an NVMe, a GPU generation whose ids are not field-verified yet
(§6.2). Both halves of that are load-bearing.

- **It may not refuse over them.** Those ids are never looked for in `lspci`, so
  they can never appear in `present_pci_ids` and would read as "dropped" on every
  run forever. A host passing a NIC through `vfio-pci` — ordinary on a
  virtualization host — could never converge, and the refusal would tell its owner
  to rescan a bus that has nothing to find.
- **It may not drop them either.** `vfio-pci.ids=` is a module parameter and the
  last occurrence wins. Our drop-in lands last by construction, so it replaces the
  whole value; writing only the ids we discovered would unbind every unmanaged
  device on the next boot.

Ignoring an unmanaged id satisfies the first and breaks the second. Carrying it
forward satisfies both, and is the only honest answer available: we cannot verify
a device we never look for. Keeping an id whose device really did go away costs
nothing — `vfio-pci` simply binds nothing.

Only the GRUB files feed the carry-forward, never `/proc/cmdline`. An id that is
only in `/proc/cmdline` has already been removed from the boot configuration, and
restoring it would resurrect a setting the provider deleted.

## The decisive check runs before the reboot is offered

After `update-grub`, the role parses the **generated** `grub.cfg` default `linux`
line and asserts every required token wins *there*. A merge failure caught at
that point is still trivially recoverable — restore
`/etc/default/grub.bak.pre-lium-cvm`; nothing has rebooted.

Caught after the reboot instead, the host comes back with `intel_iommu=on
iommu=pt`, correct vfio bindings, CC-on GPUs, the right QEMU, a healthy key
provider — and **no `kvm_intel.tdx=on`**. Attestation fails and the node scores
zero while every dashboard says the host is fine.

## The vfio modules

`/etc/modules-load.d/vfio.conf` lists them on **separate lines**:

```
vfio_pci
vfio_iommu_type1
```

Never `modprobe vfio_iommu_type1 vfio-pci`. The second argument is read as a
module **parameter**, and the kernel quietly logs `unknown parameter 'vfio-pci'
ignored` while doing nothing.

## The reboot budget

`/var/lib/lium-cvm/converge_reboots`, capped at `lium_max_converge_reboots`
(default 2).

**Arming never resets it. A clean converge does** — and `verify` owns that reset:
the run reached `verify`, `kernel.cmdline_tokens` is OK, and drift is empty.

The distinction matters in both directions:

- *Lifetime-scoped would brick a healthy host.* This role's own superset rule
  guarantees legitimate GRUB changes — add a GPU, `vfio-pci.ids=` changes, that
  is drift, that is a reboot. Two legitimate changes would exhaust a lifetime
  budget and a third would be refused forever.
- *Session-scoped still stops a real loop.* A host whose `/boot` is full, or
  which does not boot GRUB at all, never reaches a clean verify — so the counter
  climbs to the cap exactly as intended.

At the cap the role warns, prints the drifted tokens and the generated `linux`
line, ends the play, and lets `verify` report it. Clearing it deliberately:

```bash
sudo ./bootstrap.sh -e lium_reset_reboot_budget=true
```

## The resume unit

`lium-cvm-resume.service`, `Type=oneshot`, armed only by writing
`/var/lib/lium-cvm/resume.state`.

- **`ExecStopPost`, not `ExecStartPost`**, to disable itself. It runs on every
  exit path — success, failure, signal. With `ExecStartPost`, a run that fails
  its last task never reaches it, the enablement symlink survives, and weeks
  later an unrelated kernel reboot fires the whole playbook at 04:00 while a
  customer holds an open rental.
- `ExecStartPre` is `lium-resume-guard.sh`, which refuses on **three**
  independent grounds — already attempted, unchanged `boot_id`, or an exhausted
  converge budget — and **increments and persists `attempts` before returning**,
  so a crash mid-run cannot produce a second attempt.
- `lium-resume.sh` is a plain `files/` script, not a template, so it can be
  shellchecked exactly as committed. It sits directly on the reboot-recovery
  path.
- It never touches `converge_reboots`. That budget belongs to this role and must
  survive across resumes.

Post-reboot status:

```bash
systemctl is-failed lium-cvm-resume
journalctl -u lium-cvm-resume
less /var/log/lium-cvm/resume.log
```

## Test seams

All write-side, so the whole chain runs in CI against a fixture root:
`lium_grub_default_path`, `lium_grub_dropin_path`, `lium_grubcfg_path`,
`lium_update_grub_cmd`, `lium_proc_cmdline_path`. See `tests/README.md`.
