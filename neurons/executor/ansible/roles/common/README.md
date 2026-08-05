# `common`

The shared library. Four jobs: refuse unsanctioned test seams, collect every
host fact once as root, derive the CVM guard state, and own all kernel
command-line token math.

Runs under `tags: always`, so `--tags verify` still gets its facts. It is the
only role that is unconditionally reachable.

## Why it needs root

The decisive facts are all privileged. An unprivileged collector does not fail —
it silently reports every one of them as *absent*, which then reads as "nothing
to do":

| Fact | Needs root for |
|---|---|
| `pck_cert_count` | reading `/opt/intel/sgx-dcap-pccs/pckcache.db` |
| `qgsd_vsock_4050` | `ss -a --vsock` |
| `gpu_cc_mode` | the nvtrust query |
| `kp_running`, `kp_restarts`, `kp_production` | `docker inspect` / `docker logs` |

## Tri-state facts

Every fact that can genuinely fail to be read is:

```json
{"value": null, "readable": false, "reason": "..."}
```

`readable: false` means **unknown** — never zero, never absent. The distinction
decides whether an irreversible action may run at all:

| Fact | Gates | Why unknown must block |
|---|---|---|
| `pck_cert_count` | in-band SGX registration | `PCKIDRetrievalTool` flips an efivar that cannot be un-flipped. "0 certs" means register; "cannot read" means we cannot tell. |
| `gpu_cc_mode[bdf]` | the CC-mode flip | The flip resets the GPU. Never flip a mode we could not read. |
| `qemu_version`, `qemu_sha256` | the QEMU rebuild | Replacing QEMU changes RTMR0. Never replace a binary whose identity is unknown. |
| `proc_cmdline` | the reboot | Never reboot on unknown drift. |
| `guard.state` | everything destructive | Fail closed. |

Note the asymmetry that makes this useful: an absent QEMU reports
`{"value": "", "readable": true}` — a definite "not installed" — while an
unreadable one reports `readable: false`. Only the second blocks.

## What is deliberately not read

- **`/var/log/mpa_registration.log`.** It said *"Registration status indicates
  registration is completed successfully"* on the broken host (au1) and the
  healthy one (au2) alike. It is not a usable signal.
- **`platforms_registered`.** `0` is normal on a healthy host.

SGX registration truth is the `pck_cert` row count in `pckcache.db`. Nothing else.

## The guard

`files/lium-guard.sh` is the most safety-critical file in this tree. It is a
**fact collector with a derivation**, not a state machine with hand-written cells.

### The primary predicate is on disk, not in the process table

The catastrophe is a measurement change under an existing `hda.img`, not a
running process. `lium-cvm.sh stop` (`stop_cvm`, `lium-cvm.sh:559`) shuts the
guest down and returns success **without removing the VM directory**, and
`docs/host-setup.md:155` makes `rm -rf run/vms/<name>` a deliberate, separate,
manual step.

So *"stopped CVM, zero QEMU processes, intact encrypted data disk, open rental"*
is a normal, reachable, common state — and a guard that only asks "is QEMU
running?" waves it straight through and destroys the renter's data.

A running QEMU is an **additional** blocking signal, never the only one.

### The three facts

They are orthogonal, and all three are always emitted:

| Fact | Meaning |
|---|---|
| `hda_images[]` | Every `run/vms/*/hda.img` found, under the repo path and via a bounded `find` over the search roots. |
| `procs[]` | Every process whose `/proc/<pid>/comm` starts with `qemu-system`, each with `pid`, `comm`, `proc_state`, `ours`, `cmdline_head`. |
| `roots_unreadable[]` | A search root that **exists but cannot be traversed**. A root that does **not exist** is omitted, not recorded — a fresh host before the clone has no repo path, and that must never read as a permissions failure. |

### Process matching

`/proc/<pid>/comm`, **prefix** compare against `qemu-system`.

`comm` is truncated to 15 bytes, so the literal value on a real host is
`qemu-system-x86` — an exact match on `qemu-system-x86_64` would never fire.

argv is **never** the thing that finds a process. `pgrep -f` on a pattern that
also appears in your own command line killed two ssh sessions, and matching on
`qemu` false-positived on the QEMU source build. argv is read only to decide
whether an already-identified `qemu-system` process is *ours*.

### The precedence ladder

State is derived through an **ordered `if`/`elif` chain**, in exactly this order:

```
UNKNOWN  >  ZOMBIE  >  LIVE  >  FOREIGN  >  DORMANT  >  CLEAN
```

| State | Condition | Meaning |
|---|---|---|
| `UNKNOWN` | `roots_unreadable[]` non-empty | Cannot tell. Fails closed. |
| `ZOMBIE` | any `proc_state == Z` | Still holds guest RAM; not reaped until its parent dies, so polling for it never returns. |
| `LIVE` | any `ours` process | Our CVM is running. |
| `FOREIGN` | any process, none `ours` | Someone else's stack — usually an expected bare-metal rental, not an outage. |
| `DORMANT` | `hda_images[]` non-empty | A stopped CVM's data disk is still there. |
| `CLEAN` | none of the above | Nothing to protect. |

**Every state except `CLEAN` blocks destructive actions.**

The ladder exists because the naive six-way split **is not a partition**. A host
with an intact `hda.img` *and* a tenant's QEMU satisfies two descriptions at
once. Because the chain is ordered, `DORMANT`'s "and no processes" clause is
redundant — anything with processes was already claimed above it — and
harmlessly so. Written as independent `if`s it would be a second partition bug.

### Recovery is composed from the facts, never selected by the state

This is the point of the refactor. Each fact contributes its own steps:

- any `hda_images[]` entry → drain rentals → `lium-cvm.sh stop <name>` →
  `rm -rf run/vms/<name>` → re-run, plus the data-disk warning
- any `proc_state == Z` → `sudo tmux kill-session -t lium-cvm`
- a foreign process → check `rental_history` before touching the host

A host in two conditions gets both sets, in that order. If the state *name* chose
the message, the host with both an `hda.img` and a tenant's QEMU would be told
about the tenant and **never** about the `rm -rf` its data disk actually needs.

### Search roots

Owned by the script. `bootstrap.sh` runs the guard **before Ansible starts**, so
the guard can never read `group_vars`. Its built-in defaults are the single
source of truth; `group_vars/all/main.yml` mirrors them into
`lium_hda_search_roots` for the in-play include, and `tests/test_guard.sh`
asserts the two lists agree so the duplication cannot rot silently.

```bash
./files/lium-guard.sh --print-default-roots   # what the test compares against
```

### What the guard can and cannot see

It finds an `hda.img` in two ways, and the union of them is its reach:

1. **The checkout it is run from.** The script locates itself and searches
   `<checkout>/neurons/executor/dstacktee`. `bootstrap.sh` always runs the copy
   inside the checkout the provider invoked, so this covers a checkout anywhere
   on the filesystem — `/data0/lium-io` included.
2. **The search roots**, to a depth of 12.

The one shape it does **not** see: a CVM whose checkout is on a volume outside
`/home`, `/opt` and `/srv`, converged from a *different* checkout. Pass
`--repo-path` (or add the volume to `lium_hda_search_roots`) on such a host.

The depth budget was 8 and is now 12 for a specific reason. A checkout at
`/opt/lium-io` puts `hda.img` at exactly depth 8, so 8 covered that one shape and
nothing deeper — while `/home/<user>/lium-io` is depth 9, and `/home` is in the
root list *precisely* to catch a home-directory checkout. The budget defeated the
reason the root was there, and the guard answered `CLEAN` on a host holding a
renter's encrypted disk.

## Kernel command-line math

`files/lium-cmdline.sh` owns all of it. See `roles/kernel/README.md` for the
ordering guarantee the safety rests on.

It **sources** `/etc/default/grub` and `/etc/default/grub.d/*` rather than
grepping them, because the drop-in is
`GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT:-} ..."` and only shell
expansion gives its effective value. Grepping would report the literal text and
miss a clobber. Sourcing both, in order, is exactly what `grub-mkconfig` does.

Every assertion resolves a key to its **winning value**, never bare set
membership — a host with a stale `vfio_iommu_type1.dma_entry_limit=65535` would
otherwise report "our 16777216 is present" and pass.

## The destructive guard include

```yaml
- ansible.builtin.include_role:
    name: common
    tasks_from: assert_no_cvm
  vars:
    common_guard_reason: GRUB change requiring reboot
    common_guard_override_var: lium_force_reboot
```

Each destructive role names the **one** variable that accepts **its** damage.
There is no blanket override: `--force-converge` restores the destructive roles
to the play, it does not authorise anything.

## Test seams

Documented in `tests/README.md` only, never in the provider-facing README.

The gate compares each seam against its **production value** rather than asking
whether it is defined, because several of them (`lium_state_dir`,
`lium_hda_search_roots`) are ordinary settings that always have a value.
Comparing to the production value is the only rule that covers both kinds and
gives "non-default" a precise meaning. Both `lium_test_mode: true` and
`ANSIBLE_LIUM_TEST=1` are required; either alone fails closed.
