# Tests

**This is the only place test seams are documented.** The provider-facing
`../README.md` must never mention them.

## Four environments, and what each can honestly prove

| Name | What it is | What runs there |
|---|---|---|
| **CI-runner** | `ubuntu-24.04` — an **unsupported** release for this playbook | OS-agnostic tests only: lint, syntax, shellcheck, the shell units, the fixture playbooks |
| **CI-container** | `container: ubuntu:26.04` — a **supported** release with no TDX, no SGX, no NVIDIA, no GRUB | Everything OS-gated: the real `bootstrap.sh`, the preflight messages, live report generation |
| **VM** | human-run `multipass launch 26.04` | The real reboot/resume cycle and whole-playbook idempotence |
| **TDX** | a real provider host | Every positive hardware path |

The runner/container split is not fussiness. `ubuntu-latest` is Ubuntu 24.04,
which this playbook's own `os.supported` gate hard-rejects — so on the runner the
bootstrap dies at the OS gate and every BIOS-knob assertion below it never
executes. A supported-OS container is the only place those can fire for the right
reason.

## Running them

```bash
cd neurons/executor/ansible
pip install -r tests/requirements.txt

bash tests/test_forbidden_strings.sh
bash tests/test_seam_gate.sh
bash tests/test_guard.sh
bash tests/test_resume_guard.sh
bash tests/test_verify_schema.sh
bash tests/test_docs_pointer.sh

export ANSIBLE_LIUM_TEST=1
ansible-playbook -i inventory/localhost.yml tests/render_scripts.yml
ansible-playbook -i inventory/localhost.yml tests/test_command_vars.yml
ansible-playbook -i inventory/localhost.yml tests/test_grub_fixtures.yml
ansible-playbook -i inventory/localhost.yml tests/test_kernel_idempotence.yml
ansible-playbook -i inventory/localhost.yml tests/test_sgx_registration.yml
ansible-playbook -i inventory/localhost.yml tests/test_stubs.yml
ansible-playbook -i inventory/localhost.yml tests/test_repo_app_pristine.yml
ansible-playbook -i inventory/localhost.yml tests/test_maintenance_profile.yml
```

## One suite at a time, per machine

The fixture playbooks use fixed paths under `/tmp` (`/tmp/lium-grub-fixture`,
`/tmp/lium-kernel-idem`, `/tmp/lium-maint`, `/tmp/lium-repo-fixture`), so **two
runs of the suite on the same machine will delete each other's work trees**.
GitHub-hosted runners are isolated, so this is not a merge-gate concern; it
matters on a shared or self-hosted runner, and to anyone running the suite twice
at once locally.

Within a single run the cases are already isolated — each GRUB fixture owns its
own subdirectory, and the root is cleared once at the start rather than between
cases. That churn used to cause an intermittent "destination directory does not
exist" at roughly one run in three on a clean `/tmp`.

## Test seams

A seam redirects a decision away from the real host — a fake `/proc/cmdline`, a
stubbed `update-grub`, a fixture `lspci`. They exist because most of this
playbook's behaviour cannot otherwise be proved without TDX hardware.

**Both gates are required, always:**

```yaml
lium_test_mode: true          # in the playbook's own vars:
```
```bash
ANSIBLE_LIUM_TEST=1           # in the environment
```

Either alone fails closed, by name, in `common`'s very first task. The CI job's
`env:` supplies only the second, so **every seam-using playbook also sets
`lium_test_mode: true` in its own `vars:`**. A test that sets only one will fail
correctly but confusingly if you have not read this.

| Seam | Redirects |
|---|---|
| `lium_proc_cmdline_path` | what the running kernel appears to have |
| `lium_grub_default_path` | the provider's GRUB defaults |
| `lium_grub_dropin_path` | where our drop-in is written |
| `lium_grubcfg_path` | the generated bootloader configuration |
| `lium_update_grub_cmd` | the bootloader regeneration command |
| `lium_lspci_output` | the PCI device inventory |
| `lium_proc_root` | where processes are enumerated from |
| `lium_boot_id_path` | the current boot id |
| `lium_hda_search_roots` | where CVM data disks are searched for |
| `lium_state_dir` | where state and the report are written |

The gate compares each variable to its **production value** rather than asking
whether it is defined, because `lium_state_dir` and `lium_hda_search_roots` are
ordinary settings that always have a value. Comparing against production is the
only rule that covers both kinds and gives "non-default" a precise meaning.

### Scope controls, which are not seams

These narrow what a role does so it can run on a machine that is not a TDX host.
They never redirect a decision to a fake path, so they are not gated:

| Variable | Effect |
|---|---|
| `lium_kernel_reboot_enabled` | `false` stops the kernel role short of rebooting the machine running the test |
| `lium_kernel_fixture_mode` | `true` skips the writes to real `/sys` and the real `modprobe` |
| `lium_sgx_install` | `false` skips Intel's packages, which need root and network |
| `lium_start_key_provider` | `false` skips the container |
| `lium_assert_vsock`, `lium_assert_key_provider` | `false` skips assertions that only mean anything on real SGX hardware |

## Why `tests/group_vars` is a symlink

Ansible loads `group_vars/` next to the **playbook**, not from the project root.
A playbook in this directory would otherwise see none of them, and would then
fail for a completely unrelated reason.

The symlink is deliberate rather than `vars_files:`, because `vars_files`
(precedence 14) outranks play `vars:` (12) — so every per-test seam override
would silently lose to the production default. `group_vars` sits at precedence 6,
below play vars, which is what lets each test override cleanly.

## What each test proves

| Test | Proves | Where |
|---|---|---|
| `test_forbidden_strings.sh` | The "must not have" list is enforced, and three load-bearing positives are present. The **only** real enforcement of that list — `--check --diff` cannot do it, because check mode skips command tasks entirely. | CI-runner |
| `test_seam_gate.sh` | All four gate combinations: only both-open is allowed. | CI-runner |
| `test_guard.sh` | The guard blocks on **disk state first**, ignores argv, cannot self-match, fails closed on unreadable and skips benignly on absent — and composes its recovery text from facts rather than from the state name. | CI-runner |
| `test_resume_guard.sh` | The resume fires exactly once per arming, on the right boot, and never on a looping host. | CI-runner |
| `test_verify_schema.sh` | The schema enum, the README headings and the golden example describe exactly the same set of checks. | CI-runner |
| `test_docs_pointer.sh` | The docs were actually edited — and the sections that were *not* meant to go are still there. | CI-runner |
| `test_command_vars.yml` | Every external command matches a golden string. | CI-runner |
| `test_grub_fixtures.yml` | render → union → write → regenerate → assert → drift, over four shapes, with last-occurrence-wins and superset-only `vfio-pci.ids=`. | CI-runner |
| `test_kernel_idempotence.yml` | Pass 2 leaves every managed file byte-identical — and pass 1 must have changed something, or the assertion would be vacuous. | CI-runner |
| `test_sgx_registration.yml` | The manifest length rule accepts 17950 and rejects 17954. | CI-runner |
| `test_stubs.yml` | Both stubs are loud when unset **and** when set. | CI-runner |
| `test_repo_app_pristine.yml` | `repo` accepts a pristine measured surface and **refuses** a modified one, naming the file and the restore command. | CI-runner |
| `test_maintenance_profile.yml` | `--skip-tags destructive` on a host with a CVM **reaches verify** instead of aborting. | CI-runner |
| `test_preflight_messages.yml` | Preflight names each missing BIOS knob, and the OS gate did *not* fire. | CI-container |

### The one that is easy to get wrong

`test_maintenance_profile.yml` asserts **positively** that the play reached
`verify` and produced a report saying `DORMANT`.

It deliberately does not assert "no error occurred". A run that aborts in the
guard `pre_tasks` and a run that finishes correctly are **both** non-zero, because
`verify` exits non-zero on a host full of `MUST_FIX` checks. Only the report tells
them apart.

Without that test, the play-level tag rule in `site.yml` is asserted only in
prose — and its failure mode is invisible until somebody runs the playbook on a
real host that already has a CVM.

## Fixtures

| Fixture | Shape |
|---|---|
| `grub/default_only` | every argument already in `GRUB_CMDLINE_LINUX_DEFAULT` |
| `grub/linux_only` | the **au2 shape**: arguments in `GRUB_CMDLINE_LINUX`, `_DEFAULT` empty |
| `grub/split` | arguments across both |
| `grub/conflicting` | a stale `dma_entry_limit=65535` **and** a narrower pre-existing `vfio-pci.ids=` |
| `proc/cmdline_complete` | a fully converged running kernel |
| `proc/cmdline_missing_tdx` | drift: `kvm_intel.tdx=on` never took |
| `proc/cmdline_narrower_vfio_ids` | a device disappeared from the id list |
| `lspci/h200_8gpu_4nvswitch.txt` | 8 GPUs, 4 NVSwitches, plus one unrelated device |
| `lspci/empty.txt` | no CC-capable device |
| `guard/dormant_hda` | an `hda.img` with **no processes** — the case a process-only guard waves through |
| `guard/hda_plus_foreign` | an `hda.img` **and** someone else's QEMU — the case that is not a partition |
| `guard/unreadable_root` | exists but cannot be read → `UNKNOWN` |
| `guard/clean_root` | nothing at all |
| `manifest/field6_17950.bin` | the body Intel accepts |
| `manifest/field6_17954.bin` | the version-prefixed blob Intel answers with a 400 |
| `verify-report.example.json` | a healthy reference host — doubles as documentation |

`guard/unreadable_root` is chmod-ed to 000 by the test and restored in a trap.
**Running as root skips that one case**, because root bypasses file permissions
and the fixture cannot be made unreadable. The test says so out loud rather than
reporting a pass nobody earned; the CI-runner job is non-root, so it is covered
there.

## The two boxes CI cannot close

- **The real reboot/resume cycle.** A reboot terminates a CI job. The logic half
  is machine-checked by `test_resume_guard.sh`; the real cycle is `vm/RUNBOOK.md`.
- **Every positive hardware path.** No TDX, no SGX and no CC-capable GPU exist in
  CI, so those checks are `MUST_FIX`, `UNKNOWN` or `SKIPPED` there by
  construction. They belong to the hardware acceptance run.
