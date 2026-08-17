# CVM host bring-up — automated

Takes a fresh Ubuntu **25.10** or **26.04 LTS** Intel TDX host from bare OS to
CVM-ready: kernel command line, vfio bindings, GPU confidential-compute mode,
the dstack QEMU 9.2.1 build, Docker, the SGX DCAP stack and the sealing-key
provider. It is safe to re-run — every run converges from wherever the host
already is, and refuses anything that would damage a CVM that already exists.

## Run it

From a clone of this repo, on the host itself:

```bash
cd neurons/executor/ansible
sudo ./bootstrap.sh
```

That is the whole command. `bootstrap.sh` installs `ansible-core` if it is
missing, then runs the playbook against `localhost`. There is no remote
transport and no inventory of other machines.

The run may need **one reboot** to pick up new kernel command-line parameters.
It asks first (unless you pass `--yes`), arms a one-shot systemd unit, reboots,
and continues by itself. See [After a reboot](#after-a-reboot).

**Collections required: none.** Everything here is `ansible.builtin.*`, so
`pip install ansible-core` is a complete install — `ansible-galaxy` is never
needed.

## What is not automated

**BIOS settings.** The playbook cannot change firmware. If a knob is off,
preflight stops and names it:

- Intel **TDX** (and TME / TME-MT) in the CPU security menu
- Intel **SGX**, with PRMRR size ≥ 64M
- **VT-d** / Intel Virtualization Technology for Directed I/O

Also out of scope: per-executor `.env` (that is `docs/host-setup.md` §5),
datacenter port ACLs, and running or supervising the CVM itself.

## What each role does

| Role | What it does | Destructive? |
|---|---|---|
| `common` | Collects every host fact once, as root, and derives the CVM guard state. | no |
| `preflight` | Hard gates: OS, arch, sudo, TDX, SGX, VT-d, a CC-capable GPU, GRUB, disk headroom. | no |
| `kernel` | `dma_entry_limit`, vfio modules, the GRUB drop-in, drift detection, the reboot. | **yes** |
| `gpu` | nvtrust, CC mode on, vfio-pci rebinding of GPUs and NVSwitches. | **yes** |
| `docker` | Docker Engine from Docker's `noble` repo. | no |
| `qemu` | The dstack QEMU 9.2.1 build and `/etc/dstack/client.conf`. | **yes** |
| `repo` | The lium-io checkout and the CVM OS image. | no |
| `sgx_key_provider` | Intel DCAP packages, PCCS, platform registration, quote plumbing, the key-provider container. | no |
| `cvmd` | The CVM host control daemon: package, config, authorised clients, service. Optional — see [cvmd](#cvmd). | no |
| `catalog` | Variable interface only — see [Not implemented yet](#not-implemented-yet). | no |
| `verify` | Writes the machine-readable report and sets the exit code. | no |

## The tag model

Two independent dimensions, doing two different jobs.

**Audit, do not act:**

```bash
sudo ./bootstrap.sh --tags verify
```

Reaches `common` and `verify` only. Nothing is changed. Deliberately *not*
`preflight` — a hard gate firing there would abort the run before the report
existed, which is the opposite of what an audit is for.

**Maintenance.** On a host that already has a CVM, `bootstrap.sh` selects this
profile by itself and tells you it did:

```
--skip-tags destructive
```

That withholds exactly three roles — **`kernel`, `gpu`, `qemu`** — the only ones
that can change a measurement or reset a GPU. Everything else still converges:
`preflight`, `docker`, `repo`, `sgx_key_provider`, `catalog` and `verify`,
including the check that `dstacktee/app/` is unmodified. In this mode preflight
*records* a flipped BIOS knob into the report instead of aborting, so a host that
lost a setting after a firmware update produces a report naming it.

## The two locks

`--force-converge` restores the three destructive roles to the play. **It does
not authorise the damage.** Each of those roles is still blocked by its own
per-action variable, and each names exactly what it accepts:

| Variable | Accepts |
|---|---|
| `lium_force_qemu_rebuild` | that any existing `hda.img` becomes undecryptable |
| `lium_force_gpu_cc` | resetting every GPU |
| `lium_force_reboot` | an unattended reboot |

So the full form is two flags, on purpose:

```bash
sudo ./bootstrap.sh --force-converge -e lium_force_qemu_rebuild=true
```

## Why a re-run may refuse

The guard blocks whenever the host still holds CVM state. The decisive signal is
an **on-disk `run/vms/*/hda.img`**, not a running process: `lium-cvm.sh stop`
shuts the guest down and returns success **without removing the VM directory**,
so "stopped CVM, no QEMU running, intact encrypted data disk, open rental" is a
normal state — and replacing QEMU under it makes the renter's disk permanently
undecryptable.

| State | Means |
|---|---|
| `CLEAN` | No CVM state. Everything converges. |
| `DORMANT` | A stopped CVM's `hda.img` is still on disk. |
| `LIVE` | Our CVM is running. |
| `FOREIGN` | Someone else's QEMU is running — usually an expected bare-metal rental. |
| `ZOMBIE` | A QEMU zombie still holds guest RAM. |
| `UNKNOWN` | A search root exists but could not be read. The run stops. |

The refusal prints a recovery procedure composed from what was actually found,
so a host in two conditions at once gets both sets of steps.

## After a reboot

```bash
systemctl is-failed lium-cvm-resume
journalctl -u lium-cvm-resume
less /var/log/lium-cvm/resume.log
```

The resume unit runs **once per reboot it was armed for**, then disables itself
on every exit path. A later unrelated reboot cannot re-trigger it.

If a GRUB change never takes effect, the run stops after
`lium_max_converge_reboots` (default 2) rather than looping. It prints the tokens
that are still missing and what to do by hand. A converge that finishes cleanly
resets that budget, so a healthy host is never bricked by it. To clear it
manually:

```bash
sudo ./bootstrap.sh -e lium_reset_reboot_budget=true
```

## Reading the report

Every run writes `/var/lib/lium-cvm/verify-report.json`, validated against
`schemas/verify-report.schema.json`. `bootstrap.sh` prints a summary and exits
non-zero if anything is `MUST_FIX`.

```bash
sudo python3 -m json.tool /var/lib/lium-cvm/verify-report.json
```

Each check carries a stable `id`, a `status` (`OK` / `WARN` / `MUST_FIX` /
`UNKNOWN` / `SKIPPED`), what was `observed`, what was `expected`, a
`remediation`, and where the evidence came from. Fix `MUST_FIX` first; `WARN` is
worth reading but does not block. `UNKNOWN` means the check could not be read at
all — it is never silently treated as a pass.

One check is always a `WARN`: `net.acl_probe_required`. The host can prove a port
is *listening* but never that it is *reachable*, and datacenter edge ACLs
blackholing 8001/2200/19001 have taken nodes offline before. Its remediation
explains how to test from outside.

## Variables

Pass overrides with `-e`, or put them in a file and use
`--vars-file /root/lium-cvm-secrets.yml`. See `group_vars/all/secrets.example.yml`.

| Variable | Default | Purpose |
|---|---|---|
| `lium_intel_api_key` | *(unset)* | Intel PCS subscription key. Unset selects the degraded path: no local PCCS, no in-band registration, and both QPL configs point at `lium_pccs_url`. |
| `lium_pccs_url` | `https://localhost:8081/…` | PCCS to fetch DCAP collateral from. May be remote. |
| `lium_pccs_user_token` | *(unset)* | Token the local PCCS requires on `/platforms`. |
| `lium_qemu_tarball_url` | *(unset)* | Prebuilt QEMU tarball. Set it with its sha256 to skip the 10–40 minute source build. |
| `lium_qemu_tarball_sha256` | *(unset)* | Required whenever the URL is set. |
| `lium_preflight_fatal` | `true` | `false` reports BIOS gate failures instead of aborting. The maintenance profile sets it. |
| `lium_max_converge_reboots` | `2` | Reboots one converge may spend before giving up. |
| `lium_reset_reboot_budget` | `false` | `true` clears that counter manually. |
| `lium_force_qemu_rebuild` | `false` | See [The two locks](#the-two-locks). |
| `lium_force_gpu_cc` | `false` | " |
| `lium_force_reboot` | `false` | " |
| `lium_repo_path` | `/opt/lium-io` | Where the lium-io checkout lives. |
| `lium_repo_ref` | `main` | Release tag to check out. |

### cvmd

The CVM host control daemon is **optional per host**. Leave
`lium_cvmd_package_url` unset and the role says so and ends; set it and the
package is fetched against its checksum, installed, configured and started, and
the run fails if `/health` never answers.

| Variable | Default | Meaning |
|---|---|---|
| `lium_cvmd_package_url` | `""` | Release tarball from `cvmd/packaging/build.sh`. |
| `lium_cvmd_package_sha256` | `""` | Required with the URL. |
| `lium_cvmd_authorized_clients` | `[]` | `[{hotkey, scope}]` — ss58 addresses, scope `validation` or `renter`. |
| `lium_cvmd_bind_address` | `0.0.0.0` | Listen address. |
| `lium_cvmd_port` | `8443` | Listen port. |

Authorised clients are **bittensor hotkeys, not SSH keys** — cvmd authenticates
a signed request against an ss58 address. Full detail in
[`roles/cvmd/README.md`](roles/cvmd/README.md).

Run it alone with `--tags cvmd`.

### The approved artifact catalog

`catalog` stages the signed manifest cvmd starts from. It runs in the same play
as `cvmd` and after it, because the seed is cvmd configuration: written into
cvmd's config directory, verified through cvmd's own venv, and applied by
restarting cvmd. See [`roles/catalog/README.md`](roles/catalog/README.md).

| Variable | Meaning |
|---|---|
| `lium_catalog_signer` | The ss58 this host trusts. Required to have a catalog at all. |
| `lium_catalog_manifest_url` | Where the seed is fetched from, and where cvmd polls. |
| `lium_catalog_manifest_src` | Or a signed manifest from the controller. |
| `lium_catalog_manifest_sha256` | Optional extra pin on the fetched seed. |
| `lium_catalog_images_dir` | Where day-zero staged the approved OS images. |

## Where things are written

| Path | What |
|---|---|
| `/var/lib/lium-cvm/verify-report.json` | The report. |
| `/var/lib/lium-cvm/resume.state` | Armed reboot state. |
| `/var/lib/lium-cvm/converge_reboots` | Reboot budget. |
| `/var/lib/lium-cvm/qemu.sha256` | Installed QEMU fingerprint. |
| `/var/lib/lium-cvm/efivar-backup/` | SGX efivars, backed up before registration. |
| `/var/log/lium-cvm/bootstrap.log` | Full run output. |
| `/var/log/lium-cvm/resume.log` | Post-reboot continuation. |
| `/var/log/lium-cvm/qemu-build.log` | QEMU build. |
| `/var/log/lium-cvm/run-<timestamp>/` | Per-task JSON, including what actually changed. |

## Flags

| Flag | Effect |
|---|---|
| `--yes` | Do not ask before rebooting. |
| `--check` | Ansible check mode — report, change nothing. |
| `--tags` / `--skip-tags` | Passed through to `ansible-playbook`. |
| `--tag <ref>` | Check the lium-io repo out at this ref. |
| `--repo-url`, `--repo-path` | Where the checkout comes from and goes. |
| `--no-clone` | Do not touch the checkout at all. |
| `--vars-file <path>` | Extra variables, e.g. secrets. |
| `--force-converge` | Restore the destructive roles (see the two locks). |
| `--print-expected-preflight-rc` | Print the exit code a preflight failure produces, then exit. |
