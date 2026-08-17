# `qemu`

The dstack QEMU 9.2.1 build. **Destructive** — replacing QEMU changes RTMR0 and
makes any existing `hda.img` undecryptable. Withheld by the maintenance profile
and separately blocked by `lium_force_qemu_rebuild`.

## Two ways in

| | When | Cost |
|---|---|---|
| `tarball.yml` | `lium_qemu_tarball_url` is set | seconds |
| `build.yml` | otherwise | 10–40 minutes |

The tarball path has **no producer yet** — it is a follow-up. The variable
interface ships now so that when the release pipeline lands, this role needs no
changes. Setting the URL without `lium_qemu_tarball_sha256` is refused: QEMU is a
measured input, and installing one whose contents were never verified changes
RTMR0 to a value nobody can attest to.

## `--setenv=HOME=` is load-bearing

`systemd-run` starts with a minimal environment in which **`HOME` is unset**. The
build script's `set -u` then dies at line 3 — while the unit goes `inactive`
within seconds and looks *finished*, with nothing built.

`lium_qemu_systemd_run_cmd` therefore carries `--setenv=HOME=`, the rendered
script refuses to run without `HOME`, and both are asserted in CI.

## Why it waits on a marker, not the unit state

`QEMU_BUILD_DONE_OK` in `/var/log/lium-cvm/qemu-build.log`. A unit that exited
early is `inactive` and indistinguishable from one that finished. The marker is
the only signal that the build actually completed.

The unit name is `qemu-dstack-build`, matching the existing provisioning scripts,
and the role checks the legacy name too — so a half-finished build is found and
waited on rather than started a second time.

## Never `DUMP_ACPI_TABLES`

That variant is the verifier's **oracle** binary. It cannot run VMs. Building it
produces a QEMU that looks installed and fails every launch.

## `client.conf` always runs

Even when nothing was built. `/etc/dstack/client.conf` is what the launcher
actually reads; a host with the right binary and no `client.conf` falls back to
the distro QEMU, which can never reproduce RTMR0.

The `/usr/local/bin` symlink is cosmetic — it exists only so the launcher's
`check` subcommand, which looks on `PATH`, stops reporting a false MISSING on a
correctly configured host.
