# `cvmd`

Installs and configures **cvmd**, the CVM host control daemon. The daemon itself
lives in [`neurons/executor/cvmd/`](../../cvmd/); this role is how it reaches a
host.

## Variables

| Variable | Meaning |
|---|---|
| `lium_cvmd_package_url` | Where to fetch the release tarball from. Unset ⇒ the role reports that and ends. |
| `lium_cvmd_package_sha256` | Required whenever the URL is set. Verified on download. |
| `lium_cvmd_authorized_clients` | `[{hotkey, scope}]` — who may call cvmd, and in which scope. |
| `lium_cvmd_bind_address` | Listen address. `0.0.0.0` by default. |
| `lium_cvmd_port` | Listen port. `8443` by default. |
| `lium_cvmd_skew_seconds` | How far a request timestamp may sit from the host clock. |
| `lium_cvmd_max_body_bytes` | Request-body cap, enforced before the body is buffered. |
| `lium_cvmd_health_retries`, `lium_cvmd_health_delay` | How long to wait for `/health` before failing the run. |

Produce the tarball and its checksum with
[`cvmd/packaging/build.sh`](../../cvmd/packaging/build.sh), which prints exactly
the sha256 this role expects.

### The launch path (DAH-2576)

Optional as a whole — cvmd serves `/health` and `/v1/state` without any of it and
refuses a launch naming the setting it lacks. `lium_cvmd_dstack_scripts_dir` is
the switch: set it and the role requires the rest, because a host with HALF a
launch configuration looks configured and refuses every launch at the first
request.

| Variable | Meaning |
|---|---|
| `lium_cvmd_dstack_scripts_dir` | The dstacktee `scripts` directory. cvmd **imports** `dstack.py` from it as a library, so a path to the repo root will not do. |
| `lium_cvmd_run_dir` | Where cvmd keeps the VM directories it creates. Deliberately not dstacktee's own `run/vms`. |
| `lium_cvmd_catalog` | The approved artifacts — see below. |
| `lium_cvmd_key_provider_port` | dstack's key-provider port. `3443`. |
| `lium_cvmd_launch_timeout_seconds` | How long a launch waits for the guest before failing the node. |
| `lium_cvmd_teardown_timeout_seconds` | The graceful-poweroff window before cvmd signals the process group. |
| `lium_cvm_vcpus`, `lium_cvm_memory`, `lium_cvm_disk` | CVM sizing. |
| `lium_cvm_gpus` | PCI slots to pass through, `["all"]`, or `[]` for none. |
| `lium_cvm_ports` | Forwarded ports, `protocol[:address]:host:guest`. |
| `lium_cvm_env_file` | Passed as `--env-file`. Lands outside `app-compose.json`, so it does not change the compose hash. |
| `lium_cvm_ssh_guest_port` | The guest-side SSH port. Set ⇒ cvmd reports the host-key fingerprint and uses reading it as proof the CVM is up. |
| `lium_cvm_pin_numa`, `lium_cvm_hugepages` | Both change the QEMU command line and therefore the measurements. `lium-cvm.sh` passes neither. |

**Sizing is provider configuration, never an API field.** A cvmd request names
which software stack to run; the host decides how big it is. So there is nothing
a caller could send to fill a gap here — a host missing a size refuses every
launch, which is why the role refuses first.

### The catalog pins a triple

`lium_cvmd_catalog` is a list of approved artifacts, rendered to
`/etc/cvmd/catalog.json`. Each entry pins the **triple** — QEMU build, OS image
hash, compose hash — that this host is allowed to produce, plus the local paths
that produce it:

```yaml
lium_cvmd_catalog:
  - id: validation-v3
    kind: validation
    qemu: "10.1.0"
    os_image_hash: a6eafc5f007f642d8ea90c7fa8881f1e6715720ccb531941a28218f4f26d7b02
    compose_hash: ab4d14336f0762c0d8ec7631a69148246661de84ceead7a215f8a33b74fd43e6
    os_image_path: /opt/lium-io/neurons/executor/dstacktee/run/images/dstack-nvidia-0.5.11
    compose_path: /etc/cvmd/composes/validation-v3.yml
    init_script: /opt/lium-io/neurons/executor/dstacktee/app/init_script.sh
    pre_launch_script: /opt/lium-io/neurons/executor/dstacktee/app/pre_launch_script.sh
```

Hashes must be 64 lowercase hex digits — no `sha256:` prefix, no uppercase. cvmd
compares them against values it computes itself, so any other spelling would
never match and the launch would fail as "not approved", sending an operator
looking at the wrong thing. `load_catalog` refuses those spellings outright, and
the role runs it against the rendered file before it replaces the working one.

`compose_path` must point at an **already-resolved** compose. `lium-cvm.sh`
substitutes `${EXECUTOR_RUNNER_IMAGE_DIGEST}` before dstack measures the file;
under cvmd the catalog carries the resolved copy, and the compose-hash gate is
what catches it if that ever stops being true.

This is the DAH-2576 stub. DAH-2578 replaces it with the backend's signed
manifest, at which point this list stops being the source of truth.

### Authorised clients are hotkeys, not SSH keys

cvmd authenticates a **signed request** against a bittensor hotkey, so the only
thing it can act on is an ss58 address paired with a scope:

```yaml
lium_cvmd_authorized_clients:
  - hotkey: 5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX
    scope: validation
  - hotkey: 5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY
    scope: renter
```

The two scopes are disjoint. `validation` may create a validation CVM and read
state; `renter` may create and destroy a renter CVM. Neither can act in the
other's scope — the full matrix is asserted in
[`cvmd/tests/test_scope.py`](../../cvmd/tests/test_scope.py).

The DAH-2544 stub called this `lium_cvmd_authorized_client_keys` and its fixture
held an SSH public key. Nothing consumed it — setting it always failed the run
with "not implemented" — so the name and the shape both changed here rather
than keeping an interface the daemon cannot read.

## Loud in both directions

- **Unset** — one line saying cvmd was not requested, then the role ends.
- **Set** — the package is installed, the service is started, and the run
  **fails if `/health` never answers**.

There is no path that leaves a half-configured host looking finished. That was
the stub's property and it is kept.

## What it refuses, and where

cvmd is fail-closed on all of these by itself. The role refuses them first so
the failure names the variable the operator got wrong instead of reading
`systemd unit entered failed state`.

| Refused | Checked by |
|---|---|
| URL set without a checksum, or without any client | role assert |
| An entry that is not `{hotkey, scope}`, or a scope outside `{validation, renter}` | role assert |
| A hotkey that is not shaped like an ss58 address | role assert (format only) |
| The same hotkey listed twice | role assert |
| Host Python older than 3.13 | role assert, and again in `install.sh` |
| A tarball whose sha256 does not match | `get_url` |
| A tarball that matches but is not a cvmd release | role assert on the unpacked payload |
| **A hotkey whose base58 checksum is wrong** | `validate:` — see below |

### The authorized-clients file is validated by cvmd's own loader

`template:` renders the candidate file to a temporary path and installs it only
if `validate:` exits 0 — and the validate command is
`load_authorized_clients()`, the very function that refuses at daemon startup.
It does the real ss58 checksum, the scope enum and the duplicate check, so a
file that would leave cvmd unstartable never replaces the working one.

Both files are rendered **whole from a template**, never line-edited.
`lineinfile` appends when its regexp does not match, which is how a structured
config silently becomes invalid while the run still reports success.

## Idempotence

The install step records the package sha256 in `/opt/cvmd/.installed-sha256`
and is skipped while that matches *and* `/opt/cvmd/venv/bin/cvmd` exists.
Configuration is applied on every run regardless, so a changed client list takes
effect without a reinstall. The stamp is written only after `install.sh`
returns 0 — a stamp written first would mark a failed install as complete and
the next run would skip repairing it.

## Paths

`install.sh` hardcodes these, so the role's copies in `defaults/main.yml` are
names for one list rather than settings; changing one alone would move where
Ansible looks without moving where the installer writes.

| Path | Holds |
|---|---|
| `/opt/cvmd/venv` | The virtualenv and the `cvmd` entry point |
| `/etc/cvmd/config.toml` | Rendered by this role |
| `/etc/cvmd/authorized_clients.json` | Rendered by this role |
| `/etc/cvmd/tls/` | Self-signed pair, generated by `install.sh` if absent |
| `/var/lib/cvmd/` | `state.json` and the replay nonce store |

`/var/lib/cvmd` is **state, not cache**: losing it resets the replay startup
floor.
