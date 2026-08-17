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
| `lium_catalog_signer` | The ss58 whose signature makes a manifest this host's catalog — see below. Required for a host that launches anything. |
| `lium_catalog_manifest_url` | Where cvmd polls for the current manifest. |
| `lium_catalog_images_dir` | Where day-zero staged the approved OS images. |
| `lium_catalog_refresh_seconds` | How often cvmd polls. The upper bound on how long a revocation takes to reach a node nobody pushes to. |
| `lium_cvmd_key_provider_port` | dstack's key-provider port. `3443`. |
| `lium_cvmd_launch_timeout_seconds` | How long a launch waits for the guest before failing the node. |
| `lium_cvmd_teardown_timeout_seconds` | The graceful-poweroff window before cvmd signals the process group. |
| `lium_cvmd_teardown_verify_timeout_seconds` | How long the node's hardware then gets to come back — see below. |
| `lium_cvmd_teardown_memory_tolerance` | How much of the guest's configured memory must be back before the node counts as free. |
| `lium_cvm_vcpus`, `lium_cvm_memory`, `lium_cvm_disk` | CVM sizing. |
| `lium_cvm_gpus` | PCI slots to pass through, `["all"]`, or `[]` for none. |
| `lium_cvm_ports` | Forwarded ports, `protocol[:address]:host:guest`. |
| `lium_cvm_env_file` | Passed as `--env-file`. Lands outside `app-compose.json`, so it does not change the compose hash. |
| `lium_cvm_ssh_guest_port` | The guest-side SSH port. Set ⇒ cvmd reports the host-key fingerprint and uses reading it as proof the CVM is up. Must be the guest side (the last field) of one `lium_cvm_ports` entry — the role refuses a value nothing forwards, since there would be nothing to read it through. |
| `lium_cvm_pin_numa`, `lium_cvm_hugepages` | Both change the QEMU command line and therefore the measurements. `lium-cvm.sh` passes neither. |

**Sizing is provider configuration, never an API field.** A cvmd request names
which software stack to run; the host decides how big it is. So there is nothing
a caller could send to fill a gap here — a host missing a size refuses every
launch, which is why the role refuses first.

### The two teardown budgets

They measure different things, which is why there are two.

`lium_cvmd_teardown_timeout_seconds` bounds how long the **guest** is given to
power itself off before cvmd signals its process group.
`lium_cvmd_teardown_verify_timeout_seconds` bounds how long the **host** then
takes to get its hardware back: QEMU reaped with no zombie left in the group,
every `/dev/vfio` descriptor closed, the guest's memory returned, and the
forwarded ports bindable again. A teardown reports success only when all four
hold together; running out of the second budget fails the node naming the
condition that did not.

The two are not proportional. A guest that exits in seconds can leave its memory
draining for tens of minutes, because that work happens in the kernel after the
process is gone — which is exactly why confirming the process group is empty was
never enough.

Measured switch windows, for setting the budget per hardware class. The two
columns are what the two budgets bound, so read them against the settings above.

| Host | Guest | Guest powered off | Hardware back afterwards | `DELETE` end to end |
|---|---|---|---|---|
| au11 — Intel TDX, QEMU 9.2.1, `dstack-nvidia-0.5.11` | 8 vCPU, 16 GiB, no GPU passthrough | ~5 s | **0.2 s** | **5–6 s** |
| the same fleet, under `lium-cvm.sh` | 1.13 TB | — | ~43 min | — |

On a small guest the hardware is back before the first evaluation finishes: the
process, the VFIO groups, the memory and the ports were all released 0.2 s after
the stop, across three runs. The 43-minute figure is the other end of the range
and the reason the default is 1800 s rather than a minute — it is sized for the
ordinary case with room, not for the largest guests on the fleet. **A host
running TB-class CVMs needs its own value**, and the measurement to set it from
is the `memory_returned` timing in its own `last_switch` report.

One outlier is worth knowing about: a single au11 teardown held on `ports_free`
for 161 s while the other three conditions settled in 0.2 s. It did not recur in
three later runs of the same shape, and a probe of the forwarded ports across a
teardown showed them bindable 2 s after the guest stopped — with and without
`SO_REUSEADDR`, so it was not `TIME_WAIT`. It is recorded rather than explained.
`verify_released` now logs what it is still waiting for every 30 s, which is what
was missing to diagnose it at the time.

### The catalog is a signed manifest, not a list in this file

Three settings, and only one of them is required:

```yaml
lium_catalog_signer: 5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX
lium_catalog_manifest_url: https://celiumcompute.ai/api/v1/cvm-catalog/manifest
lium_catalog_images_dir: /opt/lium-io/neurons/executor/dstacktee/run/images
```

The signer is what makes the whole thing work: cvmd verifies a manifest against
**this** ss58, never against the `signer` field inside the document. A manifest
checked against its own claimed signer proves only that somebody owns a key.
Unset, this host holds no catalog and refuses every launch, saying which setting
is why.

Each manifest entry pins the **triple** — QEMU build, OS image hash, compose
hash — that this host is allowed to produce, and carries the compose and both
guest scripts as *content*. cvmd writes them under
`/var/lib/cvmd/catalog/artifacts/<id>/` on every launch, so a compose edited on
the host is put back before it can be measured. The OS image is the exception:
it is gigabytes, so the entry names a directory under `lium_catalog_images_dir`
and `cvm/measure.py` checks that directory's own `digest.txt` against the pinned
hash before QEMU starts.

The manifest's compose is an **already-resolved** one. `lium-cvm.sh` substitutes
`${EXECUTOR_RUNNER_IMAGE_DIGEST}` before dstack measures the file; the platform
resolves it before signing, and the compose-hash gate is what catches it if that
ever stops being true.

Two files, deliberately:

| | |
|---|---|
| `/etc/cvmd/manifest.json` | The **seed**, staged by the `catalog` role. Ansible owns it, cvmd only reads it. |
| `/var/lib/cvmd/catalog/manifest.json` | The **working copy**. cvmd owns it and replaces it on every successful fetch. |

They are separate so neither overwrites the other's on a converge. cvmd adopts
the seed at startup and on every refresh, but only when it is *newer*: a seed
left behind after the platform published a revocation is refused as a rollback,
which is exactly what should happen to it.

### Revocation, and why it needs an expiry

A manifest carries a `serial` and an `expires_at`, and cvmd enforces both. The
serial only goes up, so a validly signed *older* manifest — from a stale cache,
a rewound replica, anything that can serve bytes to this host — cannot put a
revoked artifact back in the catalog. The expiry is the other half: without it,
a revocation could be defeated by simply never delivering the next manifest, and
the host would keep launching the revoked stack forever while looking healthy.

The cost is that a host which cannot reach the backend eventually stops
launching. That is the intended trade: `catalog.manifest` in the verify report
goes `MUST_FIX` with the expiry in `observed` before it happens.

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
