# cvmd — CVM host daemon

The control-plane daemon that runs on a CVM host. It exposes a signed HTTPS API with two
authorized clients holding disjoint scopes, holds the node's state machine, and launches and
stops the node's CVM.

## API

| Route | Auth | Now |
|---|---|---|
| `GET /health` | none | `200 {"version": ...}` |
| `GET /v1/state` | either key | the state document plus the running CVM |
| `POST /v1/cvm` `kind=validation` | validator key | launches the validation CVM (DAH-2576) |
| `POST /v1/cvm` `kind=renter` | platform key | `501` — DAH-2580 |
| `DELETE /v1/cvm` | platform key | tears the CVM down |

A scope violation on an otherwise valid request is `403`. Everything else that fails auth is
`401`. A validly signed body that is not usable is `422`, never a scope bypass.

`DELETE` is platform-scoped even for a validation CVM. It carries no `kind`, so a validator
holding that right could destroy a *renter's* CVM — the platform owns the node's lifecycle and
the validator only asks for a validation CVM.

## Launching a CVM

```
POST /v1/cvm
{"kind": "validation",
 "qemu": "10.1.0",
 "os_image_hash": "<64 lowercase hex>",
 "compose_hash":  "<64 lowercase hex>"}
```

The body names **which software stack to run** and nothing else. Sizing — vCPUs, memory, disk,
GPUs, forwarded ports — is provider configuration read from `/etc/cvmd/config.toml`: the caller
says what to run, the host says how big it is.

The three hashes are a *pinned triple*, and they are the entire input to the CVM's measurements.
cvmd resolves them against the local catalog (`/etc/cvmd/catalog.json`), and a triple the
catalog does not carry is refused naming the component that was not approved. That is what lets
a validator detect a host serving a stack other than the one it expects to attest.

The request returns when the CVM is up, with the launch report — instance id, forwarded ports,
and the guest's SSH host-key fingerprint. Meanwhile `/v1/state` moves
`RECONCILING → LAUNCHING → VALIDATION_RUNNING`.

Readiness is that fingerprint read: a host key can only be collected once sshd inside the guest
answers through the forward, which is far later and far more meaningful than QEMU accepting a
connection. So `cvm_ssh_guest_port` must name the guest side of one `cvm_ports` entry — a value
nothing forwards is refused up front, because it would leave the launch with no readiness check
at all. On a host with no `ssh-keyscan` installed, readiness degrades to a TCP accept and the
report's fingerprint is null; the note in the report says which happened.

### The measurement gate

cvmd imports `dstacktee/scripts/dstack.py` **as a library** and calls the same functions its CLI
calls, so the bytes it measures are produced by the same code `lium-cvm.sh` produces them with.
`tests/test_dstack_library.py` prepares one CVM both ways and compares every file.

Calling the same code makes identical measurements likely; the gate makes them checked. Between
`setup_instance` writing the VM directory and QEMU starting, cvmd measures what it just built:

| Component | Read from |
|---|---|
| `compose_hash` | `sha256(<vm_dir>/shared/app-compose.json)` — the same digest dstack's own `app_compose_hash()` takes |
| `os_image_hash` | `<image_path>/digest.txt` |
| `qemu` | `get_qemu_version_string()`, which is what lands in the attested `vm_config` |

Any difference from the requested triple and the launch does not happen — the VM directory is
removed and the node is left exactly where it was. Configuration says what was *intended*; only
the files say what was *built*.

### One CVM per node

GPU passthrough is exclusive, so a node holds one CVM. The check reads `/proc` for a running TDX
guest rather than cvmd's own records — during the CVM v2 rollout the CVM already on a host is
most likely one `lium-cvm.sh` started, and cvmd's records cannot see those. A stopped CVM's
directory blocks a launch too: it still owns the node until `DELETE` removes it.

### The supervisor

`run_instance` blocks for the VM's whole life and the host API server has to stay up beside it,
so cvmd spawns `python -u -m cvmd.dstack.child`. That process **double-forks**: the survivor has
init as its parent and leads its own session and process group.

Three separate things have to hold for the CVM to outlive cvmd, and each was learned by
restarting the daemon on a real host:

| | |
|---|---|
| Its own session | so a signal aimed at cvmd's process group misses it |
| init as its parent | so it is reaped the instant it exits. As cvmd's child it lingered as a **zombie**, and a zombie is still a process-group member — `killpg(pgid, 0)` kept succeeding, so every teardown ran the full signal ladder (260s) and then reported "still present after SIGKILL" about a guest that had powered off gracefully. `os.kill(pid, 0)` has the same blind spot, and that is what `shutdown_instance` waits on |
| `KillMode=process` in the unit | because systemd kills by **cgroup**, and detaching does not leave one. With the default, `systemctl restart cvmd` logged `Killing process ... (qemu-system-x86) with signal SIGKILL` and the node came back FAILED holding a CVM directory and no CVM |

`RECONCILING` is how the daemon comes back under a guest it did not start.

`console.log` in the VM directory is the guest's serial console plus the full QEMU command line
`run_instance` prints before exec — the primary evidence for what was launched and how it booted.
The child runs with `-u` because Python block-buffers stdout to a file, and without it that one
line stays in the buffer for the VM's whole life while QEMU writes past it.

Teardown asks the guest to power off through dstack's own path, then signals the supervisor's
**process group**. dstack's own `--force` kills the pid in `runtime.json`, which is the
supervisor's — QEMU is its child, so killing that pid alone leaves a running VM holding the
guest's RAM and the GPUs while every layer above reports success. Confirming the group is gone
is the floor; the four-condition predicate (VFIO descriptors closed, RAM returned, ports
bindable) is DAH-2577, as is the per-hardware-class value for `teardown_timeout_seconds` — a
1.13 TB guest was measured taking 43 minutes to return its memory.

## Signing a request

Four headers: `X-Cvmd-Hotkey`, `X-Cvmd-Timestamp` (unix ns), `X-Cvmd-Nonce` (≥16 random bytes,
hex), `X-Cvmd-Signature` (hex, `0x` optional). The signature is over:

```
blob = sha256(
    b"cvmd-v1\x00"                       # domain separator
  | lp(method) | lp(request_target)      # request_target = path + query, exactly as received
  | lp(body_bytes)
  | lp(timestamp_ascii) | lp(nonce_ascii)
)
    where lp(x) = uint32_be(len(x)) | x
```

`src/cvmd/auth/blob.py` is the authoritative definition; `tests/fixtures/golden_vector.json` is
the reference vector for client implementations. Two deviations from architecture doc §03 are
recorded in that module's docstring — method and target are signed, and fields are
length-prefixed.

## Replay protection

A request is accepted only if it is inside the freshness window, above the startup floor, and its
`(hotkey, nonce)` pair is unseen. The nonce is fsynced **before** the request reaches a handler.

The startup floor is read once at startup and never advances during a process lifetime. That is
the whole design — a floor that advanced per request would be a strict monotonic timestamp, which
rejects the second of any two concurrent requests and every client retry. `tests/test_replay.py`
asserts both halves: replays are refused, and concurrent out-of-order requests are not.

## bittensor version

cvmd pins `bittensor==11.0.2`, deliberately independent of the executor's `9.0.0`. CVM hosts run
the Ubuntu 25.10/26.04 system Python (3.13/3.14), which 9.x does not support. The split is safe
because a 9.x signature verifies under 11.x — `tests/test_golden_vector.py` pins that with a
fixture signed under a real 9.10.1 venv. Do not "fix" the mismatch by downgrading.

Two v11 API breaks the executor's middleware predates: `bittensor.Keypair` moved to
`bittensor.sp_core.Keypair`, and `verify()` is strictly bytes-typed.

## Development

```bash
pdm use -f python3.13
pdm install
pdm run pytest
```

## Packaging

```bash
./packaging/build.sh
```

Produces `dist/cvmd-<version>.tar.gz` and prints its sha256 — the value the DAH-2544 Ansible role
takes as `lium_cvmd_package_sha256`. The tarball carries the wheel, a hash-pinned
`requirements.lock`, the unit file, a default config, and `install.sh`.
