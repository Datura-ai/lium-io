# `sgx_key_provider`

The Intel DCAP stack, platform registration, TD quote plumbing, and the
sealing-key provider container. Not destructive.

This is the role with the most field evidence behind it, because almost every
failure it prevents looks like something else.

## Two paths, because the API key has no distribution channel

The Intel PCS subscription key is account-level and legitimately shared across a
fleet — but there is no mechanism to distribute it today. It is copied by hand
from a peer host's `config/default.json`, which a new provider does not have.

| `lium_intel_api_key` | What happens |
|---|---|
| **set** | Full path: install and configure PCCS locally, run in-band registration, point both QPL configs at the local PCCS. |
| **unset** | **Degraded path**: no PCCS install, no registration. Both QPL configs point at `lium_pccs_url`, which may name a *remote* PCCS. `verify` reports `sgx.registration_pck_certs` as `SKIPPED` with the remediation. |

The in-tree precedent for the degraded path is already there: the key provider's
own `sgx_default_qcnl.conf` ships a remote PCCS URL. It is loud, documented, and
re-runnable the moment the key arrives.

## The signal that lies

`/var/log/mpa_registration.log` reported *"Registration status indicates
registration is completed successfully"* on the **broken** host and the healthy
one alike. `platforms_registered = 0` is normal on a healthy host.

**The only trustworthy signal is the `pck_cert` row count** in
`/opt/intel/sgx-dcap-pccs/pckcache.db`. A healthy reference host reads
`pck_cert=8, pck_certchain=1, platforms=1`.

Both misleading names are in `tests/forbidden-patterns.txt`.

## Never name `sgx-setup`

It lives in Ubuntu *universe*, not Intel's repository. Naming it **aborts the
whole apt transaction**, so every other package silently fails to install and the
host simply looks like it has no SGX support.

## The PCCS wizard that cannot run

`sgx-dcap-pccs`'s postinst runs an interactive wizard which fails under apt and
leaves the package half-configured. `dpkg --configure` non-interactively takes
its skip branch and clears that state — and then everything the wizard would have
done has to be done by hand:

- template `config/default.json` (mode 0600 — it holds the API key)
- **`npm install`** in `/opt/intel/sgx-dcap-pccs`; without it PCCS dies with
  `ERR_MODULE_NOT_FOUND: config`
- a self-signed certificate into `ssl_key/`
- `chown -R pccs:pccs`

## Registration is irreversible

`PCKIDRetrievalTool` flips the registration efivar to "completed", and the
manifest may not be re-extractable afterwards. So, in this order:

1. **Back up every SGX efivar first**, with `cat` redirection — `cp` fails
   *"Illegal seek"* on efivarfs. Guarded by `creates:` so a good backup is never
   overwritten, and matched by glob because the GUID suffix varies by platform: a
   hardcoded name that did not match would back up nothing while reporting
   success, right before the irreversible step.
2. Only then run the retrieval tool.

And it never runs at all unless `pck_cert_count` was **read** as 0. "0
certificates" means register; "cannot read the database" means we cannot tell,
and this cannot be undone.

## 17950, exactly

Field 6 of the tool's CSV is the PLATFORM_MANIFEST, hex-encoded. Decoded, it is
**17950 bytes** and that is what Intel accepts.

The raw efivar, after stripping only its 4-byte efivarfs attribute header, is
**17954** bytes — a 4-byte `version+size` header *plus* the body. POSTing that
returns `400 Error-Code: InvalidPlatformManifest`. This cost one host a 400 and a
round of debugging, so the length is asserted before the POST.

## No secret ever reaches an argv

`no_log` hides a value from Ansible's own output and does **nothing** about
`/proc/<pid>/cmdline`, which any local user can read with `ps aux`.

So every authenticated call uses `ansible.builtin.uri`, which passes headers
in-process. Where curl were ever unavoidable it would need a `0600` config or
header file created in the same block and removed in an `always:`. A curl
invocation with an inline header is in `tests/forbidden-patterns.txt`.

## The two defaults that reboot the guest

A CVM that boots cleanly — all GPUs initialised, zero `RmInitAdapter` errors —
and then reboots itself at ~160–190 s is almost always one of these. The guest
fetches its TD report, sends it over vhost-vsock to be turned into a quote,
presents that for the sealing key, and unseals its disk. Any failure in that
chain fails `dstack-prepare`, and the guest reboots.

**`/etc/qgs.conf` ships `#port = 4050` commented out.** qgsd then serves a unix
socket, reports `active`, and never answers the guest.
Signature: `vsock failure: Connection reset by peer (os error 104)`.

**`/etc/sgx_default_qcnl.conf` ships `"use_secure_cert": true`**, which verifies
the PCCS TLS certificate — and a local PCCS uses a self-signed one.
Signature: `[QCNL] CURL error: (60)`, `[QPL] ... 0xb033`, then
`vsock failure: failed to fill whole buffer`.

It must be `false` in **both** QPL configs: the system one used by qgsd, and the
key provider's own used by the sealing enclave. The checked-in repo copy already
ships `false`, so that half is an idempotent no-op there — the substantive work
on it is the `pccs_url` repoint.

## The end-to-end proof

`Running in PRODUCTION mode`, `RestartCount == 0`, and **zero** `error 44` /
`load_enclave ... failed` lines.

All three together, because any one alone can look healthy: a container can be
running while its enclave crash-loops. That combination is the only thing that
proves registration, PCCS, the QPL and qgsd are all correct at once.

## What CI can prove

No positive path without SGX hardware. CI proves the templates render with the
right suite, that `sgx-setup` and an inline-header curl are absent, that the
17950 rule accepts the body and rejects the version-prefixed blob, and that the
degraded path emits `SKIPPED`. The rest belongs to the hardware acceptance run.
