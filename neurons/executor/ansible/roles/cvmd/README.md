# `cvmd` (stub)

A variable interface, not an implementation. The daemon itself lands in
**DAH-2575**; this exists so day-zero provisioning does not wait on it, and so a
host provisioned today needs no re-plumbing when it arrives.

| Variable | Meaning |
|---|---|
| `lium_cvmd_package_url` | Where to fetch the cvmd package from. |
| `lium_cvmd_package_sha256` | Required whenever the URL is set. |
| `lium_cvmd_authorized_client_keys` | Public keys allowed to talk to the daemon. |

## Loud in both directions

- **Unset** — one line saying what it is waiting on, then the role ends.
- **Set** — the companion variables are asserted, then the run *fails* with
  "not implemented".

A stub that silently did nothing when set would be worse than no stub at all: it
would make a half-configured host look finished.
