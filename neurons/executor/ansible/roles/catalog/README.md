# `catalog` (stub)

A variable interface, not an implementation. The catalog work lands in
**DAH-2576 / DAH-2578**; this exists so day-zero provisioning does not wait on
it.

| Variable | Meaning |
|---|---|
| `lium_catalog_manifest_url` | Where to fetch the catalog manifest from. |
| `lium_catalog_manifest_sha256` | Required whenever the URL is set. |

## Loud in both directions

- **Unset** — one line saying what it is waiting on, then the role ends.
- **Set** — the companion variable is asserted, then the run *fails* with
  "not implemented".

The manifest decides what a host will run, so one fetched without a checksum is
not a working configuration.
