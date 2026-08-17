# `catalog`

Stages the signed artifact manifest a host starts from (DAH-2578).

The role does one thing: put a manifest the platform signed at
`/etc/cvmd/manifest.json`, having proved cvmd will accept it. It is **not** the
source of truth — cvmd polls the backend and replaces its own working copy under
`/var/lib/cvmd/catalog/`. The seed exists so a host can launch before it has
ever reached the backend, and so an air-gapped host can launch at all.

| Variable | Meaning |
|---|---|
| `lium_catalog_signer` | The ss58 this host trusts. Required whenever a manifest is named. |
| `lium_catalog_manifest_url` | Fetch the seed from here. Also what cvmd polls. |
| `lium_catalog_manifest_src` | Or copy it from the Ansible controller. |
| `lium_catalog_manifest_sha256` | Optional extra pin on the fetched seed. |

Set exactly one of `_url` and `_src`; both is a refusal, because which one wins
would depend on task order.

## Why the checksum stopped being required

The DAH-2544 stub required `lium_catalog_manifest_sha256` with the URL, on the
grounds that a manifest fetched without an integrity check is not a working
configuration. That reasoning still holds — the *check* just moved somewhere
better. The signature covers the manifest wherever it came from and whenever it
arrives; a pinned checksum covers one specific fetch, and goes stale the moment
the platform publishes a new serial, at which point every converge fails on a
host that is doing nothing wrong.

So the checksum is still accepted, and still pins the bytes when it is set. What
is *required* now is the signer, without which nothing can be verified at all.

## Loud in both directions

- **Nothing named** — one line saying what cvmd will fall back on, then the role
  ends.
- **Named** — the signer is asserted, the file is staged, and cvmd's own
  `parse_manifest` reads it back. A manifest that cvmd would refuse never
  becomes this host's seed.

The staged manifest is reported: serial, artifact ids, floors and expiry. An
already-expired one is called out rather than failed on — cvmd will refuse to
launch from it and say so, and a host whose URL works replaces it within one
refresh interval.

## Why this role restarts cvmd

cvmd adopts a newer seed at startup and on its refresh cycle, so the restart is
about *when*, not *whether*: without it a day-zero host would serve no catalog
until the first refresh tick, and the verify report at the end of the run would
say so. The handler is the role's own rather than the cvmd role's, because
handlers do not cross a play boundary and a borrowed one makes this role
runnable only in the exact play order `site.yml` happens to use.
