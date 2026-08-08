# `repo`

The lium-io checkout and the CVM OS image. Not destructive-tagged, so it **does**
run in maintenance mode — which is exactly why it is careful about the ref.

## It never moves the ref on a host that has a CVM

`neurons/executor/dstacktee/app/` is the **measured surface**. Moving the
checkout rewrites it.

That is not unsafe on its own — measurements are fixed at launch, and the
documented upgrade flow moves the ref deliberately. But doing it as a *side
effect of a maintenance run* collides with "a re-run with a CVM present performs
zero destructive actions". So on a non-`CLEAN` host the role reports the current
HEAD, leaves it alone, and emits `repo.ref_behind` as a WARN pointing at the
upgrade procedure in `docs/host-setup.md` §6.

## `app/` must be pristine — hard assert

`docker-compose*.yml`, `pre_launch_script.sh` and `init_script.sh` are hashed into
the compose the validator whitelists. **Any** local edit produces a hash nobody
has whitelisted, and every attestation fails.

The failure lists the dirty files and the exact `git checkout --` command.

## Do not copy another host's `.env`

Per-executor configuration stays manual (`docs/host-setup.md` §5) and this role
never writes it.

Copying one from another host has produced a non-whitelisted compose hash in the
field: a stale runner digest, an uncommitted patch in
`app/pre_launch_script.sh`, and a missing `/etc/dstack/client.conf` — so
`dstack.py` fell back to distro QEMU and could never reproduce RTMR0.

## The runner digest is reported, never compared

`repo.runner_digest_set` reports **presence only**. A checkout's whitelist can be
*behind* the deployed validator — on one occasion the whitelisted digest was the
broken one. Reporting a mismatch here would tempt someone into downgrading to it.
