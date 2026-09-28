# Pod secrets (not released yet)

This doc is for renters who pass secrets (API tokens, keys, credentials) to a pod. Secrets are
delivered as files, never as environment variables, so they don't show up in `docker inspect`,
`/etc/environment`, an image layer, `docker commit` or a volume backup.

## Where they are

| Path | What |
| --- | --- |
| `/run/lium/secrets/<NAME>` | One file per secret, holding the value exactly as sent (no trailing newline added) |
| `/run/lium/secrets/.ready` | Appears once every secret file is in place |

- `/run/lium/secrets` is a `tmpfs` (`noexec,nosuid,nodev`): never on disk, though on a host with
  swap the kernel may page it out (see [Lifetime](#lifetime)).
- Your secrets may use up to 1 MiB in total, with each file counted in whole 4 KiB pages (a 10-byte
  token uses 4 KiB). A set over the limit is refused when you rent, before the pod is created, and
  the error names the secret that doesn't fit.
- The directory is `0700` and each file `0400`, owned by the user your image runs as (its `USER`,
  or root when none is set). Other users in the container cannot read them.
- A secret name is letters, digits and `_`, not starting with a digit (for example `HF_TOKEN`).

## Wait for `.ready` before reading

Secrets arrive shortly after the container starts, not before. A command that reads them at boot
should wait for the marker first, with a timeout (here 2 minutes):

```sh
i=0
until [ -f /run/lium/secrets/.ready ]; do
  i=$((i + 1)); [ "$i" -gt 600 ] && { echo "secrets not delivered" >&2; exit 1; }
  sleep 0.2
done
export HF_TOKEN="$(cat /run/lium/secrets/HF_TOKEN)"
```

The marker is written only after every secret file has been written and handed to your user. If
delivery fails, the marker is never written and the rent fails.

Do not delete `.ready`, and do not empty `/run/lium/secrets`: the validator reads a pod with no
`.ready` and no files there as one whose secrets were lost.

In Python:

```python
import os, sys, time

deadline = time.monotonic() + 120
while not os.path.exists("/run/lium/secrets/.ready"):
    if time.monotonic() > deadline:
        sys.exit("secrets not delivered")
    time.sleep(0.2)
with open("/run/lium/secrets/HF_TOKEN") as f:
    hf_token = f.read()
```

Until delivery finishes, `/run/lium/secrets` is readable only by root, so a non-root process cannot
look inside it yet. `os.path.exists` and the shell `[ -f ]` treat that as "not there yet" and keep
waiting; `pathlib.Path.exists()` raises `PermissionError` instead, so do not use it for this loop.

## Lifetime

The files are on a tmpfs and are never written to the container's disk, but on a host with swap
the kernel may page them out to swap.

Rebooting the pod from the pod page creates the container again. It delivers the secrets again
only once the platform also sends them on reboot, which needs a matching platform-side change
deployed first; until then a rebooted pod comes back without `/run/lium/secrets` at all, and a new
rent is the way to get them back.

Stopping and starting the pod from the pod page, or a failed edit that puts your previous container
back, empties `/run/lium/secrets` and removes `.ready`. From then on the pod has no secrets until
you rent a new pod (or reboot this one, once the platform sends secrets on reboot). The validator
records each such start on the executor's persistent data volume, which no pod can reach and which
is kept when the executor is updated, and from then on it does not report that container as having
lost its secrets, even when your workload stops on its `.ready` timeout and Docker restarts it.

Docker can also restart the container by itself: after a host reboot, a Docker restart, or when
your main process dies (for example out of memory). Then `/run/lium/secrets` is empty, `.ready` is
gone, and the secrets are not delivered again, because they are not stored on the host. Use a
timeout when you wait for `.ready`, as in the examples above, so your workload exits with an error
after the timeout. The validator also reports such a pod as having lost its secrets,
including when your workload keeps stopping on that timeout and Docker keeps restarting it,
unless the platform started that container after a stop or a failed edit before (above).
To get them back, rent a new pod (or reboot this one, once the platform sends secrets on reboot,
as above).
