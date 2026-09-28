# Pod secrets (not released yet)

This doc is for renters who pass secrets (API tokens, keys, credentials) to a pod. Secrets are
delivered as files, never as environment variables, so they don't show up in `docker inspect`,
`/etc/environment`, an image layer, `docker commit` or a volume backup.

## Where they are

| Path | What |
| --- | --- |
| `/run/lium/secrets/<NAME>` | One file per secret, holding the value exactly as sent (no trailing newline added) |
| `/run/lium/secrets/.ready` | Appears once every secret file is in place |

- `/run/lium/secrets` is an in-memory `tmpfs` (`noexec,nosuid,nodev`).
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

Rebooting the pod from the pod page creates the container again and delivers the secrets again.

Docker can also restart the container by itself: after a host reboot, a Docker restart, or when
your main process dies (for example out of memory). Then `/run/lium/secrets` is empty, `.ready` is
gone, and the secrets are not delivered again, because they are not stored on the host. Use a
timeout when you wait for `.ready`, as in the examples above, so your workload exits with an error
after the timeout. The validator also reports such a pod as having lost its secrets,
including when your workload keeps stopping on that timeout and Docker keeps restarting it.
Reboot the pod from the pod page to get them back.
