# Pod secrets

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
should wait for the marker first:

```sh
until [ -f /run/lium/secrets/.ready ]; do sleep 0.2; done
export HF_TOKEN="$(cat /run/lium/secrets/HF_TOKEN)"
```

The marker is written only after every secret file has been written and handed to your user. If
delivery fails, the marker is never written and the rent fails.

In Python:

```python
import pathlib, time

secrets = pathlib.Path("/run/lium/secrets")
while not (secrets / ".ready").exists():
    time.sleep(0.2)
hf_token = (secrets / "HF_TOKEN").read_text()
```

## Lifetime

The files live only in the container's memory. After the container restarts (including a host
reboot), `/run/lium/secrets` is empty and `.ready` is gone. Secrets are not re-delivered after a
restart.
