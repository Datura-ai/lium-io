"""Squash a renter's image into one layer on the node, for Sysbox nodes that cannot start it as is.

Sysbox 0.7.1 on Docker's classic overlay2 store (and 0.6.6 and older on the containerd store) cannot
ID-map the relative layer paths Docker uses once an image has more layers than fit in one page of
overlay mount options (nestybox/sysbox#725, fixed upstream but not in a Sysbox release yet). A
one-layer copy of the same files starts. `docker export | docker import` keeps only the root
filesystem, so the image config is put back with `--change`; `docker import` cannot set
HEALTHCHECK, STOPSIGNAL, SHELL or ONBUILD, which are lost.
"""

import json
import shlex

import asyncssh

from core.docker_utils import df_command

FLATTENED_IMAGE_REPO = "lium-flattened"
# headroom left on the Docker disk after the copy, so the pod and the node still have room to write
FLATTEN_DISK_MARGIN_BYTES = 10 * 1024**3
# a 24 GB image takes about 5-15 minutes to copy; past this the create fails with the original error
FLATTEN_TIMEOUT_SECONDS = 1800
_DOCKER_READ_TIMEOUT_SECONDS = 120
FLATTEN_LOCK_PATH = "/tmp/lium-flatten.lock"
FLATTEN_NO_DISK_EXIT = 95

FLATTEN_NO_DISK_RENTER_SENTENCE = (
    "not enough free disk on this node to prepare this image; choose another node or a smaller image"
)


class ImageFlattenNoDisk(RuntimeError):
    renter_sentence = FLATTEN_NO_DISK_RENTER_SENTENCE


def _dockerfile_word(value: str) -> str:
    # inside double quotes the Dockerfile parser unescapes only \, " and $ (no variable expansion)
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'


def image_flatten_changes(config: dict) -> list[str]:
    """`docker import --change` lines that restore what the flatten drops from the image `Config`."""
    changes: list[str] = []
    # ENTRYPOINT first: in Dockerfile order it clears a CMD set before it
    if config.get("Entrypoint"):
        changes.append(f"ENTRYPOINT {json.dumps(config['Entrypoint'])}")
    if config.get("Cmd"):
        changes.append(f"CMD {json.dumps(config['Cmd'])}")
    for entry in config.get("Env") or ():
        key, _, value = entry.partition("=")
        if "\n" in value or "\r" in value:
            # one --change is one Dockerfile line; dropping the variable would change the program
            raise ValueError(f"environment variable {key} has a line break; cannot carry it over")
        changes.append(f"ENV {key}={_dockerfile_word(value)}")
    if config.get("WorkingDir"):
        changes.append(f"WORKDIR {_dockerfile_word(config['WorkingDir'])}")
    if config.get("User"):
        changes.append(f"USER {_dockerfile_word(config['User'])}")
    for port in sorted(config.get("ExposedPorts") or ()):
        changes.append(f"EXPOSE {port}")
    if config.get("Volumes"):
        changes.append(f"VOLUME {json.dumps(sorted(config['Volumes']))}")
    for key, value in sorted((config.get("Labels") or {}).items()):
        if "\n" in key + value or "\r" in key + value:
            continue  # metadata only (e.g. a long description); nothing at run time reads it
        changes.append(f"LABEL {_dockerfile_word(key)}={_dockerfile_word(value)}")
    return changes


def _flatten_command(image_id: str, tag: str, changes: list[str], volumes: list[str], need_bytes: int) -> str:
    """Under a node-wide lock: reuse a copy another flatten finished meanwhile, check the Docker disk
    holds `need_bytes`, create (never start) a container from the image ID, pipe its export into an
    import, tag the result only when both sides succeeded, and always remove the container and its
    volumes. Plain sh: no pipefail."""
    t = FLATTEN_TIMEOUT_SECONDS
    change_args = " ".join(f"--change {shlex.quote(c)}" for c in changes)
    # an empty volume per VOLUME path: create would otherwise copy the image's files there (extra disk)
    mounts = " ".join(f"--mount {shlex.quote(f'type=volume,dst={v},volume-nocopy')}" for v in volumes)
    root = "\"$(/usr/bin/docker info --format '{{.DockerRootDir}}')\""
    return (
        # one flatten per node: concurrent rentals would each pass the disk check and copy at once
        f"exec 9>>{FLATTEN_LOCK_PATH}; flock -w {t} 9 || exit 96; "
        f"/usr/bin/docker image inspect {shlex.quote(tag)} >/dev/null 2>&1 && exit 0; "
        f"avail=$({df_command(root)} | awk 'NR==2{{print $4}}'); "
        'case "$avail" in ""|*[!0-9]*) echo "flatten failed: unexpected df output" >&2; exit 94;; esac; '
        f'[ "$avail" -ge {int(need_bytes)} ] || exit {FLATTEN_NO_DISK_EXIT}; '
        # the ID, not the tag: a pull may move the tag after inspect and the copy would not match its name
        f"cid=$(/usr/bin/docker create --entrypoint /lium-flatten-noop {mounts} {shlex.quote(image_id)}) || exit 97; "
        'st=$(mktemp); '
        f'iid=$( (timeout {t} /usr/bin/docker export "$cid"; echo $? > "$st") '
        f"| timeout {t} /usr/bin/docker import {change_args} -); irc=$?; "
        'erc=$(cat "$st"); rm -f "$st"; /usr/bin/docker rm -fv "$cid" >/dev/null 2>&1; rrc=$?; '
        # a left-over container or volume fails the flatten untagged, so the next rental retries the cleanup
        f'if [ "$irc" = 0 ] && [ "$erc" = 0 ] && [ "$rrc" = 0 ]; then /usr/bin/docker tag "$iid" {shlex.quote(tag)}; '
        'else [ -n "$iid" ] && /usr/bin/docker rmi -f "$iid" >/dev/null 2>&1; '
        'echo "flatten failed: export=$erc import=$irc rm=$rrc" >&2; exit 98; fi'
    )


async def flatten_image_on_host(ssh_client: asyncssh.SSHClientConnection, image: str) -> str:
    """The local tag of a one-layer copy of `image`, built on the node unless one exists already.

    Keyed by the image ID, so a retry or a later rental of the same image reuses the copy. Raises
    ImageFlattenNoDisk when the Docker disk cannot hold the copy, any other error when it failed.
    """
    inspect = await ssh_client.run(
        f"/usr/bin/docker image inspect --format '{{{{json .}}}}' {shlex.quote(image)}",
        check=True,
        timeout=_DOCKER_READ_TIMEOUT_SECONDS,
    )
    info = json.loads(inspect.stdout)
    tag = f"{FLATTENED_IMAGE_REPO}:{info['Id'].removeprefix('sha256:')}"
    existing = await ssh_client.run(
        f"/usr/bin/docker image inspect {shlex.quote(tag)} >/dev/null 2>&1",
        check=False,
        timeout=_DOCKER_READ_TIMEOUT_SECONDS,
    )
    if existing.exit_status == 0:
        return tag
    config = info.get("Config") or {}
    command = _flatten_command(
        info["Id"],
        tag,
        image_flatten_changes(config),
        sorted(config.get("Volumes") or ()),
        int(info.get("Size") or 0) + FLATTEN_DISK_MARGIN_BYTES,
    )
    # waits up to one flatten for the node lock, then runs its own
    result = await ssh_client.run(command, check=False, timeout=FLATTEN_TIMEOUT_SECONDS * 3 + 120)
    if result.exit_status == FLATTEN_NO_DISK_EXIT:
        raise ImageFlattenNoDisk(FLATTEN_NO_DISK_RENTER_SENTENCE)
    if result.exit_status != 0:
        raise RuntimeError(f"flatten exited {result.exit_status}: {str(result.stderr or '')[-2000:]}")
    return tag
