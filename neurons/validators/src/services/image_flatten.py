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

from core.docker_utils import df_command, parse_df_available_bytes

FLATTENED_IMAGE_REPO = "lium-flattened"
# headroom left on the Docker disk after the copy, so the pod and the node still have room to write
FLATTEN_DISK_MARGIN_BYTES = 10 * 1024**3
# a 24 GB image takes about 5-15 minutes to copy; past this the create fails with the original error
FLATTEN_TIMEOUT_SECONDS = 1800
_DOCKER_READ_TIMEOUT_SECONDS = 120

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


def _flatten_command(image: str, tag: str, changes: list[str]) -> str:
    """Create (never start) a container from `image`, pipe its export into an import, tag the result
    only when both sides succeeded, and always remove the container. Plain sh: no pipefail."""
    t = FLATTEN_TIMEOUT_SECONDS
    change_args = " ".join(f"--change {shlex.quote(c)}" for c in changes)
    return (
        f"cid=$(/usr/bin/docker create --entrypoint /lium-flatten-noop {shlex.quote(image)}) || exit 97; "
        'st=$(mktemp); '
        f'iid=$( (timeout {t} /usr/bin/docker export "$cid"; echo $? > "$st") '
        f"| timeout {t} /usr/bin/docker import {change_args} -); irc=$?; "
        'erc=$(cat "$st"); rm -f "$st"; /usr/bin/docker rm -f "$cid" >/dev/null 2>&1; '
        f'if [ "$irc" = 0 ] && [ "$erc" = 0 ]; then /usr/bin/docker tag "$iid" {shlex.quote(tag)}; '
        'else [ -n "$iid" ] && /usr/bin/docker rmi -f "$iid" >/dev/null 2>&1; '
        'echo "flatten failed: export=$erc import=$irc" >&2; exit 98; fi'
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
    changes = image_flatten_changes(info.get("Config") or {})
    df = await ssh_client.run(
        df_command("\"$(/usr/bin/docker info --format '{{.DockerRootDir}}')\""),
        check=True,
        timeout=_DOCKER_READ_TIMEOUT_SECONDS,
    )
    if parse_df_available_bytes(df.stdout) < int(info.get("Size") or 0) + FLATTEN_DISK_MARGIN_BYTES:
        raise ImageFlattenNoDisk(FLATTEN_NO_DISK_RENTER_SENTENCE)
    await ssh_client.run(
        _flatten_command(image, tag, changes), check=True, timeout=FLATTEN_TIMEOUT_SECONDS * 2 + 120
    )
    return tag
