"""Which subnet's validator started a rental container or volume, and which ones a sweep may touch.

The validator drives the executor host's Docker daemon (the executor mounts the host socket), so
every sweep sees every container and volume on the host. A testnet executor can share a host with a
mainnet one, and each validator only knows its own backend's pods: a name-only sweep from one
network removes the other network's renter pods. Every `pod_*`/`filler_*` container, `volume_*`
volume, `lium-dind-build-*` build container and `lium-build-*` built image the validator creates
carries ``io.lium.netuid`` (plus the validator hotkey and the workload kind), and a sweep removes
only what carries its own netuid.

Resources created before the label existed carry none. They belong to mainnet — the only network
that rented at scale before the label — so only a mainnet caller removes them, exactly as it does
today; a testnet caller leaves every unlabeled or foreign-labeled resource alone.

No sweep removes another network's labeled resource: one left behind by a network whose validator
stopped visiting the host stays until an operator removes it
(`docker ps -a --filter label=io.lium.netuid=<n>`).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from services.const import FILLER_CONTAINER_PREFIX, POD_CONTAINER_PREFIX

NETUID_LABEL = "io.lium.netuid"
VALIDATOR_LABEL = "io.lium.validator"
KIND_LABEL = "io.lium.kind"

KIND_POD = "pod"
KIND_FILLER = "filler"
KIND_PROBE = "probe"
KIND_BUILD = "build"
KIND_WARM = "warm"

MAINNET_NETUID = 51

# The prefixes whose containers can hold a renter's (or the filler's) workload; `warm_` is the warm
# pool's created-never-started slot that a rental adopts by rename. `container_<miner hotkey>_*` and
# `health_check_*` are the backend's port-check containers: started by the backend with no label,
# holding no workload, so they keep the name-only rule (each network still reaps its own stale ones).
NETUID_SCOPED_CONTAINER_PREFIXES = (POD_CONTAINER_PREFIX, FILLER_CONTAINER_PREFIX, "warm_")
RENTAL_NAME_PREFIXES = (POD_CONTAINER_PREFIX, FILLER_CONTAINER_PREFIX)

# `{{.Label "k"}}` prints an empty string for a resource without the label; container and volume
# names have no spaces, so the label is whatever follows the fields before it.
NETUID_FORMAT_FIELD = '{{.Label "' + NETUID_LABEL + '"}}'


def ps_filter_names_netuid_command(*name_patterns: str) -> str:
    """`DockerCommand.ps_filter` with each line followed by the container's netuid label."""
    filters = " ".join(f'--filter "name={pattern}"' for pattern in name_patterns)
    return f"/usr/bin/docker ps -a {filters} --format '{{{{.Names}}}} {NETUID_FORMAT_FIELD}'"


def rental_labels(*, netuid: int, validator_hotkey: str | None, kind: str) -> dict[str, str]:
    labels = {NETUID_LABEL: str(netuid), KIND_LABEL: kind}
    if validator_hotkey:
        labels[VALIDATOR_LABEL] = validator_hotkey
    return labels


def label_flags(labels: dict[str, str]) -> str:
    """`--label k=v` flags for a shell `docker run`/`docker build`, each value shell-quoted."""
    return " ".join(f"--label {shlex.quote(f'{key}={value}')}" for key, value in labels.items())


def netuid_owns(netuid_label: str | None, caller_netuid: int) -> bool:
    """May a caller on ``caller_netuid`` remove a resource carrying ``netuid_label``?

    Its own label: yes. Another network's label: no. No label (legacy): only a mainnet caller.
    """
    if netuid_label:
        return netuid_label == str(caller_netuid)
    return caller_netuid == MAINNET_NETUID


def container_in_scope(name: str, netuid_label: str | None, caller_netuid: int) -> bool:
    """Whether a sweep on ``caller_netuid`` may remove this container; unscoped prefixes always may."""
    if not name.startswith(NETUID_SCOPED_CONTAINER_PREFIXES):
        return True
    return netuid_owns(netuid_label, caller_netuid)


@dataclass(frozen=True)
class LabeledName:
    name: str
    netuid_label: str | None


def parse_name_with_netuid(line: str) -> LabeledName | None:
    """One `<name> <netuid label>` line (the label may be empty); None for a blank line."""
    parts = line.split(maxsplit=1)
    if not parts:
        return None
    label = parts[1].strip() if len(parts) == 2 else ""
    return LabeledName(name=parts[0], netuid_label=label or None)


def parse_names_with_netuid(stdout: str) -> list[LabeledName]:
    parsed = (parse_name_with_netuid(line) for line in (stdout or "").splitlines())
    return [entry for entry in parsed if entry is not None]


def foreign_rental_containers(containers: list[LabeledName], caller_netuid: int) -> list[str]:
    """`pod_*`/`filler_*` containers on the host that do not carry the caller's netuid label.

    Stricter than ``container_in_scope``: an unlabeled container is foreign here even for mainnet,
    because the rental probe runs only when the backend lists nothing on the node, so any rental
    container it finds is one its own backend does not know about.
    """
    return [
        container.name
        for container in containers
        if container.name.startswith(RENTAL_NAME_PREFIXES)
        and container.netuid_label != str(caller_netuid)
    ]
