"""DAH-3257 — one SSH command for the read-only host listings a rent runs before `docker run`.

Between the image pull and `docker run`, `create_container` asks the executor host eight
questions, each its own SSH round trip: the container list (`clean_existing_containers`), the
volume list and the mounted-volume list (`clean_stale_vloopback_volumes`), the volume names
again (`reclaim_dphn_cache_for_rental`), the kernel's GPU UUID→minor map and the shared device
nodes (`build_gpu_docker_config_for_executor`), the nvidia-smi power state
(`raise_low_power_limits_to_default`) and the image's volume-encryption label. On the far half of
the fleet a round trip is 0.2–0.6 s, so the listings cost more than the work they inform.

With ``RENTAL_PRERUN_HOST_PROBE_ENABLED`` the same listings come back from ONE command, each
section tagged (``PS\\t<name>``, ``VOL\\t<name>\\t<driver>``, …) and closed by its exit status
(``PS_RC\\t0``). The consumers read the section instead of running their own listing command and
keep every removal / write / retry exactly as it is. A section whose command failed (non-zero
``_RC``) is ``None`` and that consumer runs its own command; a probe that does not parse is ``None``
as a whole and every consumer runs as before. The probe reads, never writes.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

# The probe reads the same commands the per-command path runs, so the two paths cannot drift.
from services.gpu_power_limit import _POWER_STATE_CMD
from services.nvidia_devices import (
    _GPU_DEVICE_NODES_CMD,
    _PROC_GPU_INFO_CMD,
    shared_device_nodes_command,
)

# Tags, in the order the probe prints them. Every tag ends with a `<TAG>_RC` line.
PS_TAG = "PS"
VOL_TAG = "VOL"
MNT_TAG = "MNT"
GPUPROC_TAG = "GPUPROC"
GPUDEV_TAG = "GPUDEV"
SHARED_TAG = "SHARED"
SHAREDW_TAG = "SHAREDW"
POWER_TAG = "POWER"
LABEL_TAG = "LABEL"
_ALWAYS_TAGS = (
    PS_TAG,
    VOL_TAG,
    MNT_TAG,
    GPUPROC_TAG,
    GPUDEV_TAG,
    SHARED_TAG,
    SHAREDW_TAG,
    LABEL_TAG,
)
# The per-command path never looks at these commands' exit status (a `for p in …; [ -e "$p" ] && …`
# loop exits 1 when the last node is absent; `ls … || true` exits 0), so neither does the parser.
_RC_IGNORED_TAGS = (GPUDEV_TAG, SHARED_TAG, SHAREDW_TAG)
_RC_SUFFIX = "_RC"
# How much of an unparseable probe output reaches the log line.
PROBE_OUTPUT_LOG_CAP = 512

# The docker listings, shared with the per-command path in docker_service.py (imported there, so
# the two paths run the same text).
DOCKER_PS_ALL_NAMES_CMD = '/usr/bin/docker ps -a --format "{{.Names}}"'
DOCKER_VOLUME_LS_NAME_DRIVER_CMD = '/usr/bin/docker volume ls --format "{{.Name}} {{.Driver}}"'
DOCKER_MOUNTED_VOLUME_NAMES_CMD = (
    "/usr/bin/docker ps -a -q | xargs -r /usr/bin/docker inspect --format "
    '\'{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}}{{"\\n"}}{{end}}{{end}}\''
)
# Printed (untagged) when the awk that prefixes a section's lines fails: an untagged line makes
# the parser raise, so a prefixing failure is a whole-probe fallback, never an empty listing.
PREFIX_FAILED_MARKER = "PRERUN_PROBE_PREFIX_FAILED"


def image_label_command(docker_image: str, label: str) -> str:
    """`docker image inspect` printing one label's value — the command `_image_has_encrypted_volume_label` runs."""
    return (
        "/usr/bin/docker image inspect "
        f"--format '{{{{index .Config.Labels \"{label}\"}}}}' "
        f"{shlex.quote(docker_image)}"
    )


@dataclass(frozen=True)
class PrerunHostProbe:
    """The host listings a rent reads before `docker run`, from one SSH command.

    A ``None`` field means that section's command failed on the host; its consumer runs the
    command itself (and sees the failure the way it does today).
    """

    container_names: tuple[str, ...] | None
    volumes: tuple[tuple[str, str], ...] | None  # (name, driver), as `docker volume ls` prints them
    mounted_volume_names: tuple[str, ...] | None
    gpu_proc_stdout: str | None  # raw `uuid, minor` lines for `_parse_uuid_minor_csv`
    gpu_device_nodes: tuple[str, ...] | None  # /dev/nvidiaN
    shared_nodes: tuple[str, ...] | None  # nodes every rental gets
    shared_nodes_whole_host_only: tuple[str, ...] | None  # nodes only a whole-host rental gets
    power_state_stdout: (
        str | None
    )  # raw nvidia-smi CSV for `_parse_power_state_csv`; None when not asked
    image_label_value: str | None  # the label's value, stripped; None when inspect failed

    @property
    def volume_names(self) -> tuple[str, ...] | None:
        if self.volumes is None:
            return None
        return tuple(name for name, _driver in self.volumes)

    def shared_nodes_for(self, *, is_whole_host_rental: bool) -> tuple[str, ...] | None:
        """The node list `_query_shared_nodes` would have printed for this rental shape."""
        if self.shared_nodes is None:
            return None
        if not is_whole_host_rental:
            return self.shared_nodes
        if self.shared_nodes_whole_host_only is None:
            return None
        return (*self.shared_nodes, *self.shared_nodes_whole_host_only)


def _section(tag: str, command: str) -> str:
    # `t <TAG> <command>`: the command's stdout, one `<TAG>\t<line>` per line (an empty output
    # prints nothing), then `<TAG>_RC\t<exit status>`. stderr is dropped — the consumer that falls
    # back re-runs the command and logs the real stderr.
    return f"t {tag} {shlex.quote(command)}"


def prerun_host_probe_command(*, docker_image: str, image_label: str, with_power: bool) -> str:
    """One `sh` command line printing every section; see the module docstring for the format.

    ``with_power`` adds the nvidia-smi power-state query (the customer-rental path reads it; a
    PEARL filler applies its own cap with live queries and never reads this section).
    """
    parts = [
        # awk, not sed: `\t` in a sed replacement is GNU-only; awk's "\t" is POSIX (and
        # _PROC_GPU_INFO_CMD already needs awk on the host).
        't() { tag=$1; shift; out="$(sh -c "$1" 2>/dev/null)"; rc=$?; '
        'if [ -n "$out" ]; then printf \'%s\\n\' "$out" | awk -v t="$tag" \'{ print t "\\t" $0 }\' '
        f"|| echo {PREFIX_FAILED_MARKER}; fi; "
        f'printf \'%s{_RC_SUFFIX}\\t%s\\n\' "$tag" "$rc"; }}',
        _section(PS_TAG, DOCKER_PS_ALL_NAMES_CMD),
        _section(VOL_TAG, DOCKER_VOLUME_LS_NAME_DRIVER_CMD),
        _section(MNT_TAG, DOCKER_MOUNTED_VOLUME_NAMES_CMD),
        _section(GPUPROC_TAG, _PROC_GPU_INFO_CMD),
        _section(GPUDEV_TAG, _GPU_DEVICE_NODES_CMD),
        _section(SHARED_TAG, shared_device_nodes_command(is_whole_host_rental=False)),
        _section(
            SHAREDW_TAG,
            shared_device_nodes_command(is_whole_host_rental=True, whole_host_only=True),
        ),
    ]
    if with_power:
        parts.append(_section(POWER_TAG, _POWER_STATE_CMD))
    parts.append(_section(LABEL_TAG, image_label_command(docker_image, image_label)))
    return "; ".join(parts)


class PrerunHostProbeParseError(ValueError):
    pass


def _cap(text: str) -> str:
    return text if len(text) <= PROBE_OUTPUT_LOG_CAP else text[:PROBE_OUTPUT_LOG_CAP] + " …"


def parse_prerun_host_probe(stdout: str, *, with_power: bool) -> PrerunHostProbe:
    """Split the probe output into sections; raise on anything that is not the expected shape.

    Every expected tag must close with its ``_RC`` line — a section cut short (the ssh channel
    closed, a host `sh` that died) raises, so a truncated listing can never read as "nothing
    there". A failed section (``_RC`` ≠ 0) is ``None``.
    """
    lines: dict[str, list[str]] = {}
    rcs: dict[str, str] = {}
    for raw in stdout.split("\n"):
        if raw == "":
            continue
        tag, sep, value = raw.partition("\t")
        if not sep:
            raise PrerunHostProbeParseError(
                f"prerun host probe: untagged line {raw[:80]!r} in {_cap(stdout)!r}"
            )
        if tag.endswith(_RC_SUFFIX):
            rcs[tag[: -len(_RC_SUFFIX)]] = value.strip()
        else:
            lines.setdefault(tag, []).append(value)

    expected = (*_ALWAYS_TAGS, POWER_TAG) if with_power else _ALWAYS_TAGS
    for tag in expected:
        if tag not in rcs:
            raise PrerunHostProbeParseError(
                f"prerun host probe: no {tag}{_RC_SUFFIX} in {_cap(stdout)!r}"
            )
        if not rcs[tag].isdigit():
            raise PrerunHostProbeParseError(
                f"prerun host probe: {tag}{_RC_SUFFIX} is {rcs[tag]!r} in {_cap(stdout)!r}"
            )
    unexpected = set(lines) - set(expected)
    if unexpected:
        raise PrerunHostProbeParseError(
            f"prerun host probe: unknown section(s) {sorted(unexpected)} in {_cap(stdout)!r}"
        )

    def ok(tag: str) -> bool:
        return tag in _RC_IGNORED_TAGS or rcs[tag] == "0"

    def stripped_lines(tag: str) -> tuple[str, ...]:
        return tuple(line.strip() for line in lines.get(tag, ()) if line.strip())

    volumes: tuple[tuple[str, str], ...] | None = None
    if ok(VOL_TAG):
        # `{{.Name}} {{.Driver}}` — the per-command path splits on the first space too.
        parsed = []
        for line in lines.get(VOL_TAG, ()):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                parsed.append((parts[0], parts[1]))
        volumes = tuple(parsed)

    label_value: str | None = None
    if ok(LABEL_TAG):
        # The per-command path compares the whole stripped stdout; a label value with newlines
        # arrives as several LABEL lines and is joined back before the strip.
        label_value = "\n".join(lines.get(LABEL_TAG, ())).strip()

    return PrerunHostProbe(
        container_names=stripped_lines(PS_TAG) if ok(PS_TAG) else None,
        volumes=volumes,
        mounted_volume_names=stripped_lines(MNT_TAG) if ok(MNT_TAG) else None,
        gpu_proc_stdout="\n".join(lines.get(GPUPROC_TAG, ())) if ok(GPUPROC_TAG) else None,
        gpu_device_nodes=stripped_lines(GPUDEV_TAG) if ok(GPUDEV_TAG) else None,
        shared_nodes=stripped_lines(SHARED_TAG) if ok(SHARED_TAG) else None,
        shared_nodes_whole_host_only=stripped_lines(SHAREDW_TAG) if ok(SHAREDW_TAG) else None,
        power_state_stdout=(
            "\n".join(lines.get(POWER_TAG, ())) if with_power and ok(POWER_TAG) else None
        ),
        image_label_value=label_value,
    )
