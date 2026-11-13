"""Squash a renter's image into one layer on the node, for Sysbox nodes that cannot start it as is.

Sysbox 0.7.1 on Docker's classic overlay2 store (and 0.6.6 and older on the containerd store) cannot
ID-map the relative layer paths Docker uses once an image has more layers than fit in one page of
overlay mount options (nestybox/sysbox#725, fixed upstream but not in a Sysbox release yet). A
one-layer copy of the same files starts. The disk check measures Docker's data root, so on the
containerd store (which writes images under containerd's root) it runs only when both share one
filesystem. `docker export | docker import`
keeps only the root filesystem, so the image config is put back with `--change`; SHELL (written into a
shell-form health check instead) and ONBUILD (build-time only) are not carried over.
"""

import json
import re
import shlex
import tomllib

import asyncssh

from core.docker_utils import ALPINE_HELPER_IMAGE, df_command, parse_df_available_bytes

FLATTENED_IMAGE_REPO = "lium-flattened"
# headroom left on the Docker disk after the copy, so the pod and the node still have room to write
FLATTEN_DISK_MARGIN_BYTES = 10 * 1024**3
# what a full-node create may write between its sizing and the flatten's disk check (its container, its volumes):
# the sizing keeps it free beside the copy, as the grant's GiB floors alone leave the check almost none
FLATTEN_SIZING_SLACK_BYTES = 10 * 1024**3
# a 24 GB image takes about 5-15 minutes to copy; past this the create fails with the original error
FLATTEN_TIMEOUT_SECONDS = 1800
_DOCKER_READ_TIMEOUT_SECONDS = 120
FLATTEN_LOCK_PATH = "/root/app/lium-flatten.lock"  # held by a flatten from its disk check to its tag
# touched under the node lock by every disk-measuring create as it measures: a flatten that saw no
# other create before this changed refuses, as that create's grant is not on the node yet
FLATTEN_SIZED_PATH = "/root/app/lium-flatten-sized"
# short lock over "mark a copy as in use" and "remove copies not in use", so a removal never races a reuse
FLATTEN_USE_LOCK_PATH = "/root/app/lium-flatten-use.lock"
# One lease file `<copy id>.<rental lease>` per rental between its flatten (or reuse) and the end of its
# create; a leased copy is never removed. The rental releases it when the create ends (success, failure or
# cancel), after which a container using the copy protects it; the expiry only covers a validator crash.
# Beside the validator's other node state, not on /tmp, which a renter's or provider's files can fill
# (the lock and mark files above too).
FLATTEN_LEASE_DIR = "/root/app/flatten-leases"
FLATTEN_LEASE_EXIT = 93
# the lease is written before the copy, so it covers the copy (bounded by the timeout) and the create after it
FLATTEN_LEASE_MINUTES = FLATTEN_TIMEOUT_SECONDS // 60 + 60
# rentals waiting with a new copy per node: more would let one renter hold that many abandoned copies
FLATTEN_MAX_LEASES = 2
FLATTEN_BUSY_EXIT = 92
FLATTEN_NO_DISK_EXIT = 95
FLATTEN_MEASURE_FAILED_EXIT = 90

FLATTEN_NO_DISK_RENTER_SENTENCE = (
    "not enough free disk on this node to prepare this image; choose another node or a smaller image"
)


class ImageFlattenNoDisk(RuntimeError):
    def __init__(self, renter_sentence: str = FLATTEN_NO_DISK_RENTER_SENTENCE, short_bytes: int | None = None):
        super().__init__(renter_sentence)
        self.renter_sentence = renter_sentence
        self.short_bytes = short_bytes


def flatten_no_disk_sentence(short_bytes: int) -> str:
    """The refusal with the measured shortfall; a bigger node grows the pod disk too, so only the image can shrink it."""
    gb = -(-short_bytes // 1024**3)
    return (
        f"not enough free disk on this node to prepare this image beside this pod's disk ({gb} GB short); "
        "use a smaller image or one with fewer layers"
    )


class FlattenMeasureFailed(RuntimeError):
    """Only the disk measure failed, after the node lock was held and the mark written."""


def _dockerfile_word(value: str) -> str:
    # inside double quotes the Dockerfile parser unescapes only \, " and $ (no variable expansion)
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'


def _healthcheck_change(check: dict, shell: list[str]) -> str:
    test = check.get("Test") or []
    if test[:1] == ["NONE"]:
        return "HEALTHCHECK NONE"
    options = "".join(
        f"--{flag}={check[key]}ns "  # inspect gives nanoseconds; 0 means Docker's default
        for flag, key in (
            ("interval", "Interval"),
            ("timeout", "Timeout"),
            ("start-period", "StartPeriod"),
            ("start-interval", "StartInterval"),
        )
        if check.get(key)
    )
    if check.get("Retries"):
        options += f"--retries={check['Retries']} "
    if test[:1] == ["CMD-SHELL"] and len(test) == 2:
        # Docker runs CMD-SHELL through the image's SHELL, which import cannot restore: spell the shell out
        return f"HEALTHCHECK {options}CMD {json.dumps([*shell, test[1]])}"
    if test[:1] == ["CMD"] and len(test) > 1:
        return f"HEALTHCHECK {options}CMD {json.dumps(test[1:])}"
    # one --change is one Dockerfile line; a health check that differs would change how the pod is judged
    raise ValueError(f"health check {test[:1]} cannot be carried over")


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
    if config.get("StopSignal"):
        changes.append(f"STOPSIGNAL {config['StopSignal']}")
    if config.get("Healthcheck"):
        changes.append(_healthcheck_change(config["Healthcheck"], config.get("Shell") or ["/bin/sh", "-c"]))
    for key, value in sorted((config.get("Labels") or {}).items()):
        if "\n" in key + value or "\r" in key + value:
            continue  # metadata only (e.g. a long description); nothing at run time reads it
        changes.append(f"LABEL {_dockerfile_word(key)}={_dockerfile_word(value)}")
    return changes


def _lease_path(tag: str, lease: str) -> str:
    return f"{FLATTEN_LEASE_DIR}/{shlex.quote(tag.partition(':')[2] + '.' + lease)}"


def _lease_command(tag: str, lease: str) -> str:
    """Mark `tag` as wanted by the rental `lease` until its create ends."""
    return f"mkdir -p {FLATTEN_LEASE_DIR} && touch {_lease_path(tag, lease)}"


def flatten_lease_release_command(lease: str) -> str:
    """Drop the rental's lease: from here a container using the copy, or nothing, keeps it."""
    return f"rm -f {FLATTEN_LEASE_DIR}/*.{shlex.quote(lease)}"


def _reuse_command(tag: str, lease: str) -> str:
    """Exit 0 and lease the copy when it exists; under the use lock so a cleanup cannot remove it in between.
    A copy whose lease cannot be written is not handed out: nothing would stop a cleanup removing it."""
    return (
        f"exec 8>>{FLATTEN_USE_LOCK_PATH}; flock -w {_DOCKER_READ_TIMEOUT_SECONDS} 8 || exit 96; "
        f"/usr/bin/docker image inspect {shlex.quote(tag)} >/dev/null 2>&1 && {_lease_command(tag, lease)}"
    )


def _flatten_command(
    image_id: str,
    tag: str,
    changes: list[str],
    volumes: list[str],
    need_bytes: int,
    lease: str,
    sized_mark: str = "",
    export_factor: int = 0,
    fixed_bytes: int = 0,
) -> str:
    """Under a node-wide lock: reuse a copy another flatten finished meanwhile, refuse a new copy while
    FLATTEN_MAX_LEASES other rentals still wait on theirs, remove every earlier copy
    no container uses and no rental leased in the last FLATTEN_LEASE_MINUTES (so renters cannot fill the
    disk with copies, and a rental never loses its copy before its create), check the Docker disk
    holds `need_bytes` and no create sized its disk since `sized_mark` (flatten_sized_mark_command), create (never start) a container from the image ID, pipe its export into an
    import, tag the result only when both sides succeeded, and always remove the container and its
    volumes. With `export_factor`, a counting export first refuses when the disk cannot also hold
    `export_factor` times the bytes the export really produces plus `fixed_bytes`: Size counts allocated
    blocks, so a sparse file can report far less than the export and the import write. The count and the copy
    share one FLATTEN_TIMEOUT_SECONDS deadline, the bound the lock waiters allow for. Plain sh: no pipefail."""
    t = FLATTEN_TIMEOUT_SECONDS
    change_args = " ".join(f"--change {shlex.quote(c)}" for c in changes)
    # an empty volume per VOLUME path: create would otherwise copy the image's files there (extra disk)
    mounts = " ".join(f"--mount {shlex.quote(f'type=volume,dst={v},volume-nocopy')}" for v in volumes)
    root = "\"$(/usr/bin/docker info --format '{{.DockerRootDir}}')\""
    return (
        # one flatten per node: concurrent rentals would each pass the disk check and copy at once
        f"exec 9>>{FLATTEN_LOCK_PATH}; flock -w {t} 9 || exit 96; "
        f"({_reuse_command(tag, lease)}) && exit 0; "
        f'[ "$(stat -c %y {FLATTEN_SIZED_PATH} 2>/dev/null)" = {shlex.quote(sized_mark)} ] || '
        f"{{ echo 'flatten refused: another create sized its disk meanwhile' >&2; exit {FLATTEN_NO_DISK_EXIT}; }}; "
        f"exec 8>>{FLATTEN_USE_LOCK_PATH}; flock -w {_DOCKER_READ_TIMEOUT_SECONDS} 8 || exit 96; "
        # unknown lease count (unreadable dir) is treated as full: never more waiting copies than the cap
        f"mkdir -p {FLATTEN_LEASE_DIR}; "
        f"held=$(find {FLATTEN_LEASE_DIR} -maxdepth 1 -type f ! -mmin +{FLATTEN_LEASE_MINUTES}) && "
        f'[ "$(printf %s "$held" | grep -c .)" -lt {FLATTEN_MAX_LEASES} ] || '
        f"{{ echo 'flatten refused: other rentals on this node still wait for their copies' >&2; exit {FLATTEN_BUSY_EXIT}; }}; "
        # leased before the copy: a disk that fills during the copy cannot leave the copy unleased
        f"{_lease_command(tag, lease)} || {{ echo 'flatten failed: cannot write the lease' >&2; exit {FLATTEN_LEASE_EXIT}; }}; "
        # any doubt about a lease keeps the copy: an unlistable lease dir skips the cleanup, a lease find
        # cannot read keeps its copy; rmi without -f refuses a copy a container (running or stopped) uses
        f"if ls {FLATTEN_LEASE_DIR} >/dev/null 2>&1 && [ -x {FLATTEN_LEASE_DIR} ]; then "
        f"for old in $(/usr/bin/docker image ls --format '{{{{.Repository}}}}:{{{{.Tag}}}}' {FLATTENED_IMAGE_REPO}); "
        f'do live=$(find {FLATTEN_LEASE_DIR} -maxdepth 1 -name "${{old#*:}}.*" ! -mmin +{FLATTEN_LEASE_MINUTES}) '
        '&& [ -z "$live" ] && /usr/bin/docker rmi "$old" >/dev/null 2>&1; done; '
        "else echo 'flatten: lease dir unreadable, cleanup skipped' >&2; fi; flock -u 8; "
        f"avail=$({df_command(root)} | awk 'NR==2{{print $4}}'); "
        'case "$avail" in ""|*[!0-9]*) echo "flatten failed: unexpected df output" >&2; exit 94;; esac; '
        f'[ "$avail" -ge {int(need_bytes)} ] || {{ echo "flatten avail=$avail" >&2; exit {FLATTEN_NO_DISK_EXIT}; }}; '
        # the ID, not the tag: a pull may move the tag after inspect and the copy would not match its name
        f"cid=$(/usr/bin/docker create --entrypoint /lium-flatten-noop {mounts} {shlex.quote(image_id)}) || exit 97; "
        f"end=$(( $(date +%s) + {t} )); "
        + (_export_count_check(export_factor, fixed_bytes) if export_factor else "")
        # what the count left of the one deadline (timeout 0 would mean no limit)
        + 'left=$(( end - $(date +%s) )); [ "$left" -gt 0 ] || { /usr/bin/docker rm -fv "$cid" >/dev/null 2>&1; '
        'echo "flatten failed: out of time after counting" >&2; exit 98; }; st=$(mktemp); '
        'iid=$( (timeout "$left" /usr/bin/docker export "$cid"; echo $? > "$st") '
        f'| timeout "$left" /usr/bin/docker import {change_args} -); irc=$?; '
        'erc=$(cat "$st"); rm -f "$st"; /usr/bin/docker rm -fv "$cid" >/dev/null 2>&1; rrc=$?; '
        # a left-over container or volume fails the flatten untagged, so the next rental retries the cleanup
        f'if [ "$irc" = 0 ] && [ "$erc" = 0 ] && [ "$rrc" = 0 ]; then {_lease_command(tag, lease)} 2>/dev/null; /usr/bin/docker tag "$iid" {shlex.quote(tag)}; '
        'else [ -n "$iid" ] && /usr/bin/docker rmi -f "$iid" >/dev/null 2>&1; '
        'echo "flatten failed: export=$erc import=$irc rm=$rrc" >&2; exit 98; fi'
    )


def _export_count_check(factor: int, fixed_bytes: int) -> str:
    """Count the export, stopping one byte past what the free disk allows; refuse (container removed) past it.
    The count stops early, so it never knows the real shortfall: the refusal names none."""
    t = FLATTEN_TIMEOUT_SECONDS
    gone = '/usr/bin/docker rm -fv "$cid" >/dev/null 2>&1'
    return (
        f'cap=$(( (avail - {int(fixed_bytes)}) / {int(factor)} )); st=$(mktemp); '
        f'eb=$( (timeout {t} /usr/bin/docker export "$cid"; echo $? > "$st") | head -c $((cap + 1)) | wc -c); '
        'erc=$(cat "$st"); rm -f "$st"; '
        # a wc output test cannot read refuses too (an arithmetic error would end sh)
        f'[ "$eb" -le "$cap" ] 2>/dev/null || {{ {gone}; '
        f'echo "flatten refused: the export is larger than the free disk allows" >&2; exit {FLATTEN_NO_DISK_EXIT}; }}; '
        f'[ "$erc" = 0 ] || {{ {gone}; echo "flatten failed: counting export=$erc" >&2; exit 98; }}; '
    )


# under the Docker root dir: where a cache reclaim moves volume files for its detached delete (DAH-2498)
CACHE_TRASH_DIR = "lium-cache-trash"


def flatten_sized_mark_command() -> str:
    """Prints the mtime of FLATTEN_SIZED_PATH (empty when absent); `_flatten_command` refuses once it changed."""
    return f"stat -c %y {FLATTEN_SIZED_PATH} 2>/dev/null; true"


def flatten_wait_command(with_trash: bool = False) -> str:
    root = "\"$(/usr/bin/docker info --format '{{.DockerRootDir}}')\""
    trash = f'"/proc/1/root"{root}/{CACHE_TRASH_DIR}'
    # the cache files this create's detached `rm` is still deleting (DAH-2498) are free once it ends. They are read
    # under the trash lock that `rm` takes per file, so df and du see the same deletes; a lock not had in time
    # keeps df first: a byte freed in between is then missed, never counted twice
    trash_lock = f': >>{trash}.lock 2>/dev/null && exec 7>>{trash}.lock && flock -w 60 7; ' if with_trash else ""
    # under the lock a copy holds for its whole run: no copy is half-written while df measures, and a
    # flatten that has not started copying refuses once it sees the mark (nothing to print without flock)
    command = (
        f"command -v flock >/dev/null || exit 0; exec 9>>{FLATTEN_LOCK_PATH}; "
        f"flock -w {FLATTEN_TIMEOUT_SECONDS + 300} 9 || exit 1; touch {FLATTEN_SIZED_PATH} || exit 1; {trash_lock}"
        f"{df_command(root)} || exit {FLATTEN_MEASURE_FAILED_EXIT}"
    )
    if with_trash:
        command += f'; echo "trash $(du -sxB1 {trash} 2>/dev/null | cut -f1)"'
    return command


async def wait_for_node_flatten(ssh_client: asyncssh.SSHClientConnection, with_trash: bool = False) -> int | None:
    """The Docker disk's free bytes, measured under the node flatten lock, for a create's sizing.

    A copy holds that lock from its disk check to its tag, so the measure waits for a running copy
    (whose pod may have been deleted meanwhile) and counts every copy written so far; it also marks the
    node as sized under that lock, so a flatten that has not started copying refuses (`sized_mark`).
    A disk probe taken before this, outside the lock, may predate a copy and must not size the create.
    None on a node without flock, which runs no flatten. Raises FlattenMeasureFailed when only the
    measure failed (lock held, mark written); any other error when the lock or the mark was not taken,
    which must stop the create: no other sizing is safe then. `with_trash` (the create's cache reclaim
    detached a delete) adds what that delete is still freeing."""
    result = await ssh_client.run(flatten_wait_command(with_trash), check=False, timeout=FLATTEN_TIMEOUT_SECONDS + 420)
    if result.exit_status == FLATTEN_MEASURE_FAILED_EXIT:
        raise FlattenMeasureFailed("measuring the disk under the node flatten lock failed")
    if result.exit_status != 0:
        raise RuntimeError(f"waiting for any image copy on this node failed (exit {result.exit_status})")
    out = str(result.stdout or "")
    if not out.strip():
        return None
    try:
        free = parse_df_available_bytes(out)
    except Exception as exc:
        raise FlattenMeasureFailed(str(exc)) from exc
    trash = next((line.split()[1:] for line in out.splitlines() if line.startswith("trash")), [])
    return free + (int(trash[0]) if trash and trash[0].isdigit() else 0)


# Docker switches to the relative layer paths Sysbox cannot ID-map at ~40 layers on the default data root
# and sooner on a longer one. shortcut: a fixed count; an image under it that still fails meets the flatten's own disk check
FLATTEN_BUDGET_MIN_LAYERS = 30


def _copy_bytes(size: int, driver: str) -> int:
    """The disk a new one-layer copy of an image of `size` writes on the storage `driver`, before the margin. On the
    containerd store (any driver but overlay2) the import stores a compressed blob and unpacks a snapshot, so twice
    Size: a snapshots-only image (compressed blobs gone) passes the store checks with Size ~ its unpacked data."""
    return size if driver == "overlay2" else 2 * size


async def flatten_copy_bytes(ssh_client: asyncssh.SSHClientConnection, image: str) -> int | None:
    """The disk a one-layer copy of the local `image` would take (`_copy_bytes` plus FLATTEN_DISK_MARGIN_BYTES), for
    the sizing to keep free; None when the image has too few layers to need one, its copy is already on the
    node (a flatten reuses it, writing nothing), or it cannot be read."""
    try:
        result = await ssh_client.run(
            "/usr/bin/docker image inspect --format '{{.Size}} {{len .RootFS.Layers}} {{.Id}}' "
            f"{shlex.quote(image)} && /usr/bin/docker info --format '{{{{.Driver}}}}'",
            check=True,
            timeout=_DOCKER_READ_TIMEOUT_SECONDS,
        )
        size_word, layers_word, image_id, driver = str(result.stdout).split()
        size, layers = int(size_word), int(layers_word)
    except Exception:
        return None
    if layers < FLATTEN_BUDGET_MIN_LAYERS:
        return None
    tag = f"{FLATTENED_IMAGE_REPO}:{image_id.removeprefix('sha256:')}"
    try:
        copy = await ssh_client.run(
            f"/usr/bin/docker image inspect {shlex.quote(tag)} >/dev/null 2>&1",
            check=False,
            timeout=_DOCKER_READ_TIMEOUT_SECONDS,
        )
    except Exception:
        copy = None  # unknown: keep the room for a new copy
    # shortcut: the copy may be removed before the flatten reuses it; the flatten's own disk check then refuses
    return None if copy is not None and copy.exit_status == 0 else _copy_bytes(size, driver) + FLATTEN_DISK_MARGIN_BYTES


_CONTAINERD_DEFAULT_ROOT = "/var/lib/containerd"
_CONTAINERD_NO_CONFIG_EXIT = 3


def containerd_config_command() -> str:
    """Prints /etc/containerd/config.toml on Docker's containerd image store; exits
    _CONTAINERD_NO_CONFIG_EXIT when there is none, any other non-zero when not on that store or unreadable."""
    # a symlink here would resolve inside the helper, not on the node, so it is refused rather than followed
    read = (
        "[ -L /e/containerd ] || [ -L /e/containerd/config.toml ] && exit 1; "
        f"[ -e /e/containerd/config.toml ] || exit {_CONTAINERD_NO_CONFIG_EXIT}; cat /e/containerd/config.toml"
    )
    return (
        "/usr/bin/docker info --format '{{json .DriverStatus}}' | grep -q io.containerd.snapshotter || exit 1; "
        f"/usr/bin/docker run --rm --mount type=bind,src=/etc,dst=/e,readonly {ALPINE_HELPER_IMAGE} "
        f"sh -c {shlex.quote(read)}"
    )


def containerd_root(config: str | None) -> str | None:
    """containerd's root from its config text (None: no config file, so the default root), or None when it
    cannot be told for sure (unparsable, `imports` that may override it, not an absolute path)."""
    if config is None:
        return _CONTAINERD_DEFAULT_ROOT
    try:
        doc = tomllib.loads(config)
    except tomllib.TOMLDecodeError:
        return None
    # shortcut: an imported file may set root too; refuse rather than read the imports
    if doc.get("imports"):
        return None
    # a plugin's own root_path (e.g. the overlayfs snapshotter's) puts its data on a disk this check never reads
    if any(isinstance(p, dict) and p.get("root_path") for p in (doc.get("plugins") or {}).values()):
        return None
    # shortcut: a proxy snapshotter (e.g. stargz) keeps its data under its own process's root; refuse until read
    if doc.get("proxy_plugins"):
        return None
    root = doc.get("root", _CONTAINERD_DEFAULT_ROOT)
    # a comma would split the --mount option the same-disk check binds it with
    if not isinstance(root, str) or not root.startswith("/") or "," in root:
        return None
    return root


def containerd_daemons_command() -> str:
    """Prints the arguments of every process named containerd on the node, one process per line."""
    each = 'for p in /proc/[0-9]*; do [ "$(cat "$p/comm" 2>/dev/null)" = containerd ] || continue; ' \
        'tr "\\0" " " < "$p/cmdline" && echo; done'
    return f"/usr/bin/docker run --rm --pid=host {ALPINE_HELPER_IMAGE} sh -c {shlex.quote(each)}"


_CONTAINERD_DEFAULT_ADDRESS = "/run/containerd/containerd.sock"
# where containerd.io and older Docker bundles install ctr
_CTR_PATHS = ("/usr/bin/ctr", "/usr/local/bin/ctr", "/usr/bin/docker-containerd-ctr")
# the executor container has neither ctr nor the containerd socket; it runs privileged with pid: host, so PID 1's
# mount namespace reaches the host's
HOST_NSENTER = "nsenter -t 1 -m -- "


def containerd_address(config: str | None) -> str | None:
    """containerd's socket from its config text (None: no config file, so the default), or None when unreadable."""
    if config is None:
        return _CONTAINERD_DEFAULT_ADDRESS
    try:
        grpc = tomllib.loads(config).get("grpc") or {}
    except tomllib.TOMLDecodeError:
        return None
    address = grpc.get("address") or _CONTAINERD_DEFAULT_ADDRESS if isinstance(grpc, dict) else None
    return address if isinstance(address, str) and address.startswith("/") else None


def containerd_plugins_command(address: str) -> str:
    """Lists the running daemon's plugins and the roots they export, with the host's ctr on the host's socket
    (exit 127 when the host has no ctr): containerd reads its config only at startup."""
    script = (
        f"for c in {' '.join(_CTR_PATHS)}; do "
        f'[ -x "$c" ] && exec "$c" --address {shlex.quote(address)} plugins ls -d; done; exit 127'
    )
    return f"{HOST_NSENTER}sh -c {shlex.quote(script)}"


def containerd_running_roots(listing: str) -> dict[str, str]:
    """The `root` each plugin of the running containerd exports, keyed `<type>.<id>` (its directory name), from
    `ctr plugins ls -d`."""
    roots: dict[str, str] = {}
    kind = ident = ""
    for words in (line.split() for line in listing.splitlines()):
        if words[:1] == ["Type:"]:
            kind = words[1] if len(words) == 2 else ""
        elif words[:1] == ["ID:"]:
            ident = words[1] if len(words) == 2 else ""
        elif len(words) == 2 and words[0] == "root" and kind and ident:
            roots[f"{kind}.{ident}"] = words[1]
    return roots


def containerd_runs_on_root(listing: str, root: str) -> bool:
    """The running overlayfs snapshotter and content store export exactly their directories under `root`."""
    roots = containerd_running_roots(listing)
    return all(roots.get(d) == f"{root.rstrip('/')}/{d}" for d in _CONTAINERD_DATA_DIRS if "/" not in d)


_CONTAINERD_CONFIG_ARGS = ([], ["--config", "/etc/containerd/config.toml"], ["-c", "/etc/containerd/config.toml"],
                           ["--config=/etc/containerd/config.toml"])


def containerd_daemon_reads_etc_config(daemons: str) -> bool:
    """The node runs exactly one containerd and it takes its root from /etc/containerd/config.toml: no
    --root, no other --config. A second containerd (e.g. one dockerd started itself) or any other flag
    leaves the daemon serving Docker or its root unknown."""
    lines = [line.split() for line in daemons.splitlines() if line.strip()]
    return len(lines) == 1 and lines[0][1:] in _CONTAINERD_CONFIG_ARGS


# where containerd writes an image's data under its root: the overlayfs snapshots and the content store, and the
# children it writes into (snapshots/, ingest/, blobs/sha256/). Any can be its own mount (or a symlink to one),
# so each is checked, not just the root.
_CONTAINERD_DATA_DIRS = (
    "io.containerd.snapshotter.v1.overlayfs",
    "io.containerd.snapshotter.v1.overlayfs/snapshots",
    "io.containerd.content.v1.content",
    "io.containerd.content.v1.content/ingest",
    "io.containerd.content.v1.content/blobs/sha256",
)


def same_filesystem_as_data_root_command(path: str) -> str:
    """Exit 0 only when `path` and its snapshot and content directories (and the children written into) are on the filesystem of Docker's data
    root, the disk every flatten check measures. Each is its own bind source, which Docker resolves on the node,
    so a symlink counts where it points; a missing path fails the mount and so the check."""
    dirs = [path, *(f"{path.rstrip('/')}/{sub}" for sub in _CONTAINERD_DATA_DIRS)]
    mounts = " ".join(f"--mount {shlex.quote(f'type=bind,src={d},dst=/c{i},readonly')}" for i, d in enumerate(dirs))
    same_fs = " && ".join(f'[ "$(stat -c %d /d/.)" = "$(stat -c %d /c{i}/.)" ]' for i in range(len(dirs)))
    return (
        "root=$(/usr/bin/docker info --format '{{.DockerRootDir}}') || exit 1; "
        f'/usr/bin/docker run --rm --mount "type=bind,src=$root,dst=/d,readonly" {mounts} '
        f"{ALPINE_HELPER_IMAGE} sh -c {shlex.quote(same_fs)}"
    )


# Before Docker 29.8.0 `image inspect` Size on the containerd store leaves out the unpacked snapshots, so
# the copy's disk budget would be too small (moby/moby, Docker 29.8.0 release notes)
_CONTAINERD_SIZE_MIN_DOCKER = (29, 8, 0)


def docker_counts_unpacked_size(server_version: str) -> bool:
    """Docker's server version is one whose containerd-store image Size includes the unpacked snapshots."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", server_version.strip())
    return bool(match) and tuple(map(int, match.groups())) >= _CONTAINERD_SIZE_MIN_DOCKER


async def containerd_store_refusal(ssh_client: asyncssh.SSHClientConnection) -> str | None:
    """None on Docker's containerd image store with containerd's root on the filesystem of Docker's data root, on a
    Docker whose image Size counts the unpacked snapshots; else which of those could not be confirmed. Refused
    whenever either cannot be determined, so a flatten never measures the wrong disk or budgets too little of it."""
    version = await ssh_client.run(
        "/usr/bin/docker version --format '{{.Server.Version}}'", check=False, timeout=_DOCKER_READ_TIMEOUT_SECONDS
    )
    if version.exit_status != 0 or not docker_counts_unpacked_size(str(version.stdout)):
        return f"docker version {str(version.stdout).strip()!r} does not count unpacked snapshots"
    daemons = await ssh_client.run(containerd_daemons_command(), check=False, timeout=_DOCKER_READ_TIMEOUT_SECONDS)
    if daemons.exit_status != 0 or not containerd_daemon_reads_etc_config(str(daemons.stdout)):
        return "not exactly one containerd reading /etc/containerd/config.toml"
    read = await ssh_client.run(containerd_config_command(), check=False, timeout=_DOCKER_READ_TIMEOUT_SECONDS)
    if read.exit_status not in (0, _CONTAINERD_NO_CONFIG_EXIT):
        return f"containerd config unreadable (exit {read.exit_status})"
    config = str(read.stdout) if read.exit_status == 0 else None
    root, address = containerd_root(config), containerd_address(config)
    if root is None or address is None:
        return "containerd root or socket not determinable from its config"
    # a config rewritten since the daemon started names a root it does not use
    plugins = await ssh_client.run(
        containerd_plugins_command(address), check=False, timeout=_DOCKER_READ_TIMEOUT_SECONDS
    )
    if plugins.exit_status != 0 or not containerd_runs_on_root(str(plugins.stdout), root):
        return f"running containerd plugins not under {root} (exit {plugins.exit_status})"
    same = await ssh_client.run(
        same_filesystem_as_data_root_command(root), check=False, timeout=_DOCKER_READ_TIMEOUT_SECONDS
    )
    return None if same.exit_status == 0 else f"containerd root {root} is not on Docker's data-root filesystem"


async def flatten_image_on_host(
    ssh_client: asyncssh.SSHClientConnection,
    image: str,
    *,
    lease: str,
    reserved_bytes: int = 0,
    sized_mark: str = "",
) -> str:
    """The local tag of a one-layer copy of `image`, built on the node unless one exists already.

    The copy is leased to `lease` (hex, unique per rental): the caller must run
    `flatten_lease_release_command(lease)` on the node once its create ended, however it ended.

    Keyed by the image ID, so a retry or a later rental of the same image reuses the copy. Raises
    ImageFlattenNoDisk when the Docker disk cannot hold the copy and still keep `reserved_bytes`
    (granted to the pod, not yet written) free, or a create sized its disk since `sized_mark` was read
    (flatten_sized_mark_command, before the caller checked for other creates); any other error when it failed or the node does not
    use the classic overlay2 store.
    """
    driver = await ssh_client.run(
        "/usr/bin/docker info --format '{{.Driver}}'", check=True, timeout=_DOCKER_READ_TIMEOUT_SECONDS
    )
    # the containerd store reports its snapshotter (e.g. overlayfs) and keeps images outside DockerRootDir
    if str(driver.stdout).strip() != "overlay2":
        # only the overlayfs snapshotter's directory is checked; any other snapshotter writes where nothing measures
        refusal = (
            await containerd_store_refusal(ssh_client)
            if str(driver.stdout).strip() == "overlayfs"
            else "not the overlayfs snapshotter"
        )
        if refusal:
            raise RuntimeError(f"not flattening on storage driver {str(driver.stdout).strip()!r}: {refusal}")
    inspect = await ssh_client.run(
        f"/usr/bin/docker image inspect --format '{{{{json .}}}}' {shlex.quote(image)}",
        check=True,
        timeout=_DOCKER_READ_TIMEOUT_SECONDS,
    )
    info = json.loads(inspect.stdout)
    tag = f"{FLATTENED_IMAGE_REPO}:{info['Id'].removeprefix('sha256:')}"
    existing = await ssh_client.run(
        _reuse_command(tag, lease),
        check=False,
        timeout=_DOCKER_READ_TIMEOUT_SECONDS,
    )
    if existing.exit_status == 0:
        return tag
    config = info.get("Config") or {}
    driver_name = str(driver.stdout).strip()
    fixed = FLATTEN_DISK_MARGIN_BYTES + reserved_bytes
    need_bytes = _copy_bytes(int(info.get("Size") or 0), driver_name) + fixed
    command = _flatten_command(
        info["Id"],
        tag,
        image_flatten_changes(config),
        sorted(config.get("Volumes") or ()),
        need_bytes,
        lease,
        sized_mark,
        # overlay2's Size sums apparent file sizes (sparse files at full length), so only containerd needs the count
        export_factor=0 if driver_name == "overlay2" else 2,
        fixed_bytes=fixed,
    )
    # waits up to one flatten for the node lock, then runs its own
    result = await ssh_client.run(command, check=False, timeout=FLATTEN_TIMEOUT_SECONDS * 3 + 120)
    if result.exit_status == FLATTEN_NO_DISK_EXIT:
        # only the Size check prints avail, against the need computed here; the containerd count names none
        found = re.search(r"flatten avail=(\d+)", str(result.stderr or ""))
        if found and int(found.group(1)) < need_bytes:
            short = need_bytes - int(found.group(1))
            raise ImageFlattenNoDisk(flatten_no_disk_sentence(short), short_bytes=short)
        raise ImageFlattenNoDisk(FLATTEN_NO_DISK_RENTER_SENTENCE)
    if result.exit_status != 0:
        raise RuntimeError(f"flatten exited {result.exit_status}: {str(result.stderr or '')[-2000:]}")
    return tag
