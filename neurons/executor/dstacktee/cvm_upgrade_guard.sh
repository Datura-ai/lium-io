#!/bin/bash

# CVM upgrade guard for the SGX sealing-key provider (DAH-3188 prerequisite).
#
# The key provider's enclave measurement (MRENCLAVE) seals the key every CVM on
# this host derives its encrypted data-disk key from. A rebuilt image has a new
# MRENCLAVE, so every existing CVM disk on the host becomes unreadable at its
# next boot. This script makes that rebuild impossible while any CVM disk exists:
#
#   inventory   list every CVM disk on the host (all checkouts, all VM
#               directories, stopped CVMs and disks whose manifest is gone
#               included) and say whether an upgrade is allowed
#   start       start the key provider with the exact pinned image; never builds
#               while a CVM disk exists; builds only on a host with no disks
#   upgrade     rebuild the key provider; refused while any CVM disk exists or
#               while the inventory is incomplete; keeps the previous image
#   lock        run a command while holding the host lock
#
# lium-cvm.sh sources this file as a library (new/run take the same lock, start
# the provider through cvm_guard_start and register their VM directory), so an
# upgrade and a CVM creation or start never interleave.
#
# key-provider/docker-compose.yaml only names the images; the `build:` sections
# live in key-provider/docker-compose.build.yaml, which only this guard passes
# to compose (cvm_guard_compose_build). A hand-run `docker compose build` or
# `docker compose up` in key-provider/ therefore builds nothing. The preflight
# refuses to run at all when docker-compose.yaml declares a `build:` again or
# a file compose auto-loads (an override file, compose.yaml) sits beside it.
#
# Host state (override for tests with the LIUM_CVM_* variables):
#   /var/lock/lium-cvm.lock                the host lock (flock)
#   /var/lib/lium-cvm/vm-dirs              every VM directory root ever used
#   /var/lib/lium-cvm/key-provider.image   the pinned image id
#
# Exit codes: 0 allowed/done · 1 usage or tool error (docker unreachable, flock
#             missing, a failed build, a `build:` in docker-compose.yaml) · 3
#             refused, CVM disks exist · 4 refused,
#             inventory incomplete · 5 lock busy · 6 pinned image missing or not
#             the one running

CVM_GUARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CVM_GUARD_KP_DIR="${LIUM_CVM_KEY_PROVIDER_DIR:-$CVM_GUARD_DIR/key-provider}"
CVM_GUARD_STATE_DIR="${LIUM_CVM_STATE_DIR:-/var/lib/lium-cvm}"
CVM_GUARD_LOCK_FILE="${LIUM_CVM_LOCK_FILE:-/var/lock/lium-cvm.lock}"
CVM_GUARD_LOCK_WAIT="${LIUM_CVM_LOCK_WAIT:-60}"
# Directories swept for VM directories that no checkout registered. Filesystems
# mounted below them are swept too (the mount table is read for that). The
# sweep runs on `upgrade` and on a `start` that has no pinned image; a `start`
# with a pin never builds, so it skips the walk.
CVM_GUARD_SWEEP_ROOTS="${LIUM_CVM_SWEEP_ROOTS:-/home /root /opt /srv /mnt /data}"
CVM_GUARD_MOUNTS="${LIUM_CVM_MOUNTS_FILE:-/proc/self/mounts}"
# Mount types that never hold a CVM disk.
CVM_GUARD_PSEUDO_FS="overlay proc sysfs cgroup cgroup2 devpts devtmpfs mqueue debugfs tracefs securityfs pstore bpf autofs fusectl configfs hugetlbfs binfmt_misc nsfs rpc_pipefs fuse.lxcfs squashfs"
CVM_GUARD_REGISTRY="$CVM_GUARD_STATE_DIR/vm-dirs"
CVM_GUARD_PIN_FILE="$CVM_GUARD_STATE_DIR/key-provider.image"
CVM_GUARD_IMAGE="lium-key-provider:local"
CVM_GUARD_AESMD_IMAGE="lium-aesmd:local"
CVM_GUARD_CONTAINER="dstack-key-provider"
CVM_GUARD_AESMD_CONTAINER="dstack-aesmd"
CVM_GUARD_SERVICE="gramine-sealing-key-provider"
CVM_GUARD_COMPOSE_FILE="docker-compose.yaml"
CVM_GUARD_BUILD_FILE="docker-compose.build.yaml"
# Image names docker compose gave the builds before this guard named them.
CVM_GUARD_LEGACY_IMAGES="key-provider-gramine-sealing-key-provider key-provider_gramine-sealing-key-provider"
CVM_GUARD_LEGACY_AESMD_IMAGES="key-provider-aesmd key-provider_aesmd"
CVM_GUARD_LOCK_FD=9
CVM_GUARD_LOCKED=0

# Filled by cvm_guard_inventory: one "state<TAB>path" line per CVM disk,
# and the roots the sweep could not read.
CVM_GUARD_DISKS=()
CVM_GUARD_UNREADABLE=()
CVM_GUARD_NO_DISK_YET=()

cvm_guard_say() { echo "[cvm-guard] $*"; }
cvm_guard_err() { echo "[cvm-guard] ERROR: $*" >&2; }

# Runtime compose: docker-compose.yaml alone, named with -f so an override
# file or COMPOSE_FILE in the environment never reaches the guard's own calls.
cvm_guard_compose() {
    (cd "$CVM_GUARD_KP_DIR" && docker compose -f "$CVM_GUARD_COMPOSE_FILE" "$@")
}

# The only build path. The build file is passed explicitly, so nothing outside
# this function can rebuild through compose.
cvm_guard_compose_build() {
    (cd "$CVM_GUARD_KP_DIR" && docker compose -f "$CVM_GUARD_COMPOSE_FILE" -f "$CVM_GUARD_BUILD_FILE" build "$@")
}

# Files compose loads on its own when run in key-provider/ without -f (its
# default names, docker-compose.yml among them and preferred over .yaml, and
# the override names). A `build:` in any of them lets a hand-run
# `docker compose build` rebuild the enclave, so none may exist.
CVM_GUARD_AUTOLOAD_FILES="docker-compose.yml docker-compose.override.yaml docker-compose.override.yml compose.yaml compose.yml compose.override.yaml compose.override.yml"

# docker-compose.yaml must not declare a build:, and no auto-loaded compose
# file may sit beside it: either lets `docker compose build` or `up` by hand
# rebuild the key provider past this guard.
cvm_guard_check_compose_file() {
    local file="$CVM_GUARD_KP_DIR/$CVM_GUARD_COMPOSE_FILE" extra
    if [ ! -f "$file" ]; then
        cvm_guard_err "$file is missing"
        return 1
    fi
    if grep -qE '^[[:space:]]+build:' "$file"; then
        cvm_guard_err "$file declares a 'build:' section. With it, a hand-run 'docker compose build' or 'docker compose up' rebuilds the key provider past this guard."
        cvm_guard_err "Builds belong in $CVM_GUARD_KP_DIR/$CVM_GUARD_BUILD_FILE only; remove the 'build:' section and run this command again."
        return 1
    fi
    # shellcheck disable=SC2086 # space-separated list on purpose
    for extra in $CVM_GUARD_AUTOLOAD_FILES; do
        [ -e "$CVM_GUARD_KP_DIR/$extra" ] || continue
        cvm_guard_err "$CVM_GUARD_KP_DIR/$extra exists. docker compose loads that file on its own, so a 'build:' in it rebuilds the key provider past this guard."
        cvm_guard_err "Remove it (the build definition is $CVM_GUARD_BUILD_FILE, which only this guard passes) and run this command again."
        return 1
    done
}

# The tools the guard needs, before any verdict that could be misread: a docker
# daemon that is down would otherwise look like a lost image, and a missing
# flock like a busy lock.
cvm_guard_preflight() {
    if ! command -v flock >/dev/null 2>&1; then
        cvm_guard_err "flock (util-linux) is not installed"
        return 1
    fi
    cvm_guard_check_compose_file || return 1
    if ! docker info >/dev/null 2>&1; then
        cvm_guard_err "docker is not reachable (daemon down, or run with sudo)"
        return 1
    fi
}

# --- lock -------------------------------------------------------------------

# Take the host lock. Idempotent inside one process. Exit 5 when another
# holder keeps it past LIUM_CVM_LOCK_WAIT seconds.
cvm_guard_lock() {
    [ "$CVM_GUARD_LOCKED" = 1 ] && return 0
    if ! command -v flock >/dev/null 2>&1; then
        cvm_guard_err "flock (util-linux) is not installed"
        return 1
    fi
    local lock_dir
    lock_dir="$(dirname "$CVM_GUARD_LOCK_FILE")"
    mkdir -p "$lock_dir" 2>/dev/null || true
    if ! eval "exec $CVM_GUARD_LOCK_FD>>\"\$CVM_GUARD_LOCK_FILE\""; then
        cvm_guard_err "cannot open lock file $CVM_GUARD_LOCK_FILE (run with sudo)"
        return 1
    fi
    if ! flock -w "$CVM_GUARD_LOCK_WAIT" "$CVM_GUARD_LOCK_FD"; then
        cvm_guard_err "another lium-cvm.sh or cvm_upgrade_guard.sh holds $CVM_GUARD_LOCK_FILE; waited ${CVM_GUARD_LOCK_WAIT}s. Let it finish, then retry."
        return 5
    fi
    CVM_GUARD_LOCKED=1
    return 0
}

cvm_guard_unlock() {
    [ "$CVM_GUARD_LOCKED" = 1 ] || return 0
    flock -u "$CVM_GUARD_LOCK_FD"
    eval "exec $CVM_GUARD_LOCK_FD>&-"
    CVM_GUARD_LOCKED=0
}

# --- registry ----------------------------------------------------------------

# Record a VM directory root so every later inventory on this host sees it,
# whichever checkout runs the inventory.
cvm_guard_register_vms_dir() {
    local vms_dir="$1"
    mkdir -p "$CVM_GUARD_STATE_DIR" 2>/dev/null || {
        cvm_guard_err "cannot create $CVM_GUARD_STATE_DIR (run with sudo)"
        return 1
    }
    touch "$CVM_GUARD_REGISTRY"
    grep -qxF -- "$vms_dir" "$CVM_GUARD_REGISTRY" 2>/dev/null && return 0
    echo "$vms_dir" >>"$CVM_GUARD_REGISTRY"
}

# --- inventory ---------------------------------------------------------------

# The one predicate for a CVM disk, used by the registry walk and the sweep
# alike: a directory holding hda.img. runtime.json (written while QEMU runs)
# and the manifest (vm-manifest.json, the layout scripts/dstack.py writes) only
# refine the state: running, stopped, or orphan when the CVM is not running and
# its manifest is gone (state unknown; the disk still holds sealed data and
# still blocks an upgrade).
cvm_guard_classify_dir() {
    local dir="$1" state
    [ -f "$dir/hda.img" ] || {
        [ -f "$dir/vm-manifest.json" ] && CVM_GUARD_NO_DISK_YET+=("$dir")
        return 0
    }
    if [ -f "$dir/runtime.json" ]; then
        state=running
    elif [ ! -f "$dir/vm-manifest.json" ]; then
        state=orphan
    else
        state=stopped
    fi
    CVM_GUARD_DISKS+=("$state	$dir")
}

# Mount points below $1 whose filesystem can hold files (the mount table, with
# the pseudo filesystems dropped). Octal escapes in mount paths are decoded.
cvm_guard_submounts() {
    local root="$1" target fstype
    [ -r "$CVM_GUARD_MOUNTS" ] || return 0
    while read -r _ target fstype _; do
        case " $CVM_GUARD_PSEUDO_FS " in *" $fstype "*) continue ;; esac
        target="$(printf '%b' "$target")"
        case "$target" in "$root"/*) echo "$target" ;; esac
    done <"$CVM_GUARD_MOUNTS"
}

# The directory of every hda.img under CVM_GUARD_SWEEP_ROOTS, one per line.
# find's errors (unreadable directories) are appended to the file named by $1.
# `find -xdev` stays on one filesystem, so every real filesystem mounted below
# a root is swept as its own root.
cvm_guard_sweep_dirs() {
    local err_file="$1" root line found
    local sweep=()
    # shellcheck disable=SC2086 # the roots are a space-separated list on purpose
    for root in $CVM_GUARD_SWEEP_ROOTS; do
        [ -d "$root" ] || continue
        sweep+=("$root")
        while IFS= read -r line; do
            [ -n "$line" ] && sweep+=("$line")
        done < <(cvm_guard_submounts "$root")
    done
    for root in "${sweep[@]}"; do
        # The trailing slash makes find enter a root that is a symlink
        # (/data -> /mnt/nvme0); in -P mode a bare symlink start point is skipped.
        found="$(find "$root/" -xdev -type f -name hda.img -print 2>>"$err_file")" || true
        while IFS= read -r line; do
            [ -n "$line" ] && dirname "$line"
        done <<<"$found"
    done
}

# Fill CVM_GUARD_DISKS / CVM_GUARD_UNREADABLE from: this checkout's run/vms,
# every root in the registry (the registry walk), and cvm_guard_sweep_dirs for
# any hda.img the registry does not know about (the sweep).
cvm_guard_inventory() {
    CVM_GUARD_DISKS=()
    CVM_GUARD_UNREADABLE=()
    CVM_GUARD_NO_DISK_YET=()
    local -A seen=()
    local root dir line err_file

    local roots=("$CVM_GUARD_DIR/run/vms")
    if [ -e "$CVM_GUARD_REGISTRY" ]; then
        if [ -r "$CVM_GUARD_REGISTRY" ]; then
            while IFS= read -r line; do
                [ -n "$line" ] || continue
                if [ ! -e "$line" ]; then
                    # An unmounted disk looks the same as a removed checkout.
                    CVM_GUARD_UNREADABLE+=("$line (registered VM root is missing: mount it, or delete the line from $CVM_GUARD_REGISTRY if the checkout is gone)")
                    continue
                fi
                roots+=("$line")
            done <"$CVM_GUARD_REGISTRY"
        else
            CVM_GUARD_UNREADABLE+=("$CVM_GUARD_REGISTRY (registry not readable)")
        fi
    fi

    for root in "${roots[@]}"; do
        [ -e "$root" ] || continue
        if [ ! -r "$root" ] || [ ! -x "$root" ]; then
            CVM_GUARD_UNREADABLE+=("$root")
            continue
        fi
        for dir in "$root"/*/; do
            [ -d "$dir" ] || continue
            dir="${dir%/}"
            [ -n "${seen[$dir]:-}" ] && continue
            seen[$dir]=1
            cvm_guard_classify_dir "$dir"
        done
    done

    err_file="$(mktemp)"
    while IFS= read -r dir; do
        [ -n "$dir" ] || continue
        [ -n "${seen[$dir]:-}" ] && continue
        seen[$dir]=1
        cvm_guard_classify_dir "$dir"
    done < <(cvm_guard_sweep_dirs "$err_file")
    if [ -s "$err_file" ]; then
        while IFS= read -r line; do
            CVM_GUARD_UNREADABLE+=("$line")
        done <"$err_file"
    fi
    rm -f "$err_file"
}

# Print the inventory. Exit 0 = no disk, 3 = disks exist, 4 = incomplete.
cvm_guard_report() {
    local entry state dir
    cvm_guard_say "CVM disk inventory (this checkout, registered VM directories, sweep of: $CVM_GUARD_SWEEP_ROOTS)"
    if [ ${#CVM_GUARD_DISKS[@]} -eq 0 ]; then
        cvm_guard_say "no CVM disk found on this host"
    else
        cvm_guard_say "${#CVM_GUARD_DISKS[@]} CVM disk(s) exist on this host:"
        for entry in "${CVM_GUARD_DISKS[@]}"; do
            state="${entry%%	*}"
            dir="${entry#*	}"
            printf '  %-8s %s\n' "$state" "$dir/hda.img"
        done
        case " ${CVM_GUARD_DISKS[*]%%	*} " in *" orphan "*)
            cvm_guard_say "orphan = hda.img with no vm-manifest.json beside it and not running (state unknown; the disk still blocks)"
            ;;
        esac
    fi
    if [ ${#CVM_GUARD_NO_DISK_YET[@]} -gt 0 ]; then
        cvm_guard_say "created, no data disk yet (does not block): ${CVM_GUARD_NO_DISK_YET[*]}"
    fi
    if [ ${#CVM_GUARD_UNREADABLE[@]} -gt 0 ]; then
        cvm_guard_say "inventory INCOMPLETE, could not read:"
        for entry in "${CVM_GUARD_UNREADABLE[@]}"; do echo "  $entry"; done
        return 4
    fi
    [ ${#CVM_GUARD_DISKS[@]} -eq 0 ] && return 0
    return 3
}

# The exact manual steps that unblock an upgrade. The guard never runs them.
cvm_guard_removal_steps() {
    local entry state dir name checkout
    echo
    cvm_guard_say "The upgrade is refused while these disks exist. Each data disk is lost when it is removed."
    cvm_guard_say "When a CVM's data is no longer needed, remove it by hand (drain rentals first):"
    for entry in "${CVM_GUARD_DISKS[@]}"; do
        state="${entry%%	*}"
        dir="${entry#*	}"
        name="$(basename "$dir")"
        # <checkout>/run/vms/<name> is the layout lium-cvm.sh writes; a swept
        # directory elsewhere gets only the rm line.
        checkout="$(cd "$dir/../../.." 2>/dev/null && pwd)" || checkout=""
        if [ "$state" = running ] && [ -x "$checkout/lium-cvm.sh" ]; then
            printf '  sudo %q stop %q\n' "$checkout/lium-cvm.sh" "$name"
        fi
        # The rm line is printed only for a directory the CVM stack made: it has
        # the manifest (vm-manifest.json) or QEMU's runtime.json beside hda.img.
        # An orphan hda.img may belong to something else; name it, remove nothing.
        if [ "$state" = orphan ]; then
            printf '  # %q: disk with no vm-manifest.json beside it; confirm it is a Lium CVM before you remove anything\n' "$dir"
        else
            printf '  sudo rm -rf %q\n' "$dir"
        fi
    done
    cvm_guard_say "then run the upgrade again."
}

# --- image pin ---------------------------------------------------------------

cvm_guard_image_id() {
    docker image inspect --format '{{.Id}}' "$1" 2>/dev/null
}

cvm_guard_pinned_id() {
    [ -r "$CVM_GUARD_PIN_FILE" ] || return 1
    awk 'NR==1 {print $1}' "$CVM_GUARD_PIN_FILE"
}

cvm_guard_write_pin() {
    local id="$1"
    mkdir -p "$CVM_GUARD_STATE_DIR" || return 1
    printf '%s %s\n' "$id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$CVM_GUARD_PIN_FILE"
    # A second tag keeps the image when a later build moves :local.
    docker tag "$id" "lium-key-provider:pinned-${id#sha256:}" >/dev/null 2>&1 || true
    cvm_guard_say "pinned key-provider image $id in $CVM_GUARD_PIN_FILE"
}

# On a host that predates the pin file, the image in use is the one the
# dstack-key-provider container runs, else the image compose named before this
# guard. Pins it. Returns 1 when nothing is there to adopt.
cvm_guard_adopt_existing_image() {
    local id legacy
    id="$(docker inspect --format '{{.Image}}' "$CVM_GUARD_CONTAINER" 2>/dev/null)" || id=""
    if [ -z "$id" ]; then
        id="$(cvm_guard_image_id "$CVM_GUARD_IMAGE")" || id=""
    fi
    if [ -z "$id" ]; then
        # shellcheck disable=SC2086 # space-separated list on purpose
        for legacy in $CVM_GUARD_LEGACY_IMAGES; do
            id="$(cvm_guard_image_id "$legacy")" || id=""
            [ -n "$id" ] && break
        done
    fi
    [ -n "$id" ] || return 1
    docker tag "$id" "$CVM_GUARD_IMAGE" >/dev/null || return 1
    cvm_guard_say "adopted the key-provider image already on this host: $id"
    cvm_guard_write_pin "$id"
}

# The pinned image is gone. $2 is the inventory verdict: with disks (or an
# incomplete inventory) the way out is the image; on an empty host it is an
# explicit upgrade. Nothing is built here either way.
cvm_guard_recovery_guidance() {
    local id="$1" rc="$2"
    if [ "$rc" -eq 0 ]; then
        cvm_guard_err "the pinned key-provider image $id is not on this host. No CVM disk exists, so a rebuild is safe,"
        cvm_guard_err "but start never builds over a pin: run '$CVM_GUARD_DIR/cvm_upgrade_guard.sh upgrade' to rebuild and pin the new image."
        return 0
    fi
    cvm_guard_err "the pinned key-provider image $id is not on this host and CVM disks exist (or the inventory is incomplete)."
    cvm_guard_err "A rebuild would give a new MRENCLAVE and make every CVM data disk unreadable, so nothing is built."
    cvm_guard_err "Recovery: restore that exact image (docker load from your backup, or docker tag <id> $CVM_GUARD_IMAGE),"
    cvm_guard_err "then run this command again. If the image is gone for good and the CVM data is no longer needed,"
    cvm_guard_err "remove the CVM disks by hand ('$CVM_GUARD_DIR/cvm_upgrade_guard.sh inventory' lists the commands) and run '$CVM_GUARD_DIR/cvm_upgrade_guard.sh upgrade'."
}

# The aesmd sidecar holds no key material, so building it is harmless; still,
# a host that predates the guard keeps the sidecar it has.
cvm_guard_ensure_aesmd() {
    local id legacy
    docker image inspect "$CVM_GUARD_AESMD_IMAGE" >/dev/null 2>&1 && return 0
    id="$(docker inspect --format '{{.Image}}' "$CVM_GUARD_AESMD_CONTAINER" 2>/dev/null)" || id=""
    if [ -z "$id" ]; then
        # shellcheck disable=SC2086 # space-separated list on purpose
        for legacy in $CVM_GUARD_LEGACY_AESMD_IMAGES; do
            id="$(cvm_guard_image_id "$legacy")" || id=""
            [ -n "$id" ] && break
        done
    fi
    if [ -n "$id" ]; then
        docker tag "$id" "$CVM_GUARD_AESMD_IMAGE" || return 1
        cvm_guard_say "adopted the aesmd image already on this host: $id"
        return 0
    fi
    cvm_guard_say "building the aesmd sidecar (it holds no key material)"
    cvm_guard_compose_build aesmd
}

# Start (or keep) the key provider on the exact pinned image. Builds only when
# the host has no pin, nothing to adopt, and a complete inventory with no disk.
# The inventory (and its sweep) runs only when a build could be the outcome.
cvm_guard_start() {
    local rc=0 pinned current running
    cvm_guard_preflight || return 1
    cvm_guard_lock || return $?

    if ! pinned="$(cvm_guard_pinned_id)" || [ -z "$pinned" ]; then
        if ! cvm_guard_adopt_existing_image; then
            cvm_guard_inventory
            cvm_guard_report >/dev/null || rc=$?
            if [ "$rc" -ne 0 ]; then
                cvm_guard_report
                cvm_guard_err "no key-provider image on this host and no pin, but CVM disks exist or the inventory is incomplete: refusing to build."
                cvm_guard_recovery_guidance "(none recorded)" "$rc"
                return 6
            fi
            cvm_guard_say "first start on a host with no CVM disk: building the key provider"
            cvm_guard_compose_build || return 1
            pinned="$(cvm_guard_image_id "$CVM_GUARD_IMAGE")" || pinned=""
            [ -n "$pinned" ] || { cvm_guard_err "build produced no $CVM_GUARD_IMAGE"; return 1; }
            cvm_guard_write_pin "$pinned"
        else
            pinned="$(cvm_guard_pinned_id)"
        fi
    fi

    # :local must be the pinned image; a stray rebuild moved it, so move it back.
    current="$(cvm_guard_image_id "$CVM_GUARD_IMAGE")" || current=""
    if [ "$current" != "$pinned" ]; then
        if ! docker image inspect "$pinned" >/dev/null 2>&1; then
            cvm_guard_inventory
            cvm_guard_report >/dev/null || rc=$?
            cvm_guard_recovery_guidance "$pinned" "$rc"
            return 6
        fi
        cvm_guard_say "$CVM_GUARD_IMAGE was ${current:-<none>}; retagging the pinned image $pinned"
        docker tag "$pinned" "$CVM_GUARD_IMAGE" || return 1
    fi

    cvm_guard_ensure_aesmd || return 1

    cvm_guard_say "starting the key provider from $pinned (no build)"
    cvm_guard_compose up -d --no-build || return 1

    running="$(docker inspect --format '{{.Image}}' "$CVM_GUARD_CONTAINER" 2>/dev/null)" || running=""
    if [ "$running" != "$pinned" ]; then
        cvm_guard_err "container $CVM_GUARD_CONTAINER runs image ${running:-<none>} but the pin is $pinned"
        return 6
    fi
    cvm_guard_say "key provider up on the pinned image $pinned"
    return 0
}

# Rebuild the key provider. Refused while any CVM disk exists (3) or the
# inventory is incomplete (4). Keeps the previous image under a dated tag.
cvm_guard_upgrade() {
    local dry_run=0 rc=0 pinned new_id stamp
    case "${1:-}" in
    "") ;;
    --dry-run) dry_run=1 ;;
    *) cvm_guard_err "unknown argument '$1' (only --dry-run is accepted)"; return 1 ;;
    esac
    cvm_guard_preflight || return 1
    cvm_guard_lock || return $?
    cvm_guard_inventory
    cvm_guard_report || rc=$?
    if [ "$rc" -eq 3 ]; then
        cvm_guard_removal_steps
        return 3
    fi
    if [ "$rc" -eq 4 ]; then
        cvm_guard_err "refusing to upgrade: the inventory is incomplete. Fix the paths above (or remove stale lines from $CVM_GUARD_REGISTRY) and retry."
        return 4
    fi

    if [ "$dry_run" = 1 ]; then
        pinned="$(cvm_guard_pinned_id)" || pinned=""
        cvm_guard_say "dry run: no CVM disk on this host, an upgrade would rebuild the key provider (current pin: ${pinned:-none})"
        return 0
    fi

    if pinned="$(cvm_guard_pinned_id)" && [ -n "$pinned" ]; then
        :
    elif cvm_guard_adopt_existing_image; then
        pinned="$(cvm_guard_pinned_id)"
    else
        pinned=""
    fi

    if [ -n "$pinned" ] && docker image inspect "$pinned" >/dev/null 2>&1; then
        stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        docker tag "$pinned" "lium-key-provider:pre-upgrade-$stamp" || return 1
        cvm_guard_say "previous image $pinned kept as lium-key-provider:pre-upgrade-$stamp"
    fi

    cvm_guard_say "no CVM disk on this host: rebuilding the key provider"
    cvm_guard_compose_build || return 1
    new_id="$(cvm_guard_image_id "$CVM_GUARD_IMAGE")" || new_id=""
    [ -n "$new_id" ] || { cvm_guard_err "build produced no $CVM_GUARD_IMAGE"; return 1; }
    cvm_guard_write_pin "$new_id"
    cvm_guard_ensure_aesmd || return 1
    cvm_guard_compose up -d --no-build || return 1
    cvm_guard_say "key provider rebuilt: $new_id"
    cvm_guard_say "new MRENCLAVE: docker compose -f $CVM_GUARD_KP_DIR/$CVM_GUARD_COMPOSE_FILE logs $CVM_GUARD_SERVICE | grep -m1 mr_enclave"
    return 0
}

cvm_guard_usage() {
    cat <<EOF
Usage: $0 <command>

  inventory            list every CVM disk on this host; exit 0 none, 3 disks exist, 4 incomplete
  start                start the key provider on the exact pinned image (builds only on an empty host)
  upgrade [--dry-run]  rebuild the key provider; refused while any CVM disk exists
  lock -- <cmd...>     run <cmd> while holding $CVM_GUARD_LOCK_FILE

State: $CVM_GUARD_STATE_DIR (vm-dirs registry, key-provider.image pin), lock $CVM_GUARD_LOCK_FILE.
Override with LIUM_CVM_STATE_DIR, LIUM_CVM_LOCK_FILE, LIUM_CVM_LOCK_WAIT (s), LIUM_CVM_SWEEP_ROOTS.
EOF
}

cvm_guard_main() {
    local cmd="${1:-}" rc
    shift || true
    case "$cmd" in
    inventory)
        rc=0
        cvm_guard_inventory
        cvm_guard_report || rc=$?
        [ "$rc" -eq 3 ] && cvm_guard_removal_steps
        return $rc
        ;;
    start)
        cvm_guard_start
        ;;
    upgrade)
        cvm_guard_upgrade "$@"
        ;;
    lock)
        [ "${1:-}" = "--" ] && shift
        [ $# -gt 0 ] || { cvm_guard_usage; return 1; }
        command -v flock >/dev/null 2>&1 || { cvm_guard_err "flock (util-linux) is not installed"; return 1; }
        cvm_guard_lock || return $?
        "$@"
        ;;
    help | -h | --help | "")
        cvm_guard_usage
        [ -n "$cmd" ]
        ;;
    *)
        cvm_guard_err "unknown command: $cmd"
        cvm_guard_usage
        return 1
        ;;
    esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    cvm_guard_main "$@"
fi
