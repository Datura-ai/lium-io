#!/usr/bin/env bash
# Decide whether this host still holds CVM state, and say what to do about it.
#
# THE CATASTROPHE THIS EXISTS TO PREVENT
# --------------------------------------
# Changing a measured input (QEMU, the OS image, the compose) while an encrypted
# data disk exists makes that disk permanently undecryptable — by design. See
# neurons/executor/dstacktee/docs/host-setup.md line 160.
#
# So the PRIMARY predicate is an on-disk `run/vms/*/hda.img`, NOT a running
# process. `lium-cvm.sh stop` (stop_cvm, lium-cvm.sh:559) shuts the guest down
# and returns success WITHOUT removing the VM directory, and host-setup.md:155
# makes `rm -rf run/vms/<name>` a deliberate, separate, manual step. "Stopped
# CVM, zero QEMU processes, intact hda.img, open rental" is therefore a normal,
# reachable, common state — and a guard that only asks "is QEMU running?" waves
# it straight through.
#
# STRUCTURE
# ---------
# This is a fact collector with a derivation, not a state machine with
# hand-written cells. It emits three ORTHOGONAL facts — hda_images[], procs[],
# roots_unreadable[] — and then derives one state from them through an ordered
# precedence ladder. The naive six-way split is not a partition: a host with an
# intact hda.img AND a tenant's QEMU satisfies two descriptions at once.
#
# The `recovery` text is therefore COMPOSED FROM THE FACTS, never selected by
# the state name. A host in two conditions gets both sets of steps.
#
# Every state except CLEAN blocks destructive actions.
#
# Search roots are owned by THIS SCRIPT. bootstrap.sh runs it before Ansible
# starts, so it can never read group_vars. group_vars/all/main.yml mirrors these
# defaults into lium_hda_search_roots for the in-play include, and
# tests/test_guard.sh asserts the two lists agree so the duplication cannot rot.

set -Eeuo pipefail

DEFAULT_SEARCH_ROOTS="/home,/opt,/srv"
DEFAULT_REPO_PATH="/opt/lium-io"
DSTACKTEE_SUBPATH="neurons/executor/dstacktee"

# 12, not 8. A checkout at /opt/lium-io puts hda.img at exactly depth 8 below
# the search root, so 8 covered that one case and NOTHING deeper. The canonical
# /home/<user>/lium-io is depth 9 — and /home is in the root list specifically
# to catch a home-directory checkout, so the budget defeated the reason the root
# was there. A host with its checkout on a data volume is on record.
#
# The cost of scanning deeper is a slower find; the cost of stopping short is a
# CLEAN verdict on a host holding a renter's encrypted disk.
FIND_MAXDEPTH=12

# Per-root ceiling. An unbounded find under a failing disk hangs the guard, and
# a guard that never answers is a guard that never blocks. A root that times out
# is recorded UNREADABLE, which the ladder below turns into UNKNOWN.
FIND_TIMEOUT_SEC="${LIUM_FIND_TIMEOUT_SEC:-120}"

SEARCH_ROOTS="${LIUM_HDA_SEARCH_ROOTS:-$DEFAULT_SEARCH_ROOTS}"
REPO_PATH="${LIUM_REPO_PATH:-$DEFAULT_REPO_PATH}"
PROC_ROOT="${LIUM_PROC_ROOT:-/proc}"

# Test seam, exactly like LIUM_PROC_ROOT. Point it at a hand-written mounts file
# to drive the per-filesystem sweep below over fixtures, or at /dev/null to
# switch the sweep off and sandbox a case to its declared roots. The playbook
# never sets it.
MOUNTS_PATH="${LIUM_MOUNTS_PATH:-/proc/mounts}"

MODE="json"

usage() {
  cat <<'USAGE'
Usage: lium-guard.sh [--json|--state|--reason|--recovery|--unreadable|--print-default-roots]
                     [--roots a,b,c] [--repo-path PATH] [--proc-root PATH]

  --json                 full fact + derivation object (default)
  --state                just the state word, for shell callers with no JSON parser
  --reason               one-line explanation of the state
  --recovery             the fact-composed recovery procedure
  --unreadable           newline-separated list of roots that exist but cannot be read
  --print-default-roots  the built-in search roots, comma separated (tests assert on this)
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --json) MODE="json" ;;
    --state) MODE="state" ;;
    --reason) MODE="reason" ;;
    --recovery) MODE="recovery" ;;
    --unreadable) MODE="unreadable" ;;
    --print-default-roots) printf '%s\n' "$DEFAULT_SEARCH_ROOTS"; exit 0 ;;
    --roots) SEARCH_ROOTS="${2:?--roots needs a value}"; shift ;;
    --repo-path) REPO_PATH="${2:?--repo-path needs a value}"; shift ;;
    --proc-root) PROC_ROOT="${2:?--proc-root needs a value}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'lium-guard.sh: unknown argument: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
  shift
done

# --- JSON helpers ------------------------------------------------------------
# Emitted by hand because jq is not a dependency of this tree and this script
# runs before anything is installed.
json_escape() {
  printf '%s' "$1" | LC_ALL=C sed \
    -e 's/\\/\\\\/g' \
    -e 's/"/\\"/g' \
    -e 's/\t/\\t/g' \
    -e 's/\r/\\r/g' \
    -e ':a' -e 'N' -e '$!ba' -e 's/\n/\\n/g'
}

json_array_of_strings() {
  # Each argument becomes one element.
  local first=1 item out=""
  for item in "$@"; do
    if [ "$first" -eq 1 ]; then first=0; else out="${out}, "; fi
    out="${out}\"$(json_escape "$item")\""
  done
  printf '[%s]' "$out"
}

# --- fact 1: hda images ------------------------------------------------------
HDA_IMAGES=()
ROOTS_UNREADABLE=()

collect_hda_under() {
  # A root that does NOT exist is omitted, not recorded. A fresh host before the
  # clone has no repo path at all, and that must never read as a permissions
  # failure — it is the normal first-run shape.
  #
  # Any arguments after the root are passed straight to find, which is how the
  # per-filesystem sweep below adds -xdev.
  local root="$1"; shift
  local stderr_file found rc=0
  [ -e "$root" ] || return 0

  stderr_file="$(mktemp)"
  found="$(timeout "$FIND_TIMEOUT_SEC" \
             find "$root" -maxdepth "$FIND_MAXDEPTH" "$@" \
             -path '*/run/vms/*/hda.img' -type f -print 2>"$stderr_file")" || rc=$?

  # Anything on find's stderr means part of the tree could not be traversed, and
  # a non-zero exit covers the case stderr does not: timeout's 124, where find
  # was killed mid-walk and its silence means nothing.
  # Fail closed: an unreadable root becomes UNKNOWN, which blocks.
  if [ "$rc" -ne 0 ] || [ -s "$stderr_file" ]; then
    ROOTS_UNREADABLE+=("$root")
  fi
  rm -f "$stderr_file"

  if [ -n "$found" ]; then
    while IFS= read -r line; do
      [ -n "$line" ] && HDA_IMAGES+=("$line")
    done <<<"$found"
  fi
}

# Where this script actually lives, so a checkout in an unexpected place is
# found without anyone having to pass --repo-path. bootstrap.sh runs from inside
# the checkout, so its location is the one piece of information that is always
# correct — and relying on the /opt/lium-io default instead is what let a
# home-directory checkout read as CLEAN.
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CHECKOUT="$(cd -- "${SELF_DIR}/../../../../../.." 2>/dev/null && pwd || printf '')"

collect_hda_under "${REPO_PATH}/${DSTACKTEE_SUBPATH}"
if [ -n "$LOCAL_CHECKOUT" ]; then
  collect_hda_under "${LOCAL_CHECKOUT}/${DSTACKTEE_SUBPATH}"
fi
IFS=',' read -r -a _roots <<<"$SEARCH_ROOTS"
for _root in "${_roots[@]}"; do
  [ -n "$_root" ] && collect_hda_under "$_root"
done

# --- fact 1b: every locally mounted filesystem --------------------------------
# The static roots are the CONVENTIONAL locations, and conventions are not where
# hosts actually are. A production host with its checkout on a data volume put a
# 388 GB renter disk at /data0/lium-io/…/hda.img — outside all three roots, and
# outside REPO_PATH and LOCAL_CHECKOUT too on the day-zero shape, where the
# operator bootstraps a FRESH clone at /opt/lium-io. Stop the CVM and there is no
# process to fall back on either, so all five sources miss and the answer is
# CLEAN. CLEAN is precisely the verdict that authorises rebuilding QEMU, and
# rebuilding QEMU makes that disk permanently undecryptable.
#
# Raising FIND_MAXDEPTH fixed how DEEP the search goes. This fixes WHERE it
# starts. Each filesystem is swept once with -xdev, so the union is complete and
# nothing is walked twice — measured at 2.2 s for / plus 0.03 s for a 14 TB data
# volume, once per bootstrap.
#
# The type list is a DENY list. An unrecognised filesystem gets searched,
# because failing to find a disk yields CLEAN and destroys data, while searching
# something exotic only costs time and is bounded by FIND_TIMEOUT_SEC. The three
# categories denied are ones where searching is never right: kernel pseudo
# filesystems hold no files; overlay and squashfs are container image layers,
# never a QEMU data disk; and network mounts hang on `[ -e ]` itself when the
# server is gone — before any timeout can bound them — while a multi-hundred-GB
# raw image that QEMU does direct I/O against is never on one.
DENY_FS_TYPES="proc sysfs devtmpfs devpts tmpfs ramfs cgroup cgroup2 securityfs
pstore efivarfs bpf tracefs debugfs configfs fusectl mqueue hugetlbfs
binfmt_misc autofs nsfs rpc_pipefs selinuxfs overlay squashfs vfat
nfs nfs4 cifs smb3 smbfs ceph glusterfs afs 9p fuse.sshfs fuse.s3fs"

MOUNT_ROOTS=()
if [ -r "$MOUNTS_PATH" ]; then
  while read -r _dev _mnt _fstype _rest; do
    [ -n "${_mnt:-}" ] || continue
    case " $DENY_FS_TYPES " in *" $_fstype "*) continue ;; esac
    # /proc/mounts octal-escapes spaces and tabs in mount points.
    _mnt="$(printf '%b' "$_mnt")"
    MOUNT_ROOTS+=("$_mnt")
  done < "$MOUNTS_PATH"
fi

if [ "${#MOUNT_ROOTS[@]}" -gt 0 ]; then
  mapfile -t MOUNT_ROOTS < <(printf '%s\n' "${MOUNT_ROOTS[@]}" | sort -u)
  for _mroot in "${MOUNT_ROOTS[@]}"; do
    collect_hda_under "$_mroot" -xdev
  done
fi

# De-duplicate: the repo path may itself sit under one of the search roots.
if [ "${#HDA_IMAGES[@]}" -gt 0 ]; then
  mapfile -t HDA_IMAGES < <(printf '%s\n' "${HDA_IMAGES[@]}" | sort -u)
fi
if [ "${#ROOTS_UNREADABLE[@]}" -gt 0 ]; then
  mapfile -t ROOTS_UNREADABLE < <(printf '%s\n' "${ROOTS_UNREADABLE[@]}" | sort -u)
fi

# --- fact 2: qemu-system processes -------------------------------------------
# Matched on /proc/<pid>/comm with a PREFIX compare.
#
# `comm` is truncated to 15 bytes, so the literal value on a real host is
# `qemu-system-x86` — an exact match on `qemu-system-x86_64` would NEVER fire.
#
# argv is deliberately never pattern-matched. `pgrep -f`/`pkill -f` on a pattern
# that also appears in your own command line killed two ssh sessions (traps #8),
# and matching on `qemu` false-positived on the QEMU source build (traps #11).
# argv is READ here, but only to decide whether an already-identified
# qemu-system process is ours — it is never the thing that finds the process.
COMM_PREFIX="qemu-system"

PROC_PIDS=()
PROC_COMMS=()
PROC_STATES=()
PROC_OURS=()
PROC_CMDHEADS=()

for _procdir in "$PROC_ROOT"/[0-9]*; do
  [ -d "$_procdir" ] || continue
  [ -r "$_procdir/comm" ] || continue

  _comm="$(cat "$_procdir/comm" 2>/dev/null || true)"
  case "$_comm" in
    "$COMM_PREFIX"*) ;;
    *) continue ;;
  esac

  _pid="$(basename "$_procdir")"

  # Process state is field 3 of /proc/<pid>/stat, after the parenthesised comm.
  # A `Z` (zombie) QEMU still holds guest RAM and is not reaped until its parent
  # dies — polling it stalls forever (traps #8).
  _state="?"
  if [ -r "$_procdir/stat" ]; then
    _state="$(sed -e 's/^.*) //' -e 's/ .*$//' "$_procdir/stat" 2>/dev/null || printf '?')"
  fi

  _cmdline=""
  if [ -r "$_procdir/cmdline" ]; then
    _cmdline="$(tr '\0' ' ' <"$_procdir/cmdline" 2>/dev/null || true)"
  fi

  _ours="false"
  case "$_cmdline" in
    */run/vms/*) _ours="true" ;;
  esac

  PROC_PIDS+=("$_pid")
  PROC_COMMS+=("$_comm")
  PROC_STATES+=("${_state:-?}")
  PROC_OURS+=("$_ours")
  PROC_CMDHEADS+=("$(printf '%.160s' "$_cmdline")")
done

# --- fact 3: incidental context ----------------------------------------------
TMUX_LIUM_CVM="false"
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t lium-cvm >/dev/null 2>&1; then
    TMUX_LIUM_CVM="true"
  fi
fi

MEM_TOTAL_GB=0
MEM_USED_GB=0
if [ -r /proc/meminfo ]; then
  _kb_total="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || printf '0')"
  _kb_avail="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || printf '0')"
  MEM_TOTAL_GB=$(( _kb_total / 1024 / 1024 ))
  MEM_USED_GB=$(( (_kb_total - _kb_avail) / 1024 / 1024 ))
fi

# --- derivation: the precedence ladder ---------------------------------------
# ORDERED if/elif, in exactly this order:
#
#   UNKNOWN > ZOMBIE > LIVE > FOREIGN > DORMANT > CLEAN
#
# Written as independent `if`s this would be a second partition bug. Because the
# chain is ordered, DORMANT's "and no processes" clause is redundant — anything
# with processes was already claimed above it — and harmlessly so.
ANY_ZOMBIE="false"
ANY_OURS="false"
for _i in "${!PROC_PIDS[@]}"; do
  [ "${PROC_STATES[$_i]}" = "Z" ] && ANY_ZOMBIE="true"
  [ "${PROC_OURS[$_i]}" = "true" ] && ANY_OURS="true"
done

if [ "${#ROOTS_UNREADABLE[@]}" -gt 0 ]; then
  STATE="UNKNOWN"
  REASON="a search root exists but could not be read, so the presence of a CVM cannot be determined: ${ROOTS_UNREADABLE[*]}"
elif [ "$ANY_ZOMBIE" = "true" ]; then
  STATE="ZOMBIE"
  REASON="a qemu-system process is a zombie and still holds guest memory"
elif [ "$ANY_OURS" = "true" ]; then
  STATE="LIVE"
  REASON="a qemu-system process is running against our run/vms/ tree"
elif [ "${#PROC_PIDS[@]}" -gt 0 ]; then
  STATE="FOREIGN"
  REASON="a qemu-system process is running that is not ours — usually an expected bare-metal rental, not an outage"
elif [ "${#HDA_IMAGES[@]}" -gt 0 ]; then
  STATE="DORMANT"
  REASON="a stopped CVM's encrypted data disk is still on disk; changing a measured input now makes it permanently undecryptable"
else
  STATE="CLEAN"
  REASON="no CVM state found"
fi

# --- recovery: composed from the FACTS, never selected by the state name ------
RECOVERY_LINES=()

if [ "${#ROOTS_UNREADABLE[@]}" -gt 0 ]; then
  RECOVERY_LINES+=("Make these paths readable, then re-run — the guard fails closed rather than guess:")
  for _r in "${ROOTS_UNREADABLE[@]}"; do
    RECOVERY_LINES+=("  ls -ld ${_r}")
  done
fi

if [ "${#HDA_IMAGES[@]}" -gt 0 ]; then
  RECOVERY_LINES+=("An encrypted CVM data disk exists. Removing it DESTROYS the renter's data —")
  RECOVERY_LINES+=("that is by design and it is unrecoverable. Follow docs/host-setup.md section 6:")
  RECOVERY_LINES+=("  1. Drain any open rental on this node first.")
  for _img in "${HDA_IMAGES[@]}"; do
    _vmdir="$(dirname "$_img")"
    _vmname="$(basename "$_vmdir")"
    RECOVERY_LINES+=("  2. sudo ./lium-cvm.sh stop ${_vmname}")
    RECOVERY_LINES+=("  3. sudo rm -rf ${_vmdir}")
  done
  RECOVERY_LINES+=("  4. Re-run bootstrap.sh.")
fi

if [ "$ANY_ZOMBIE" = "true" ]; then
  RECOVERY_LINES+=("A zombie qemu-system process still holds guest memory. It is not reaped until its")
  RECOVERY_LINES+=("parent dies, so polling for it never returns. Kill the parent:")
  RECOVERY_LINES+=("  sudo tmux kill-session -t lium-cvm")
fi

if [ "$ANY_OURS" = "true" ]; then
  RECOVERY_LINES+=("Our CVM is running. Stop it before converging anything measured:")
  RECOVERY_LINES+=("  sudo ./lium-cvm.sh stop <name>")
fi

if [ "$ANY_OURS" = "false" ] && [ "${#PROC_PIDS[@]}" -gt 0 ]; then
  RECOVERY_LINES+=("A qemu-system process is running that does not point at our run/vms/ tree.")
  RECOVERY_LINES+=("This is what an active bare-metal rental looks like — expected revenue, not an")
  RECOVERY_LINES+=("outage. Confirm against rental_history before doing anything to this host.")
fi

if [ "${#RECOVERY_LINES[@]}" -eq 0 ]; then
  RECOVERY_LINES+=("No action needed.")
fi

RECOVERY_TEXT="$(printf '%s\n' "${RECOVERY_LINES[@]}")"

# --- output ------------------------------------------------------------------
case "$MODE" in
  state) printf '%s\n' "$STATE"; exit 0 ;;
  reason) printf '%s\n' "$REASON"; exit 0 ;;
  recovery) printf '%s\n' "$RECOVERY_TEXT"; exit 0 ;;
  unreadable)
    if [ "${#ROOTS_UNREADABLE[@]}" -gt 0 ]; then
      printf '%s\n' "${ROOTS_UNREADABLE[@]}"
    fi
    exit 0
    ;;
esac

procs_json="["
for _i in "${!PROC_PIDS[@]}"; do
  [ "$_i" -gt 0 ] && procs_json="${procs_json},"
  procs_json="${procs_json}{\"pid\": ${PROC_PIDS[$_i]}"
  procs_json="${procs_json}, \"comm\": \"$(json_escape "${PROC_COMMS[$_i]}")\""
  procs_json="${procs_json}, \"proc_state\": \"$(json_escape "${PROC_STATES[$_i]}")\""
  procs_json="${procs_json}, \"ours\": ${PROC_OURS[$_i]}"
  procs_json="${procs_json}, \"cmdline_head\": \"$(json_escape "${PROC_CMDHEADS[$_i]}")\"}"
done
procs_json="${procs_json}]"

hda_json="$(json_array_of_strings ${HDA_IMAGES[@]+"${HDA_IMAGES[@]}"})"
unreadable_json="$(json_array_of_strings ${ROOTS_UNREADABLE[@]+"${ROOTS_UNREADABLE[@]}"})"
roots_json="$(json_array_of_strings ${_roots[@]+"${_roots[@]}"})"
mount_roots_json="$(json_array_of_strings ${MOUNT_ROOTS[@]+"${MOUNT_ROOTS[@]}"})"

cat <<JSON
{
  "state": "$(json_escape "$STATE")",
  "reason": "$(json_escape "$REASON")",
  "recovery": "$(json_escape "$RECOVERY_TEXT")",
  "hda_images": ${hda_json},
  "procs": ${procs_json},
  "roots_unreadable": ${unreadable_json},
  "search_roots": ${roots_json},
  "mount_roots": ${mount_roots_json},
  "repo_path": "$(json_escape "$REPO_PATH")",
  "tmux_lium_cvm": ${TMUX_LIUM_CVM},
  "mem_used_gb": ${MEM_USED_GB},
  "mem_total_gb": ${MEM_TOTAL_GB}
}
JSON
