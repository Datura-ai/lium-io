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
Usage: lium-guard.sh [--json|--state|--reason|--recovery|--unreadable|--qemu-procs|--print-default-roots]
                     [--roots a,b,c] [--repo-path PATH] [--proc-root PATH]

  --json                 full fact + derivation object (default)
  --state                just the state word, for shell callers with no JSON parser
  --reason               one-line explanation of the state
  --recovery             the fact-composed recovery procedure
  --unreadable           newline-separated list of roots that exist but cannot be read
  --qemu-procs           one line per qemu-system process: "pid state threads cpu_ticks".
                         Skips the filesystem sweep, so it is cheap enough to poll.
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
    --qemu-procs) MODE="qemu-procs" ;;
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

# `--qemu-procs` answers one question — is a QEMU still on this host — and its
# caller is a POLL LOOP that asks every few seconds for up to an hour while a
# TDX teardown drains. The disk sweep costs seconds per call and cannot change
# that answer, so it is skipped. Every other mode still pays for it, because
# every other mode reports a state, and a state derived without the disk facts
# would be CLEAN on a host holding a renter's encrypted disk.
SKIP_HDA_SCAN="false"
[ "$MODE" = "qemu-procs" ] && SKIP_HDA_SCAN="true"

IFS=',' read -r -a _roots <<<"$SEARCH_ROOTS"
if [ "$SKIP_HDA_SCAN" = "false" ]; then
  collect_hda_under "${REPO_PATH}/${DSTACKTEE_SUBPATH}"
  if [ -n "$LOCAL_CHECKOUT" ]; then
    collect_hda_under "${LOCAL_CHECKOUT}/${DSTACKTEE_SUBPATH}"
  fi
  for _root in "${_roots[@]}"; do
    [ -n "$_root" ] && collect_hda_under "$_root"
  done
fi

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
if [ "$SKIP_HDA_SCAN" = "false" ] && [ -r "$MOUNTS_PATH" ]; then
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
PROC_PPIDS=()
PROC_OURS=()
PROC_ARGV_READABLE=()
PROC_CMDHEADS=()
PROC_THREADS=()
PROC_CPU_TICKS=()

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
  # Everything after the parenthesised comm is "state ppid pgrp ...", so both
  # fields come from one read. The ppid matters for a zombie: the recovery turns
  # on whether there is still a parent left to kill.
  # Threads and CPU time come from the same read, because together they are the
  # only way to tell a teardown that is GRINDING from one that is WEDGED, and
  # that distinction is what decides whether waiting is the right move.
  #
  # A TDX guest's memory is handed back in STEPS, one per memory-backend-ram
  # object — a 1.13 TB guest sat at a dead-flat `free` for 22 minutes, jumped by
  # half, sat flat another 21, then finished. So a flat memory reading proves
  # nothing. The surviving thread's utime+stime climbing at roughly one core is
  # what proves progress.
  _state="?"
  _ppid="0"
  _threads="0"
  _cpu_ticks="0"
  if [ -r "$_procdir/stat" ]; then
    # Everything after the parenthesised comm. The comm itself may contain
    # spaces and parentheses, so the split is on the LAST ') ', not the first.
    _rest="$(sed -e 's/^.*) //' "$_procdir/stat" 2>/dev/null || printf '')"
    if [ -n "$_rest" ]; then
      # Field numbers within _rest: 1 state, 2 ppid, 12 utime, 13 stime,
      # 18 num_threads. A fixture with a short stat line yields empty, which
      # awk prints as the 0 default rather than an empty string that would
      # then fail the numeric guards below.
      read -r _state _ppid _cpu_ticks _threads <<<"$(printf '%s\n' "$_rest" | awk '{
        printf "%s %s %s %s\n", ($1 == "" ? "?" : $1), ($2 == "" ? 0 : $2),
                                (($12 == "" ? 0 : $12) + ($13 == "" ? 0 : $13)),
                                ($18 == "" ? 0 : $18)
      }')"
    fi
  fi
  case "$_ppid" in ''|*[!0-9]*) _ppid="0" ;; esac
  case "$_threads" in ''|*[!0-9]*) _threads="0" ;; esac
  case "$_cpu_ticks" in ''|*[!0-9]*) _cpu_ticks="0" ;; esac
  [ -n "${_state:-}" ] || _state="?"

  # A ZOMBIE'S ARGV IS ALWAYS EMPTY. The kernel frees it when the process exits,
  # so /proc/<pid>/cmdline reads as 0 bytes while the entry still exists. Reading
  # that emptiness as "does not point at run/vms/, therefore not ours" is how a
  # host with our own dead CVM was described to the operator as somebody's
  # active bare-metal rental — "expected revenue, not an outage". Observed on a
  # production host. Absence of evidence is tracked separately from evidence of
  # absence, and only the latter may claim a process belongs to a tenant.
  _cmdline=""
  _argv_readable="false"
  if [ -r "$_procdir/cmdline" ]; then
    _cmdline="$(tr '\0' ' ' <"$_procdir/cmdline" 2>/dev/null || true)"
    [ -n "$_cmdline" ] && _argv_readable="true"
  fi

  _ours="false"
  case "$_cmdline" in
    */run/vms/*) _ours="true" ;;
  esac

  PROC_PIDS+=("$_pid")
  PROC_COMMS+=("$_comm")
  PROC_STATES+=("${_state:-?}")
  PROC_PPIDS+=("$_ppid")
  PROC_OURS+=("$_ours")
  PROC_ARGV_READABLE+=("$_argv_readable")
  PROC_CMDHEADS+=("$(printf '%.160s' "$_cmdline")")
  PROC_THREADS+=("$_threads")
  PROC_CPU_TICKS+=("$_cpu_ticks")
done

# Answered before the derivation, because the derivation needs the disk facts
# this mode deliberately skipped. Exit status is the count, capped at 250, so a
# poll loop can branch on it without parsing: 0 means the host is free of QEMU.
if [ "$MODE" = "qemu-procs" ]; then
  for _i in "${!PROC_PIDS[@]}"; do
    printf '%s %s %s %s\n' \
      "${PROC_PIDS[$_i]}" "${PROC_STATES[$_i]}" \
      "${PROC_THREADS[$_i]}" "${PROC_CPU_TICKS[$_i]}"
  done
  _n="${#PROC_PIDS[@]}"
  [ "$_n" -gt 250 ] && _n=250
  exit "$_n"
fi

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
ANY_ZOMBIE_ORPHANED="false"
ANY_FOREIGN_WITH_ARGV="false"
ANY_ARGV_UNREADABLE="false"
for _i in "${!PROC_PIDS[@]}"; do
  [ "${PROC_STATES[$_i]}" = "Z" ] && ANY_ZOMBIE="true"
  [ "${PROC_OURS[$_i]}" = "true" ] && ANY_OURS="true"
  if [ "${PROC_STATES[$_i]}" = "Z" ] && [ "${PROC_PPIDS[$_i]}" = "1" ]; then
    ANY_ZOMBIE_ORPHANED="true"
  fi
  if [ "${PROC_ARGV_READABLE[$_i]}" = "true" ]; then
    [ "${PROC_OURS[$_i]}" = "false" ] && ANY_FOREIGN_WITH_ARGV="true"
  else
    ANY_ARGV_UNREADABLE="true"
  fi
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
  RECOVERY_LINES+=("A zombie qemu-system process is TEARING DOWN A TDX GUEST. It still holds that")
  RECOVERY_LINES+=("guest's memory, and it is not reaped until its last thread exits.")
  _first_zombie_pid=""
  for _i in "${!PROC_PIDS[@]}"; do
    [ "${PROC_STATES[$_i]}" = "Z" ] || continue
    [ -n "$_first_zombie_pid" ] || _first_zombie_pid="${PROC_PIDS[$_i]}"
    RECOVERY_LINES+=("  pid ${PROC_PIDS[$_i]}, parent pid ${PROC_PPIDS[$_i]}, threads ${PROC_THREADS[$_i]}, cpu ticks ${PROC_CPU_TICKS[$_i]}")
  done
  RECOVERY_LINES+=("")
  RECOVERY_LINES+=("DO NOT REBOOT THIS HOST TO CLEAR IT. A soft reboot asks the running kernel to")
  RECOVERY_LINES+=("finish the very teardown it would then be waiting on, and to shut down the")
  RECOVERY_LINES+=("same GPUs that teardown is still unmapping. On a production host this left")
  RECOVERY_LINES+=("the old kernel wedged for over fourteen minutes WITH THE NETWORK STILL UP —")
  RECOVERY_LINES+=("ping answered, port 22 accepted the connection and never sent a banner — and")
  RECOVERY_LINES+=("it took a datacentre power-cycle to recover. Rebooting turns a wait into an")
  RECOVERY_LINES+=("outage plus a support ticket.")
  RECOVERY_LINES+=("")
  RECOVERY_LINES+=("WAIT INSTEAD. It reaps itself. How long scales with the guest's memory:")
  RECOVERY_LINES+=("about a minute for a near-empty guest, and 43 minutes measured for 1.13 TB.")
  # No backticks in these strings: they are inside double quotes, so a backtick
  # pair is command substitution and the host would run whatever it wrapped.
  RECOVERY_LINES+=("Memory comes back in STEPS, one per memory-backend-ram object, so a free(1)")
  RECOVERY_LINES+=("reading that sits flat for twenty minutes is the normal shape, not a hang.")
  RECOVERY_LINES+=("To prove it is still working, watch the cpu ticks climb — roughly 100 per")
  RECOVERY_LINES+=("second is one full core of page reclaim:")
  RECOVERY_LINES+=("  watch -n5 ./lium-guard.sh --qemu-procs")
  RECOVERY_LINES+=("If the ticks stop climbing for many minutes, the host needs an OUT-OF-BAND")
  RECOVERY_LINES+=("power-cycle from the datacentre — never a soft reboot from inside.")
  # Which instruction is correct depends on facts already collected, so it is
  # chosen from them. The unconditional `tmux kill-session -t lium-cvm` named a
  # session this host did not have and a parent that was already init.
  if [ "$TMUX_LIUM_CVM" = "true" ]; then
    RECOVERY_LINES+=("")
    RECOVERY_LINES+=("The lium-cvm tmux session is its parent, so once the teardown finishes,")
    RECOVERY_LINES+=("killing that session reaps the entry:")
    RECOVERY_LINES+=("  sudo tmux kill-session -t lium-cvm")
  elif [ "$ANY_ZOMBIE_ORPHANED" = "true" ]; then
    RECOVERY_LINES+=("")
    RECOVERY_LINES+=("Its parent is already init (pid 1), so there is no parent left to kill, and")
    RECOVERY_LINES+=("kill -9 does nothing to a process that has already exited. Init will reap it")
    RECOVERY_LINES+=("the moment the last thread is gone. See what is still running:")
    RECOVERY_LINES+=("  ls /proc/${_first_zombie_pid:-<pid>}/task")
  else
    RECOVERY_LINES+=("")
    RECOVERY_LINES+=("Once the teardown finishes, the parent pid shown above reaps it: a zombie is")
    RECOVERY_LINES+=("cleared when its parent exits or reaps it, and not before.")
  fi
fi

if [ "$ANY_OURS" = "true" ]; then
  RECOVERY_LINES+=("Our CVM is running. Stop it before converging anything measured:")
  RECOVERY_LINES+=("  sudo ./lium-cvm.sh stop <name>")
fi

# Gated on argv having actually been READ. "Its command line does not mention
# run/vms/" only means "someone else's" when there was a command line to look at.
if [ "$ANY_FOREIGN_WITH_ARGV" = "true" ]; then
  RECOVERY_LINES+=("A qemu-system process is running that does not point at our run/vms/ tree.")
  RECOVERY_LINES+=("This is what an active bare-metal rental looks like — expected revenue, not an")
  RECOVERY_LINES+=("outage. Confirm against rental_history before doing anything to this host.")
fi

if [ "$ANY_ARGV_UNREADABLE" = "true" ]; then
  RECOVERY_LINES+=("A qemu-system process has no readable command line — a zombie's is always")
  RECOVERY_LINES+=("empty — so whether it was ours or a tenant's cannot be told from it. Do not")
  RECOVERY_LINES+=("read that silence as 'not ours': check run/vms/ and rental_history instead.")
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
  procs_json="${procs_json}, \"ppid\": ${PROC_PPIDS[$_i]}"
  procs_json="${procs_json}, \"threads\": ${PROC_THREADS[$_i]}"
  procs_json="${procs_json}, \"cpu_ticks\": ${PROC_CPU_TICKS[$_i]}"
  procs_json="${procs_json}, \"ours\": ${PROC_OURS[$_i]}"
  procs_json="${procs_json}, \"argv_readable\": ${PROC_ARGV_READABLE[$_i]}"
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
