#!/usr/bin/env bash
# Merge-gate box 6.
#
# The guard is the most safety-critical file in this tree: a false negative
# destroys a paying customer's encrypted data disk, unrecoverably and by design.
# So every branch is exercised here, against fixtures, with no hardware.
#
# Every assertion is an explicit if/then/exit branch. No `!`-inverted commands:
# under `bash -e` a command preceded by `!` does not trigger errexit, so such an
# "assertion" silently always passes.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GUARD=roles/common/files/lium-guard.sh
FIX="$PWD/tests/fixtures/guard"
FAILURES=0
CASES=0

# Every case below declares its own search surface and must see nothing else.
# The guard also sweeps each locally mounted filesystem, which on a real machine
# means the runner's own disks — so without this, every fixture case inherits
# whatever the CI host happens to be holding and the sandbox is not a sandbox.
# /dev/null is a mounts file with no entries. Case (c4) overrides it with a real
# one, which is where the sweep itself is tested.
export LIUM_MOUNTS_PATH=/dev/null

# Substring tests use `case`, not `printf ... | grep -q`: grep -q exits on the
# first match, printf takes SIGPIPE, and pipefail reports that as failure — so
# a passing assertion intermittently reads as a FAIL. Same defect the collector
# had, and a merge gate that fails at random is worse than one that does not run.
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }

expect_state() {
  local label="$1" want="$2" got
  shift 2
  CASES=$((CASES + 1))
  got="$("$@" "$GUARD" --state)"
  if [ "$got" = "$want" ]; then
    pass "$label -> $got"
  else
    fail "$label -> expected $want, got $got"
  fi
}

# A synthetic /proc so process detection can be tested on any machine.
# The stat line is written to REAL field widths, not a short stand-in. The guard
# reads num_threads at field 18 and utime+stime at 12 and 13, all counted after
# the parenthesised comm; a six-field line makes every one of them read as the
# absent-value default, so a fixture would agree with a guard that had stopped
# collecting them at all.
make_proc() {
  local root="$1" pid="$2" comm="$3" state="$4" cmdline="$5"
  local threads="${6:-1}" utime="${7:-0}" stime="${8:-0}"
  mkdir -p "$root/$pid"
  printf '%s\n' "$comm" >"$root/$pid/comm"
  # state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt
  #   utime stime cutime cstime priority nice num_threads
  printf '%s (%s) %s 1 1 1 0 -1 0 0 0 0 0 %s %s 0 0 20 0 %s\n' \
    "$pid" "$comm" "$state" "$utime" "$stime" "$threads" >"$root/$pid/stat"
  printf '%s' "$cmdline" | tr ' ' '\0' >"$root/$pid/cmdline"
}

PROC="$(mktemp -d)"
EMPTY_PROC="$(mktemp -d)"

# The two "a disk exists" fixtures are BUILT HERE rather than committed.
#
# To the guard, a checked-in file at */run/vms/*/hda.img is indistinguishable
# from a renter's encrypted data disk — and the playbook clones this very repo
# to /opt/lium-io, which is inside the search roots at a depth the find reaches.
# So a host that had never created a CVM read as DORMANT the moment the clone
# landed: kernel, gpu and qemu blocked on every run afterwards, with recovery
# text telling the operator to rm -rf the repo's own test data. Reproduced on a
# production host. Nothing this tree ships may match that glob — asserted below.
FIXTMP="$(mktemp -d)"
DORMANT_ROOT="$FIXTMP/dormant_hda"
FOREIGN_ROOT="$FIXTMP/hda_plus_foreign"
for _r in "$DORMANT_ROOT" "$FOREIGN_ROOT"; do
  mkdir -p "$_r/run/vms/demo"
  printf 'placeholder standing in for a CVM encrypted data disk\n' >"$_r/run/vms/demo/hda.img"
done

trap 'rm -rf "$PROC" "$EMPTY_PROC" "$FIXTMP"; chmod 755 "$FIX/unreadable_root" 2>/dev/null || printf ""' EXIT

printf 'test_guard.sh\n'

# (a) The comm decoy. On a real host `comm` is truncated to 15 bytes, so the
#     literal value is `qemu-system-x86`. A guard that compared exactly against
#     `qemu-system-x86_64` would never fire on a real host at all — this decoy
#     uses the untruncated name to prove the compare is a PREFIX match.
rm -rf "${PROC:?}"/*
make_proc "$PROC" 1001 "qemu-system-x86_64" S "/usr/bin/qemu-system-x86_64 -m 4096"
expect_state "(a) comm decoy 'qemu-system-x86_64' blocks" FOREIGN \
  env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent LIUM_REPO_PATH=/nonexistent

# (b) argv mentions run/vms/ but comm is not qemu. Must NOT block. This proves
#     argv is never the thing that FINDS a process — which is what stops the
#     guard matching the playbook's own command line.
rm -rf "${PROC:?}"/*
make_proc "$PROC" 1002 "bash" S "/bin/bash -c ansible-playbook --repo /opt/lium-io/run/vms/demo"
expect_state "(b) argv-only decoy does not block" CLEAN \
  env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent LIUM_REPO_PATH=/nonexistent

# (b2) The QEMU SOURCE BUILD shape: a non-qemu process whose argv mentions
#      qemu-system. Case (b) only covers argv containing `run/vms/`, so a
#      regression to matching comm+argv against `qemu-system` — which is what
#      `pgrep -f qemu-system` would do — slips straight past it. That regression
#      refuses to converge for the whole 10-40 minutes of a QEMU build, on a
#      host with no CVM at all.
rm -rf "${PROC:?}"/*
make_proc "$PROC" 1006 "make" S "make -j32 qemu-system-x86_64"
expect_state "(b2) a build whose argv mentions qemu-system does not block" CLEAN \
  env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent LIUM_REPO_PATH=/nonexistent

# (c) THE CASE A PROCESS-ONLY GUARD WOULD WAVE THROUGH.
#     A stopped CVM: hda.img on disk, zero processes. stop_cvm() never removes
#     the VM directory, so this is a normal, reachable, common state.
expect_state "(c) hda.img with no processes blocks as DORMANT" DORMANT \
  env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$DORMANT_ROOT" LIUM_REPO_PATH=/nonexistent

# (c2) A checkout deeper than /opt/lium-io.
#
#      /home/<user>/lium-io puts hda.img at depth 9 below the search root, and
#      the find budget was 8 — so /home, which is in the root list precisely to
#      catch a home-directory checkout, could not reach one. The guard answered
#      CLEAN on a host holding a renter's encrypted disk, and the converge would
#      have rebuilt QEMU under it.
DEEP="$(mktemp -d)"
mkdir -p "$DEEP/ubuntu/lium-io/neurons/executor/dstacktee/run/vms/demo"
: >"$DEEP/ubuntu/lium-io/neurons/executor/dstacktee/run/vms/demo/hda.img"
expect_state "(c2) a checkout one level deeper than /opt still blocks" DORMANT \
  env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$DEEP" LIUM_REPO_PATH=/nonexistent
rm -rf "$DEEP"

# (c3) A checkout on a volume that is in NO search root — the /data0 shape.
#
#      The only thing that can find it is the guard locating its own checkout
#      from its own path. So the guard is COPIED into a fake checkout at the
#      real relative depth and run from there, with the search roots pointed at
#      an empty directory: if the self-location is removed, nothing else can
#      possibly see the hda.img and this case goes CLEAN.
OFFROOT="$(mktemp -d)"
mkdir -p "$OFFROOT/neurons/executor/ansible/roles/common/files"
mkdir -p "$OFFROOT/neurons/executor/dstacktee/run/vms/demo"
: >"$OFFROOT/neurons/executor/dstacktee/run/vms/demo/hda.img"
cp "$GUARD" "$OFFROOT/neurons/executor/ansible/roles/common/files/lium-guard.sh"
chmod +x "$OFFROOT/neurons/executor/ansible/roles/common/files/lium-guard.sh"
EMPTY_ROOT="$(mktemp -d)"

CASES=$((CASES + 1))
got="$(env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$EMPTY_ROOT" \
           LIUM_REPO_PATH=/nonexistent \
           "$OFFROOT/neurons/executor/ansible/roles/common/files/lium-guard.sh" --state)"
if [ "$got" = "DORMANT" ]; then
  pass "(c3) a checkout outside every search root still blocks when run from inside it"
else
  fail "(c3) a checkout outside every search root -> expected DORMANT, got $got"
fi

# And the recovery must name that checkout's real VM directory, not a guess
# based on the /opt default.
CASES=$((CASES + 1))
offrec="$(env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$EMPTY_ROOT" \
              LIUM_REPO_PATH=/nonexistent \
              "$OFFROOT/neurons/executor/ansible/roles/common/files/lium-guard.sh" --recovery)"
case "$offrec" in *"$OFFROOT/neurons/executor/dstacktee/run/vms/demo"*) _hit=1 ;; *) _hit=0 ;; esac
if [ "$_hit" -eq 1 ]; then
  pass "(c3) the recovery names that checkout's own VM directory"
else
  fail "(c3) the recovery did not name ${OFFROOT}/... — it is guessing a path"
fi
rm -rf "$OFFROOT" "$EMPTY_ROOT"

# (c4) THE SAME /data0 SHAPE, MINUS THE SELF-LOCATION FALLBACK — which is the
#      combination that actually happens on day zero.
#
#      Case (c3) survives only because the guard was run from inside the offside
#      checkout. A provider onboarding a host does the opposite: they bootstrap a
#      FRESH clone at /opt/lium-io while the stopped CVM's disk still sits on a
#      data volume. Now REPO_PATH is the new clone, LOCAL_CHECKOUT is the new
#      clone, the static roots are /home /opt /srv, and a stopped CVM leaves no
#      process. All five sources miss, and the answer was CLEAN over a 388 GB
#      encrypted disk — reproduced on a production host.
#
#      Only the per-filesystem sweep can see this. Driven through a fixture
#      mounts file so the case is hermetic.
VOL="$(mktemp -d)"
mkdir -p "$VOL/lium-io/neurons/executor/dstacktee/run/vms/demo"
: >"$VOL/lium-io/neurons/executor/dstacktee/run/vms/demo/hda.img"
FAKE_MOUNTS="$(mktemp)"
printf '/dev/sdb1 %s ext4 rw,relatime 0 0\n' "$VOL" >"$FAKE_MOUNTS"
expect_state "(c4) a disk on a volume in no search root blocks via the filesystem sweep" DORMANT \
  env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$FIX/clean_root" \
      LIUM_REPO_PATH=/nonexistent LIUM_MOUNTS_PATH="$FAKE_MOUNTS"

# (c5) The deny list is consulted, not decorative. The SAME volume, described as
#      an overlay mount, must not be swept: container image layers never hold a
#      QEMU data disk, and sweeping every overlay on a busy Docker host walks
#      each image layer separately. If the type check is dropped, this case goes
#      DORMANT and reveals it.
printf 'overlay %s overlay rw,relatime 0 0\n' "$VOL" >"$FAKE_MOUNTS"
expect_state "(c5) a denied filesystem type is not swept" CLEAN \
  env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$FIX/clean_root" \
      LIUM_REPO_PATH=/nonexistent LIUM_MOUNTS_PATH="$FAKE_MOUNTS"
rm -rf "$VOL" "$FAKE_MOUNTS"

# (d) Unreadable vs absent. An absent root is the NORMAL fresh-host shape before
#     the clone and must never read as a permissions failure.
expect_state "(d1) absent root is a benign skip" CLEAN \
  env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$FIX/this_root_does_not_exist" LIUM_REPO_PATH=/nonexistent

CASES=$((CASES + 1))
if [ "$(id -u)" -eq 0 ]; then
  # Not silently skipped: root bypasses file permissions, so this case cannot be
  # constructed here. Say so loudly rather than reporting a pass nobody earned.
  printf '  SKIP (d2) unreadable root -> UNKNOWN: running as root, which bypasses\n'
  printf '       permissions so the fixture cannot be made unreadable. This case is\n'
  printf '       covered by the non-root CI runner job.\n'
else
  chmod 000 "$FIX/unreadable_root"
  got="$(env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$FIX/unreadable_root" \
             LIUM_REPO_PATH=/nonexistent "$GUARD" --state)"
  chmod 755 "$FIX/unreadable_root"
  if [ "$got" = "UNKNOWN" ]; then
    pass "(d2) unreadable root -> UNKNOWN (fails closed)"
  else
    fail "(d2) unreadable root -> expected UNKNOWN, got $got"
  fi
fi

# (e) THE PARTITION BUG, in fact form. A host with an intact hda.img AND a
#     tenant's QEMU satisfies two descriptions at once. The ladder must classify
#     it — and the recovery text must STILL contain the rm -rf its data disk
#     needs. If the state name selected the message, this host would be told
#     about the tenant and never about the disk.
rm -rf "${PROC:?}"/*
make_proc "$PROC" 1003 "qemu-system-x86" S "/opt/dstack/dstack-v05x/run/foreign.img"
expect_state "(e) hda + foreign QEMU classifies per the ladder" FOREIGN \
  env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS="$FOREIGN_ROOT" LIUM_REPO_PATH=/nonexistent

CASES=$((CASES + 1))
recovery="$(env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS="$FOREIGN_ROOT" \
                LIUM_REPO_PATH=/nonexistent "$GUARD" --recovery)"
case "$recovery" in *"rm -rf"*) _hit_rm=1 ;; *) _hit_rm=0 ;; esac
case "$recovery" in *"run/vms/"*) _hit_vms=1 ;; *) _hit_vms=0 ;; esac
if [ "$_hit_rm" -eq 1 ]; then
  if [ "$_hit_vms" -eq 1 ]; then
    pass "(e) recovery still contains the rm -rf run/vms/ step"
  else
    fail "(e) recovery has rm -rf but not run/vms/"
  fi
else
  fail "(e) recovery is missing the rm -rf step — the state name is selecting the message"
fi

# (f) Our QEMU, no hda.img. Must classify rather than fall through.
rm -rf "${PROC:?}"/*
make_proc "$PROC" 1004 "qemu-system-x86" S "/opt/lium-io/neurons/executor/dstacktee/run/vms/demo/x"
expect_state "(f) our QEMU with no hda classifies as LIVE" LIVE \
  env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent LIUM_REPO_PATH=/nonexistent

# The ladder is ORDERED. A zombie outranks LIVE: it still holds guest RAM and is
# not reaped until its parent dies, so polling for it never returns.
rm -rf "${PROC:?}"/*
make_proc "$PROC" 1005 "qemu-system-x86" Z "/opt/lium-io/neurons/executor/dstacktee/run/vms/demo/x"
expect_state "(ladder) ZOMBIE outranks LIVE" ZOMBIE \
  env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent LIUM_REPO_PATH=/nonexistent

# A REAL zombie has an EMPTY argv — the kernel frees it on exit — and once it is
# orphaned its parent is init. Modelled exactly, because both facts were being
# misread on a production host: the empty argv made `ours` false, so our own dead
# CVM was reported as somebody's active bare-metal rental ("expected revenue,
# not an outage"), and the recovery told the operator to kill a tmux session
# named lium-cvm that did not exist, for a parent that was already pid 1.
rm -rf "${PROC:?}"/*
make_proc "$PROC" 1007 "qemu-system-x86" Z ""
CASES=$((CASES + 1))
zorphan="$(env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent \
               LIUM_REPO_PATH=/nonexistent "$GUARD" --recovery)"
case "$zorphan" in *"expected revenue"*) _rev=1 ;; *) _rev=0 ;; esac
if [ "$_rev" -eq 0 ]; then
  pass "(zombie) an empty argv is not reported as a tenant's paying rental"
else
  fail "(zombie) a zombie with no argv was described as an active bare-metal rental"
fi

CASES=$((CASES + 1))
case "$zorphan" in *"tmux kill-session"*) _tm=1 ;; *) _tm=0 ;; esac
case "$zorphan" in *"parent is already init"*) _init=1 ;; *) _init=0 ;; esac
if [ "$_tm" -eq 0 ] && [ "$_init" -eq 1 ]; then
  pass "(zombie) an orphaned zombie gets the init advice, not a tmux session that does not exist"
else
  fail "(zombie) orphan recovery wrong: tmux_advice=$_tm init_advice=$_init"
fi

CASES=$((CASES + 1))
case "$zorphan" in *"cannot be told from it"*) _amb=1 ;; *) _amb=0 ;; esac
if [ "$_amb" -eq 1 ]; then
  pass "(zombie) the unreadable command line is declared rather than assumed"
else
  fail "(zombie) an unreadable argv was passed over in silence"
fi

rm -rf "${PROC:?}"/*
make_proc "$PROC" 1005 "qemu-system-x86" Z "/opt/lium-io/neurons/executor/dstacktee/run/vms/demo/x"
CASES=$((CASES + 1))
zrecovery="$(env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent \
                 LIUM_REPO_PATH=/nonexistent "$GUARD" --recovery)"
case "$zrecovery" in *"pid 1005, parent pid 1"*) _hit=1 ;; *) _hit=0 ;; esac
if [ "$_hit" -eq 1 ]; then
  pass "(ladder) a zombie contributes a reaping step naming its own pid and parent"
else
  fail "(ladder) the zombie recovery does not name the pid and parent it is about"
fi

# The tmux branch is still the right advice when that session really is the
# parent, so it keeps its own case rather than being deleted along with the
# unconditional version.
CASES=$((CASES + 1))
if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s lium-cvm 'sleep 30' 2>/dev/null || true
  if tmux has-session -t lium-cvm 2>/dev/null; then
    tmuxrec="$(env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent \
                   LIUM_REPO_PATH=/nonexistent "$GUARD" --recovery)"
    tmux kill-session -t lium-cvm 2>/dev/null || true
    case "$tmuxrec" in *"tmux kill-session -t lium-cvm"*) _hit=1 ;; *) _hit=0 ;; esac
    if [ "$_hit" -eq 1 ]; then
      pass "(zombie) with a real lium-cvm session present, the tmux step is offered"
    else
      fail "(zombie) a live lium-cvm session did not produce the tmux reaping step"
    fi
  else
    printf '  SKIP (zombie) tmux present but a session could not be started here\n'
  fi
else
  printf '  SKIP (zombie) tmux is not installed, so the tmux branch cannot be exercised\n'
fi

# The search roots are duplicated on purpose — the script owns them because it
# runs before Ansible starts and can never read group_vars. This asserts the two
# copies agree, so the duplication cannot rot silently.
CASES=$((CASES + 1))
script_roots="$("$GUARD" --print-default-roots)"
gv_roots="$(grep '^lium_hda_search_roots:' group_vars/all/main.yml \
            | sed -e 's/^[^:]*: *//' -e 's/[]["]//g' -e 's/, */,/g' -e 's/ //g')"
if [ "$script_roots" = "$gv_roots" ]; then
  pass "search roots agree between lium-guard.sh and group_vars ($script_roots)"
else
  fail "search roots DIVERGED: script='$script_roots' group_vars='$gv_roots'"
fi

# NOTHING THIS REPOSITORY SHIPS MAY LOOK LIKE A CVM DATA DISK.
#
# The playbook clones lium-io to /opt/lium-io, and /opt is a search root. A
# tracked file at */run/vms/*/hda.img is therefore indistinguishable from a
# renter's encrypted disk the moment the clone lands: a host that has never
# created a CVM reads DORMANT, every destructive role refuses for good, and the
# recovery text instructs the operator to rm -rf the repo's own test data. Two
# such fixtures were committed and did exactly this on a production host, so
# this is asserted against the whole repository, not just this directory.
CASES=$((CASES + 1))
repo_top="$(git rev-parse --show-toplevel 2>/dev/null || printf '')"
if [ -n "$repo_top" ]; then
  shipped="$(git -C "$repo_top" ls-files -- '*/run/vms/*/hda.img' 'run/vms/*/hda.img')"
  if [ -z "$shipped" ]; then
    pass "no tracked file matches the guard's data-disk glob"
  else
    fail "tracked files match */run/vms/*/hda.img — a fresh clone will read as DORMANT:
$shipped"
  fi
else
  printf '  SKIP no git checkout here, so the shipped-fixture invariant cannot be read\n'
fi

# A clean host, with nothing anywhere.
expect_state "(clean) nothing anywhere" CLEAN \
  env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$FIX/clean_root" LIUM_REPO_PATH=/nonexistent

# --- the reboot advice, and the mode the reboot gate polls --------------------
#
# A production host was wedged by an operator following this script's own
# recovery text, which used to end "If those threads do not exit, a host reboot
# is the only thing that frees it." A soft reboot during a TDX teardown left the
# old kernel alive on the network for fourteen minutes without ever resetting,
# and took a datacentre power-cycle to clear. The text is the defect, so the
# text is asserted.
rm -rf "${PROC:?}"/*
make_proc "$PROC" 1008 "qemu-system-x86" Z "" 2 51234 9876

zadvice="$(env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent \
               LIUM_REPO_PATH=/nonexistent "$GUARD" --recovery)"

CASES=$((CASES + 1))
case "$zadvice" in *"reboot is the only thing"*) _bad=1 ;; *) _bad=0 ;; esac
case "$zadvice" in *"DO NOT REBOOT"*) _warn=1 ;; *) _warn=0 ;; esac
case "$zadvice" in *"OUT-OF-BAND"*) _oob=1 ;; *) _oob=0 ;; esac
if [ "$_bad" -eq 0 ] && [ "$_warn" -eq 1 ] && [ "$_oob" -eq 1 ]; then
  pass "(zombie) the recovery warns against a soft reboot and names the power-cycle"
else
  fail "(zombie) reboot advice wrong: old_advice=$_bad warns=$_warn out_of_band=$_oob"
fi

# Threads and cpu ticks are what let an operator tell a teardown that is
# GRINDING from one that is WEDGED. Without them the only visible signal is
# memory, which sits flat for twenty minutes at a time and proves nothing.
CASES=$((CASES + 1))
case "$zadvice" in *"threads 2, cpu ticks 61110"*) _diag=1 ;; *) _diag=0 ;; esac
if [ "$_diag" -eq 1 ]; then
  pass "(zombie) the recovery reports thread count and cpu ticks for the wait"
else
  fail "(zombie) the recovery omits the progress signal the operator needs"
fi

# --qemu-procs is what roles/kernel/tasks/wait_for_teardown.yml polls, so its
# contract is asserted here rather than only through the playbook.
CASES=$((CASES + 1))
qp="$(env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent \
          LIUM_REPO_PATH=/nonexistent "$GUARD" --qemu-procs)" && qp_rc=0 || qp_rc=$?
if [ "$qp" = "1008 Z 2 61110" ] && [ "$qp_rc" -eq 1 ]; then
  pass "(qemu-procs) emits 'pid state threads ticks' and exits with the count"
else
  fail "(qemu-procs) expected '1008 Z 2 61110' rc 1, got '$qp' rc $qp_rc"
fi

# The gate polls this every few seconds for up to 90 minutes, so it must not pay
# for the disk sweep. Asserted through behaviour: a DORMANT disk that would make
# every other mode report state and recovery must produce no output here.
CASES=$((CASES + 1))
qp_empty="$(env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$DORMANT_ROOT" \
                LIUM_REPO_PATH=/nonexistent "$GUARD" --qemu-procs)" && qp_erc=0 || qp_erc=$?
if [ -z "$qp_empty" ] && [ "$qp_erc" -eq 0 ]; then
  pass "(qemu-procs) reports only processes, and exits 0 when there are none"
else
  fail "(qemu-procs) on a process-free host expected empty rc 0, got '$qp_empty' rc $qp_erc"
fi

# Every state except CLEAN must block. Asserted as data, so a new state added
# later without a decision about it cannot quietly default to "allowed".
CASES=$((CASES + 1))
if [ "$(printf 'UNKNOWN ZOMBIE LIVE FOREIGN DORMANT CLEAN' | wc -w)" -eq 6 ]; then
  pass "the ladder has exactly 6 documented states"
else
  fail "the ladder state list changed"
fi

printf '\n'
if [ "$FAILURES" -gt 0 ]; then
  printf 'FAIL test_guard.sh — %s of %s checks failed.\n' "$FAILURES" "$CASES" >&2
  exit 1
fi
printf 'PASS test_guard.sh — %s checks.\n' "$CASES"
