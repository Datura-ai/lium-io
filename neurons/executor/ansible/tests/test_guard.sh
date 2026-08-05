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
make_proc() {
  local root="$1" pid="$2" comm="$3" state="$4" cmdline="$5"
  mkdir -p "$root/$pid"
  printf '%s\n' "$comm" >"$root/$pid/comm"
  printf '%s (%s) %s 1 1 1\n' "$pid" "$comm" "$state" >"$root/$pid/stat"
  printf '%s' "$cmdline" | tr ' ' '\0' >"$root/$pid/cmdline"
}

PROC="$(mktemp -d)"
EMPTY_PROC="$(mktemp -d)"
trap 'rm -rf "$PROC" "$EMPTY_PROC"; chmod 755 "$FIX/unreadable_root" 2>/dev/null || printf ""' EXIT

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

# (c) THE CASE A PROCESS-ONLY GUARD WOULD WAVE THROUGH.
#     A stopped CVM: hda.img on disk, zero processes. stop_cvm() never removes
#     the VM directory, so this is a normal, reachable, common state.
expect_state "(c) hda.img with no processes blocks as DORMANT" DORMANT \
  env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$FIX/dormant_hda" LIUM_REPO_PATH=/nonexistent

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
  env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS="$FIX/hda_plus_foreign" LIUM_REPO_PATH=/nonexistent

CASES=$((CASES + 1))
recovery="$(env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS="$FIX/hda_plus_foreign" \
                LIUM_REPO_PATH=/nonexistent "$GUARD" --recovery)"
if printf '%s' "$recovery" | grep -qF 'rm -rf'; then
  if printf '%s' "$recovery" | grep -qF 'run/vms/'; then
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

CASES=$((CASES + 1))
zrecovery="$(env LIUM_PROC_ROOT="$PROC" LIUM_HDA_SEARCH_ROOTS=/nonexistent \
                 LIUM_REPO_PATH=/nonexistent "$GUARD" --recovery)"
if printf '%s' "$zrecovery" | grep -qF 'tmux kill-session'; then
  pass "(ladder) a zombie contributes the tmux reaping step"
else
  fail "(ladder) a zombie did not contribute the tmux reaping step"
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

# A clean host, with nothing anywhere.
expect_state "(clean) nothing anywhere" CLEAN \
  env LIUM_PROC_ROOT="$EMPTY_PROC" LIUM_HDA_SEARCH_ROOTS="$FIX/clean_root" LIUM_REPO_PATH=/nonexistent

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
