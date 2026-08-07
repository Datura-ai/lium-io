#!/usr/bin/env bash
# Ordering invariants that are silent when broken.
#
# Both cases here were REAL defects, found by running the playbook on real
# hardware, and neither produced a syntax error, a lint warning, or a failing
# test. The tasks were all present and individually correct — only their order
# was wrong, which is exactly the class of defect that survives review.
#
# A structural assertion on line numbers is a blunt instrument, and it is the
# right one here: the invariant IS the order. Nothing subtler would have caught
# either bug, and both cost a full converge on a real host to discover.
#
# Every assertion is an explicit if/then branch. No `!`-inverted commands: under
# `bash -e` a command preceded by `!` does not trigger errexit, so such an
# "assertion" silently always passes.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

FAILURES=0
CASES=0

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }

# Line number of the first line matching a fixed string, or empty when absent.
#
# grep runs as an `if` CONDITION and its output is trimmed with a parameter
# expansion, so there is no pipeline and no bare failing command. Written the
# obvious way — `a="$(grep ... | cut -d: -f1)"` — a missing marker makes grep
# exit 1, pipefail propagates it, and `set -e` kills the whole script AT THE
# ASSIGNMENT: exit 1 with not one line of output, before the "marker not found"
# branch below could ever report which marker was missing. A test that fails
# silently is the thing this tree keeps finding and removing.
line_of() {
  local hit
  if hit="$(grep -n -F -m1 -- "$2" "$1" 2>/dev/null)"; then
    printf '%s' "${hit%%:*}"
  fi
}

# Assert that `first` appears before `second` in `file`. A MISSING marker is a
# failure, never a skip: renaming a task must break this test loudly rather than
# silently disable the invariant it guards.
assert_before() {
  local label="$1" file="$2" first="$3" second="$4" a b
  CASES=$((CASES + 1))
  a="$(line_of "$file" "$first")"
  b="$(line_of "$file" "$second")"
  if [ -z "$a" ]; then
    fail "$label — marker not found in $file: '$first'"
    return
  fi
  if [ -z "$b" ]; then
    fail "$label — marker not found in $file: '$second'"
    return
  fi
  if [ "$a" -lt "$b" ]; then
    pass "$label (line $a before line $b)"
  else
    fail "$label — '$first' is at line $a, AFTER '$second' at line $b"
  fi
}

GPU=roles/gpu/tasks/main.yml
KERNEL=roles/kernel/tasks/reboot.yml

# --- 1. nvtrust is installed before the gate that needs it -------------------
#
# The gate refuses when CC mode cannot be read. On a host that has never run
# this playbook it cannot be read, because the tool that reads it is nvtrust —
# which these tasks install. Ordered the other way the role refused with
# "Install or repair nvtrust first, then re-run" while the next task was the one
# that installs nvtrust, so re-running changed nothing and the converge aborted
# on every fresh host. Observed on au11.
assert_before "nvtrust is installed before CC mode is required to be readable" \
  "$GPU" "Install nvtrust at the pinned tag" "Refuse to act on an unreadable CC mode"

# The facts are collected in play 1, BEFORE that clone exists, so the bundle
# still says "nvtrust tool is not available" for every device. Without a refresh
# between the two, moving the install earlier fixes nothing: the gate would
# still judge the stale snapshot and still refuse.
assert_before "the facts are re-read after installing nvtrust" \
  "$GPU" "Re-read the host facts now that nvtrust is installed" \
  "Refuse to act on an unreadable CC mode"

assert_before "the re-read happens after the install, not before it" \
  "$GPU" "Install nvtrust at the pinned tag" \
  "Re-read the host facts now that nvtrust is installed"

# --- 2. the teardown gate runs before the reboot is scheduled ----------------
#
# A soft reboot issued into an in-flight TDX teardown does not complete: the old
# kernel stayed alive on the network for 14+ minutes without reaching platform
# reset, and only a datacentre power-cycle recovered it. A gate that runs after
# the reboot has been scheduled protects nothing.
assert_before "the teardown gate runs before the reboot is scheduled" \
  "$KERNEL" "Refuse to reboot into a live or draining TDX guest" "Schedule the reboot"

# The hold is what stops the rest of site.yml running in the seconds before the
# host goes down — without it Ansible walks into the gpu play, which flips CC
# mode across eight GPUs, and the shutdown kills it partway through.
assert_before "the run holds after scheduling the reboot" \
  "$KERNEL" "Schedule the reboot" "Hold the run while the host goes down"

assert_before "surviving the hold is treated as a failure" \
  "$KERNEL" "Hold the run while the host goes down" "The scheduled reboot never happened"

printf '\n'
if [ "$FAILURES" -gt 0 ]; then
  printf 'FAIL test_task_ordering.sh — %s of %s checks failed.\n' "$FAILURES" "$CASES" >&2
  exit 1
fi
printf 'PASS test_task_ordering.sh — %s checks.\n' "$CASES"
