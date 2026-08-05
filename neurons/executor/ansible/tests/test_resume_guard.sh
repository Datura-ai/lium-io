#!/usr/bin/env bash
# Merge-gate box 4, logic half.
#
# The real reboot cycle cannot run in CI — a reboot terminates the job. What CAN
# be proved here is the logic that makes the resume fire exactly once, on the
# right boot, and never on a host that is looping. The real cycle is proved on a
# human-run VM; see tests/vm/RUNBOOK.md.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GUARD=roles/kernel/files/lium-resume-guard.sh
FAILURES=0
CASES=0

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }

STATE_DIR="$(mktemp -d)"
BOOT_ID_FILE="${STATE_DIR}/current_boot_id"
trap 'rm -rf "$STATE_DIR"' EXIT

write_state() {
  cat >"${STATE_DIR}/resume.state" <<EOF
{
  "armed_boot_id": "$1",
  "armed_at": "2026-08-05T09:00:00Z",
  "attempts": $2,
  "playbook_args": "--yes",
  "bootstrap_path": "/bin/true"
}
EOF
}

run_guard() {
  LIUM_STATE_DIR="$STATE_DIR" LIUM_BOOT_ID_PATH="$BOOT_ID_FILE" \
  LIUM_MAX_CONVERGE_REBOOTS=2 "$GUARD" >/dev/null 2>&1
}

expect() {
  local label="$1" want="$2" rc=0
  CASES=$((CASES + 1))
  run_guard || rc=$?
  if [ "$want" = "allow" ] && [ "$rc" -eq 0 ]; then
    pass "$label"
  elif [ "$want" = "refuse" ] && [ "$rc" -ne 0 ]; then
    pass "$label"
  else
    fail "$label (wanted $want, rc=$rc)"
  fi
}

printf 'test_resume_guard.sh\n'
printf 'BOOT-CURRENT\n' >"$BOOT_ID_FILE"

# 1. A clean arm: a different boot, and never attempted.
write_state "BOOT-PREVIOUS" 0
rm -f "${STATE_DIR}/converge_reboots"
expect "(1) clean arm is allowed" allow

# The attempt must be CONSUMED AND PERSISTED before ExecStart runs, so a crash
# mid-run cannot produce a second attempt. Verified by reading the file back.
CASES=$((CASES + 1))
if grep -q '"attempts": 1' "${STATE_DIR}/resume.state"; then
  pass "(1) the attempt was persisted before returning success"
else
  fail "(1) attempts was not persisted — a crash mid-run could yield a second attempt"
fi

# 2. Already attempted.
expect "(2) attempts >= 1 is refused" refuse

# 3. The boot id has not changed, so nothing actually rebooted. Without this, a
#    manual `systemctl start` on the same boot re-runs the whole playbook —
#    potentially while a paying CVM is live.
write_state "BOOT-CURRENT" 0
expect "(3) unchanged boot id is refused" refuse

# 4. The converge budget is exhausted: this host is looping. "Exactly once per
#    arming" is satisfied on every iteration of a loop, so it is not the property
#    that stops one.
write_state "BOOT-PREVIOUS" 0
printf '3\n' >"${STATE_DIR}/converge_reboots"
expect "(4) an exhausted converge budget is refused" refuse

# A budget at, but not over, the cap must still allow the resume the reboot it
# was just armed for — otherwise the second legitimate reboot could never finish.
write_state "BOOT-PREVIOUS" 0
printf '2\n' >"${STATE_DIR}/converge_reboots"
expect "(4b) a budget at the cap still allows the arming it belongs to" allow

# No state file at all: nothing was armed, so there is nothing to resume.
rm -f "${STATE_DIR}/resume.state"
expect "(5) a missing resume state is refused" refuse

printf '\n'
if [ "$FAILURES" -gt 0 ]; then
  printf 'FAIL test_resume_guard.sh — %s of %s checks failed.\n' "$FAILURES" "$CASES" >&2
  exit 1
fi
printf 'PASS test_resume_guard.sh — %s checks.\n' "$CASES"
