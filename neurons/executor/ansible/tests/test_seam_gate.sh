#!/usr/bin/env bash
# Test seams must be unreachable in production.
#
# Every seam redirects a decision away from the real host — a fake /proc/cmdline,
# a stubbed update-grub, a fixture lspci. One reachable by accident on a provider
# machine would make the playbook converge against something that is not the
# host it is running on.
#
# The gate requires BOTH `lium_test_mode: true` AND the environment variable
# ANSIBLE_LIUM_TEST=1. Either alone must fail closed. This asserts all four
# combinations, so a change that makes one of them sufficient is caught.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

FAILURES=0
CASES=0
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }

run_probe() {
  local test_mode="$1" env_flag="$2"
  ANSIBLE_LIUM_TEST="$env_flag" \
    ansible-playbook -i inventory/localhost.yml \
      -e "lium_test_mode=${test_mode}" \
      tests/_seam_probe.yml >"$WORK/out.log" 2>&1
}

expect() {
  local label="$1" want="$2" test_mode="$3" env_flag="$4" rc=0
  CASES=$((CASES + 1))
  run_probe "$test_mode" "$env_flag" || rc=$?
  if [ "$want" = "allow" ] && [ "$rc" -eq 0 ]; then
    pass "$label"
  elif [ "$want" = "refuse" ] && [ "$rc" -ne 0 ]; then
    if grep -q 'Test seam' "$WORK/out.log"; then
      pass "$label (refused by the seam gate, by name)"
    else
      fail "$label refused, but NOT by the seam gate — check why it failed"
    fi
  else
    fail "$label (wanted $want, rc=$rc)"
  fi
}

printf 'test_seam_gate.sh\n'
rm -rf /tmp/lium-seam-probe

expect "(1) neither gate open is refused"            refuse false ""
expect "(2) lium_test_mode alone is refused"         refuse true  ""
expect "(3) ANSIBLE_LIUM_TEST alone is refused"      refuse false "1"
expect "(4) both gates open is allowed"              allow  true  "1"

printf '\n'
if [ "$FAILURES" -gt 0 ]; then
  printf 'FAIL test_seam_gate.sh — %s of %s checks failed.\n' "$FAILURES" "$CASES" >&2
  exit 1
fi
printf 'PASS test_seam_gate.sh — %s checks.\n' "$CASES"
