#!/usr/bin/env bash
# Drive the parts of tests/vm/RUNBOOK.md that can be scripted.
#
# RUN THIS INSIDE A THROWAWAY VM. It reboots the machine it runs on.
#
# It cannot be one continuous script, because step 2 reboots. So it runs in
# phases, records where it got to, and is re-invoked after each reboot — which is
# also a fair imitation of what a provider actually experiences.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PHASE_FILE=/var/lib/lium-cvm/vm-test-phase
DO_REBOOT=0
DO_IDEMPOTENCE=0
DO_ARTIFACTS=0
SAFE_TAGS="preflight,gpu,qemu,sgx,repo,stubs"

while [ $# -gt 0 ]; do
  case "$1" in
    --reboot) DO_REBOOT=1 ;;
    --idempotence) DO_IDEMPOTENCE=1 ;;
    --artifacts) DO_ARTIFACTS=1 ;;
    --all) DO_REBOOT=1; DO_IDEMPOTENCE=1; DO_ARTIFACTS=1 ;;
    *) printf 'Unknown flag: %s\n' "$1" >&2; exit 64 ;;
  esac
  shift
done

if [ "$(id -u)" -ne 0 ]; then
  printf 'Run this as root: sudo %s\n' "$0" >&2
  exit 1
fi

FAILURES=0
pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }

phase() {
  if [ -r "$PHASE_FILE" ]; then cat "$PHASE_FILE"; else printf '0'; fi
}
set_phase() {
  mkdir -p "$(dirname "$PHASE_FILE")"
  printf '%s\n' "$1" >"$PHASE_FILE"
}

CURRENT="$(phase)"
printf 'run-vm-tests.sh — phase %s\n\n' "$CURRENT"

# --- phase 0: converge, and reboot if the kernel command line changed ---------
if [ "$CURRENT" = "0" ] && [ "$DO_REBOOT" -eq 1 ]; then
  printf 'Phase 0: converging the kernel role. This machine will reboot.\n'
  set_phase 1
  ./bootstrap.sh --yes --skip-tags "$SAFE_TAGS"
  # If we are still here, nothing needed a reboot.
  printf 'No reboot was required; the kernel command line was already correct.\n'
  set_phase 2
  CURRENT=2
fi

# --- phase 1: we are back from the reboot ------------------------------------
if [ "$CURRENT" = "1" ]; then
  printf 'Phase 1: checking the resume unit.\n'

  if systemctl is-enabled lium-cvm-resume >/dev/null 2>&1; then
    fail "lium-cvm-resume is still ENABLED. ExecStopPost should have disabled it on every exit path."
  else
    pass "the resume unit disabled itself"
  fi

  if [ -r /var/log/lium-cvm/resume.log ]; then
    runs="$(grep -c 'resuming' /var/log/lium-cvm/resume.log || printf '0')"
    if [ "$runs" -eq 1 ]; then
      pass "the resume ran exactly once"
    else
      fail "the resume ran $runs time(s); it must run exactly once per arming"
    fi
  else
    fail "no /var/log/lium-cvm/resume.log — the unit never ran"
  fi

  if [ -r /var/lib/lium-cvm/resume.state ]; then
    printf '  note: resume.state still present (the converge did not finish cleanly)\n'
  else
    pass "resume.state was cleared after a clean finish"
  fi

  set_phase 2
  CURRENT=2
fi

# --- phase 2: idempotence -----------------------------------------------------
if [ "$CURRENT" = "2" ] && [ "$DO_IDEMPOTENCE" -eq 1 ]; then
  printf '\nPhase 2: idempotence.\n'
  LOG=/tmp/lium-idempotence.log
  # The exit code is not the signal here: verify exits non-zero on any MUST_FIX,
  # which a plain VM always has. What matters is the changed count.
  set +e
  ./bootstrap.sh --yes --skip-tags "$SAFE_TAGS" 2>&1 | tee "$LOG"
  set -e

  changed="$(grep -oE 'changed=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)"
  if [ "${changed:-1}" -eq 0 ]; then
    pass "a second converge changed nothing (changed=0)"
  else
    fail "a second converge reported changed=${changed:-unknown}; the playbook is not idempotent"
  fi
fi

# --- phase 3: the forensic trail ---------------------------------------------
if [ "$DO_ARTIFACTS" -eq 1 ]; then
  printf '\nPhase 3: forensic artifacts.\n'
  for f in /var/log/lium-cvm/bootstrap.log /var/lib/lium-cvm/verify-report.json; do
    if [ -r "$f" ]; then
      pass "$f exists"
    else
      fail "$f is missing"
    fi
  done

  # The --tree manifest is what tells you which tasks actually changed
  # something, run by run. Without it a post-mortem has only prose.
  if compgen -G '/var/log/lium-cvm/run-*' >/dev/null; then
    pass "per-run task manifests exist ($(find /var/log/lium-cvm -maxdepth 1 -name 'run-*' | wc -l) run(s))"
  else
    fail "no /var/log/lium-cvm/run-* directories — the --tree manifest is missing"
  fi

  if [ -r /var/lib/lium-cvm/converge_reboots ]; then
    pass "the reboot budget is recorded ($(cat /var/lib/lium-cvm/converge_reboots))"
  else
    printf '  note: no converge_reboots file — no reboot has been spent yet\n'
  fi
fi

printf '\n'
if [ "$FAILURES" -gt 0 ]; then
  printf 'FAIL run-vm-tests.sh — %s check(s) failed.\n' "$FAILURES" >&2
  exit 1
fi
printf 'PASS run-vm-tests.sh. Paste this transcript into the pull request.\n'
printf 'Steps 3a, 3b and 5 of RUNBOOK.md are still manual — they need a second\n'
printf 'reboot, a forced drift and a fabricated hda.img respectively.\n'
