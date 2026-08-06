#!/usr/bin/env bash
# The fact collector must return the SAME answer for the same host, every time.
#
# THE DEFECT THIS EXISTS TO CATCH
# -------------------------------
# present_pci_ids was built with `printf '%s' "$LSPCI_TEXT" | grep -qF "$id"`.
# `grep -q` exits the moment it matches, printf loses the rest of its write to
# SIGPIPE and returns 141, and `set -o pipefail` makes the pipeline report that
# 141 — so a SUCCESSFUL match reads as "not found".
#
# It is a race against the scheduler, so it is not reproducible by running the
# thing once. On a live 8xH200 host, 30 runs with no change between them
# returned the correct pair of ids 20 times, a single id 8 times, and an empty
# list twice, while `lspci -n` listed all 12 devices on every one of those runs.
#
# present_pci_ids is what `vfio-pci.ids=` is built from, in the kernel role that
# writes GRUB and in the verify role that audits it. A run that loses 10de:2335
# persists a boot line binding only the NVSwitches, and after the reboot not one
# of the eight GPUs can be given to a CVM. Nothing fails loudly; the host simply
# stops earning.
#
# The fixture is deliberately LARGE. The bug's window is the gap between grep
# matching and printf finishing its write, so a handful of lines fits in the
# pipe buffer, printf wins, and the old code passes. Thousands of lines with the
# ids near the top makes the old code fail essentially always.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

FACTS=roles/common/files/lium-facts.sh
ITERATIONS="${LIUM_TEST_ITERATIONS:-25}"
FAILURES=0
CASES=0

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

printf 'test_facts_determinism.sh\n'

# An 8xH200 + 4xNVSwitch host, padded to a size no pipe buffer swallows whole.
LSPCI="$WORK/lspci.txt"
{
  printf '00:00.0 0600: 8086:09a2\n'
  for bdf in 05 06 07 08; do printf '%s:00.0 0680: 10de:22a3 (rev a1)\n' "$bdf"; done
  for bdf in 0f 34 48 5a 87 ae c2 d7; do printf '%s:00.0 0302: 10de:2335 (rev a1)\n' "$bdf"; done
  for i in $(seq 1 4000); do printf 'ff:%02x.0 0c03: 1b36:000d\n' $((i % 256)); done
} >"$LSPCI"

run_ids() {
  # The collector probes Intel's API with a 6 s timeout. Pointed at a closed
  # local port it answers instantly with the same "false", which keeps this loop
  # to a few seconds instead of six per iteration.
  env LIUM_INTEL_API_HOST=127.0.0.1 \
      LIUM_INTEL_API_PORT=1 \
      LIUM_LSPCI_OUTPUT="$LSPCI" \
      LIUM_STATE_DIR="$WORK/state" \
      LIUM_REPO_PATH="$WORK/norepo" \
      LIUM_NVTRUST_TOOL="$WORK/no-nvtrust" \
      LIUM_PROC_CMDLINE_PATH="$WORK/cmdline" \
      LIUM_GRUB_DEFAULT_PATH="$WORK/grub" \
      LIUM_GRUB_DROPIN_DIR="$WORK/grub.d" \
      "$FACTS" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["present_pci_ids"], len(d["gpu_bdfs"]), len(d["nvswitch_bdfs"]))'
}

: >"$WORK/cmdline"
: >"$WORK/grub"
mkdir -p "$WORK/grub.d" "$WORK/state"

RESULTS="$WORK/results.txt"
: >"$RESULTS"
for _ in $(seq 1 "$ITERATIONS"); do
  run_ids >>"$RESULTS"
done

DISTINCT="$(sort -u "$RESULTS" | wc -l)"
CASES=$((CASES + 1))
if [ "$DISTINCT" -eq 1 ]; then
  pass "$ITERATIONS runs over one fixture agree ($(head -n 1 "$RESULTS"))"
else
  fail "the collector is NON-DETERMINISTIC — $DISTINCT distinct answers over $ITERATIONS runs:
$(sort "$RESULTS" | uniq -c)"
fi

# Agreeing on the WRONG answer every time would satisfy the check above.
EXPECTED="['10de:2335', '10de:22a3'] 8 4"
CASES=$((CASES + 1))
if [ "$(head -n 1 "$RESULTS")" = "$EXPECTED" ]; then
  pass "every id present in the fixture is reported, with the right device counts"
else
  fail "expected <$EXPECTED>, got <$(head -n 1 "$RESULTS")>"
fi

printf '\n'
if [ "$FAILURES" -gt 0 ]; then
  printf 'FAIL test_facts_determinism.sh — %s of %s checks failed.\n' "$FAILURES" "$CASES" >&2
  exit 1
fi
printf 'PASS test_facts_determinism.sh — %s checks.\n' "$CASES"
