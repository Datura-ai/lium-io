#!/usr/bin/env bash
# Every command this tree tells an operator to run must actually be accepted.
#
# WHY THIS EXISTS
# ---------------
# Two remediation messages instructed the operator to run
# `sudo ./bootstrap.sh -e <var>=<value>`. bootstrap.sh had no `-e`, so it
# answered with `Unknown flag: -e`, a usage dump, and exit 64 — at exactly the
# moment the operator was already stuck. Advice that the entry point rejects is
# worse than no advice: it reads as "you are holding it wrong" when the tool is.
#
# The check is derived, not a fixed list. It greps the tree for the flags the
# tree itself recommends, so a message added later is covered without anyone
# remembering to extend this file.
#
# --print-expected-preflight-rc is what makes this runnable anywhere: it is
# answered in section 2 of bootstrap.sh, before the apt-get, architecture and
# OS guards, and before any side effect. So argument parsing is exercised for
# real on a developer laptop that could never run the playbook.
#
# Every assertion is an explicit if/then branch. No `!`-inverted commands: under
# `bash -e` a command preceded by `!` does not trigger errexit, so such an
# "assertion" silently always passes.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BOOTSTRAP=./bootstrap.sh
FAILURES=0
CASES=0

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }

EXPECTED_RC="$("$BOOTSTRAP" --print-expected-preflight-rc)"

# The baseline: the query itself works with no extra flags. Without this, a
# bootstrap.sh that rejected EVERYTHING would make every case below fail with a
# confusing message about -e.
CASES=$((CASES + 1))
if [ "$EXPECTED_RC" = "2" ]; then
  pass "--print-expected-preflight-rc answers before any side effect"
else
  fail "--print-expected-preflight-rc returned '$EXPECTED_RC', expected 2"
fi

# -e must be accepted, repeatable, and must not disturb the query.
CASES=$((CASES + 1))
rc=0
got="$("$BOOTSTRAP" -e lium_test_one=1 -e lium_test_two=2 --print-expected-preflight-rc)" || rc=$?
if [ "$rc" -eq 0 ] && [ "$got" = "$EXPECTED_RC" ]; then
  pass "-e is accepted and repeatable"
else
  fail "-e rejected: rc=$rc output='$got'"
fi

CASES=$((CASES + 1))
rc=0
got="$("$BOOTSTRAP" --extra-vars lium_test_one=1 --print-expected-preflight-rc)" || rc=$?
if [ "$rc" -eq 0 ] && [ "$got" = "$EXPECTED_RC" ]; then
  pass "--extra-vars is accepted as the long form"
else
  fail "--extra-vars rejected: rc=$rc output='$got'"
fi

# A value-taking flag with no value must still fail, loudly. Accepting -e must
# not turn a typo into a silently ignored argument.
CASES=$((CASES + 1))
rc=0
"$BOOTSTRAP" -e >/dev/null 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
  pass "-e with no value is refused (rc=$rc)"
else
  fail "-e with no value was accepted"
fi

# THE REAL ASSERTION: every `bootstrap.sh -e <var>=<value>` this tree prints at
# an operator must be a command bootstrap.sh accepts.
#
# tests/ is excluded so this file's own examples above are not treated as
# recommendations to the operator.
CASES=$((CASES + 1))
mapfile -t recommended < <(
  grep -rhoE "bootstrap\.sh -e [A-Za-z0-9_]+=[A-Za-z0-9_.:/-]+" \
    roles group_vars site.yml bootstrap.sh 2>/dev/null \
  | sed -E 's/^bootstrap\.sh -e //' | sort -u
)

if [ "${#recommended[@]}" -eq 0 ]; then
  fail "found no 'bootstrap.sh -e <var>=<value>' recommendations to check — the grep has stopped matching"
else
  bad=""
  for ev in "${recommended[@]}"; do
    rc=0
    "$BOOTSTRAP" -e "$ev" --print-expected-preflight-rc >/dev/null 2>&1 || rc=$?
    [ "$rc" -ne 0 ] && bad="${bad}    ${ev} (rc=${rc})"$'\n'
  done
  if [ -z "$bad" ]; then
    pass "all ${#recommended[@]} recommended -e invocations are accepted"
  else
    fail "the tree recommends invocations bootstrap.sh rejects:
$bad"
  fi
fi

# --- the supported-release list is duplicated, so pin the duplication ---------
#
# bootstrap.sh gates the OS before ansible-core exists, so it cannot read
# group_vars and has to carry its own copy. Same shape as the guard's search
# roots in test_guard.sh: the duplication is legitimate, the DRIFT is not.
#
# Adding a release to lium_os_matrix without adding it here leaves the entry
# point rejecting a host it is meant to support, with a message that contradicts
# preflight — and the run dies before any playbook output exists to explain it.
CASES=$((CASES + 1))
mapfile -t script_releases < <("$BOOTSTRAP" --print-supported-releases)

# The keys of lium_os_matrix: quoted, two-space-indented mapping names directly
# under it. Anchored on the indent so a nested key can never be read as a
# release, and range-limited so a later block's keys cannot leak in.
#
# sed, not awk. Ubuntu's default awk is mawk, and the three-argument match()
# with a capture array that this wants is a gawk extension — it would fail
# on the very container the supported-OS CI job runs in.
mapfile -t gv_releases < <(
  sed -n '/^lium_os_matrix:/,/^[^[:space:]#]/ s/^  "\([^"]*\)":.*/\1/p' \
    group_vars/all/main.yml | sort
)

mapfile -t script_sorted < <(printf '%s\n' "${script_releases[@]}" | sort)

if [ "${#gv_releases[@]}" -eq 0 ]; then
  fail "found no releases under lium_os_matrix in group_vars/all/main.yml — the parser has stopped matching"
elif [ "${script_sorted[*]}" = "${gv_releases[*]}" ]; then
  pass "bootstrap.sh and lium_os_matrix agree on the supported releases (${script_sorted[*]})"
else
  fail "supported-release drift:
    bootstrap.sh SUPPORTED_VERSIONS : ${script_sorted[*]}
    group_vars lium_os_matrix       : ${gv_releases[*]}
  Both must list the same releases. bootstrap.sh gates before ansible-core is
  installed, so it cannot read group_vars and must carry its own copy."
fi

printf '\n'
if [ "$FAILURES" -gt 0 ]; then
  printf 'FAIL test_bootstrap_args.sh — %s of %s checks failed.\n' "$FAILURES" "$CASES" >&2
  exit 1
fi
printf 'PASS test_bootstrap_args.sh — %s checks.\n' "$CASES"
