#!/usr/bin/env bash
# Merge-gate box 8. The docs were actually edited as claimed.
#
# Two failure modes matter here and they pull in opposite directions:
#   - the manual sections survive, so providers follow a procedure the playbook
#     has already replaced and diverges from;
#   - the retirement takes the sections that were NOT meant to go with it.
# So this asserts both that the old ones are gone and that the kept ones remain.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

HOST_SETUP=../dstacktee/docs/host-setup.md
DSTACKTEE_README=../dstacktee/README.md
DOCKERIGNORE=../.dockerignore

FAILURES=0
CASES=0

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }

assert_absent() {
  local label="$1" pattern="$2" file="$3"
  CASES=$((CASES + 1))
  if grep -qE "$pattern" "$file"; then
    fail "$label — '$pattern' is still in $file"
  else
    pass "$label"
  fi
}

assert_present() {
  local label="$1" pattern="$2" file="$3"
  CASES=$((CASES + 1))
  if grep -qE "$pattern" "$file"; then
    pass "$label"
  else
    fail "$label — '$pattern' is missing from $file"
  fi
}

printf 'test_docs_pointer.sh\n'

for f in "$HOST_SETUP" "$DSTACKTEE_README" "$DOCKERIGNORE"; do
  if [ ! -r "$f" ]; then
    printf '  FAIL %s is missing\n' "$f" >&2
    exit 1
  fi
done

# The four manual bring-up sections are gone.
assert_absent "host-setup.md section 1 heading removed" '^## 1\. ' "$HOST_SETUP"
assert_absent "host-setup.md section 2 heading removed" '^## 2\. ' "$HOST_SETUP"
assert_absent "host-setup.md section 3 heading removed" '^## 3\. ' "$HOST_SETUP"
assert_absent "host-setup.md section 4 heading removed" '^## 4\. ' "$HOST_SETUP"

# Replaced by a pointer that actually tells a provider what to run.
assert_present "host-setup.md has the automated bring-up section" '^## 1.4\. Host bring-up . automated' "$HOST_SETUP"
assert_present "host-setup.md has the bootstrap command" 'sudo \./bootstrap\.sh' "$HOST_SETUP"
assert_present "host-setup.md points at the playbook README" 'ansible/README\.md' "$HOST_SETUP"

# BIOS is the one thing the playbook cannot do, so the doc must still say so.
assert_present "host-setup.md still names the TDX BIOS setting" 'TDX' "$HOST_SETUP"
assert_present "host-setup.md still names the SGX BIOS setting" 'SGX' "$HOST_SETUP"
assert_present "host-setup.md still names VT-d" 'VT-d' "$HOST_SETUP"

# Everything that was NOT retired must survive.
assert_present "host-setup.md keeps section 5" '^## 5\. ' "$HOST_SETUP"
assert_present "host-setup.md keeps section 6" '^## 6\. ' "$HOST_SETUP"
assert_present "host-setup.md keeps Troubleshooting" '^## Troubleshooting' "$HOST_SETUP"

# Section 6 carries the warning that the data on an old disk is unrecoverable.
# That is the sentence the whole destructive guard exists to honour.
assert_present "host-setup.md keeps the data-disk warning" 'unrecoverable|undecryptable' "$HOST_SETUP"

# First-time providers are sent to the playbook, not the retired procedure.
assert_present "dstacktee README points at the bootstrap command" 'bootstrap\.sh' "$DSTACKTEE_README"

# The executor Dockerfile does `COPY . .`, so the playbook must be excluded from
# the build context — and the pattern has to be on its OWN line. The file ended
# without a trailing newline, so a naive append produced `dstacktee/ansible/`,
# which matches nothing.
assert_present "dockerignore excludes the ansible tree" '^ansible/$' "$DOCKERIGNORE"
assert_present "dockerignore still excludes dstacktee" '^dstacktee/$' "$DOCKERIGNORE"
assert_absent  "dockerignore has no merged dstacktee/ansible pattern" 'dstacktee/ansible' "$DOCKERIGNORE"

printf '\n'
if [ "$FAILURES" -gt 0 ]; then
  printf 'FAIL test_docs_pointer.sh — %s of %s checks failed.\n' "$FAILURES" "$CASES" >&2
  exit 1
fi
printf 'PASS test_docs_pointer.sh — %s checks.\n' "$CASES"
