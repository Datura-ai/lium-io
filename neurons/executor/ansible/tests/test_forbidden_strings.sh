#!/usr/bin/env bash
# Enforce tests/forbidden-patterns.txt.
#
# This is the ONLY mechanism in this tree that actually prevents a banned flag
# from reappearing. `--check --diff` cannot do it: check mode skips command and
# shell tasks entirely, so Ansible never sees what they would have run.
#
# Every assertion below is an explicit `if <condition>; then ...; exit 1; fi`.
# There is deliberately no `! grep` anywhere — under `bash -e` a command
# preceded by `!` does not trigger the errexit trap, so such an "assertion"
# silently always passes.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PATTERN_FILE="tests/forbidden-patterns.txt"
WORKFLOWS_DIR="../../../.github/workflows"

FAILURES=0

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  FAILURES=$((FAILURES + 1))
}

# Search scopes. The pattern file, every README, and the whole tests/ tree are
# excluded — see the header of forbidden-patterns.txt for why that is a
# correctness requirement and not a convenience.
tree_files() {
  find roles -type d \( -name tasks -o -name files -o -name templates \) -exec find {} -type f \; 2>/dev/null
  find group_vars -type f 2>/dev/null
  printf '%s\n' bootstrap.sh site.yml
}

ci_files() {
  for f in "$WORKFLOWS_DIR/test.yml" "$WORKFLOWS_DIR/qemu-2604-compile.yml"; do
    [ -f "$f" ] && printf '%s\n' "$f"
  done
}

# Emit `path:lineno:content` for every line that is not a whole-line comment.
#
# Whole-line only, on purpose. A trailing-comment stripper would have to know
# where a `#` is quoted, and getting that wrong in the permissive direction is
# how a real command hides behind a fake comment.
strip_comments() {
  local file
  for file in "$@"; do
    [ -f "$file" ] || continue
    awk -v f="$file" '
      { line = $0
        sub(/^[[:space:]]+/, "", line)
        if (line ~ /^#/) next          # shell and YAML comment
        if (line ~ /^\{#/) next        # Jinja comment opener
        if (line ~ /^-#\}/) next       # Jinja comment closer
        print f ":" NR ":" $0
      }' "$file"
  done
}

files_for_scope() {
  case "$1" in
    tree) tree_files ;;
    ci) ci_files ;;
    all) tree_files; ci_files ;;
  esac
}

[ -r "$PATTERN_FILE" ] || { printf 'FAIL: %s is missing\n' "$PATTERN_FILE" >&2; exit 1; }

RULES=0

while IFS=$'\t' read -r sense kind scope pattern why; do
  case "$sense" in
    ''|'#'*) continue ;;
  esac
  [ -n "${pattern:-}" ] || continue
  RULES=$((RULES + 1))

  mapfile -t targets < <(files_for_scope "$scope" | sort -u)
  # A scope with no files is a failure, never a skip. The CI-scoped rules exist
  # because the workflow files sit OUTSIDE this job's working directory, and an
  # earlier version of this rule was unenforced against exactly the files that
  # had the defect. Silently passing when they cannot be found would recreate
  # that hole the first time somebody moved or renamed one.
  if [ "${#targets[@]}" -eq 0 ]; then
    fail "no files found for scope '$scope' — rule for '$pattern' is unenforced"
    continue
  fi

  case "$kind" in
    fixed) GREP_ARGS=(-F) ;;
    regex) GREP_ARGS=(-E) ;;
    *) fail "unknown pattern kind '$kind' for '$pattern'"; continue ;;
  esac

  if [ "$sense" = "deny" ]; then
    # Deny rules match EFFECTIVE content only: comment lines are stripped first.
    #
    # The thing being prevented is the playbook DOING one of these, not a
    # comment saying "never do this" — and those comments are exactly where the
    # field evidence for each ban is recorded, right next to the code it
    # constrains. Stripping them removes no protection: every command this tree
    # runs is built in group_vars/all/commands.yml or in a task's `cmd:`, none
    # of which is a comment.
    #
    # `require` rules deliberately do NOT strip comments — see below.
    if hits="$(strip_comments "${targets[@]}" | grep "${GREP_ARGS[@]}" -n -- "$pattern" 2>/dev/null)"; then
      fail "banned pattern '$pattern' found in effective (non-comment) content:"
      printf '%s\n' "$hits" | sed 's/^/    /' >&2
      printf '  why: %s\n' "$why" >&2
    fi
  else
    if grep "${GREP_ARGS[@]}" -q -- "$pattern" "${targets[@]}" 2>/dev/null; then
      : # present, as required
    else
      fail "required pattern '$pattern' is MISSING from scope '$scope'"
      printf '  why: %s\n' "$why" >&2
    fi
  fi
done <"$PATTERN_FILE"

if [ "$RULES" -lt 10 ]; then
  printf 'FAIL: only %s rules were parsed from %s; the pattern file looks truncated\n' \
    "$RULES" "$PATTERN_FILE" >&2
  exit 1
fi

if [ "$FAILURES" -gt 0 ]; then
  printf '\n%s rule(s) failed.\n' "$FAILURES" >&2
  exit 1
fi

printf 'PASS test_forbidden_strings.sh — %s rules enforced.\n' "$RULES"
