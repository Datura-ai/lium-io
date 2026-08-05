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
#
# MATCHING IS PER FILE, ON RAW LINES.
#
# An earlier version piped every file through a comment stripper that prefixed
# each line with "path:lineno:" and then grepped THAT. Two of the rules are
# anchored with `^` — and those are precisely the two `!`-inverted rules the
# whole exercise exists to enforce — so the anchor bound to the file path and
# they could never fire. The rules that mattered most were the ones that were
# inert.
#
# grep is therefore run against each file directly, where `^` means what it
# says, and comments are filtered out of the MATCHES afterwards.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PATTERN_FILE="tests/forbidden-patterns.txt"
WORKFLOWS_DIR="../../../.github/workflows"
REQUIRED_WORKFLOWS=(test.yml qemu-2604-compile.yml)

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
  printf '%s\n' site.yml
}

bootstrap_files() {
  printf '%s\n' bootstrap.sh
}

# The command catalogue and the variable file, named separately so a `require`
# rule can insist a value lives WHERE IT IS USED. Scoped at `tree` these rules
# were satisfied by any non-comment mention anywhere — including a runtime error
# message that happens to quote the flag — which made them decorative.
commands_file() {
  printf '%s\n' group_vars/all/commands.yml
}

groupvars_files() {
  find group_vars -type f 2>/dev/null
}

ci_files() {
  local f
  for f in "${REQUIRED_WORKFLOWS[@]}"; do
    printf '%s\n' "${WORKFLOWS_DIR}/${f}"
  done
}

files_for_scope() {
  case "$1" in
    tree) tree_files; bootstrap_files ;;
    bootstrap) bootstrap_files ;;
    commands) commands_file ;;
    groupvars) groupvars_files ;;
    ci) ci_files ;;
    all) tree_files; bootstrap_files; ci_files ;;
  esac
}

# Is this line a whole-line comment?
#
# Whole-line only, on purpose. A trailing-comment stripper would have to know
# where a `#` is quoted, and getting that wrong in the permissive direction is
# how a real command hides behind a fake comment.
is_comment_line() {
  local line="${1#"${1%%[![:space:]]*}"}"   # strip leading whitespace
  case "$line" in
    '#'*) return 0 ;;      # shell and YAML comment
    '{#'*) return 0 ;;     # Jinja comment opener
    '-#}'*) return 0 ;;    # Jinja comment closer
    *) return 1 ;;
  esac
}

# Every match of PATTERN in FILE that is not on a comment line.
# Prints "file:lineno:content". grep is given the file directly, so an anchored
# pattern anchors on the CODE.
matches_in_file() {
  local kind="$1" pattern="$2" file="$3" rc=0 out lineno content
  [ -f "$file" ] || return 0

  case "$kind" in
    fixed) out="$(grep -F -n -- "$pattern" "$file")" || rc=$? ;;
    regex) out="$(grep -E -n -- "$pattern" "$file")" || rc=$? ;;
  esac

  # grep exits 0 on a match, 1 on none, and >=2 on an ERROR — a malformed
  # pattern, say. Treating an error as "no match" would silently disable the
  # rule, which is the failure mode this whole file exists to prevent.
  #
  # The error is RETURNED, never reported from here. This function is always
  # called inside a command substitution, which is a subshell — so calling
  # fail() here would increment a copy of FAILURES that dies with the subshell,
  # print its message to stderr, and still exit 0. That is a third variant of
  # "looks like it is asserting and is not", in the one file whose entire job is
  # to stop exactly that. Anything added here must return, not report.
  if [ "$rc" -ge 2 ]; then
    return 2
  fi
  [ "$rc" -eq 0 ] || return 0

  while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    lineno="${hit%%:*}"
    content="${hit#*:}"
    if is_comment_line "$content"; then
      continue
    fi
    printf '%s:%s:%s\n' "$file" "$lineno" "$content"
  done <<<"$out"
}

[ -r "$PATTERN_FILE" ] || { printf 'FAIL: %s is missing\n' "$PATTERN_FILE" >&2; exit 1; }

# The CI-scoped rules exist because the workflow files sit OUTSIDE this job's
# working directory, and an earlier version of this rule was unenforced against
# exactly the files that had the defect. A missing workflow file is a failure,
# never a skip — and each is checked individually, so renaming one cannot
# silently shrink the scope to the other.
for wf in "${REQUIRED_WORKFLOWS[@]}"; do
  if [ ! -f "${WORKFLOWS_DIR}/${wf}" ]; then
    fail "workflow ${wf} not found at ${WORKFLOWS_DIR} — every CI-scoped rule is unenforced"
  fi
done

RULES=0

while IFS=$'\t' read -r sense kind scope pattern why; do
  case "$sense" in
    ''|'#'*) continue ;;
  esac
  [ -n "${pattern:-}" ] || continue
  RULES=$((RULES + 1))

  mapfile -t targets < <(files_for_scope "$scope" | sort -u)
  if [ "${#targets[@]}" -eq 0 ]; then
    fail "no files found for scope '$scope' — rule for '$pattern' is unenforced"
    continue
  fi

  case "$kind" in
    fixed|regex) ;;
    *) fail "unknown pattern kind '$kind' for '$pattern'"; continue ;;
  esac

  if [ "$sense" = "deny" ]; then
    hits=""
    for target in "${targets[@]}"; do
      mrc=0
      found="$(matches_in_file "$kind" "$pattern" "$target")" || mrc=$?
      if [ "$mrc" -ge 2 ]; then
        fail "grep failed (rc=$mrc) applying '$pattern' to $target — the rule is NOT being enforced"
        continue
      fi
      [ -n "$found" ] && hits="${hits}${found}"$'\n'
    done
    if [ -n "${hits// /}" ] && [ -n "$(printf '%s' "$hits" | tr -d '[:space:]')" ]; then
      fail "banned pattern '$pattern' found in effective (non-comment) content:"
      printf '%s' "$hits" | sed 's/^/    /' >&2
      printf '  why: %s\n' "$why" >&2
    fi
  else
    # `require` rules ALSO ignore comments. Otherwise the comment explaining why
    # a string must be present satisfies the requirement, and deleting the real
    # thing goes unnoticed.
    present=0
    for target in "${targets[@]}"; do
      mrc=0
      found="$(matches_in_file "$kind" "$pattern" "$target")" || mrc=$?
      if [ "$mrc" -ge 2 ]; then
        fail "grep failed (rc=$mrc) applying '$pattern' to $target — the rule is NOT being enforced"
        continue
      fi
      if [ -n "$found" ]; then
        present=1
        break
      fi
    done
    if [ "$present" -eq 0 ]; then
      fail "required pattern '$pattern' is MISSING from the effective content of scope '$scope'"
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
