#!/usr/bin/env bash
# release_notes_update.sh <tag> — put the "CVM attestation" section (approved runner digest + expected compose
# hash, printed by compose_hash.py --release-notes) into the GitHub release of <tag>. Run by the release-notes
# job of .github/workflows/executor_cd_prod.yml from the tag's checkout, after the images are published (DAH-3602).
#
#   release exists, exactly one section, equal to this checkout's → nothing to do (exit 0)
#   release exists, section differs or the heading is doubled   → the section is replaced (the tag was moved to a
#                                                                  tree with another hash, the digest is stale
#                                                                  while the hash is current, or an edit doubled
#                                                                  the heading; conflicting instructions are worse
#                                                                  than none, and the job must not stay red on
#                                                                  re-runs after `deploy` published the images)
#   release exists, no section                                  → the section is appended
#   no release for the tag                                      → created with GitHub's generated notes, the section last
#
# The section is the heading line up to the next "## " heading or the end of the body. The whole generated section
# is compared, not only the hash (a release body that quotes the new hash somewhere else, or a section with the new
# hash and an old digest, does not pass), and text a human wrote after the section survives the replacement.
# Needs GH_TOKEN and python3; nothing else.
set -euo pipefail

tag="${1:?usage: release_notes_update.sh <tag>}"
here="$(cd "$(dirname "$0")" && pwd)"
heading="## CVM attestation"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

python3 "$here/compose_hash.py" --release-notes > "$work/section.md"
hash="$(python3 "$here/compose_hash.py")"

# the section of a body: the heading line through the line before the next "## " heading (or EOF);
# the heading is matched with trailing whitespace ignored (a hand edit may leave a space after it)
section_of() {
  awk -v h="$heading" '
    { line = $0; sub(/[ \t]+$/, "", line) }
    line == h { p = 1; print; next }
    p && /^## / { exit }
    p' "$1"
}
# the body with the section cut out; trailing blank lines dropped so the append below adds exactly one
without_section() {
  awk -v h="$heading" '
    { line = $0; sub(/[ \t]+$/, "", line) }
    line == h { p = 1; next }
    p && /^## / { p = 0 }
    p { next }
    /^$/ { blanks++; next }
    { for (; blanks > 0; blanks--) print ""; print }' "$1"
}
has_heading() { grep -qxE "$heading[[:space:]]*" "$1"; }
heading_count() { grep -cxE "$heading[[:space:]]*" "$1" || true; }
# stdin with its trailing blank lines dropped (the section in a body ends where the next heading starts, after blanks)
trim_trailing_blank() { awk '/^$/ { blanks++; next } { for (; blanks > 0; blanks--) print ""; print }'; }
# exit 0 when the body carries exactly one section and it equals this checkout's, line for line: a current hash
# next to a stale digest, or a heading an edit doubled, is not "already there"
section_is_current() {
  [ "$(heading_count "$1")" -eq 1 ] \
    && cmp -s <(section_of "$1" | trim_trailing_blank) <(trim_trailing_blank < "$work/section.md")
}

# a body edited in the web UI comes back with CRLF; tr makes the heading match
if gh release view "$tag" --json body --jq .body > "$work/raw.md" 2> "$work/view.err"; then
  tr -d '\r' < "$work/raw.md" > "$work/body.md"
  if section_is_current "$work/body.md"; then
    echo "release $tag already carries this checkout's CVM attestation section (hash $hash)"
    exit 0
  fi
  if has_heading "$work/body.md"; then
    echo "release $tag carries a CVM attestation section that is not this checkout's (another hash or digest, or a doubled heading): replacing it"
  else
    echo "release $tag has no CVM attestation section: appending it"
  fi
  { without_section "$work/body.md"; echo; cat "$work/section.md"; } > "$work/notes.md"
  gh release edit "$tag" --notes-file "$work/notes.md"
elif grep -qi "release not found" "$work/view.err"; then
  # no release yet for the tag: GitHub's generated notes first, the section after them, one create call
  echo "no release for $tag: creating it with generated notes and the section"
  gh api "repos/{owner}/{repo}/releases/generate-notes" -f tag_name="$tag" --jq .body | tr -d '\r' > "$work/generated.md"
  { cat "$work/generated.md"; echo; cat "$work/section.md"; } > "$work/notes.md"
  gh release create "$tag" --verify-tag --notes-file "$work/notes.md"
else
  # auth, rate limit, 5xx: creating here would collide with a release that exists; say what gh said
  cat "$work/view.err" >&2
  echo "cannot read the release of $tag (not a 'release not found'): re-run the job" >&2
  exit 1
fi

# prove the section landed as generated, read back from GitHub
gh release view "$tag" --json body --jq .body | tr -d '\r' > "$work/after.md"
section_is_current "$work/after.md" || {
  echo "the CVM attestation section read back from release $tag is not the generated one" >&2
  exit 1
}
section_of "$work/after.md"
