#!/usr/bin/env bash
# Every image a tracked compose file pulls must be in its registry. Compose itself resolves the refs
# (`docker compose config`), with each env template copied to .env the way a provider installs; each ref
# then gets an anonymous `docker manifest inspect`, the way a provider pulls.
#
#   check_image_pins.sh              check every pulled ref
#   check_image_pins.sh --base REF   check only the refs this tree pulls that REF does not
#
# Not checked: an image a service builds itself, a `:local` tag (tagged on the machine by the build
# scripts), and a ref whose tag or digest comes from a variable the templates leave blank (set at install).
set -euo pipefail

list_refs() (
  cd "$1"
  for t in $(git ls-files '*.env.template' '*.env.example'); do
    [ -e "$(dirname "$t")/.env" ] || cp "$t" "$(dirname "$t")/.env"
  done
  for f in $(git ls-files | grep -E '(^|/)(docker-)?compose[^/]*\.ya?ml$'); do
    files=(-f "$f")
    # an override file: loaded on top of its base, as e2e/Makefile does
    [ "$f" = e2e/docker-compose.gpu.yml ] && files=(-f e2e/docker-compose.e2e.yml -f "$f")
    docker compose "${files[@]}" config --format json 2>/dev/null \
      | jq -r '.services[] | select(.build == null) | .image // empty' \
      || { echo "::error file=$f::docker compose cannot load $f" >&2; exit 1; }
  done | { grep -vE '(:local|[@:])$' || true; } | sort -u
)

refs=$(list_refs .)
if [ "${1:-}" = --base ]; then
  base_tree=$(mktemp -d)
  git worktree add -q --detach "$base_tree" "$2"
  base_refs=$(list_refs "$base_tree")
  rm -rf "$base_tree" && git worktree prune
  refs=$(comm -13 <(echo "$base_refs") <(echo "$refs"))
fi

rc=0
for ref in $refs; do
  if err=$(docker manifest inspect "$ref" 2>&1 >/dev/null); then
    echo "ok       $ref"
  else
    echo "::error::$ref is not readable anonymously: ${err:-no output}"
    rc=1
  fi
done
[ -n "$refs" ] || echo "no image refs to check"
exit $rc
