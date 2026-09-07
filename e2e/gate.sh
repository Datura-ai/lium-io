#!/usr/bin/env bash
# gate.sh — the e2e merge gate as one command: build → up → every suite → logs → down.
#
# What makes it a gate rather than `make e2e`:
#   - every step runs under a hard timeout (GNU timeout, SIGKILL 30 s after SIGTERM), so a hung image build or a
#     stack that never becomes healthy can hold a CI runner or a pod for minutes, never for an hour;
#   - every suite in tests/<name> runs even after an earlier one failed, so one push shows every failure;
#   - artifacts/ always holds timings.txt, summary.md, <suite>-junit.xml, compose.log, compose-ps.txt and the
#     suites' JSON dumps (cycle-result.json, rental-*.json), pass or fail; CI uploads the directory and posts summary.md on the PR;
#   - the stack is torn down (volumes too) on every exit path; KEEP_UP=1 keeps it for debugging.
# Exit 0 only when build, up and every suite passed. CI's e2e-gate job runs this (`make e2e-full`); a GPU box or
# a Lium DinD pod is meant to run the same script with E2E_GPU=1 (README: the GPU path has not been run yet).
# (Same gate as lium-platform/e2e/gate.sh; keep the two in step.)
#
# Env: SUITES (default: every tests/<name>; each needs a test-<name> Makefile target), T_BUILD T_UP T_SUITE
# T_MISC (timeout durations, defaults 25m 8m 20m 3m), KEEP_UP=1.
set -uo pipefail
cd "$(dirname "$0")"
A=artifacts
mkdir -p "$A"
: > "$A/timings.txt"
rm -f "$A"/*-junit.xml "$A/summary.md"
SUITES=${SUITES:-$(for d in tests/*/; do basename "$d"; done | tr '\n' ' ')}
T_BUILD=${T_BUILD:-25m}; T_UP=${T_UP:-8m}; T_SUITE=${T_SUITE:-20m}; T_MISC=${T_MISC:-3m}
FAILED=""

step() {  # step <name> <timeout> <make target...>
  local name=$1 t=$2; shift 2
  local t0=$SECONDS rc status
  echo "::group::$name"
  timeout -k 30 "$t" make "$@"; rc=$?
  echo "::endgroup::"
  case $rc in 0) status=pass ;; 124|137) status="TIMEOUT(>$t)" ;; *) status="FAIL(rc=$rc)" ;; esac
  printf '%s\t%ds\t%s\n' "$name" "$((SECONDS - t0))" "$status" >> "$A/timings.txt"
  echo "gate: $name $status in $((SECONDS - t0))s"
  [ $rc -eq 0 ] || FAILED="$FAILED $name"
  return $rc
}

summary() {  # artifacts/summary.md from timings.txt and the junit files
  python3 - "$A" "$FAILED" <<'PY'
import glob, os, sys
import xml.etree.ElementTree as ET
a, failed = sys.argv[1], sys.argv[2].split()
rows, red = [], []
for f in sorted(glob.glob(f"{a}/*-junit.xml")):
    suite = os.path.basename(f).removesuffix("-junit.xml")
    root = ET.parse(f).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    n = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    for tc in root.iter("testcase"):
        if tc.find("failure") is not None or tc.find("error") is not None:
            red.append(f"{suite}: {tc.get('classname', '')}::{tc.get('name', '')}")
    passed = n["tests"] - n["failures"] - n["errors"] - n["skipped"]
    rows.append((suite, f"{passed} passed, {n['failures'] + n['errors']} failed, {n['skipped']} skipped"))
timings = [l.split("\t") for l in open(f"{a}/timings.txt").read().splitlines() if l]
out = ["| step | result | time |", "|---|---|---|"]
for name, secs, status in timings:
    detail = dict(rows).get(name.removeprefix("test-"), "")
    mark = "✅" if status == "pass" else "❌"
    out.append(f"| {name} | {mark} {status}{' — ' + detail if detail else ''} | {secs} |")
if red:
    out += ["", "Failed tests:", *[f"- `{r}`" for r in red[:20]]]
    if len(red) > 20:
        out.append(f"- … and {len(red) - 20} more (see the junit files in the artifact)")
verdict = "**e2e gate: PASS**" if not failed else f"**e2e gate: FAIL** ({', '.join(failed)})"
open(f"{a}/summary.md", "w").write("\n".join([verdict, "", *out, ""]))
print("\n".join([verdict, *out]))
PY
}

finish() {
  # logs before down: compose.log is the only record of why a container died
  timeout -k 30 "$T_MISC" make logs >/dev/null 2>&1 || true
  summary
  if [ -z "${KEEP_UP:-}" ]; then timeout -k 30 "$T_MISC" make down >/dev/null 2>&1 || echo "gate: make down failed — check for leftover lium-e2e containers"; fi
  [ -z "$FAILED" ]
  exit $?
}

step build "$T_BUILD" build || finish
step up "$T_UP" up || finish
for s in $SUITES; do step "test-$s" "$T_SUITE" "test-$s" || true; done   # one push, every failure
finish
