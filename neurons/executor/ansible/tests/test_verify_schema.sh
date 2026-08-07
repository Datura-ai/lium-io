#!/usr/bin/env bash
# Merge-gate box 7, schema half.
#
# The schema alone cannot enforce that every id has a DOCUMENTED meaning — it
# can only say which ids are legal. This closes that gap by asserting four
# independent artifacts describe exactly the same set of checks:
#
#   1. the schema's `id` enum          — what a consumer may receive
#   2. roles/verify/README.md headings — what a human can look up
#   3. the golden example              — what a consumer is shown
#   4. checks.json.j2                  — what the role ACTUALLY emits
#
# An id in the schema but not the README is an id nobody can act on. An id the
# role emits that the schema does not list is a report that fails validation on
# somebody else's host.
#
# (4) was missing, and it is the only one taken from the code rather than from a
# description of the code. Without it a new check could be added to the template
# and to nothing else: all three of the others still agreed with each other, this
# test still passed, and the first real host to run it emitted an id its own
# schema rejected. That happened while adding sgx.qpl_config_parses.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SCHEMA=schemas/verify-report.schema.json
README=roles/verify/README.md
GOLDEN=tests/fixtures/verify-report.example.json
TEMPLATE=roles/verify/templates/checks.json.j2

for f in "$SCHEMA" "$README" "$GOLDEN" "$TEMPLATE"; do
  if [ ! -r "$f" ]; then
    printf 'FAIL: %s is missing\n' "$f" >&2
    exit 1
  fi
done

printf 'test_verify_schema.sh\n'

python3 - "$SCHEMA" "$README" "$GOLDEN" "$TEMPLATE" <<'PY'
import json, re, sys

schema_path, readme_path, golden_path, template_path = sys.argv[1:5]

with open(schema_path) as fh:
    schema = json.load(fh)
enum = set(schema["properties"]["checks"]["items"]["properties"]["id"]["enum"])

with open(readme_path) as fh:
    documented = set(re.findall(r"^### (\S+)$", fh.read(), re.M))

with open(golden_path) as fh:
    golden = json.load(fh)
emitted = {check["id"] for check in golden["checks"]}

# Read straight from the template, so this one source cannot drift from the
# other three by agreeing with a stale copy of itself.
with open(template_path) as fh:
    templated = set(re.findall(r"^\s*'id':\s*'([^']+)'", fh.read(), re.M))

failures = []

if len(enum) < 10:
    failures.append(f"the schema enum has only {len(enum)} ids, which looks truncated")

# A regex that stops matching would make the template comparison vacuously true.
if len(templated) < 10:
    failures.append(
        f"only {len(templated)} ids were parsed out of {template_path}; "
        "the id regex has stopped matching and the comparison proves nothing")

for label, missing in [
    ("in the schema enum but NOT documented in the README", enum - documented),
    ("documented in the README but NOT in the schema enum", documented - enum),
    ("in the schema enum but NOT emitted in the golden example", enum - emitted),
    ("emitted in the golden example but NOT in the schema enum", emitted - enum),
    ("emitted by checks.json.j2 but NOT in the schema enum", templated - enum),
    ("in the schema enum but NOT emitted by checks.json.j2", enum - templated),
]:
    if missing:
        failures.append(f"{len(missing)} id(s) {label}: {', '.join(sorted(missing))}")

# Every check carries a remediation, because that text is the support channel:
# providers run this unattended with no Lium access to their host.
for check in golden["checks"]:
    if not check.get("remediation", "").strip():
        failures.append(f"check {check['id']} has an empty remediation")

# The status vocabulary must keep UNKNOWN distinct. Collapsing it into a pass or
# a failure is the exact defect the tri-state facts exist to prevent.
statuses = set(schema["properties"]["checks"]["items"]["properties"]["status"]["enum"])
if statuses != {"OK", "WARN", "MUST_FIX", "UNKNOWN", "SKIPPED"}:
    failures.append(f"the status vocabulary changed: {sorted(statuses)}")

if failures:
    for failure in failures:
        print(f"  FAIL {failure}", file=sys.stderr)
    sys.exit(1)

print(f"  ok   {len(enum)} ids agree across the schema, the README, the golden example and checks.json.j2")
print(f"  ok   every check carries a remediation")
print(f"  ok   the status vocabulary still distinguishes UNKNOWN")
PY

printf 'PASS test_verify_schema.sh\n'
