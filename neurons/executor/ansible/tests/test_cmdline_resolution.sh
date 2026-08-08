#!/usr/bin/env bash
# lium-cmdline.sh: the two vfio-pci.ids questions, resolved separately.
#
# A pure-shell unit, deliberately NOT another kernel-role fixture case. It pins
# the parser directly, which is where the answers are decided, and it therefore
# also runs in the container job where ansible-core cannot start.
#
# The distinction under test:
#
#   existing_vfio_ids       UNION over /etc/default/grub, the drop-ins and
#                           /proc/cmdline. Answers "was this id ever
#                           configured?" — the basis for the narrowing refusal.
#
#   grub_existing_vfio_ids  LAST-OCCURRENCE-WINS over the line 10_linux emits,
#                           `${GRUB_CMDLINE_LINUX} ${GRUB_CMDLINE_LINUX_DEFAULT}`.
#                           Answers "what does the next boot get?" — the basis
#                           for the carry-forward.
#
# Both have already been got wrong once. Case (3) is the regression: with the
# ids split across the two variables, a `local IFS=,` leak dropped the second
# variable's ids from the carry-forward, and a union would have carried the
# first variable's ids that the host does not actually boot with.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CMDLINE=roles/common/files/lium-cmdline.sh
FAILURES=0
CASES=0

pass() { printf '  ok   %s\n' "$1"; }
fail() { printf '  FAIL %s\n' "$1" >&2; FAILURES=$((FAILURES + 1)); }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/grub.d"

# Present in /proc/cmdline and nowhere else, in every case below. It must always
# reach the union and never the carry-forward: it is already gone from the boot
# configuration, and restoring it would resurrect a setting the provider removed.
printf 'BOOT_IMAGE=/vmlinuz root=/dev/sda1 vfio-pci.ids=10de:beef\n' >"$WORK/proc"

# Writes the two GRUB variables, then prints "<carry-forward>|<union>" as two
# comma-joined lists so a case can be asserted with one string compare.
resolve() {
  printf 'GRUB_CMDLINE_LINUX="%s"\nGRUB_CMDLINE_LINUX_DEFAULT="%s"\n' \
    "$1" "$2" >"$WORK/grub"
  "$CMDLINE" analyze \
    --required nohibernate \
    --grub-default "$WORK/grub" \
    --grub-dropin-dir "$WORK/grub.d" \
    --proc-cmdline "$WORK/proc" \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(",".join(d["grub_existing_vfio_ids"]) + "|" + ",".join(d["existing_vfio_ids"]))
'
}

check() {
  local label="$1" linux_line="$2" default_line="$3" want="$4" got
  CASES=$((CASES + 1))
  got="$(resolve "$linux_line" "$default_line")"
  if [ "$got" = "$want" ]; then
    pass "$label"
  else
    fail "$label
       GRUB_CMDLINE_LINUX         = $linux_line
       GRUB_CMDLINE_LINUX_DEFAULT = $default_line
       want (carry|union) = $want
       got  (carry|union) = $got"
  fi
}

check "(1) ids in GRUB_CMDLINE_LINUX only are carried forward" \
  'kvm_intel.tdx=on vfio-pci.ids=10de:2335,15b3:1017' \
  'quiet' \
  '10de:2335,15b3:1017|10de:2335,10de:beef,15b3:1017'

check "(2) ids in GRUB_CMDLINE_LINUX_DEFAULT only are carried forward" \
  'quiet' \
  'nohibernate vfio-pci.ids=15b3:1017' \
  '15b3:1017|10de:beef,15b3:1017'

# The regression. _DEFAULT is emitted last, so the host boots with 144d:a80a
# ALONE — 10de:2335 is already unbound there. Carrying the union would re-bind
# a device the provider stopped passing through, and on a host whose NVMe is
# the id in question that takes the disk away from the OS.
check "(3) ids in BOTH variables: only the winning one is carried" \
  'vfio-pci.ids=10de:2335' \
  'vfio-pci.ids=144d:a80a' \
  '144d:a80a|10de:2335,10de:beef,144d:a80a'

check "(4) an id only in /proc/cmdline reaches the union, never the carry-forward" \
  'quiet' \
  'nohibernate' \
  '|10de:beef'

# Upper case is a working configuration — vfio-pci parses the ids with %x — so
# the parser must report exactly what it read and leave case folding to the
# caller, which compares lower-cased on both sides.
check "(5) case is preserved, not silently folded" \
  'vfio-pci.ids=10DE:2335' \
  'quiet' \
  '10DE:2335|10DE:2335,10de:beef'

printf '\n'
if [ "$FAILURES" -gt 0 ]; then
  printf 'FAIL test_cmdline_resolution.sh — %s of %s checks failed.\n' "$FAILURES" "$CASES" >&2
  exit 1
fi
printf 'PASS test_cmdline_resolution.sh — %s checks.\n' "$CASES"
