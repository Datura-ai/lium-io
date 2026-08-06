#!/usr/bin/env bash
# ExecStartPre for lium-cvm-resume.service.
#
# Exiting non-zero here aborts the unit BEFORE ExecStart, so this is the thing
# that makes the resume fire exactly once, on the right boot, and never on a
# host that is looping.
#
# THREE INDEPENDENT REFUSALS
# --------------------------
# 1. attempts >= 1   — it already ran. The counter is incremented and PERSISTED
#                      here, before ExecStart, so a crash mid-run cannot yield a
#                      second attempt. Moving the increment into the resume
#                      script itself would break exactly that property.
#
# 2. boot_id unchanged — no reboot actually happened. Without this, a manual
#                      `systemctl start` on the same boot would re-run the whole
#                      playbook, potentially while a paying CVM is live.
#
# 3. converge budget exhausted — the host is looping. "Exactly once per arming"
#                      is satisfied on every iteration of a loop, so it is not
#                      the property that stops one.

set -Eeuo pipefail

STATE_DIR="${LIUM_STATE_DIR:-/var/lib/lium-cvm}"
STATE_FILE="${STATE_DIR}/resume.state"
BUDGET_FILE="${STATE_DIR}/converge_reboots"
BOOT_ID_PATH="${LIUM_BOOT_ID_PATH:-/proc/sys/kernel/random/boot_id}"
MAX_CONVERGE_REBOOTS="${LIUM_MAX_CONVERGE_REBOOTS:-2}"

refuse() {
  printf 'lium-cvm-resume: NOT resuming: %s\n' "$1" >&2
  exit 1
}

[ -r "$STATE_FILE" ] || refuse "no resume state at ${STATE_FILE}"
[ -r "$BOOT_ID_PATH" ] || refuse "cannot read the current boot id at ${BOOT_ID_PATH}"

# resume.state is written by arm_resume.yml in a fixed one-key-per-line shape,
# so these extractions need no JSON parser and therefore no dependency on a
# host that has not finished being provisioned.
json_str() {
  sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p;T;q" "$STATE_FILE"
}
json_num() {
  sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p;T;q" "$STATE_FILE"
}

ARMED_BOOT_ID="$(json_str armed_boot_id)"
ATTEMPTS="$(json_num attempts)"
[ -n "$ATTEMPTS" ] || ATTEMPTS=0

CURRENT_BOOT_ID="$(tr -d '\n' <"$BOOT_ID_PATH")"

# 1. Already attempted.
if [ "$ATTEMPTS" -ge 1 ]; then
  refuse "this arming has already been attempted (attempts=${ATTEMPTS}). Re-run bootstrap.sh by hand if the run did not finish."
fi

# 2. Same boot — nothing actually rebooted.
if [ -n "$ARMED_BOOT_ID" ] && [ "$CURRENT_BOOT_ID" = "$ARMED_BOOT_ID" ]; then
  refuse "the boot id is unchanged (${CURRENT_BOOT_ID}), so no reboot has happened since arming."
fi

# 3. The host is looping.
BUDGET=0
if [ -r "$BUDGET_FILE" ]; then
  BUDGET="$(tr -cd '0-9' <"$BUDGET_FILE")"
  [ -n "$BUDGET" ] || BUDGET=0
fi
if [ "$BUDGET" -gt "$MAX_CONVERGE_REBOOTS" ]; then
  refuse "the converge reboot budget is exhausted (${BUDGET} > ${MAX_CONVERGE_REBOOTS}). This host is looping; fix the bootloader by hand."
fi

# Consume the attempt and PERSIST IT before returning success. Everything after
# this point is allowed to crash: it still cannot produce a second attempt.
TMP="${STATE_FILE}.tmp"
sed "s/\"attempts\"[[:space:]]*:[[:space:]]*[0-9][0-9]*/\"attempts\": $((ATTEMPTS + 1))/" \
  "$STATE_FILE" >"$TMP"
chmod 600 "$TMP"
mv -f "$TMP" "$STATE_FILE"
# `sync -f`, not bare `sync`. Only this file's durability matters here, and a
# bare sync flushes EVERY mounted filesystem. This runs from ExecStartPre of a
# Type=oneshot unit at boot, where a single wedged mount — an unreachable NFS
# share, a failing disk — blocks it in uninterruptible sleep. Paired with
# TimeoutStartSec in the unit, so neither half can hang the host on its own.
sync -f "$STATE_FILE"

printf 'lium-cvm-resume: resuming (boot %s, armed on %s, attempt %s)\n' \
  "$CURRENT_BOOT_ID" "$ARMED_BOOT_ID" "$((ATTEMPTS + 1))"
