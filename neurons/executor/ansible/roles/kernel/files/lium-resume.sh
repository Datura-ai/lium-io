#!/usr/bin/env bash
# ExecStart for lium-cvm-resume.service — continue the converge after a reboot.
#
# A PLAIN SCRIPT, NOT A TEMPLATE. It reads everything it needs from
# resume.state at runtime. A templated shell script cannot be shellchecked as
# committed, and this one sits directly on the reboot-recovery path — the one
# place where a syntax error means an unattended host never comes back to
# finish provisioning.
#
# It deliberately does NOT touch converge_reboots. That budget belongs to the
# kernel role and must survive across resumes; a resume that reset it would
# defeat the loop protection it exists to provide.

set -Eeuo pipefail

STATE_DIR="${LIUM_STATE_DIR:-/var/lib/lium-cvm}"
STATE_FILE="${STATE_DIR}/resume.state"

[ -r "$STATE_FILE" ] || { printf 'lium-cvm-resume: no state file, nothing to do\n'; exit 0; }

json_str() {
  sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$STATE_FILE" | head -1
}

BOOTSTRAP="$(json_str bootstrap_path)"
PLAYBOOK_ARGS="$(json_str playbook_args)"

if [ -z "$BOOTSTRAP" ] || [ ! -x "$BOOTSTRAP" ]; then
  printf 'lium-cvm-resume: bootstrap script is missing or not executable: %s\n' "$BOOTSTRAP" >&2
  exit 1
fi

printf 'lium-cvm-resume: continuing with %s --resume --no-clone %s\n' "$BOOTSTRAP" "$PLAYBOOK_ARGS"

RC=0
# shellcheck disable=SC2086  # PLAYBOOK_ARGS is our own flag list and must split.
"$BOOTSTRAP" --resume --no-clone $PLAYBOOK_ARGS || RC=$?

if [ "$RC" -eq 0 ]; then
  printf 'lium-cvm-resume: converge finished; disarming\n'
  rm -f "$STATE_FILE"
  if systemctl disable lium-cvm-resume.service >/dev/null 2>&1; then
    printf 'lium-cvm-resume: unit disabled\n'
  else
    # Not fatal: ExecStopPost disables it too, on every exit path.
    printf 'lium-cvm-resume: could not disable the unit here; ExecStopPost will\n' >&2
  fi
else
  # The state file stays so an operator can see what was armed and when. The
  # unit still disables itself via ExecStopPost, so it cannot fire again.
  printf 'lium-cvm-resume: converge exited %s; the unit disables itself either way\n' "$RC" >&2
fi

exit "$RC"
