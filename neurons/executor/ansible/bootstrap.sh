#!/usr/bin/env bash
# One command, from bare Ubuntu TDX host to CVM-ready.
#
# Modelled on scripts/install_executor_on_ubuntu.sh: same log/ok/die helpers,
# same ERR trap, same refusal to proceed without passwordless sudo, same
# non-interactive apt environment.
#
# ORDER IS LOAD-BEARING. Read the numbered comments before moving anything.

set -Eeuo pipefail

RED=$'\e[1;31m'; GRN=$'\e[1;32m'; BLU=$'\e[1;34m'; YEL=$'\e[1;33m'; RST=$'\e[0m'
log()  { printf "${BLU}==>${RST} %s\n" "$*"; }
ok()   { printf "${GRN}OK${RST}  %s\n" "$*"; }
warn() { printf "${YEL}!!${RST}  %s\n" "$*"; }
die()  { printf "${RED}XX  %s${RST}\n" "$*" >&2; exit 1; }

trap 'die "Failed at: ${BASH_COMMAND}"' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GUARD="${SCRIPT_DIR}/roles/common/files/lium-guard.sh"

LIUM_STATE_DIR=/var/lib/lium-cvm
LIUM_LOG_DIR=/var/log/lium-cvm
# The lock is taken on the state DIRECTORY, not on a file inside it.
#
# Section 4 creates that directory root-owned 0755 — but `exec 9>` is performed
# by the shell itself, with the CALLER's privileges, never sudo's. A lock FILE
# inside a root-owned directory is therefore unopenable for write by exactly the
# non-root provider this script exists to support, and the run dies at the
# redirection, before section 6, with a bare "Permission denied".
#
# Opening a DIRECTORY read-only succeeds for anyone who can traverse it, and
# flock(2) places its lock on the inode regardless of the fd's access mode. So
# `exec 9<` on the state dir locks identically for root and for a sudo-capable
# user, with no file to create, chown, or leave behind.
LOCK_DIR="$LIUM_STATE_DIR"

# The exit code a preflight aggregate failure produces. CI reads this through
# --print-expected-preflight-rc rather than hardcoding it, so a change here moves
# the CI assertion with it. A constant assigned inside this script does NOT reach
# a CI step's environment — that is why this is a flag and not a variable.
EXPECTED_PREFLIGHT_RC=2

# DUPLICATED FROM group_vars/all/main.yml's lium_os_matrix, and it has to be:
# this gate runs before ansible-core is installed, so nothing here can read a
# group_vars file. tests/test_bootstrap_args.sh asserts the two lists agree via
# --print-supported-releases, exactly as tests/test_guard.sh does for the
# guard's search roots. Without that assertion, adding a release to the matrix
# leaves this entry point rejecting the host at section 3 with a message
# claiming it is unsupported while preflight would have passed it.
SUPPORTED_VERSIONS=("25.10" "26.04")
# 2.18, not 2.16. `meta: end_role` is used by four roles and only exists from
# 2.18; `systemd_service` needs 2.17. An older core already on PATH passes the
# install step's early return, clears this gate, and then dies at the first
# end_role — in play 4 or 5, AFTER the kernel play has written the GRUB drop-in
# and possibly rebooted. A half-applied converge is exactly what this tool must
# never produce.
MIN_ANSIBLE_CORE="2.18"

usage() {
  cat <<'USAGE'
Usage: sudo ./bootstrap.sh [options]

  --tag <ref>            check the lium-io repo out at this ref
  --repo-url <url>       where to clone the repo from
  --repo-path <path>     where the checkout lives (default /opt/lium-io)
  --vars-file <path>     extra variables, e.g. secrets (see group_vars/all/secrets.example.yml)
  -e <name=value>        one extra variable, passed through to ansible-playbook.
                         Repeatable. NEVER put a secret here — argv is world
                         readable through ps and /proc/<pid>/cmdline. Use
                         --vars-file for anything sensitive.
  --check                Ansible check mode: report, change nothing
  --tags <list>          passed through to ansible-playbook
  --skip-tags <list>     passed through to ansible-playbook
  --no-clone             do not touch the repo checkout
  --yes                  do not ask before rebooting
  --force-converge       restore the destructive roles to the play (see below)
  --resume               internal: continuation after a reboot
  --print-expected-preflight-rc
                         print the exit code a preflight failure produces, then exit
  --print-supported-releases
                         print the Ubuntu releases this script accepts, then exit
                         (tests assert these match group_vars' lium_os_matrix)

--force-converge only restores the destructive ROLES. It does not authorise the
damage. Each role is still blocked by its own variable:
    lium_force_qemu_rebuild   lium_force_gpu_cc   lium_force_reboot
USAGE
}

# --- 1. Argument parsing. Pure: touches nothing. -----------------------------
REPO_REF=""; REPO_URL=""; REPO_PATH=""; VARS_FILE=""
CHECK_MODE=0; TAGS=""; SKIP_TAGS=""; NO_CLONE=0; ASSUME_YES=0
EXTRA_VARS=()
FORCE_CONVERGE=0; RESUME=0; PRINT_RC=0; PRINT_RELEASES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) REPO_REF="${2:?--tag needs a value}"; shift ;;
    --repo-url) REPO_URL="${2:?--repo-url needs a value}"; shift ;;
    --repo-path) REPO_PATH="${2:?--repo-path needs a value}"; shift ;;
    --vars-file) VARS_FILE="${2:?--vars-file needs a value}"; shift ;;
    # Passed through to ansible-playbook. Several remediation messages in this
    # tree instruct the operator to run `bootstrap.sh -e <var>=<value>`, and
    # without this they were told to run a command the entry point rejected
    # with exit 64 and a usage dump.
    -e|--extra-vars) EXTRA_VARS+=("${2:?-e needs a name=value}"); shift ;;
    --check) CHECK_MODE=1 ;;
    --tags) TAGS="${2:?--tags needs a value}"; shift ;;
    --skip-tags) SKIP_TAGS="${2:?--skip-tags needs a value}"; shift ;;
    --no-clone) NO_CLONE=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --force-converge) FORCE_CONVERGE=1 ;;
    --resume) RESUME=1 ;;
    --print-expected-preflight-rc) PRINT_RC=1 ;;
    --print-supported-releases) PRINT_RELEASES=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown flag: %s\n' "$1" >&2; usage >&2; exit 64 ;;
  esac
  shift
done

# --- 2. The pure query, answered before ANY side effect. ---------------------
# It must not require root, must not create anything, and must not contend for
# the lock. CI uses it to read the expected rc, and then asserts that the
# state directories did NOT exist before the real run — an assertion this query
# would otherwise satisfy on its own, making it self-satisfying.
if [ "$PRINT_RC" -eq 1 ]; then
  printf '%s\n' "$EXPECTED_PREFLIGHT_RC"
  exit 0
fi

# One release per line, in declaration order. Same contract as the guard's
# --print-default-roots, and pure for the same reasons.
if [ "$PRINT_RELEASES" -eq 1 ]; then
  printf '%s\n' "${SUPPORTED_VERSIONS[@]}"
  exit 0
fi

# --- 3. Guards, all before installing anything. ------------------------------
command -v apt-get >/dev/null 2>&1 || die "This installer needs apt-get (Ubuntu 25.10 or 26.04)."

ARCH="$(uname -m)"
[ "$ARCH" = "x86_64" ] || die "Unsupported architecture: ${ARCH}. Intel TDX hosts are x86_64."

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  command -v sudo >/dev/null 2>&1 || die "Run as root, or install sudo and configure it passwordless."
  if sudo -n true 2>/dev/null; then
    # An ARRAY, not a string. As a string, `$SUDO --preserve-env=X cmd` turns
    # the flag itself into the command when running AS ROOT, where SUDO is
    # empty. That only shows up on a real root run — which is what the
    # supported-OS container job is for.
    SUDO=(sudo -n)
  else
    die "Passwordless sudo is required (or run as root). Refusing to prompt: this runs unattended."
  fi
else
  SUDO=()
fi

# Check the OS before installing anything — a 24.04 provider must not sit
# through an apt install to learn their release is unsupported. Same message
# preflight uses.
OS_VERSION_ID=""
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  OS_VERSION_ID="${VERSION_ID:-}"
fi
OS_SUPPORTED=0
for _v in "${SUPPORTED_VERSIONS[@]}"; do
  [ "$OS_VERSION_ID" = "$_v" ] && OS_SUPPORTED=1
done
if [ "$OS_SUPPORTED" -ne 1 ]; then
  die "os.supported: Ubuntu 25.10 and 26.04 LTS only; found '${OS_VERSION_ID:-unknown}'. See docs/host-setup.md section 1"
fi

# --- 4. First privileged action: create the directories we are about to use. --
# Before the flock (which opens a file inside one of them) and before the first
# tee (which writes into the other). Under `set -Eeuo pipefail` doing this in the
# other order kills the very first run on a fresh host — the only kind of host
# this tool exists for.
"${SUDO[@]}" install -d -m 0755 "$LIUM_STATE_DIR" "$LIUM_LOG_DIR"

# --- 5. Concurrency lock. ----------------------------------------------------
# Two providers, or a provider and the resume unit, must never converge at once.
#
# The resume path WAITS rather than failing: a boot-time resume can legitimately
# race a provider's own re-run or a slow shutdown still holding the lock, and
# failing there would burn the unit's single attempt on a contention that is not
# an attempt at anything. If the wait expires, the resume fails having consumed
# its attempt — the correct conservative outcome, because something else is
# already converging this host.
#
# The `attempts` increment stays in lium-resume-guard.sh, before ExecStart. That
# placement is what makes exactly-once crash-safe; do not move it here.
exec 9<"$LOCK_DIR"
if [ "$RESUME" -eq 1 ]; then
  if ! flock -w 120 9; then
    die "another bootstrap held the lock for over 120s; this host is already being converged"
  fi
else
  if ! flock -n 9; then
    die "another bootstrap is already running (lock: ${LOCK_DIR})"
  fi
fi

BOOTSTRAP_LOG="${LIUM_LOG_DIR}/bootstrap.log"

# --- 6. Install what Ansible needs. ------------------------------------------
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
APT=("${SUDO[@]}" apt-get -y -qq -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)

install_ansible() {
  if command -v ansible-playbook >/dev/null 2>&1; then
    local banner
    banner="$(ansible-playbook --version)"
    ok "ansible-core present: ${banner%%$'\n'*}"
    return 0
  fi
  log "Installing ansible-core, git, curl, ca-certificates"
  local tries=0
  until "${APT[@]}" update; do
    tries=$((tries + 1))
    if [ "$tries" -ge 5 ]; then die "apt update failed after 5 attempts"; fi
    sleep $((2 * tries))
  done
  "${APT[@]}" install --no-install-recommends ansible-core git curl ca-certificates
  ok "ansible-core installed"
}

install_ansible

command -v ansible-playbook >/dev/null 2>&1 \
  || die "ansible-core is still not on PATH after installation"

# Collections are deliberately NOT installed: this tree needs none. If that ever
# changes it needs a collections/requirements.yml installed with
# `ansible-galaxy collection install -r` HERE and in CI — pip cannot install them.
#
# Parsed with bash's own regex rather than `... | head -1 | grep -oE ... | head -1`.
# `head` exits after one line, the producer takes SIGPIPE, and `set -o pipefail`
# reports that 141 — which under `set -e` aborts the script at an ASSIGNMENT,
# before the emptiness check below ever runs. `|| true` is not available here:
# swallowing a status in the entry point is exactly what the forbidden-pattern
# list bans, because this script's exit code is a contract.
CORE_VERSION=""
if [[ "$(ansible-playbook --version)" =~ ([0-9]+\.[0-9]+\.[0-9]+) ]]; then
  CORE_VERSION="${BASH_REMATCH[1]}"
fi
if [ -z "$CORE_VERSION" ]; then
  die "cannot determine the installed ansible-core version"
fi
# sort -V with no early-exiting reader, then the first line by expansion.
CORE_OK="$(printf '%s\n%s\n' "$MIN_ANSIBLE_CORE" "$CORE_VERSION" | sort -V)"
CORE_OK="${CORE_OK%%$'\n'*}"
if [ "$CORE_OK" != "$MIN_ANSIBLE_CORE" ]; then
  die "ansible-core ${CORE_VERSION} is older than the required ${MIN_ANSIBLE_CORE}.
    This playbook uses 'meta: end_role', which does not exist before 2.18. An older
    core would run the first few plays and then fail partway through, leaving the
    host half-converged. Upgrade ansible-core, or remove it so this script can
    install a supported one."
fi

# --- 7. Guard, then choose the run profile. ----------------------------------
[ -x "$GUARD" ] || die "guard script is missing or not executable: ${GUARD}"

GUARD_ARGS=()
[ -n "$REPO_PATH" ] && GUARD_ARGS+=(--repo-path "$REPO_PATH")
GUARD_STATE="$("${SUDO[@]}" "$GUARD" "${GUARD_ARGS[@]+"${GUARD_ARGS[@]}"}" --state)"

MAINTENANCE=0
case "$GUARD_STATE" in
  CLEAN)
    ok "Guard: CLEAN — no CVM state on this host, converging everything"
    ;;
  UNKNOWN)
    # Failing closed is right for a DESTRUCTIVE ACTION. It is wrong for CHOOSING
    # A RUN PROFILE: silently downgrading a fresh-host day-zero run to a
    # near-no-op behind a confusing banner is worse than stopping. So this is a
    # hard failure that names the exact path.
    printf '%s\n' "$("${SUDO[@]}" "$GUARD" "${GUARD_ARGS[@]+"${GUARD_ARGS[@]}"}" --recovery)" >&2
    printf 'Unreadable search roots:\n' >&2
    "${SUDO[@]}" "$GUARD" "${GUARD_ARGS[@]+"${GUARD_ARGS[@]}"}" --unreadable >&2
    die "guard could not determine this host's state, so the run profile cannot be chosen safely.
    Make the paths above readable and re-run. If you are certain this host has no CVM,
    re-run with --force-converge."
    ;;
  LIVE|DORMANT|FOREIGN|ZOMBIE)
    if [ "$FORCE_CONVERGE" -eq 1 ]; then
      warn "Guard: ${GUARD_STATE} — but --force-converge was given, so the destructive roles stay in the play."
    else
      MAINTENANCE=1
      warn "Guard: ${GUARD_STATE} — this host still holds CVM state."
      printf '\n'
      printf '  Selecting the MAINTENANCE profile: --skip-tags destructive\n'
      printf '  Withheld, because each can change a measurement or reset a GPU:\n'
      printf '    kernel  — kernel command line and GRUB (needs a reboot)\n'
      printf '    gpu     — confidential-compute mode (resets every GPU)\n'
      printf '    qemu    — the QEMU build (changes RTMR0)\n'
      printf '  Still converging: preflight, docker, repo, sgx_key_provider, catalog, verify.\n'
      printf '\n'
      "${SUDO[@]}" "$GUARD" "${GUARD_ARGS[@]+"${GUARD_ARGS[@]}"}" --recovery
      printf '\n'
    fi
    ;;
  *)
    die "guard returned an unrecognised state: '${GUARD_STATE}'"
    ;;
esac

if [ "$FORCE_CONVERGE" -eq 1 ]; then
  printf '\n'
  warn "--force-converge restores the destructive roles to the play."
  printf '  It does NOT authorise the damage. Each role is still blocked by its own\n'
  printf '  variable, and each names exactly what it accepts:\n'
  printf '    -e lium_force_qemu_rebuild=true   accepts that any existing hda.img becomes undecryptable\n'
  printf '    -e lium_force_gpu_cc=true         accepts resetting every GPU\n'
  printf '    -e lium_force_reboot=true         accepts an unattended reboot\n'
  printf '\n'
fi

# --- 8. Build and run the playbook command. ----------------------------------
# Captured once, and used both to name the run directory and to decide in
# section 9 whether the report on disk belongs to THIS run.
RUN_STARTED_EPOCH="$(date +%s)"
RUN_DIR="${LIUM_LOG_DIR}/run-${RUN_STARTED_EPOCH}"

PLAY_ARGS=(-i "${SCRIPT_DIR}/inventory/localhost.yml" "${SCRIPT_DIR}/site.yml" --diff)

[ "$CHECK_MODE" -eq 1 ] && PLAY_ARGS+=(--check)
[ -n "$TAGS" ] && PLAY_ARGS+=(--tags "$TAGS")

SKIP_ALL="$SKIP_TAGS"
if [ "$MAINTENANCE" -eq 1 ]; then
  if [ -n "$SKIP_ALL" ]; then
    SKIP_ALL="${SKIP_ALL},destructive"
  else
    SKIP_ALL="destructive"
  fi
  PLAY_ARGS+=(-e "lium_preflight_fatal=false")
fi
[ -n "$SKIP_ALL" ] && PLAY_ARGS+=(--skip-tags "$SKIP_ALL")

[ -n "$REPO_REF" ] && PLAY_ARGS+=(-e "lium_repo_ref=${REPO_REF}")
[ -n "$REPO_URL" ] && PLAY_ARGS+=(-e "lium_repo_url=${REPO_URL}")
[ -n "$REPO_PATH" ] && PLAY_ARGS+=(-e "lium_repo_path=${REPO_PATH}")
[ "$NO_CLONE" -eq 1 ] && PLAY_ARGS+=(-e "lium_repo_clone=false")
[ "$ASSUME_YES" -eq 1 ] && PLAY_ARGS+=(-e "lium_assume_yes=true")
if [ -n "$VARS_FILE" ]; then
  [ -r "$VARS_FILE" ] || die "--vars-file is not readable: ${VARS_FILE}"
  PLAY_ARGS+=(-e "@${VARS_FILE}")
fi

# LAST, so an explicit -e beats the values bootstrap.sh derived from its own
# flags: with repeated --extra-vars, ansible takes the later one. That ordering
# is the point of the flag — an operator reaching for -e is overriding a default
# on purpose, and silently losing to a flag they did not pass would be worse
# than not having the option at all.
if [ "${#EXTRA_VARS[@]}" -gt 0 ]; then
  for _ev in "${EXTRA_VARS[@]}"; do
    PLAY_ARGS+=(-e "$_ev")
  done
fi

log "Running the playbook. Full output also goes to ${BOOTSTRAP_LOG}"
printf '\n'

# Exported rather than written as prefix assignments, so they survive sudo's
# environment reset via --preserve-env.
export ANSIBLE_CONFIG="${SCRIPT_DIR}/ansible.cfg"

# The per-run forensic manifest: one JSON file per host recording what each task
# actually did. `--tree` is an ADHOC-ONLY option — `ansible-playbook` rejects it
# outright — so the callback is enabled by name and pointed at the run directory
# instead. The `tree` callback ships with ansible-core, so this needs no
# collection.
export ANSIBLE_CALLBACKS_ENABLED=tree
export ANSIBLE_CALLBACK_TREE_DIR="$RUN_DIR"

LIUM_PRESERVE=ANSIBLE_CONFIG,ANSIBLE_CALLBACKS_ENABLED,ANSIBLE_CALLBACK_TREE_DIR
SUDO_PLAY=()
if [ "${#SUDO[@]}" -gt 0 ]; then
  SUDO_PLAY=("${SUDO[@]}" "--preserve-env=${LIUM_PRESERVE}")
fi

# `|| PLAY_RC=$?`, NOT `set +e`.
#
# `set +e` does not disable an ERR trap. Bash runs the ERR trap under the same
# CONDITIONS as errexit, but independently of whether errexit is switched on —
# so the trap above would fire here and `die` would exit 1, throwing away the
# playbook's real status. A command on the left of `||` is genuinely exempt.
#
# With `pipefail` set, the pipeline's status is ansible-playbook's (tee exits 0),
# so this needs no PIPESTATUS gymnastics.
PLAY_RC=0
{ "${SUDO_PLAY[@]}" ansible-playbook "${PLAY_ARGS[@]}" 2>&1 \
    | "${SUDO[@]}" tee -a "$BOOTSTRAP_LOG"; } || PLAY_RC=$?

# --- 9. Report, then exit with the playbook's status. ------------------------
REPORT="${LIUM_STATE_DIR}/verify-report.json"

# A report older than this run belongs to a PREVIOUS one, and printing it as the
# current reading of the host is a lie the operator cannot see through.
#
# Observed on au11: a play that failed after the reboot but before `verify`
# printed the pre-reboot report's "5 tokens missing" while /proc/cmdline already
# carried them — sending the operator to debug something that had just been
# fixed. The old `else` branch could never catch it, because it only fires when
# the file is ABSENT, and any host that has converged once already has one.
#
# mtime against the run's start, not the boot id: a second failed run on the
# same boot leaves a same-boot report that is still stale. `if cmd` rather than a
# bare assignment, so a missing or unreadable file is answered here instead of
# tripping the ERR trap.
REPORT_MTIME=0
if REPORT_MTIME_RAW="$("${SUDO[@]}" stat -c %Y "$REPORT" 2>/dev/null)"; then
  REPORT_MTIME="$REPORT_MTIME_RAW"
fi
REPORT_IS_CURRENT=0
if [ "$REPORT_MTIME" -ge "$RUN_STARTED_EPOCH" ]; then
  REPORT_IS_CURRENT=1
fi

printf '\n'
if [ -r "$REPORT" ] && [ "$REPORT_IS_CURRENT" -eq 0 ]; then
  warn "The verify report at ${REPORT} is from an EARLIER run, so it is not shown."
  warn "This run ended before the verify role rewrote it. Its contents describe the"
  warn "host as it was at $(date -d "@${REPORT_MTIME}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || printf 'an earlier time'), not as it is now."
  warn "Fix what the playbook output above reports, then re-run to get a current report."
elif [ -r "$REPORT" ]; then
  log "Verify report: ${REPORT}"
  if command -v python3 >/dev/null 2>&1; then
    "${SUDO[@]}" python3 - "$REPORT" <<'PYSUM'
import json, sys
with open(sys.argv[1]) as fh:
    report = json.load(fh)
summary = report.get("summary", {})
print("  {ok} OK   {warn} WARN   {must_fix} MUST_FIX   {unknown} UNKNOWN   {skipped} SKIPPED".format(
    ok=summary.get("ok", 0), warn=summary.get("warn", 0),
    must_fix=summary.get("must_fix", 0), unknown=summary.get("unknown", 0),
    skipped=summary.get("skipped", 0)))
for check in report.get("checks", []):
    if check.get("status") in ("MUST_FIX", "UNKNOWN"):
        print("  [{status}] {id}: {observed}".format(**check))
        print("      fix: {remediation}".format(**check))
PYSUM
  fi
elif [ "$CHECK_MODE" -eq 1 ]; then
  # Check mode writes no files, so the absence of the report says nothing about
  # whether verify ran. It usually did — the summary is in the output above,
  # and the report content appears there as a diff. Claiming the run "did not
  # reach the verify role" here is simply false, and it sends an operator
  # hunting for a failure that did not happen.
  warn "No verify report written: --check makes no changes, including to ${REPORT}."
  warn "The verify summary above is still the real reading of this host."
else
  warn "No verify report at ${REPORT} — the run did not reach the verify role."
fi

printf '\n'
if [ "$PLAY_RC" -eq 0 ]; then
  ok "Host provisioning complete."
  exit 0
fi

# Exit with the PLAYBOOK's status, not a flat 1.
#
# The exit code is part of this script's contract: CI compares it against
# --print-expected-preflight-rc, and a provider's own automation gates on it.
# Collapsing every failure to 1 would make that constant meaningless and would
# throw away the one signal a scripted caller has.
printf "${RED}XX  Playbook exited %s. See %s and the report above.${RST}\n" \
  "$PLAY_RC" "$BOOTSTRAP_LOG" >&2
exit "$PLAY_RC"
