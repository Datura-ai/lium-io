#!/bin/sh
# Rental-container SSH bootstrap (DAH-2341: race-safe against images that
# start sshd themselves).
#
# The container image's own entrypoint may generate host keys and start sshd
# concurrently with this script (e.g. runpod-style /start.sh). This script
# therefore never assumes it owns sshd bring-up; it converges on an invariant
# instead: "host keys exist and sshd is listening on :22". Who got there
# first does not matter.
#
# Decision flow:
#   1. sshd already running          -> converge (no keygen), verify, exit.
#   2. sshd binary present           -> wait up to a grace period for the
#      image to start sshd itself; if it appears -> converge, verify, exit.
#   3. fallback: install sshd if missing, then (under a lock shared with
#      lium images' /start.sh) generate missing host keys, harden config,
#      start sshd. A failed start is re-checked: if sshd is running anyway
#      (lost a benign start race), that is success.
#   4. verify: sshd running AND listening on port 22 — the exit code of this
#      script reflects the invariant, not individual command exit codes.
#
# Callers treat a non-zero exit as bootstrap failure (fatal for templates
# with ships_sshd=True), so exit 1 only when sshd is genuinely not serving.
set -eu

# LIUM_* environment overrides exist for tests (which cannot write /run or
# /root) and for operational tuning; production execs never set them.
RUN_DIR="${LIUM_RUN_DIR:-/run}"
SSH_DIR="${LIUM_SSH_DIR:-/root/.ssh}"
SSHD_CONFIG="${LIUM_SSHD_CONFIG:-/etc/ssh/sshd_config}"

WATCHDOG_PIDFILE="$RUN_DIR/sshd-watchdog.pid"
WATCHDOG_LOG="/tmp/sshd-watchdog.log"
SLEEP_SECONDS=30
SCRIPT_PATH="$0"

# Non-negative integer (capped) or the fallback — the timing knobs feed
# `[ -lt/-ge ]` tests that would abort the script under `set -eu` on malformed
# input, and `docker exec` inherits the image's ENV, so a hostile image must
# not be able to stretch the wait loops via oversized LIUM_* values.
int_or() {
    case "${1:-}" in
        *[!0-9]* | "")
            printf '%s' "$2"
            return 0
            ;;
    esac
    if [ "$1" -gt "$3" ]; then
        printf '%s' "$3"
    else
        printf '%s' "$1"
    fi
}

# Shared with lium images' /start.sh setup_ssh — keep the path in sync.
LOCK_DIR="$RUN_DIR/lium-ssh-setup.lock"
LOCK_TIMEOUT_SECS="$(int_or "${LIUM_SSH_LOCK_TIMEOUT_SECS:-}" 60 300)"
LOCK_HELD=0

# Grace period to let a self-starting image bring sshd up before the fallback
# takes over. Resolved in main: --grace flag > LIUM_SSHD_GRACE_SECS > default
# (only applies when an sshd binary ships in the image — an image without
# sshd cannot self-start it, so waiting would be pure latency).
DEFAULT_GRACE_SECS="$(int_or "${LIUM_SSHD_GRACE_SECS:-}" 10 300)"
VERIFY_TIMEOUT_SECS="$(int_or "${LIUM_SSHD_VERIFY_SECS:-}" 15 120)"

CONFIG_CHANGED=0

log() {
    printf '%s\n' "$1"
}

is_sshd_running() {
    if command -v pgrep >/dev/null 2>&1; then
        if pgrep -x sshd >/dev/null 2>&1; then
            return 0
        fi
    fi

    if command -v ps >/dev/null 2>&1; then
        if ps -ef 2>/dev/null | grep '[s]shd' >/dev/null 2>&1; then
            return 0
        fi

        if ps 2>/dev/null | grep '[s]shd' >/dev/null 2>&1; then
            return 0
        fi
    fi

    return 1
}

# A listener on :22 (hex 0016) in LISTEN state (0A). The remote address of a
# listening socket is all zeroes, which keeps established/outbound rows with
# port 22 on the far side from matching.
is_sshd_listening() {
    checked=0
    # Intentionally unquoted: space-separated list of files.
    for tcp_file in ${LIUM_PROC_TCP_FILES:-/proc/net/tcp /proc/net/tcp6}; do
        [ -r "$tcp_file" ] || continue
        checked=1
        if grep -E ':0016 0+:0+ 0A ' "$tcp_file" >/dev/null 2>&1; then
            return 0
        fi
    done

    # No readable /proc/net/tcp* — cannot observe sockets; do not fail the
    # bootstrap on missing observability alone.
    if [ "$checked" -eq 0 ]; then
        return 0
    fi

    return 1
}

get_sshd_binary() {
    # Exclusive override (absolute path) — when set, the standard locations
    # are not consulted at all, so tests fully control binary presence.
    if [ -n "${LIUM_SSHD_BIN:-}" ]; then
        if [ -x "$LIUM_SSHD_BIN" ]; then
            printf '%s\n' "$LIUM_SSHD_BIN"
            return 0
        fi
        return 1
    fi

    if [ -x /usr/sbin/sshd ]; then
        printf '%s\n' "/usr/sbin/sshd"
        return 0
    fi

    if command -v sshd >/dev/null 2>&1; then
        command -v sshd
        return 0
    fi

    return 1
}

install_sshd() {
    if command -v apt-get >/dev/null 2>&1; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y openssh-server
        return 0
    fi

    if command -v apk >/dev/null 2>&1; then
        apk add --no-cache openssh
        return 0
    fi

    if command -v dnf >/dev/null 2>&1; then
        dnf install -y openssh-server
        return 0
    fi

    if command -v yum >/dev/null 2>&1; then
        yum install -y openssh-server
        return 0
    fi

    log "No supported package manager found to install sshd"
    return 1
}

ensure_sshd_installed() {
    if get_sshd_binary >/dev/null 2>&1; then
        return 0
    fi

    install_sshd

    if get_sshd_binary >/dev/null 2>&1; then
        return 0
    fi

    log "sshd binary is unavailable after installation attempt"
    return 1
}

acquire_lock() {
    waited=0
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        if [ "$waited" -ge "$LOCK_TIMEOUT_SECS" ]; then
            log "SSH setup lock busy for ${LOCK_TIMEOUT_SECS}s; proceeding without it"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    LOCK_HELD=1
    # Never leak the lock, whatever exit path `set -eu` takes.
    trap release_lock EXIT
}

release_lock() {
    if [ "$LOCK_HELD" -eq 1 ]; then
        rmdir "$LOCK_DIR" 2>/dev/null || true
        LOCK_HELD=0
    fi
}

harden_sshd_config() {
    sshd_config="$SSHD_CONFIG"
    [ -f "$sshd_config" ] || return 0
    if grep -q '^# lium-hardened$' "$sshd_config" 2>/dev/null; then
        return 0
    fi
    # sshd honors the first matching directive — comment out any existing
    # occurrences before appending, otherwise our values would be ignored on
    # base images that ship explicit settings. A read-only /etc/ssh must not
    # abort the whole bootstrap (the exit code belongs to the sshd invariant,
    # not to hardening), so failures here are logged and skipped.
    if ! sed -i -E 's/^[[:space:]]*(PasswordAuthentication|ChallengeResponseAuthentication|KbdInteractiveAuthentication)[[:space:]]+.*/# lium-disabled &/I' "$sshd_config" 2>/dev/null; then
        log "Could not harden sshd config (read-only?); skipping"
        return 0
    fi
    if ! printf '\n# lium-hardened\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nChallengeResponseAuthentication no\n' >> "$sshd_config" 2>/dev/null; then
        log "Could not append hardened sshd config; skipping"
        return 0
    fi
    CONFIG_CHANGED=1
}

# A running sshd only re-reads its config on SIGHUP/restart. When the config
# was hardened after the daemon came up (image-started sshd), nudge the master
# via its pidfile — never pgrep, which would also hit per-session processes.
reload_sshd_if_config_changed() {
    if [ "$CONFIG_CHANGED" -ne 1 ]; then
        return 0
    fi

    for pidfile in "$RUN_DIR/sshd.pid" /var/run/sshd.pid; do
        if [ -f "$pidfile" ]; then
            sshd_pid="$(cat "$pidfile" 2>/dev/null || true)"
            case "${sshd_pid:-}" in
                *[!0-9]* | "") continue ;;
            esac
            if kill -0 "$sshd_pid" 2>/dev/null; then
                kill -HUP "$sshd_pid" 2>/dev/null || true
                log "Sent SIGHUP to sshd (pid $sshd_pid) to load hardened config"
                return 0
            fi
        fi
    done

    log "Hardened sshd config but found no sshd pidfile to reload"
}

prepare_ssh_dir() {
    mkdir -p "$RUN_DIR/sshd"
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR"
}

prepare_sshd_runtime() {
    prepare_ssh_dir
    ssh-keygen -A
    harden_sshd_config
}

start_sshd_if_needed() {
    if is_sshd_running; then
        return 0
    fi

    sshd_bin="$(get_sshd_binary)"
    if ! "$sshd_bin"; then
        # Lost a benign start race (the image's sshd bound :22 between our
        # check and our start) — only a real failure if sshd is still down.
        if is_sshd_running; then
            log "sshd was started concurrently; continuing"
            return 0
        fi
        return 1
    fi
}

# Adopt an sshd somebody else (image entrypoint, previous bootstrap) started:
# no keygen, just the pieces this script still guarantees — key-injection
# dir, config hardening, and the restart watchdog.
converge_on_running_sshd() {
    prepare_ssh_dir
    harden_sshd_config
    reload_sshd_if_config_changed
    spawn_watchdog
}

# Terminal adopt path: converge, prove the invariant, and exit with it.
adopt_and_finish() {
    log "$1"
    converge_on_running_sshd
    verify_sshd
    exit 0
}

wait_for_image_sshd() {
    grace="$1"
    waited=0
    while [ "$waited" -lt "$grace" ]; do
        if is_sshd_running; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    is_sshd_running
}

verify_sshd() {
    waited=0
    while [ "$waited" -lt "$VERIFY_TIMEOUT_SECS" ]; do
        if is_sshd_running && is_sshd_listening; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    running=no
    listening=no
    if is_sshd_running; then running=yes; fi
    if is_sshd_listening; then listening=yes; fi
    log "sshd verification failed after ${VERIFY_TIMEOUT_SECS}s"
    log "(running=$running, listening=$listening)"
    return 1
}

watchdog_loop() {
    while true; do
        if ! is_sshd_running; then
            if ensure_sshd_installed; then
                acquire_lock
                prepare_sshd_runtime
                if ! start_sshd_if_needed; then
                    log "Failed to restart sshd from watchdog"
                fi
                release_lock
            else
                log "Watchdog could not install sshd"
            fi
        fi

        sleep "$SLEEP_SECONDS"
    done
}

spawn_watchdog() {
    if [ -f "$WATCHDOG_PIDFILE" ]; then
        watchdog_pid="$(cat "$WATCHDOG_PIDFILE" 2>/dev/null || true)"
        if [ -n "${watchdog_pid:-}" ] && kill -0 "$watchdog_pid" 2>/dev/null; then
            log "sshd watchdog already running with pid $watchdog_pid"
            return 0
        fi
    fi

    rm -f "$WATCHDOG_PIDFILE"

    if command -v nohup >/dev/null 2>&1; then
        nohup sh "$SCRIPT_PATH" --watchdog-loop >> "$WATCHDOG_LOG" 2>&1 &
    else
        sh "$SCRIPT_PATH" --watchdog-loop >> "$WATCHDOG_LOG" 2>&1 &
    fi

    watchdog_pid=$!
    printf '%s\n' "$watchdog_pid" > "$WATCHDOG_PIDFILE"
    log "Started sshd watchdog with pid $watchdog_pid"
}

if [ "${1:-}" = "--watchdog-loop" ]; then
    printf '%s\n' "$$" > "$WATCHDOG_PIDFILE"
    watchdog_loop
    exit 0
fi

GRACE_SECS=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --grace)
            shift
            GRACE_SECS="${1:-}"
            ;;
        *)
            log "Ignoring unknown argument: $1"
            ;;
    esac
    shift || break
done
GRACE_SECS="$(int_or "${GRACE_SECS:-}" "" 300)"

# 1. Somebody already brought sshd up (image entrypoint won outright, or this
#    is a re-run) — adopt it, never touch host keys.
if is_sshd_running; then
    adopt_and_finish "sshd already running; adopting existing daemon"
fi

# 2. The image ships an sshd binary, so its entrypoint may be about to start
#    it (possibly delayed behind dockerd/pre-start work). Give it a grace
#    period before the fallback claims ownership — this is what removes the
#    concurrent-keygen window against images we do not control.
if get_sshd_binary >/dev/null 2>&1; then
    if [ -z "${GRACE_SECS:-}" ]; then
        GRACE_SECS="$DEFAULT_GRACE_SECS"
    fi
    if [ "$GRACE_SECS" -gt 0 ]; then
        log "Waiting up to ${GRACE_SECS}s for image-provided sshd"
        if wait_for_image_sshd "$GRACE_SECS"; then
            adopt_and_finish "Image-provided sshd came up; adopting it"
        fi
        log "No image-provided sshd after ${GRACE_SECS}s; falling back to own bring-up"
    fi
fi

# 3. Fallback: this script owns sshd bring-up.
ensure_sshd_installed
acquire_lock
if is_sshd_running; then
    # The image's setup slipped in while we were installing/locking.
    release_lock
    adopt_and_finish "sshd appeared before fallback bring-up; adopting it"
fi
prepare_sshd_runtime
if ! start_sshd_if_needed; then
    log "Fallback sshd start failed"
fi
release_lock
spawn_watchdog

# 4. The exit code reflects the invariant, not who did what.
verify_sshd
