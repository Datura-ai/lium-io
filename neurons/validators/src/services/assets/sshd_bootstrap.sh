#!/bin/sh
set -eu

WATCHDOG_PIDFILE="/run/sshd-watchdog.pid"
WATCHDOG_LOG="/tmp/sshd-watchdog.log"
SLEEP_SECONDS=30
SCRIPT_PATH="$0"

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

get_sshd_binary() {
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

harden_sshd_config() {
    sshd_config="/etc/ssh/sshd_config"
    [ -f "$sshd_config" ] || return 0
    if grep -q '^# lium-hardened$' "$sshd_config" 2>/dev/null; then
        return 0
    fi
    # sshd honors the first matching directive — comment out any existing
    # occurrences before appending, otherwise our values would be ignored on
    # base images that ship explicit settings.
    sed -i -E 's/^[[:space:]]*(PasswordAuthentication|ChallengeResponseAuthentication|KbdInteractiveAuthentication)[[:space:]]+.*/# lium-disabled &/I' "$sshd_config"
    printf '\n# lium-hardened\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nChallengeResponseAuthentication no\n' >> "$sshd_config"
}

prepare_sshd_runtime() {
    mkdir -p /run/sshd
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    ssh-keygen -A
    harden_sshd_config
}

start_sshd_if_needed() {
    if is_sshd_running; then
        return 0
    fi

    sshd_bin="$(get_sshd_binary)"
    "$sshd_bin"
}

watchdog_loop() {
    while true; do
        if ! is_sshd_running; then
            if ensure_sshd_installed; then
                prepare_sshd_runtime
                if ! start_sshd_if_needed; then
                    log "Failed to restart sshd from watchdog"
                fi
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

ensure_sshd_installed
prepare_sshd_runtime
start_sshd_if_needed
spawn_watchdog
