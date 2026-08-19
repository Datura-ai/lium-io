#!/usr/bin/env bash
# Local proof of the guest-SSH image (DAH-2684). Runs the container against THIS machine as
# if it were the guest — the same `pid: host` / `network_mode: host` / `privileged` /
# `/:/host` / `/dev/pts:/dev/pts` shape the backend writes into every renter compose — and
# checks that an SSH session lands on the host, not in the container.
#
# Needs docker and a free TCP port (default 2200; override with LIUM_SSH_TEST_PORT). It
# does not touch a CVM: the mechanism (chroot into the guest root, nsenter into PID 1's
# namespaces) is identical on any Linux host, which is what makes this runnable anywhere.
#
# Usage:  bash tests/test_guest_ssh.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${LIUM_SSH_TEST_PORT:-2200}"
IMAGE="lium-cvm-ssh:test-$$"
NAME="lium-cvm-ssh-test-$$"
VOLUME="lium-cvm-ssh-test-hostkeys-$$"
WORK="$(mktemp -d)"
PASS=0
FAIL=0

cleanup() {
    docker rm -f "$NAME" >/dev/null 2>&1
    docker volume rm "$VOLUME" >/dev/null 2>&1
    docker rmi "$IMAGE" >/dev/null 2>&1
    rm -rf "$WORK"
}
trap cleanup EXIT

ok()   { PASS=$((PASS + 1)); printf '  ok    %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf '  FAIL  %s%s\n' "$1" "${2:+ — $2}"; }
check() { if [ "$1" = "true" ]; then ok "$2"; else bad "$2" "${3:-}"; fi; }

ssh-keygen -q -t ed25519 -N '' -f "$WORK/renter" -C renter@test
ssh-keygen -q -t ed25519 -N '' -f "$WORK/second" -C second@test
ssh-keygen -q -t ed25519 -N '' -f "$WORK/stranger" -C stranger@test
KEYS="$(cat "$WORK/renter.pub")
$(cat "$WORK/second.pub")"

SSH_OPTS=(-o StrictHostKeyChecking=no -o "UserKnownHostsFile=$WORK/known" -o BatchMode=yes -o ConnectTimeout=5)
as_renter() { ssh -i "$WORK/renter" "${SSH_OPTS[@]}" -p "$PORT" root@127.0.0.1 "$@"; }

run_container() {
    docker rm -f "$NAME" >/dev/null 2>&1
    docker run -d --name "$NAME" "$@" \
        -e "LIUM_SSH_PORT=$PORT" -e "LIUM_SSH_AUTHORIZED_KEYS=$KEYS" "$IMAGE" >/dev/null
    sleep 2
}
last_log_line() { docker logs "$NAME" 2>&1 | tail -1; }

echo "== build"
docker build -q -t "$IMAGE" "$HERE" >/dev/null
ok "image builds"

echo "== every missing precondition is refused with its reason"
run_container --network=host --privileged -v /:/host -v /dev/pts:/dev/pts
case "$(last_log_line)" in *"pid: host"*) ok "no pid host -> refused" ;; *) bad "no pid host -> refused" "$(last_log_line)" ;; esac
run_container --pid=host --network=host -v /:/host -v /dev/pts:/dev/pts
case "$(last_log_line)" in *"privileged: true"*) ok "not privileged -> refused" ;; *) bad "not privileged -> refused" "$(last_log_line)" ;; esac
run_container --pid=host --privileged -v /:/host -v /dev/pts:/dev/pts
case "$(last_log_line)" in *"network_mode: host"*) ok "private network -> refused" ;; *) bad "private network -> refused" "$(last_log_line)" ;; esac
run_container --pid=host --network=host --privileged -v /dev/pts:/dev/pts
case "$(last_log_line)" in *"/:/host"*) ok "no guest root -> refused" ;; *) bad "no guest root -> refused" "$(last_log_line)" ;; esac
run_container --pid=host --network=host --privileged -v /:/host
case "$(last_log_line)" in *"/dev/pts:/dev/pts"*) ok "private devpts -> refused" ;; *) bad "private devpts -> refused" "$(last_log_line)" ;; esac
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --pid=host --network=host --privileged -v /:/host -v /dev/pts:/dev/pts \
    -e "LIUM_SSH_PORT=$PORT" -e "LIUM_SSH_AUTHORIZED_KEYS=" "$IMAGE" >/dev/null
sleep 1
case "$(last_log_line)" in *"LIUM_SSH_AUTHORIZED_KEYS is empty"*) ok "no keys -> refused" ;; *) bad "no keys -> refused" "$(last_log_line)" ;; esac

echo "== the real shape"
run_container --pid=host --network=host --privileged -v /:/host -v /dev/pts:/dev/pts -v "$VOLUME:/etc/ssh/hostkeys"
FP1="$(docker logs "$NAME" 2>&1 | sed -n 's/.*host key fingerprint: //p' | head -1)"
check "$([ -n "$FP1" ] && echo true || echo false)" "sshd started and printed a host-key fingerprint" "$(last_log_line)"

HOST_MNT="$(readlink /proc/1/ns/mnt 2>/dev/null || sudo -n readlink /proc/1/ns/mnt 2>/dev/null || true)"
# shellcheck disable=SC2016  # expanded on the guest, not here
SESSION="$(as_renter 'echo "$(hostname)|$(readlink /proc/self/ns/mnt)|$(id -u)|$([ -d /host ] && echo container || echo guest)"' 2>/dev/null || true)"
check "$([ "${SESSION%%|*}" = "$(hostname)" ] && echo true || echo false)" "session hostname is the guest's" "$SESSION"
if [ -n "$HOST_MNT" ]; then
    check "$([ "$(echo "$SESSION" | cut -d'|' -f2)" = "$HOST_MNT" ] && echo true || echo false)" "session is in PID 1's mount namespace" "$SESSION"
fi
check "$([ "$(echo "$SESSION" | cut -d'|' -f3)" = "0" ] && echo true || echo false)" "session is root" "$SESSION"
check "$([ "$(echo "$SESSION" | cut -d'|' -f4)" = "guest" ] && echo true || echo false)" "session sees the guest root, not the container" "$SESSION"

TTY="$(ssh -i "$WORK/renter" "${SSH_OPTS[@]}" -tt -p "$PORT" root@127.0.0.1 'tty' 2>/dev/null | tr -d '\r' | head -1 || true)"
case "$TTY" in /dev/pts/*) ok "an interactive session gets a working terminal ($TTY)" ;; *) bad "an interactive session gets a working terminal" "$TTY" ;; esac

WORDS="$(as_renter 'echo "a b c" | wc -w' 2>/dev/null | tr -d ' ' || true)"
check "$([ "$WORDS" = "3" ] && echo true || echo false)" "a quoted remote command runs in the guest" "$WORDS"

SECOND="$(ssh -i "$WORK/second" "${SSH_OPTS[@]}" -p "$PORT" root@127.0.0.1 'echo second-ok' 2>/dev/null || true)"
check "$([ "$SECOND" = "second-ok" ] && echo true || echo false)" "the second authorized key logs in too" "$SECOND"

STRANGER_RC=0
ssh -i "$WORK/stranger" "${SSH_OPTS[@]}" -p "$PORT" root@127.0.0.1 true 2>/dev/null || STRANGER_RC=$?
check "$([ "$STRANGER_RC" -ne 0 ] && echo true || echo false)" "an unlisted key is refused" "rc=$STRANGER_RC"

PW_RC=0
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no "${SSH_OPTS[@]}" -p "$PORT" root@127.0.0.1 true 2>/dev/null || PW_RC=$?
check "$([ "$PW_RC" -ne 0 ] && echo true || echo false)" "password authentication is refused" "rc=$PW_RC"

PROBE="/tmp/lium-cvm-ssh-probe-$$"
echo "hello-from-renter" > "$WORK/probe.txt"
scp -q -i "$WORK/renter" "${SSH_OPTS[@]}" -P "$PORT" "$WORK/probe.txt" "root@127.0.0.1:$PROBE" 2>/dev/null || true
UPLOADED="$(as_renter "cat $PROBE && rm -f $PROBE" 2>/dev/null || true)"
check "$([ "$UPLOADED" = "hello-from-renter" ] && echo true || echo false)" "scp/sftp writes the guest filesystem" "$UPLOADED"

echo "== the host key survives a container restart"
run_container --pid=host --network=host --privileged -v /:/host -v /dev/pts:/dev/pts -v "$VOLUME:/etc/ssh/hostkeys"
FP2="$(docker logs "$NAME" 2>&1 | sed -n 's/.*host key fingerprint: //p' | head -1)"
check "$([ -n "$FP1" ] && [ "$FP1" = "$FP2" ] && echo true || echo false)" "same fingerprint after restart" "$FP1 vs $FP2"

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
