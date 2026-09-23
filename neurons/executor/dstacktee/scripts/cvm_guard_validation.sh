#!/bin/bash

# Validation of cvm_upgrade_guard.sh on a real SGX/TDX host (DAH-3188 prerequisite).
# Prints one PASS/FAIL line per case and a summary; deletes no CVM disk, ever.
# Run every phase from the dstacktee directory of the checkout under test, as root.
#
#   sudo scripts/cvm_guard_validation.sh phase1 <cvm>   # with <cvm> created and running:
#                                                       #   refused upgrade (running, then stopped), missing pinned
#                                                       #   image, restart on the pinned image; then reboot the host
#   sudo scripts/cvm_guard_validation.sh phase2 <cvm>   # after the host reboot: the CVM boots and keeps its data
#   sudo scripts/cvm_guard_validation.sh phase3 <cvm>   # after you removed every listed disk by hand: the empty-host
#                                                       #   upgrade, new MRENCLAVE, a new <cvm> created and started
#   sudo scripts/cvm_guard_validation.sh phase4 <cvm>   # after `lium-cvm.sh stop <cvm>` + `run`: the new CVM keeps its data
#
# Data preservation is judged by three signals from outside the guest: the serial
# console shows no "Failed to open encrypted data disk", the executor API answers on
# EXTERNAL_PORT, and the SSH host key on SSH_PUBLIC_PORT is the one recorded before
# (the key lives on the encrypted data disk). With GUEST_EXEC set to a command that
# runs a shell line inside the guest (for example "ssh -p 2200 root@127.0.0.1"), a
# marker file under /var/lib is written in phase1 and read in phase2/phase4 as well.
#
# State between phases: run/validation/ (fingerprints, mr_enclave, the console logs).
# Paste the whole output on the prerequisite PR.

set -u

DSTACKTEE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$DSTACKTEE/cvm_upgrade_guard.sh"
LIUM_CVM="$DSTACKTEE/lium-cvm.sh"
KP_DIR="$DSTACKTEE/key-provider"
STATE="${VALIDATION_DIR:-$DSTACKTEE/run/validation}"
PIN_FILE="${LIUM_CVM_STATE_DIR:-/var/lib/lium-cvm}/key-provider.image"
BOOT_WAIT="${BOOT_WAIT:-420}"
PHASE="${1:-}"
CVM="${2:-}"
PASS_N=0
FAIL_N=0

say() { echo "== $*"; }
indent() { printf '%s\n' "$1" | sed 's/^/    /'; }
pass() { PASS_N=$((PASS_N + 1)); echo "PASS  $*"; }
fail() { FAIL_N=$((FAIL_N + 1)); echo "FAIL  $*"; }
check() { # check <description> <command...>: PASS when the command exits 0
    local what="$1"
    shift
    if "$@"; then pass "$what"; else fail "$what (rc=$?)"; fi
}
summary() {
    echo
    echo "== $PHASE: $PASS_N passed, $FAIL_N failed"
    [ "$FAIL_N" -eq 0 ]
}

need_cvm() {
    [ -n "$CVM" ] || { echo "usage: $0 $PHASE <cvm-name>" >&2; exit 1; }
}

load_env() {
    [ -f "$DSTACKTEE/.env" ] || { echo ".env not found in $DSTACKTEE" >&2; exit 1; }
    set -a
    # shellcheck disable=SC1091
    source "$DSTACKTEE/.env"
    set +a
    : "${EXTERNAL_PORT:?EXTERNAL_PORT missing in .env}" "${SSH_PUBLIC_PORT:?SSH_PUBLIC_PORT missing in .env}"
}

pinned_id() { awk 'NR==1 {print $1}' "$PIN_FILE" 2>/dev/null; }
container_image() { docker inspect --format '{{.Image}}' dstack-key-provider 2>/dev/null; }
kp_mr_enclave() {
    (cd "$KP_DIR" && docker compose logs gramine-sealing-key-provider 2>/dev/null) | grep -o '"mr_enclave": *"[0-9a-f]*"' | tail -1 | grep -o '[0-9a-f]\{64\}'
}
ssh_fingerprint() {
    ssh-keyscan -T 10 -p "$SSH_PUBLIC_PORT" 127.0.0.1 2>/dev/null | ssh-keygen -lf - 2>/dev/null | awk '{print $2}' | sort | tr '\n' ' '
}
api_answers() { curl -s -o /dev/null -m 10 "http://127.0.0.1:$EXTERNAL_PORT/" ; }
disk_error_in() { grep -q 'Failed to open encrypted data disk' "$1" 2>/dev/null; }
no_disk_error_in() { ! disk_error_in "$1"; }
differs() { [ -n "$1" ] && [ -n "$2" ] && [ "$1" != "$2" ]; }
same_nonempty() { [ -n "$1" ] && [ "$1" = "$2" ]; }
pre_upgrade_tag_exists() { docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^lium-key-provider:pre-upgrade-'; }

# Start the CVM in the background; its serial console goes to $STATE/<cvm>.<phase>.console.log.
start_cvm_bg() {
    local log="$STATE/$CVM.$PHASE.console.log"
    : >"$log"
    (cd "$DSTACKTEE" && nohup "$LIUM_CVM" run "$CVM" >"$log" 2>&1 </dev/null &)
    echo "$log"
}

wait_for_api() {
    local log="$1" waited=0
    while [ "$waited" -lt "$BOOT_WAIT" ]; do
        if api_answers; then return 0; fi
        if disk_error_in "$log"; then return 2; fi
        sleep 10
        waited=$((waited + 10))
    done
    return 1
}

guest_marker_write() {
    [ -n "${GUEST_EXEC:-}" ] || return 0
    $GUEST_EXEC "echo cvm-guard-$(date -u +%s) > /var/lib/cvm-guard-marker && cat /var/lib/cvm-guard-marker" >"$STATE/marker.txt" 2>/dev/null
}
guest_marker_check() {
    [ -n "${GUEST_EXEC:-}" ] || return 0
    local now
    now="$($GUEST_EXEC "cat /var/lib/cvm-guard-marker" 2>/dev/null)"
    [ -n "$now" ] && [ "$now" = "$(cat "$STATE/marker.txt" 2>/dev/null)" ]
}

phase1() {
    need_cvm
    load_env
    mkdir -p "$STATE"
    local out rc pin_before img_before mr_before console fp tmp_state images_before
    pin_before="$(pinned_id)"
    img_before="$(container_image)"
    say "phase1 on $(hostname): CVM $CVM, pinned image ${pin_before:-<none>}, container image ${img_before:-<none>}"

    say "case 1: the inventory lists the running CVM and refuses"
    out="$("$GUARD" inventory 2>&1)"; rc=$?
    indent "$out"
    check "inventory exits 3 with a disk" [ "$rc" -eq 3 ]
    check "inventory names run/vms/$CVM/hda.img" grep -q "run/vms/$CVM/hda.img" <<<"$out"

    say "case 2: the upgrade is refused while the CVM runs and changes nothing"
    out="$("$GUARD" upgrade 2>&1)"; rc=$?
    indent "$out"
    check "upgrade exits 3" [ "$rc" -eq 3 ]
    check "upgrade prints the manual removal line" grep -q "sudo rm -rf .*run/vms/$CVM" <<<"$out"
    check "pinned image unchanged" [ "$(pinned_id)" = "$pin_before" ]
    check "container image unchanged" [ "$(container_image)" = "$img_before" ]
    check "hda.img still present" [ -f "$DSTACKTEE/run/vms/$CVM/hda.img" ]

    say "case 2b: a hand-run docker compose build in key-provider builds nothing"
    images_before="$(docker images -q | sort -u)"
    out="$(cd "$KP_DIR" && docker compose build 2>&1)"; rc=$?
    indent "$out"
    check "compose build exits 0 with nothing to build" [ "$rc" -eq 0 ]
    check "image set unchanged" [ "$(docker images -q | sort -u)" = "$images_before" ]
    check "lium-key-provider:local still the pinned image" [ "$(docker image inspect --format '{{.Id}}' lium-key-provider:local 2>/dev/null)" = "$pin_before" ]

    say "case 3: record the guest's identity before the reboot"
    check "executor API answers on $EXTERNAL_PORT" api_answers
    fp="$(ssh_fingerprint)"
    echo "$fp" >"$STATE/ssh_fingerprint.txt"
    check "SSH host key recorded ($fp)" [ -n "$fp" ]
    kp_mr_enclave >"$STATE/mr_enclave.before.txt"
    mr_before="$(cat "$STATE/mr_enclave.before.txt")"
    check "key-provider mr_enclave recorded (${mr_before:-<not in logs>})" [ -n "$mr_before" ]
    if [ -n "${GUEST_EXEC:-}" ]; then check "marker written in the guest" guest_marker_write; fi

    say "case 4: a stopped CVM still blocks the upgrade"
    (cd "$DSTACKTEE" && "$LIUM_CVM" stop "$CVM" --timeout 120 --force) >"$STATE/$CVM.stop.log" 2>&1
    out="$("$GUARD" inventory 2>&1)"; rc=$?
    indent "$out"
    check "inventory exits 3 with the CVM stopped" [ "$rc" -eq 3 ]
    check "inventory marks it stopped" grep -q "stopped .*run/vms/$CVM/hda.img" <<<"$out"
    out="$("$GUARD" upgrade 2>&1)"; rc=$?
    check "upgrade exits 3 with the CVM stopped" [ "$rc" -eq 3 ]
    check "pinned image unchanged" [ "$(pinned_id)" = "$pin_before" ]

    say "case 5: a missing pinned image is refused, nothing is built"
    tmp_state="$(mktemp -d)"
    cp "${LIUM_CVM_STATE_DIR:-/var/lib/lium-cvm}/vm-dirs" "$tmp_state/vm-dirs" 2>/dev/null || echo "$DSTACKTEE/run/vms" >"$tmp_state/vm-dirs"
    echo "sha256:$(printf '0%.0s' $(seq 64)) validation" >"$tmp_state/key-provider.image"
    images_before="$(docker images -q | sort -u)"
    out="$(LIUM_CVM_STATE_DIR="$tmp_state" "$GUARD" start 2>&1)"; rc=$?
    indent "$out"
    check "start exits 6 with the pinned image missing" [ "$rc" -eq 6 ]
    check "start prints recovery guidance" grep -q "Recovery" <<<"$out"
    check "no image was built" [ "$(docker images -q | sort -u)" = "$images_before" ]
    check "real pin untouched" [ "$(pinned_id)" = "$pin_before" ]
    rm -rf "$tmp_state"

    say "case 6: the CVM starts again on the pinned image"
    console="$(start_cvm_bg)"
    wait_for_api "$console"; rc=$?
    check "CVM boots and the API answers (console: $console)" [ "$rc" -eq 0 ]
    check "no encrypted-disk error on the console" no_disk_error_in "$console"
    check "key provider runs the pinned image" [ "$(container_image)" = "$pin_before" ]
    check "SSH host key unchanged" [ "$(ssh_fingerprint)" = "$fp" ]

    summary
    echo "Next: reboot the host, then: sudo $0 phase2 $CVM"
}

phase2() {
    need_cvm
    load_env
    local console rc pin fp_before
    pin="$(pinned_id)"
    fp_before="$(cat "$STATE/ssh_fingerprint.txt" 2>/dev/null)"
    say "phase2 on $(hostname) after reboot: CVM $CVM, pinned image ${pin:-<none>}, uptime $(cut -d. -f1 /proc/uptime)s"
    [ -n "$fp_before" ] || { echo "no $STATE/ssh_fingerprint.txt: run phase1 first" >&2; exit 1; }

    say "case 7: after a refused upgrade and a host reboot the CVM keeps its data"
    console="$(start_cvm_bg)"
    wait_for_api "$console"; rc=$?
    check "key provider came up on the pinned image (no build)" [ "$(container_image)" = "$pin" ]
    check "CVM boots and the API answers (console: $console)" [ "$rc" -eq 0 ]
    check "no encrypted-disk error on the console" no_disk_error_in "$console"
    check "SSH host key identical to phase1 ($fp_before)" [ "$(ssh_fingerprint)" = "$fp_before" ]
    check "key-provider mr_enclave unchanged" same_nonempty "$(kp_mr_enclave)" "$(cat "$STATE/mr_enclave.before.txt")"
    if [ -n "${GUEST_EXEC:-}" ]; then check "marker still in the guest" guest_marker_check; fi

    summary
    echo "Next: drain, then stop $CVM and remove its disk BY HAND with the lines 'sudo ./cvm_upgrade_guard.sh inventory' prints;"
    echo "      then: sudo $0 phase3 $CVM"
}

phase3() {
    need_cvm
    load_env
    local out rc pin_before pin_after mr_before mr_after console fp
    pin_before="$(pinned_id)"
    mr_before="$(cat "$STATE/mr_enclave.before.txt" 2>/dev/null)"
    say "phase3 on $(hostname): empty-host upgrade, pinned image before ${pin_before:-<none>}"

    say "case 8: the inventory is empty and the upgrade goes through"
    out="$("$GUARD" inventory 2>&1)"; rc=$?
    indent "$out"
    if [ "$rc" -ne 0 ]; then
        fail "inventory is not empty (rc=$rc): remove the listed disks by hand first; this script never does"
        summary
        return 1
    fi
    pass "inventory exits 0"
    out="$("$GUARD" upgrade 2>&1)"; rc=$?
    indent "$out"
    check "upgrade exits 0" [ "$rc" -eq 0 ]
    pin_after="$(pinned_id)"
    check "new image pinned ($pin_after)" differs "$pin_after" "$pin_before"
    check "container runs the new image" [ "$(container_image)" = "$pin_after" ]
    check "previous image kept as lium-key-provider:pre-upgrade-*" pre_upgrade_tag_exists
    sleep 15
    mr_after="$(kp_mr_enclave)"
    echo "$mr_after" >"$STATE/mr_enclave.after.txt"
    check "new mr_enclave ${mr_after:-<not in logs yet>} differs from ${mr_before:-<unknown>}" differs "$mr_after" "$mr_before"

    say "case 9: a new CVM on the upgraded key provider"
    (cd "$DSTACKTEE" && "$LIUM_CVM" new "$CVM") >"$STATE/$CVM.new.log" 2>&1
    check "lium-cvm.sh new $CVM" [ -f "$DSTACKTEE/run/vms/$CVM/vm-manifest.json" ]
    console="$(start_cvm_bg)"
    wait_for_api "$console"; rc=$?
    check "new CVM boots and the API answers (console: $console)" [ "$rc" -eq 0 ]
    check "no encrypted-disk error on the console" no_disk_error_in "$console"
    fp="$(ssh_fingerprint)"
    echo "$fp" >"$STATE/ssh_fingerprint.new.txt"
    check "new SSH host key recorded ($fp)" [ -n "$fp" ]
    if [ -n "${GUEST_EXEC:-}" ]; then check "marker written in the new guest" guest_marker_write; fi

    summary
    echo "Next: sudo ./lium-cvm.sh stop $CVM; then: sudo $0 phase4 $CVM"
    echo "Also record on the PR: the validator log line 'Attestation verified' for this executor, and the verifier's key_provider_info id (= $mr_after)."
}

phase4() {
    need_cvm
    load_env
    local console rc fp_before
    fp_before="$(cat "$STATE/ssh_fingerprint.new.txt" 2>/dev/null)"
    [ -n "$fp_before" ] || { echo "no $STATE/ssh_fingerprint.new.txt: run phase3 first" >&2; exit 1; }
    say "phase4 on $(hostname): the new CVM across a stop/start"

    say "case 10: the new CVM keeps its data across a reboot"
    console="$(start_cvm_bg)"
    wait_for_api "$console"; rc=$?
    check "CVM boots and the API answers (console: $console)" [ "$rc" -eq 0 ]
    check "no encrypted-disk error on the console" no_disk_error_in "$console"
    check "SSH host key identical to phase3 ($fp_before)" [ "$(ssh_fingerprint)" = "$fp_before" ]
    check "key provider still on the pinned image" [ "$(container_image)" = "$(pinned_id)" ]
    if [ -n "${GUEST_EXEC:-}" ]; then check "marker still in the new guest" guest_marker_check; fi

    summary
}

case "$PHASE" in
phase1) phase1 ;;
phase2) phase2 ;;
phase3) phase3 ;;
phase4) phase4 ;;
*)
    sed -n '3,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
