#!/usr/bin/env bash
# Collect every host fact this playbook needs, once, as root, and emit one JSON
# object on stdout.
#
# TRI-STATE
# ---------
# Every fact that can genuinely fail to be read is emitted as
#
#     {"value": ..., "readable": true|false, "reason": "..."}
#
# `readable: false` means value is null — UNKNOWN. It is never collapsed into
# zero and never into absent. That distinction is the whole point: "0 PCK certs"
# means the platform is not registered and registration should run; "cannot read
# pckcache.db" means we cannot tell, and running registration anyway flips an
# efivar that cannot be un-flipped.
#
# The role hard-fails on `readable: false` for any fact gating a destructive or
# irreversible action. An unprivileged collector would silently degrade every
# one of these to "absent", which is exactly the bug this shape prevents.
#
# WHAT IS DELIBERATELY NOT READ
# -----------------------------
#   /var/log/mpa_registration.log — it said "registration is completed
#     successfully" on the BROKEN host (au1) and the healthy one (au2) alike. It
#     is not a usable signal (traps.md#1).
#   platforms_registered           — 0 is normal on a healthy host.
#
# SGX registration truth is the pck_cert row count in pckcache.db. Nothing else.

set -Eeuo pipefail

PROC_CMDLINE_PATH="${LIUM_PROC_CMDLINE_PATH:-/proc/cmdline}"
GRUB_DEFAULT_PATH="${LIUM_GRUB_DEFAULT_PATH:-/etc/default/grub}"
GRUB_DROPIN_DIR="${LIUM_GRUB_DROPIN_DIR:-/etc/default/grub.d}"
STATE_DIR="${LIUM_STATE_DIR:-/var/lib/lium-cvm}"
REPO_PATH="${LIUM_REPO_PATH:-/opt/lium-io}"
LSPCI_OUTPUT="${LIUM_LSPCI_OUTPUT:-}"
NVTRUST_TOOL="${LIUM_NVTRUST_TOOL:-/opt/nvtrust/host_tools/python/nvidia_gpu_tools.py}"
CC_GPU_IDS="${LIUM_CC_GPU_PCI_IDS:-10de:2335,10de:2336,10de:2330,10de:233a}"
NVSWITCH_IDS="${LIUM_NVSWITCH_PCI_IDS:-10de:22a3}"
INTEL_API_HOST="${LIUM_INTEL_API_HOST:-api.trustedservices.intel.com}"
INTEL_API_PORT="${LIUM_INTEL_API_PORT:-443}"
DSTACKTEE="${REPO_PATH}/neurons/executor/dstacktee"

json_escape() {
  printf '%s' "$1" | LC_ALL=C sed \
    -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g' -e 's/\r/\\r/g' \
    -e ':a' -e 'N' -e '$!ba' -e 's/\n/\\n/g'
}

json_array_of_strings() {
  local first=1 item out=""
  for item in "$@"; do
    if [ "$first" -eq 1 ]; then first=0; else out="${out}, "; fi
    out="${out}\"$(json_escape "$item")\""
  done
  printf '[%s]' "$out"
}

FIELDS=()

plain() {  # name, already-valid-json-value
  FIELDS+=("\"$1\": $2")
}

plain_str() {  # name, string
  FIELDS+=("\"$1\": \"$(json_escape "$2")\"")
}

tri_ok() {  # name, already-valid-json-value
  FIELDS+=("\"$1\": {\"value\": $2, \"readable\": true, \"reason\": \"\"}")
}

tri_str_ok() {  # name, string
  FIELDS+=("\"$1\": {\"value\": \"$(json_escape "$2")\", \"readable\": true, \"reason\": \"\"}")
}

tri_unknown() {  # name, reason
  FIELDS+=("\"$1\": {\"value\": null, \"readable\": false, \"reason\": \"$(json_escape "$2")\"}")
}

# --- OS / hardware identity (trivially readable, so plain) --------------------
OS_ID=""; OS_VERSION_ID=""
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  OS_ID="${ID:-}"
  OS_VERSION_ID="${VERSION_ID:-}"
fi
plain_str os_id "$OS_ID"
plain_str os_version_id "$OS_VERSION_ID"
plain_str kernel "$(uname -r 2>/dev/null || printf '')"
plain_str arch "$(uname -m 2>/dev/null || printf '')"

if [ -r /proc/sys/kernel/random/boot_id ]; then
  tri_str_ok boot_id "$(cat /proc/sys/kernel/random/boot_id)"
else
  tri_unknown boot_id "/proc/sys/kernel/random/boot_id is not readable"
fi

# --- TDX / SGX / IOMMU platform ----------------------------------------------
if [ -e /sys/module/kvm_intel/parameters/tdx ]; then
  if [ -r /sys/module/kvm_intel/parameters/tdx ]; then
    tri_str_ok tdx_param "$(cat /sys/module/kvm_intel/parameters/tdx)"
  else
    tri_unknown tdx_param "/sys/module/kvm_intel/parameters/tdx exists but is not readable"
  fi
else
  # Absent is a real answer: the kvm_intel module is not loaded with TDX support.
  tri_str_ok tdx_param "N"
fi

SGX_DEVS=0
[ -e /dev/sgx_enclave ] && SGX_DEVS=$((SGX_DEVS + 1))
[ -e /dev/sgx_provision ] && SGX_DEVS=$((SGX_DEVS + 1))
tri_ok sgx_devices "$SGX_DEVS"

if [ -d /sys/class/iommu ]; then
  _n="$(find /sys/class/iommu -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)"
  tri_ok iommu_present "$([ "$_n" -gt 0 ] && printf 'true' || printf 'false')"
else
  tri_ok iommu_present false
fi

plain boot_grub_present "$([ -f /boot/grub/grub.cfg ] && printf 'true' || printf 'false')"
plain update_grub_present "$(command -v update-grub >/dev/null 2>&1 && printf 'true' || printf 'false')"

# --- SGX registration: the pck_cert row count, and nothing else --------------
PCKCACHE=/opt/intel/sgx-dcap-pccs/pckcache.db
if [ ! -e "$PCKCACHE" ]; then
  tri_unknown pck_cert_count "PCCS cache database $PCKCACHE does not exist (PCCS not installed or not configured)"
elif ! command -v python3 >/dev/null 2>&1; then
  tri_unknown pck_cert_count "python3 is not available to read $PCKCACHE"
else
  _pck="$(python3 -c "
import sqlite3, sys
try:
    c = sqlite3.connect('file:$PCKCACHE?mode=ro', uri=True)
    print(c.execute('select count(*) from pck_cert').fetchone()[0])
except Exception:
    sys.exit(1)
" 2>/dev/null || true)"
  if [ -n "$_pck" ]; then
    tri_ok pck_cert_count "$_pck"
  else
    tri_unknown pck_cert_count "cannot read pck_cert from $PCKCACHE; cannot distinguish 'not registered' from 'cannot tell', and registration cannot be undone"
  fi
fi

plain_str pccs_active "$(systemctl is-active pccs 2>/dev/null || printf 'unknown')"

if command -v ss >/dev/null 2>&1; then
  _v="$(ss -a --vsock 2>/dev/null | grep -c ':4050' || true)"
  tri_ok qgsd_vsock_4050 "$([ "${_v:-0}" -gt 0 ] && printf 'true' || printf 'false')"
else
  tri_unknown qgsd_vsock_4050 "ss is not available (install iproute2) so vsock listeners cannot be enumerated"
fi

QCNL=/etc/sgx_default_qcnl.conf
if [ -r "$QCNL" ]; then
  _sec="$(grep -oiE '"use_secure_cert"[[:space:]]*:[[:space:]]*(true|false)' "$QCNL" 2>/dev/null \
          | grep -oiE 'true|false' | head -1 || true)"
  if [ -n "$_sec" ]; then
    tri_str_ok qcnl_use_secure_cert "$_sec"
  else
    tri_unknown qcnl_use_secure_cert "$QCNL has no use_secure_cert key"
  fi
else
  tri_unknown qcnl_use_secure_cert "$QCNL is not readable"
fi

# Bounded reachability probe. An egress filter in front of Intel's API turns
# in-band registration into an opaque hang, so it is worth knowing up front.
if timeout 6 bash -c "exec 3<>/dev/tcp/${INTEL_API_HOST}/${INTEL_API_PORT}" 2>/dev/null; then
  tri_ok intel_api_reachable true
else
  tri_ok intel_api_reachable false
fi

# --- GPUs --------------------------------------------------------------------
GPU_BDFS=(); NVSWITCH_BDFS=(); NOT_VFIO=()
LSPCI_TEXT=""
LSPCI_READABLE=true
LSPCI_REASON=""

if [ -n "$LSPCI_OUTPUT" ]; then
  if [ -r "$LSPCI_OUTPUT" ]; then
    LSPCI_TEXT="$(cat "$LSPCI_OUTPUT")"
  else
    LSPCI_READABLE=false
    LSPCI_REASON="seam file $LSPCI_OUTPUT is not readable"
  fi
elif command -v lspci >/dev/null 2>&1; then
  LSPCI_TEXT="$(lspci -n 2>/dev/null || true)"
else
  LSPCI_READABLE=false
  LSPCI_REASON="cannot enumerate PCI devices: lspci not found (install pciutils)"
fi

# Substring test done in the shell, NEVER as `printf ... | grep -q`.
#
# `grep -q` exits the instant it matches, so printf loses the rest of its write
# to SIGPIPE and returns 141. Under `set -o pipefail` — which this script sets —
# the PIPELINE then reports 141, so a SUCCESSFUL match reads as "not found". It
# is a race with the scheduler, not a constant: over 30 runs on one unchanged
# 8xH200 host this dropped an id 8 times and returned an empty list twice, while
# `lspci -n` listed all 12 devices every time.
#
# That matters more here than anywhere else in this tree: present_pci_ids is what
# `vfio-pci.ids=` is built from. A run that silently loses 10de:2335 writes a
# GRUB line binding only the NVSwitches, and after the reboot not one of the
# eight GPUs is available to any CVM.
str_contains_any() {
  local haystack="$1" _needle
  local IFS=','
  for _needle in $2; do
    [ -n "$_needle" ] || continue
    case "$haystack" in *"$_needle"*) return 0 ;; esac
  done
  return 1
}

if [ "$LSPCI_READABLE" = "true" ]; then
  while IFS= read -r _line; do
    [ -n "$_line" ] || continue
    # Parameter expansion, not `| cut -f1`: this runs once per PCI device, and
    # two forked processes per line is the whole cost of the collector on a
    # machine with a long device list.
    _bdf="${_line%% *}"
    case "$_bdf" in
      *:*:*) ;;                       # already domain-qualified
      *) _bdf="0000:${_bdf}" ;;       # lspci -n omits the domain
    esac
    if str_contains_any "$_line" "$NVSWITCH_IDS"; then
      NVSWITCH_BDFS+=("$_bdf")
    elif str_contains_any "$_line" "$CC_GPU_IDS"; then
      GPU_BDFS+=("$_bdf")
    else
      continue
    fi
    _drv="$(basename "$(readlink -f "/sys/bus/pci/devices/${_bdf}/driver" 2>/dev/null || printf 'none')")"
    [ "$_drv" = "vfio-pci" ] || NOT_VFIO+=("$_bdf")
  done <<<"$LSPCI_TEXT"
  tri_ok lspci_available true
else
  tri_unknown lspci_available "$LSPCI_REASON"
fi

# Which of the configured PCI ids are actually PRESENT on this host. This is
# what `vfio-pci.ids=` is built from — it takes vendor:device pairs, not BDFs —
# and what the superset rule compares any pre-existing list against.
PRESENT_IDS=()
if [ "$LSPCI_READABLE" = "true" ]; then
  for _id in $(printf '%s,%s' "$CC_GPU_IDS" "$NVSWITCH_IDS" | tr ',' ' '); do
    [ -n "$_id" ] || continue
    if str_contains_any "$LSPCI_TEXT" "$_id"; then
      PRESENT_IDS+=("$_id")
    fi
  done
fi
plain present_pci_ids "$(json_array_of_strings ${PRESENT_IDS[@]+"${PRESENT_IDS[@]}"})"

plain gpu_bdfs "$(json_array_of_strings ${GPU_BDFS[@]+"${GPU_BDFS[@]}"})"
plain nvswitch_bdfs "$(json_array_of_strings ${NVSWITCH_BDFS[@]+"${NVSWITCH_BDFS[@]}"})"
plain devices_not_vfio "$(json_array_of_strings ${NOT_VFIO[@]+"${NOT_VFIO[@]}"})"

# CC mode per GPU. Unknown here BLOCKS the flip — never flip a mode we cannot
# read. `--gpu-bdf=` is required: a bare --query-cc-mode only lists devices.
cc_json="{"
cc_first=1
for _bdf in ${GPU_BDFS[@]+"${GPU_BDFS[@]}"}; do
  [ "$cc_first" -eq 1 ] && cc_first=0 || cc_json="${cc_json},"
  if [ -r "$NVTRUST_TOOL" ] && command -v python3 >/dev/null 2>&1; then
    _q="$(timeout 60 python3 "$NVTRUST_TOOL" --gpu-bdf="$_bdf" --query-cc-mode 2>&1 \
          | grep -oiE 'CC mode is (on|off)|broken' | tail -1 || true)"
    case "$_q" in
      *[Oo]n) cc_json="${cc_json}\"$(json_escape "$_bdf")\": {\"value\": \"on\", \"readable\": true, \"reason\": \"\"}" ;;
      *[Oo]ff) cc_json="${cc_json}\"$(json_escape "$_bdf")\": {\"value\": \"off\", \"readable\": true, \"reason\": \"\"}" ;;
      *) cc_json="${cc_json}\"$(json_escape "$_bdf")\": {\"value\": null, \"readable\": false, \"reason\": \"nvtrust query failed for $(json_escape "$_bdf"); refusing to flip a mode we cannot read\"}" ;;
    esac
  else
    cc_json="${cc_json}\"$(json_escape "$_bdf")\": {\"value\": null, \"readable\": false, \"reason\": \"nvtrust tool $(json_escape "$NVTRUST_TOOL") is not available\"}"
  fi
done
cc_json="${cc_json}}"
plain gpu_cc_mode "$cc_json"

# --- kernel command line -----------------------------------------------------
DMA_PARAM=/sys/module/vfio_iommu_type1/parameters/dma_entry_limit
if [ -r "$DMA_PARAM" ]; then
  tri_ok dma_entry_limit "$(cat "$DMA_PARAM")"
else
  tri_unknown dma_entry_limit "$DMA_PARAM is not readable (the vfio_iommu_type1 module may not be loaded)"
fi
plain dma_persisted "$([ -f /etc/modprobe.d/vfio-dma.conf ] && printf 'true' || printf 'false')"

if [ -r "$PROC_CMDLINE_PATH" ]; then
  tri_str_ok proc_cmdline "$(tr -d '\n' <"$PROC_CMDLINE_PATH")"
else
  tri_unknown proc_cmdline "cannot read $PROC_CMDLINE_PATH; refusing to reboot on unknown drift"
fi

# This reads the GRUB files a SECOND time — lium-cmdline.sh in the same
# directory also parses them — and the difference is deliberate, not drift.
#
# lium-cmdline.sh answers "what does the next boot get?": last occurrence wins,
# and the caller can exclude our own drop-in. That is the right question for
# deciding what to WRITE.
#
# The scan below answers "was this id ever configured?": every occurrence in
# both variables, our drop-in INCLUDED. That is the right question for the
# narrowing refusal, which must see ids a previous run of this playbook wrote.
# Substituting either answer for the other reintroduces a defect: refusing over
# a healthy host, or shortening the list on a wedged GPU.
_grub_linux=""; _grub_default=""
if [ -r "$GRUB_DEFAULT_PATH" ]; then
  while IFS=$'\t' read -r _k _v; do
    case "$_k" in
      LINUX) _grub_linux="$_v" ;;
      DEFAULT) _grub_default="$_v" ;;
    esac
  done < <(
    set +u
    GRUB_CMDLINE_LINUX=""; GRUB_CMDLINE_LINUX_DEFAULT=""
    # shellcheck source=/dev/null
    . "$GRUB_DEFAULT_PATH"
    if [ -d "$GRUB_DROPIN_DIR" ]; then
      for _f in "$GRUB_DROPIN_DIR"/*; do
        [ -r "$_f" ] || continue
        # shellcheck source=/dev/null
        . "$_f"
      done
    fi
    printf 'LINUX\t%s\n' "$GRUB_CMDLINE_LINUX"
    printf 'DEFAULT\t%s\n' "$GRUB_CMDLINE_LINUX_DEFAULT"
  )
fi

_existing_ids=()
for _src in "$_grub_linux" "$_grub_default"; do
  for _tok in $_src; do
    case "$_tok" in
      vfio-pci.ids=*)
        _ids="${_tok#vfio-pci.ids=}"
        IFS=',' read -r -a _arr <<<"$_ids"
        for _id in "${_arr[@]}"; do
          [ -n "$_id" ] && _existing_ids+=("$_id")
        done
        ;;
    esac
  done
done
if [ "${#_existing_ids[@]}" -gt 0 ]; then
  mapfile -t _existing_ids < <(printf '%s\n' "${_existing_ids[@]}" | sort -u)
fi
plain grub_existing_vfio_ids "$(json_array_of_strings ${_existing_ids[@]+"${_existing_ids[@]}"})"

# --- QEMU --------------------------------------------------------------------
QEMU_BIN="${LIUM_QEMU_PREFIX:-/opt/qemu-dstack}/bin/qemu-system-x86_64"
if [ -x "$QEMU_BIN" ]; then
  _qv="$("$QEMU_BIN" --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || true)"
  if [ -n "$_qv" ]; then
    tri_str_ok qemu_version "$_qv"
  else
    tri_unknown qemu_version "cannot read the installed QEMU version; refusing to replace a binary whose identity is unknown"
  fi
  _qs="$(sha256sum "$QEMU_BIN" 2>/dev/null | cut -d' ' -f1 || true)"
  if [ -n "$_qs" ]; then
    tri_str_ok qemu_sha256 "$_qs"
  else
    tri_unknown qemu_sha256 "cannot hash $QEMU_BIN; refusing to replace a binary whose identity is unknown"
  fi
else
  # Not installed is a definite answer, not an unknown one.
  tri_str_ok qemu_version ""
  tri_str_ok qemu_sha256 ""
fi

# Read client.conf directly. `lium-cvm.sh check` looks for qemu-system-x86_64 on
# PATH (lium-cvm.sh:145) while the launcher uses client.conf, so it reports a
# false MISSING on a correctly configured host.
_conf_path=""
if [ -r /etc/dstack/client.conf ]; then
  _conf_path="$(grep -hs '^path' /etc/dstack/client.conf | tr -d ' ' | cut -d= -f2 || true)"
fi
plain_str qemu_client_conf_path "$_conf_path"

_qemu_recorded=""
[ -r "${STATE_DIR}/qemu.sha256" ] && _qemu_recorded="$(cat "${STATE_DIR}/qemu.sha256")"
plain_str qemu_recorded_sha256 "$_qemu_recorded"

# --- Docker and the key provider ---------------------------------------------
plain_str docker_version "$(docker --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || printf '')"
plain_str kp_running "$(docker inspect -f '{{.State.Running}}' dstack-key-provider 2>/dev/null || printf 'unknown')"
plain_str kp_restarts "$(docker inspect -f '{{.RestartCount}}' dstack-key-provider 2>/dev/null || printf 'unknown')"
_kp_logs=""
if command -v docker >/dev/null 2>&1; then
  _kp_logs="$(docker logs dstack-key-provider 2>&1 || true)"
fi
plain kp_production "$(printf '%s' "$_kp_logs" | grep -c 'PRODUCTION mode' || true)"
plain kp_aesm_error "$(printf '%s' "$_kp_logs" | grep -ciE 'error 44|load_enclave.*failed' || true)"

# --- repo --------------------------------------------------------------------
plain repo_present "$([ -d "$DSTACKTEE" ] && printf 'true' || printf 'false')"

_dirty=()
if [ -d "${REPO_PATH}/.git" ] && command -v git >/dev/null 2>&1; then
  while IFS= read -r _l; do
    [ -n "$_l" ] && _dirty+=("$_l")
  done < <(git -C "$REPO_PATH" status --porcelain neurons/executor/dstacktee/app 2>/dev/null || true)
fi
plain app_dirty_files "$(json_array_of_strings ${_dirty[@]+"${_dirty[@]}"})"

_repo_ref=""
if [ -d "${REPO_PATH}/.git" ] && command -v git >/dev/null 2>&1; then
  _repo_ref="$(git -C "$REPO_PATH" rev-parse HEAD 2>/dev/null || printf '')"
fi
plain_str repo_head "$_repo_ref"

_os_digest=""
for _d in "$DSTACKTEE"/run/images/*/digest.txt; do
  [ -r "$_d" ] || continue
  _os_digest="$(head -1 "$_d")"
  break
done
plain_str os_image_digest "$_os_digest"
plain_str env_runner_digest "$(grep -hs '^EXECUTOR_RUNNER_IMAGE_DIGEST=' "$DSTACKTEE/.env" 2>/dev/null | cut -d= -f2 || printf '')"
plain_str env_external_port "$(grep -hs '^EXTERNAL_PORT=' "$DSTACKTEE/.env" 2>/dev/null | cut -d= -f2 | awk '{print $1}' || printf '')"

# --- disk headroom -----------------------------------------------------------
# Reported against the nearest EXISTING ancestor, because on a fresh host
# neither the build directory nor the repo path exists yet and `df` on a missing
# path would report nothing at all rather than the filesystem that will hold it.
free_gb_for() {
  local path="$1"
  while [ -n "$path" ] && [ ! -d "$path" ]; do
    path="$(dirname "$path")"
    [ "$path" = "/" ] && break
  done
  df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -cd '0-9'
}

_bs="$(free_gb_for "${LIUM_QEMU_BUILD_DIR:-/usr/local/lib/lium-cvm}")"
if [ -n "$_bs" ]; then tri_ok build_space_gb "$_bs"; else tri_unknown build_space_gb "df could not report free space for the build directory"; fi

_is="$(free_gb_for "$REPO_PATH")"
if [ -n "$_is" ]; then tri_ok image_space_gb "$_is"; else tri_unknown image_space_gb "df could not report free space for $REPO_PATH"; fi

# --- emit --------------------------------------------------------------------
printf '{\n'
_n=${#FIELDS[@]}
for _i in "${!FIELDS[@]}"; do
  if [ "$((_i + 1))" -lt "$_n" ]; then
    printf '  %s,\n' "${FIELDS[$_i]}"
  else
    printf '  %s\n' "${FIELDS[$_i]}"
  fi
done
printf '}\n'
