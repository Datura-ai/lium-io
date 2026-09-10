#!/bin/bash
set -e

# Lium Executor — Sysbox Setup
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Datura-ai/lium-io/main/neurons/executor/nvidia_docker_sysbox_setup.sh | sudo bash
#   or: cd lium-io/neurons/executor && sudo bash nvidia_docker_sysbox_setup.sh
#   sudo bash nvidia_docker_sysbox_setup.sh --check   only the preflight, one PASS/FIX line per requirement; exit 1 on any FIX
# Env:
#   SYSBOX_SKIP_KERNEL_CHECK=1  install even when the ID-mapped mounts check rejects the host
#   EXECUTOR_PORT / SSH_PORT    the ports the preflight checks (else neurons/executor/.env next to this script, else 8080 / 2200)

SYSBOX_VERSION="0.6.6"
SYSBOX_DEB_URL="https://github.com/nestybox/sysbox/releases/download/v${SYSBOX_VERSION}/sysbox-ce_${SYSBOX_VERSION}-0.linux_amd64.deb"
SYSBOX_SHA="87cfa5cad97dc5dc1a243d6d88be1393be75b93a517dc1580ecd8a2801c2777a"
VERIFY_IMAGE="daturaai/compute-subnet-executor:latest"
DOWNLOADED_DEB=""

G='\033[0;32m' Y='\033[1;33m' R='\033[0;31m' B='\033[1;34m' N='\033[0m'
ok()   { echo -e "  ${G}✓${N} $1"; }
warn() { echo -e "  ${Y}!${N} $1"; }
fail() { echo -e "  ${R}✗${N} $1"; }
step() { echo -e "\n${B}[$1/$2]${N} $3"; }

# an `if`, not `[ … ] && rm`: under `set -e` the failing test made the EXIT trap end every run
# with status 1, including "Nothing to do." and SUCCESS
cleanup() { if [ -n "$DOWNLOADED_DEB" ]; then rm -f "$DOWNLOADED_DEB"; fi; }
trap cleanup EXIT

version_ge() {
    # a dotted version string ("29.5.1", "5.15.0-91-generic") against a major/minor floor
    local major minor
    major=${1%%.*} minor=${1#*.} minor=${minor%%.*}
    [ "$major" -gt "$2" ] 2>/dev/null || { [ "$major" -eq "$2" ] && [ "$minor" -ge "$3" ]; } 2>/dev/null
}

docker_server_version() {
    # the DAEMON version — it writes the OCI spec sysbox has to accept, and it can differ from the client
    docker version --format '{{.Server.Version}}' 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
}

docker_version_ge() { version_ge "$(docker_server_version)" "$1" "$2"; }

version3_ge() {
    # "a.b.c" >= "x.y.z", all three components numeric (driver versions: 580.65.06 vs the validators' minimum)
    awk -v a="$1" -v b="$2" 'BEGIN {
        n = split(a, x, "."); split(b, y, ".")
        for (i = 1; i <= 3; i++) { if (x[i] + 0 > y[i] + 0) exit 0; if (x[i] + 0 < y[i] + 0) exit 1 }
        exit 0 }'
}

kernel_supports_idmapped() {
    # overlayfs over ID-mapped mounts landed in 5.19; without it sysbox falls back to shiftfs
    version_ge "$(uname -r)" 5 19
}

apt_install() {
    # apt's output is kept and shown on failure: with `set -e` a silenced apt-get ended the
    # script after "Installing packages" with nothing on screen (DAH-2768)
    local log
    log=$(mktemp)
    if ! apt-get "$@" > "$log" 2>&1; then
        fail "apt-get $* failed:"
        tail -n 20 "$log" | sed 's/^/      /'
        rm -f "$log"
        return 1
    fi
    rm -f "$log"
}

ensure_nvidia_container_toolkit_repo() {
    # nvidia-container-toolkit ships from NVIDIA's apt repository, not Ubuntu's. Drivers installed
    # from the Ubuntu archive or a .run file leave no such repository behind, so add it (NVIDIA's
    # documented sequence) unless any apt source already points at it.
    local root="${APT_ROOT:-}"
    local keyring="$root/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    local list="$root/etc/apt/sources.list.d/nvidia-container-toolkit.list"
    if grep -rqs "nvidia.github.io/libnvidia-container" "$root/etc/apt/sources.list.d/" "$root/etc/apt/sources.list"; then
        ok "NVIDIA container toolkit apt repository already configured."
        return 0
    fi
    # downloads land in a temp file first: in a `curl | gpg` pipeline a failed curl is invisible
    local tmp
    tmp=$(mktemp)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o "$tmp" \
        || { rm -f "$tmp"; fail "Could not fetch the NVIDIA container toolkit signing key (https://nvidia.github.io/libnvidia-container/gpgkey)."; return 1; }
    gpg --dearmor --yes -o "$keyring" "$tmp" \
        || { rm -f "$tmp"; fail "Could not import the NVIDIA container toolkit signing key into $keyring."; return 1; }
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list -o "$tmp" \
        || { rm -f "$tmp"; fail "Could not fetch the NVIDIA container toolkit apt source list."; return 1; }
    sed "s#deb https://#deb [signed-by=$keyring] https://#g" "$tmp" > "$list"
    rm -f "$tmp"
    ok "NVIDIA container toolkit apt repository added."
}

sysbox_idmapped_report() {
    # what sysbox-mgr itself decided this boot: "yes", "no", or empty when unavailable
    journalctl -u sysbox-mgr -b --no-pager 2>/dev/null \
        | grep -o 'Overlayfs on ID-mapped mounts supported by kernel: [a-z]*' \
        | tail -1 | awk '{print $NF}'
}

abort_on_active_rentals() {
    # a rental blocks every path below, so check before anything that costs the node time or bandwidth
    docker ps --filter "name=pod_" --format '{{.Names}}' 2>/dev/null | grep -q . || return 0
    fail "Active rentals found (pod_* containers). Cannot proceed."
    docker ps --filter "name=pod_" --format "    - {{.Names}}" 2>/dev/null
    exit 1
}

nvidia_hook_symptom() {
    fail "Sysbox then falls back to shiftfs and runs the container rootfs from another path, so the NVIDIA hook fails with:"
    fail "  nvidia-container-cli: mount error: .../merged/proc/driver/nvidia: no such file or directory"
}

fail_no_idmapped() {
    fail "Sysbox reports it cannot use ID-mapped mounts, although kernel $(uname -r) supports them."
    nvidia_hook_symptom
    fail "Usual causes: Docker's data-root sits on a filesystem without ID-map support (ZFS, some btrfs setups),"
    fail "or sysbox-mgr runs with ID-mapped mounts disabled. Check:"
    fail "  docker info --format '{{.Driver}}'   (overlay2 expected)"
    fail "  journalctl -u sysbox-mgr -b | grep -i id-mapped"
}

# ── Preflight ───────────────────────────────────────────
# One PASS / FIX line per requirement, each FIX with the command that fixes it. The host checks
# run before anything is installed (a FIX there ends the run with nothing changed); `--check`
# runs them plus the checks on what this script installs. Paths under HOST_ROOT so tests can
# point them at a fixture tree; commands are stubbed on PATH.

MIN_NVIDIA_DRIVER="580.65.06"   # validators' MIN_NVIDIA_DRIVER_VERSION: an idle node below it earns nothing after the cutoff
MIN_DISK_TO_VRAM_RATE="1.5"     # validators' MIN_DISK_TO_VRAM_RATE (rental_price.py): idle pay needs total disk >= 1.5x total VRAM
PREFLIGHT_PASS=0 PREFLIGHT_FIX=0 PREFLIGHT_SKIP=0

pf_pass() { PREFLIGHT_PASS=$((PREFLIGHT_PASS + 1)); echo -e "  ${G}PASS${N} $1"; }
pf_skip() { PREFLIGHT_SKIP=$((PREFLIGHT_SKIP + 1)); echo -e "  ${Y}SKIP${N} $1"; }
pf_fix() {
    # $1 what is wrong; every further argument is one line of the fix
    PREFLIGHT_FIX=$((PREFLIGHT_FIX + 1))
    echo -e "  ${R}FIX${N}  $1"
    shift
    local line
    for line in "$@"; do echo "         $line"; done
    return 1
}

host_path() { echo "${HOST_ROOT:-}$1"; }

self_cmd() {
    # how to run this script again: the file when there is one, else the one-liner (curl | bash). A leading VAR=VALUE
    # argument goes after sudo (sudo's env_reset drops variables set before it); the rest are the script's options.
    local env=""
    case "${1:-}" in *=*) env="$1 "; shift ;; esac
    if [ -f "$0" ]; then
        echo "sudo ${env}bash $0${1:+ $*}"
    else
        echo "curl -fsSL https://raw.githubusercontent.com/Datura-ai/lium-io/main/neurons/executor/nvidia_docker_sysbox_setup.sh | sudo ${env}bash${1:+ -s -- $*}"
    fi
}

os_release_field() {
    # VERSION_ID / ID from /etc/os-release, empty when the file or the key is missing
    local file
    file=$(host_path /etc/os-release)
    [ -r "$file" ] && sed -n "s/^$1=\"\{0,1\}\([^\"]*\)\"\{0,1\}$/\1/p" "$file" | head -1
}

daemon_feature_off() {
    # true when /etc/docker/daemon.json sets features.<$1> to false (jq when present; jq itself
    # is one of the packages this script installs, so fall back to grep before it exists)
    local file
    file=$(host_path /etc/docker/daemon.json)
    [ -r "$file" ] || return 1
    if command -v jq &>/dev/null; then
        [ "$(jq -r --arg k "$1" '.features[$k]' "$file" 2>/dev/null)" = "false" ]
    else
        grep -Eq "\"$1\"[[:space:]]*:[[:space:]]*false" "$file"
    fi
}

check_root() {
    [ "$(id -u)" -eq 0 ] && { pf_pass "Running as root."; return 0; }
    pf_fix "Not running as root — the checks read Docker's socket and the kernel modules." \
        "$(self_cmd "${PREFLIGHT_MODE_FLAG:-}")"
}

check_arch() {
    local arch
    arch=$(uname -m)
    [ "$arch" = "x86_64" ] && { pf_pass "Architecture $arch."; return 0; }
    pf_fix "Architecture $arch — sysbox ships for x86_64 only." "Use an x86_64 host."
}

check_kernel() {
    local kernel version_id
    kernel=$(uname -r)
    if kernel_supports_idmapped; then
        pf_pass "Kernel $kernel (>= 5.19, ID-mapped mounts available)."
        return 0
    fi
    if [ "${SYSBOX_SKIP_KERNEL_CHECK:-0}" = "1" ]; then
        pf_pass "Kernel $kernel accepted because SYSBOX_SKIP_KERNEL_CHECK=1 is set."
        return 0
    fi
    version_id=$(os_release_field VERSION_ID)
    case "$version_id" in
        22.04) pf_fix "Kernel $kernel is below 5.19 — sysbox cannot pass GPUs through without ID-mapped mounts." \
                   "sudo apt-get install -y linux-generic-hwe-22.04 && sudo reboot" \
                   "Before rebooting: stop any rentals; 'dkms status' must list the nvidia module or the driver will not load on the new kernel." \
                   "If this kernel is known to carry the ID-mapped mounts backport: $(self_cmd SYSBOX_SKIP_KERNEL_CHECK=1)" ;;
        20.04) pf_fix "Kernel $kernel is below 5.19 and Ubuntu 20.04 tops out at 5.15 even with HWE." \
                   "Upgrade the host to Ubuntu 22.04 or newer (sudo do-release-upgrade), then re-run this script." ;;
        *)     pf_fix "Kernel $kernel is below 5.19 — sysbox cannot pass GPUs through without ID-mapped mounts." \
                   "Install a 5.19+ kernel for your distribution and reboot, then re-run this script." ;;
    esac
}

check_docker() {
    local version
    if ! command -v docker &>/dev/null; then
        pf_fix "Docker is not installed." "curl -fsSL https://get.docker.com | sudo sh"
        return 1
    fi
    if ! docker ps &>/dev/null; then
        pf_fix "Docker daemon is not running (docker ps failed)." "sudo systemctl enable --now docker"
        return 1
    fi
    version=$(docker_server_version)
    if [ -z "$version" ]; then
        pf_fix "Docker is running but reported no server version." "docker version   # then sudo systemctl restart docker"
        return 1
    fi
    if docker_version_ge 29 0 && ! docker_version_ge 29 2; then
        pf_pass "Docker $version (29.0–29.1 is untested with sysbox; 28.x and 29.2+ are)."
    else
        pf_pass "Docker $version."
    fi
}

check_docker_features() {
    # Docker 29.2 routes --gpus through CDI and 29.5 gives containers a time namespace; sysbox
    # accepts neither. The install step writes both keys; this reports whether they are set.
    local version missing=""
    version=$(docker_server_version)
    [ -n "$version" ] || { pf_skip "Docker 29 settings — Docker is not running."; return 0; }
    if ! docker_version_ge 29 2; then
        pf_pass "Docker $version needs no daemon.json features (cdi / time-namespaces appear in 29.2 / 29.5)."
        return 0
    fi
    daemon_feature_off cdi || missing="cdi"
    if docker_version_ge 29 5 && ! daemon_feature_off time-namespaces; then
        missing="${missing:+$missing, }time-namespaces"
    fi
    if [ -z "$missing" ]; then
        pf_pass "Docker $version has the sysbox settings in /etc/docker/daemon.json (features.cdi$(docker_version_ge 29 5 && echo ' and features.time-namespaces') = false)."
        return 0
    fi
    local block='{"features":{"cdi":false}}'
    docker_version_ge 29 5 && block='{"features":{"cdi":false,"time-namespaces":false}}'
    pf_fix "Docker $version without features.$missing = false in /etc/docker/daemon.json — sysbox rejects its containers." \
        "$(self_cmd)   # writes the features block and restarts Docker; stop rentals first" \
        "or by hand: add $block to /etc/docker/daemon.json && sudo systemctl restart docker"
}

check_nvidia_driver() {
    local driver
    if ! command -v nvidia-smi &>/dev/null; then
        pf_fix "nvidia-smi not found — no NVIDIA driver installed." \
            "sudo apt-get install -y nvidia-driver-580-server && sudo reboot   # Ubuntu; or your vendor's driver package"
        return 1
    fi
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -z "$driver" ] || ! ls "$(host_path /proc/driver/nvidia)" &>/dev/null; then
        pf_fix "NVIDIA driver is installed but not loaded (nvidia-smi gives no driver version or /proc/driver/nvidia is missing)." \
            "sudo reboot   # then check 'nvidia-smi'; if it still fails: sudo dkms autoinstall && sudo reboot"
        return 1
    fi
    if version3_ge "$driver" "$MIN_NVIDIA_DRIVER"; then
        pf_pass "NVIDIA driver $driver ($(nvidia-smi --list-gpus 2>/dev/null | grep -c '^GPU') GPU(s))."
        return 0
    fi
    pf_fix "NVIDIA driver $driver is below $MIN_NVIDIA_DRIVER, the validators' minimum — an idle node below it earns nothing." \
        "sudo apt-get install -y nvidia-driver-580-server && sudo reboot   # stop rentals first"
}

check_nvidia_toolkit() {
    local version
    version=$(nvidia-container-cli --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [ -n "$version" ]; then
        pf_pass "NVIDIA container toolkit (nvidia-container-cli $version)."
        return 0
    fi
    pf_fix "NVIDIA container toolkit is not installed — Docker cannot pass GPUs into containers." \
        "$(self_cmd)   # adds NVIDIA's apt repository and installs nvidia-container-toolkit"
}

check_iptables_modules() {
    # The validators' Docker-in-Docker probe runs legacy iptables inside the pod; on an nftables
    # host without these modules its dockerd fails with "can't initialize iptables table 'nat'",
    # sshd never starts and the node is scored as having no sysbox (ticket-0309: three reinstalls).
    local mod missing=""
    for mod in ip_tables iptable_nat iptable_filter; do
        grep -q "^$mod " "$(host_path /proc/modules)" 2>/dev/null && continue
        [ -d "$(host_path "/sys/module/$mod")" ] && continue
        missing="${missing:+$missing }$mod"
    done
    if [ -z "$missing" ]; then
        pf_pass "Legacy iptables modules loaded (ip_tables iptable_nat iptable_filter) for the validators' Docker-in-Docker probe."
        return 0
    fi
    pf_fix "Kernel modules $missing are not loaded — the validators' Docker-in-Docker probe needs legacy iptables and reports sysbox missing without them." \
        "sudo modprobe -a ip_tables iptable_nat iptable_filter   # -a: without it modprobe reads the 2nd and 3rd name as parameters of the 1st" \
        "printf 'ip_tables\\niptable_nat\\niptable_filter\\n' | sudo tee /etc/modules-load.d/lium-iptables.conf >/dev/null   # survives reboots"
}

check_disk_for_vram() {
    # The validators compare the TOTAL size of the filesystem the executor sees as / (its
    # rootfs lives under Docker's data-root) with total GPU VRAM; free space is not the rule.
    local vram_mib vram_gb data_root total_kb total_gb needed_gb
    command -v nvidia-smi &>/dev/null || { pf_skip "Disk >= ${MIN_DISK_TO_VRAM_RATE}x VRAM — no NVIDIA driver, VRAM unknown."; return 0; }
    vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{s += $1} END {print s + 0}')
    [ "${vram_mib:-0}" -gt 0 ] 2>/dev/null || { pf_skip "Disk >= ${MIN_DISK_TO_VRAM_RATE}x VRAM — nvidia-smi reports no GPU memory."; return 0; }
    data_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
    if [ -z "$data_root" ] || [ ! -d "$data_root" ]; then data_root=/var/lib/docker; fi
    [ -d "$data_root" ] || data_root=/
    total_kb=$(df -Pk "$data_root" 2>/dev/null | awk 'NR == 2 {print $2}')
    [ "${total_kb:-0}" -gt 0 ] 2>/dev/null || { pf_skip "Disk >= ${MIN_DISK_TO_VRAM_RATE}x VRAM — cannot read the filesystem size of $data_root."; return 0; }
    vram_gb=$(awk -v m="$vram_mib" 'BEGIN {printf "%.1f", m / 1024}')
    total_gb=$(awk -v k="$total_kb" 'BEGIN {printf "%.1f", k / 1024 / 1024}')
    needed_gb=$(awk -v v="$vram_gb" -v r="$MIN_DISK_TO_VRAM_RATE" 'BEGIN {printf "%.1f", v * r}')
    # the validator rounds VRAM and disk to 0.1 GB, then compares VRAM x 1.5 unrounded with the disk
    if awk -v t="$total_gb" -v v="$vram_gb" -v r="$MIN_DISK_TO_VRAM_RATE" 'BEGIN {exit !(t >= v * r)}'; then
        pf_pass "Disk ${total_gb} GB on $data_root >= ${needed_gb} GB (${MIN_DISK_TO_VRAM_RATE}x of ${vram_gb} GB VRAM) — idle pay eligible."
        return 0
    fi
    pf_fix "Disk ${total_gb} GB on $data_root is below ${needed_gb} GB (${MIN_DISK_TO_VRAM_RATE}x of ${vram_gb} GB VRAM) — the node is listed but earns nothing while idle." \
        "Add disk, or move Docker's data-root to a filesystem of at least ${needed_gb} GB (\"data-root\" in /etc/docker/daemon.json, then sudo systemctl restart docker)."
}

preflight_ports() {
    # EXECUTOR_PORT / SSH_PORT from the environment, else the executor .env next to this script, else the defaults
    local env_file=""
    [ -f "$0" ] && env_file="$(dirname "$0")/.env"   # piped from curl there is no file next to the script
    if [ -z "${EXECUTOR_PORT:-}" ] && [ -n "$env_file" ] && [ -r "$env_file" ]; then
        EXECUTOR_PORT=$(sed -n 's/^EXTERNAL_PORT=\([0-9]*\).*/\1/p' "$env_file" | head -1)
    fi
    if [ -z "${SSH_PORT:-}" ] && [ -n "$env_file" ] && [ -r "$env_file" ]; then
        SSH_PORT=$(sed -n 's/^SSH_PORT=\([0-9]*\).*/\1/p' "$env_file" | head -1)
    fi
    echo "${EXECUTOR_PORT:-8080} ${SSH_PORT:-2200}"
}

port_listener() {
    # the process listening on TCP $1, e.g. 'users:(("sshd",pid=812,fd=3))'; empty when the port is free
    ss -Hltnp "sport = :$1" 2>/dev/null | awk '{print $NF}' | head -1
}

ufw_blocks_port() {
    # true when ufw is active and no ALLOW rule covers TCP $1 (single port or lo:hi range)
    local status
    status=$(ufw status 2>/dev/null) || return 1
    echo "$status" | grep -q '^Status: active' || return 1
    ! echo "$status" | awk -v p="$1" '
        $2 == "ALLOW" || $3 == "ALLOW" {
            split($1, spec, "/"); if (spec[2] != "" && spec[2] != "tcp") next
            n = split(spec[1], range, ":")
            if ((n == 1 && range[1] == p) || (n == 2 && range[1] + 0 <= p + 0 && p + 0 <= range[2] + 0)) found = 1
        }
        END { exit !found }'
}

check_ports() {
    local ports p listener label fixed=0
    read -r -a ports <<< "$(preflight_ports)"
    for p in "${ports[0]}:executor" "${ports[1]}:SSH"; do
        label=${p#*:} p=${p%%:*}
        listener=$(port_listener "$p")
        if [ -n "$listener" ] && ! echo "$listener" | grep -Eq 'docker-proxy|dockerd'; then
            pf_fix "TCP $p ($label port) is already in use by $listener — the executor cannot bind it." \
                "Stop that process, or choose another port (EXTERNAL_PORT / SSH_PORT in neurons/executor/.env) and open it instead."
            fixed=1
            continue
        fi
        if ufw_blocks_port "$p"; then
            pf_fix "TCP $p ($label port) is blocked by ufw — validators cannot reach the node." "sudo ufw allow $p/tcp"
            fixed=1
            continue
        fi
        if [ -n "$listener" ]; then
            pf_pass "TCP $p ($label port) is served by Docker and not blocked by ufw."
        else
            pf_pass "TCP $p ($label port) is free and not blocked by ufw."
        fi
    done
    echo "         Cloud firewalls and routers are outside this host: after 'docker compose up', from ANY other machine run"
    echo "           nc -vz <this host's public IP> ${ports[0]} ${ports[1]}    # both must say 'succeeded'"
    return $fixed
}

check_sysbox() {
    # installed, registered as a Docker runtime, and a container starts under it (the docs' probe)
    if ! command -v sysbox-runc &>/dev/null; then
        pf_fix "sysbox-runc is not installed — validators reject a node without it." "$(self_cmd)"
        return 1
    fi
    if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q sysbox-runc; then
        pf_fix "sysbox-runc is installed but not registered in Docker's runtimes." \
            "$(self_cmd)   # writes the runtime into /etc/docker/daemon.json and restarts Docker"
        return 1
    fi
    if ! docker run --rm --runtime=sysbox-runc alpine echo ok &>/dev/null; then
        pf_fix "sysbox-runc is registered but 'docker run --rm --runtime=sysbox-runc alpine echo ok' fails." \
            "$(self_cmd)   # re-applies the Docker 29 settings and re-verifies; then: journalctl -u sysbox-mgr --no-pager -n 20"
        return 1
    fi
    pf_pass "sysbox-runc $(sysbox-runc --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo installed) runs a container."
}

preflight_summary() {
    echo ""
    echo "  Preflight: $PREFLIGHT_PASS PASS, $PREFLIGHT_FIX FIX, $PREFLIGHT_SKIP SKIP."
    [ "$PREFLIGHT_FIX" -eq 0 ]
}

preflight_host() {
    # what this script cannot install for you; a FIX here stops the run before anything changes
    check_root || true
    check_arch || true
    check_kernel || true
    check_docker || true
    check_nvidia_driver || true
    check_iptables_modules || true
    check_disk_for_vram || true
    check_ports || true
}

preflight_stack() {
    # what this script installs: reported in --check mode, done in install mode
    check_nvidia_toolkit || true
    check_docker_features || true
    check_sysbox || true
}


# sourced by the tests: functions only, nothing below runs
if [ -n "${SYSBOX_SETUP_LIB:-}" ]; then
    return 0
fi

PREFLIGHT_MODE_FLAG=""
case "${1:-}" in
    "") ;;
    --check) PREFLIGHT_MODE_FLAG="--check" ;;
    -h|--help)
        echo "Usage: sudo bash nvidia_docker_sysbox_setup.sh [--check]"
        echo "  (no option)  preflight the host, then install sysbox + NVIDIA container toolkit, configure Docker, verify"
        echo "  --check      preflight only: host requirements and what this script installs, PASS/FIX per line, exit 1 on any FIX"
        echo "Env: SYSBOX_SKIP_KERNEL_CHECK=1, EXECUTOR_PORT, SSH_PORT (see the header of this script)"
        exit 0
        ;;
    *)  fail "Unknown option: $1 (use --check or --help)"; exit 2 ;;
esac

# ── 1. Pre-flight ────────────────────────────────────────

if [ -n "$PREFLIGHT_MODE_FLAG" ]; then
    echo -e "\n${B}Preflight${N} — host requirements, then what this script installs"
    preflight_host
    preflight_stack
    preflight_summary && exit 0
    echo "  Fix the lines above, then run: $(self_cmd)"
    exit 1
fi

step 1 7 "Pre-flight checks"

preflight_host
preflight_summary || { echo "  Fix the lines above and re-run. Nothing was installed."; exit 1; }

echo -e "\n  This will install sysbox, configure Docker, restart Docker, and verify."
if [ -t 0 ]; then
    read -rp "  Continue? [Y/n]: " c; [ "$c" = "n" ] || [ "$c" = "N" ] && { ok "Aborted."; exit 0; }
fi

# ── 2. Already working? ─────────────────────────────────

if command -v sysbox-runc &>/dev/null && docker info 2>/dev/null | grep -q sysbox-runc; then
    # pull first: without the image the real test cannot run and the host would be judged on kernel version alone
    if ! docker image inspect "$VERIFY_IMAGE" &>/dev/null; then
        abort_on_active_rentals
        echo "  Pulling $VERIFY_IMAGE to test the current setup..."
        docker pull "$VERIFY_IMAGE" &>/dev/null || true
    fi
    if docker run --rm --runtime=sysbox-runc --gpus all "$VERIFY_IMAGE" nvidia-smi &>/dev/null; then
        ok "Sysbox is already working. Nothing to do."
        exit 0
    fi
fi

# Reached only when the real GPU test above did not pass, so a working host is never rejected here.
# The preflight already settled the kernel version, so a "no" from sysbox-mgr points at the
# filesystem under Docker's data-root or at sysbox-mgr's own configuration.
if [ "${SYSBOX_SKIP_KERNEL_CHECK:-0}" = "1" ]; then
    warn "SYSBOX_SKIP_KERNEL_CHECK=1 — installing without the ID-mapped mounts check."
elif [ "$(sysbox_idmapped_report)" = "no" ]; then
    fail_no_idmapped
    exit 1
fi

SKIP_INSTALL=false
command -v sysbox-runc &>/dev/null && SKIP_INSTALL=true && warn "Sysbox installed but not working. Reconfiguring..."

# ── 3. Check running containers ─────────────────────────

step 2 7 "Checking running containers"

EXECUTOR_COMPOSE_DIR=""
all_containers=$(docker ps --format "{{.Names}}" 2>/dev/null || true)
has_validator=false has_executor=false has_unknown=false
unknown_list=""

for name in $all_containers; do
    case "$name" in
        pod_*)       ;;  # handled by abort_on_active_rentals below
        container_*) has_validator=true ;;
        executor-*|executor_*)
            has_executor=true
            if [ -z "$EXECUTOR_COMPOSE_DIR" ]; then
                EXECUTOR_COMPOSE_DIR=$(docker inspect "$name" --format '{{ index .Config.Labels "com.docker.compose.project.working_dir" }}' 2>/dev/null || true)
            fi
            ;;
        *)           has_unknown=true; unknown_list="$unknown_list\n    - $name" ;;
    esac
done

abort_on_active_rentals

if [ "$has_validator" = true ]; then
    fail "Validator check in progress (container_* containers). Wait ~30 seconds and retry."
    exit 1
fi

if [ "$has_unknown" = true ]; then
    fail "Unknown containers found — stop them manually before proceeding:"
    echo -e "$unknown_list"
    exit 1
fi

find_executor_compose() {
    # 1. Label from container
    [ -n "$EXECUTOR_COMPOSE_DIR" ] && [ -f "$EXECUTOR_COMPOSE_DIR/docker-compose.yml" ] && return 0
    # 2. Current directory
    [ -f ./docker-compose.yml ] && EXECUTOR_COMPOSE_DIR="$(pwd)" && return 0
    # 3. Known paths
    for d in \
        "$HOME/compute-subnet/neurons/executor" \
        /root/compute-subnet/neurons/executor \
        /home/*/compute-subnet/neurons/executor \
        "$HOME/executor" \
        /root/executor; do
        [ -f "$d/docker-compose.yml" ] && EXECUTOR_COMPOSE_DIR="$d" && return 0
    done
    return 1
}

if [ "$has_executor" = true ]; then
    find_executor_compose || EXECUTOR_COMPOSE_DIR=""
    warn "Executor containers will be stopped and restarted after setup."
    if [ -n "$EXECUTOR_COMPOSE_DIR" ] && [ -f "$EXECUTOR_COMPOSE_DIR/docker-compose.yml" ]; then
        docker compose -f "$EXECUTOR_COMPOSE_DIR/docker-compose.yml" down 2>/dev/null \
            || { fail "Failed to stop executor. Run 'docker compose down' manually from: $EXECUTOR_COMPOSE_DIR"; exit 1; }
        ok "Executor stopped via compose ($EXECUTOR_COMPOSE_DIR)."
    else
        # Compose file missing — stop and remove executor containers directly
        executor_ids=$(docker ps --filter "name=executor" -q 2>/dev/null || true)
        if [ -n "$executor_ids" ]; then
            # shellcheck disable=SC2086  # one id per word
            docker stop $executor_ids > /dev/null 2>&1
            # shellcheck disable=SC2086
            docker rm $executor_ids > /dev/null 2>&1
        fi
        ok "Executor containers stopped (compose file not found — restart manually after setup)."
        EXECUTOR_COMPOSE_DIR=""  # Clear so we don't try to restart
    fi
fi

# Check for stopped containers (sysbox requires zero containers)
stopped=$(docker ps -a -q 2>/dev/null || true)
if [ -n "$stopped" ]; then
    warn "Removing stopped containers (sysbox requires none)..."
    # shellcheck disable=SC2086  # one id per word
    docker rm -f $stopped > /dev/null 2>&1
    ok "Stopped containers removed."
fi

ok "Docker is clear for sysbox installation."

# ── 4. Install ──────────────────────────────────────────

step 3 7 "Installing packages"

ensure_nvidia_container_toolkit_repo || exit 1
apt_install update -qq || exit 1
apt_install install -y -qq nvidia-container-toolkit jq || exit 1
ok "nvidia-container-toolkit, jq"

if [ "$SKIP_INSTALL" = false ]; then
    LOCAL_DEB="./sysbox-ce_${SYSBOX_VERSION}-0.linux_amd64.deb"
    if [ -f "$LOCAL_DEB" ]; then
        SYSBOX_DEB="$LOCAL_DEB"
    else
        SYSBOX_DEB=$(mktemp /tmp/sysbox-ce.XXXXXX.deb)
        DOWNLOADED_DEB="$SYSBOX_DEB"
        ok "Downloading sysbox v${SYSBOX_VERSION}..."
        wget -q -O "$SYSBOX_DEB" "$SYSBOX_DEB_URL"
        actual=$(sha256sum "$SYSBOX_DEB" | cut -d' ' -f1)
        [ "$actual" = "$SYSBOX_SHA" ] || { fail "Checksum mismatch!"; exit 1; }
    fi
    apt_install install -y -qq "$SYSBOX_DEB" || exit 1
    ok "Sysbox v${SYSBOX_VERSION} installed."
else
    ok "Sysbox already installed, skipping."
fi

# ── 5. Configure Docker ────────────────────────────────

step 4 7 "Configuring Docker"

CONFIG='{"runtimes":{"sysbox-runc":{"path":"/usr/bin/sysbox-runc"},"nvidia":{"path":"nvidia-container-runtime","runtimeArgs":[]}},"features":{"cdi":false}}'

# Docker >= 29.5 gives containers a private time namespace and puts it in the OCI spec; sysbox-runc
# does not know that namespace type and rejects the whole spec. Set only on daemons that know the
# flag — an unknown feature key on an older daemon is untested and would cost us the restart below.
if docker_version_ge 29 5; then
    CONFIG=$(echo "$CONFIG" | jq -c '.features["time-namespaces"] = false')
    ok "Docker >= 29.5: disabling time namespaces for sysbox."
fi

mkdir -p /etc/docker
if [ -f /etc/docker/daemon.json ]; then
    cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
    jq --argjson p "$CONFIG" '. * $p' /etc/docker/daemon.json > /tmp/daemon.json.tmp \
        || { fail "Failed to merge daemon.json (invalid JSON?). Backup: daemon.json.bak"; exit 1; }
    [ -s /tmp/daemon.json.tmp ] || { fail "Merged daemon.json is empty. Backup: daemon.json.bak"; exit 1; }
    mv /tmp/daemon.json.tmp /etc/docker/daemon.json
    ok "Merged into daemon.json (backup: daemon.json.bak)."
else
    echo "$CONFIG" | jq . > /etc/docker/daemon.json
    ok "Created daemon.json."
fi

if docker_version_ge 29 2; then
    rm -f /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml
    systemctl disable --now nvidia-cdi-refresh.path nvidia-cdi-refresh.service 2>/dev/null || true
    ok "Cleaned up CDI specs (Docker >= 29.2)."
fi

# ── 6. Restart Docker ──────────────────────────────────

step 5 7 "Restarting Docker"

systemctl restart docker
for _ in $(seq 1 30); do docker ps &>/dev/null && break; sleep 1; done
docker ps &>/dev/null || { fail "Docker failed to restart. Check: journalctl -u docker.service"; exit 1; }
ok "Docker is running."

# ── 7. Verify ──────────────────────────────────────────

step 6 7 "Verifying sysbox + GPU"

if docker run --rm --runtime=sysbox-runc --gpus all "$VERIFY_IMAGE" nvidia-smi &>/dev/null; then
    echo ""
    echo -e "  ${G}╔══════════════════════════════════════════╗${N}"
    echo -e "  ${G}║  SUCCESS: Sysbox is working with GPUs!   ║${N}"
    echo -e "  ${G}╚══════════════════════════════════════════╝${N}"
    echo ""
    ok "Your executor now supports Docker-in-Docker."

    # ── 8. Restart executor ──────────────────────────────
    if [ -n "$EXECUTOR_COMPOSE_DIR" ]; then
        step 7 7 "Restarting executor"
        if docker compose -f "$EXECUTOR_COMPOSE_DIR/docker-compose.yml" up -d 2>/dev/null; then
            ok "Executor restarted ($EXECUTOR_COMPOSE_DIR)."
        else
            warn "Failed to restart executor. Run manually: cd $EXECUTOR_COMPOSE_DIR && docker compose up -d"
        fi
    fi
else
    echo ""
    fail "Verification FAILED. Diagnostics:"
    echo ""
    echo "    Docker daemon:       $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo unknown)"
    echo "    Kernel:              $(uname -r)"
    echo "    ID-mapped mounts:    $(sysbox_idmapped_report | grep . || echo 'not reported by sysbox-mgr')"
    echo "    NVIDIA driver:       $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo FAILED)"
    echo "    /proc/driver/nvidia: $(ls /proc/driver/nvidia &>/dev/null && echo exists || echo MISSING)"
    echo "    sysbox-runc:         $(sysbox-runc --version 2>/dev/null | head -1 || echo 'not found')"
    echo "    CDI specs:           $(ls /var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml 2>/dev/null || echo none)"
    echo "    daemon.json cdi:     $(jq -r '.features.cdi // "not set"' /etc/docker/daemon.json 2>/dev/null)"
    echo "    daemon.json time-ns: $(jq -r '.features["time-namespaces"] | if . == null then "not set" else tostring end' /etc/docker/daemon.json 2>/dev/null)"
    echo ""
    if docker_version_ge 29 5; then
        fail "Docker >= 29.5 puts a time namespace in the OCI spec, which sysbox-runc rejects with"
        fail "  'namespace {\"time\" \"\"} does not exist'. The time-namespaces feature above turns that off;"
        fail "  if the error is still there, downgrade the daemon to Docker 28.x."
    fi
    fail "Check: journalctl -u docker.service --no-pager -n 20"
    fail "Check: journalctl -u sysbox-mgr.service --no-pager -n 20"
    exit 1
fi
