#!/bin/bash

# DStack CVM Management Script
# Provides commands to check system requirements, create CVMs, and run CVMs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OS_IMAGE_NAME=${OS:-dstack-nvidia-0.5.11}
# DAH-2338: default to the official upstream image (NVIDIA driver 595.58.03 = R595 TRD1,
# sysbox baked in). To run the legacy Lium-built 0.5.5 image, set OS=dstack-nvidia-0.5.5
# and OS_IMAGE_URL to the private-ml-sdk v0.5.5 release asset.
OS_IMAGE_URL="${OS_IMAGE_URL:-https://github.com/Dstack-TEE/meta-dstack/releases/download/v0.5.11/$OS_IMAGE_NAME.tar.gz}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_PROVIDER_DIR="$THIS_DIR/key-provider"
SCRIPTS_DIR="$THIS_DIR/scripts"
RUN_DIR="$THIS_DIR/run"
VMS_DIR="$RUN_DIR/vms"
IMAGE_DIR="$RUN_DIR/images"
INIT_SCRIPT="$THIS_DIR/app/init_script.sh"
PRE_LAUNCH_SCRIPT="$THIS_DIR/app/pre_launch_script.sh"
PORT_BIND_IP=0.0.0.0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if service is running
service_running() {
    systemctl is-active --quiet "$1" 2>/dev/null
}

# Function to check if docker container is running
container_running() {
    docker ps --format "table {{.Names}}" | grep -q "$1" 2>/dev/null
}

# Resolve the QEMU binary the launcher will actually use: dstack.py reads
# [qemu] path from the client.conf chain (later files override earlier ones)
# and only falls back to PATH when no config sets it.
resolve_qemu_path() {
    local resolved=""
    local conf
    for conf in /etc/dstack/client.conf "$HOME/.config/dstack/client.conf" "$THIS_DIR/.dstack/client.conf"; do
        [ -f "$conf" ] || continue
        local configured
        configured=$(awk '/^[[:space:]]*\[/ { section = $0 }
            section ~ /\[qemu\]/ && /^[[:space:]]*path[[:space:]]*=/ { sub(/^[^=]*=[[:space:]]*/, ""); print }' "$conf" | tail -n1)
        [ -n "$configured" ] && resolved="$configured"
    done
    if [ -n "$resolved" ]; then
        echo "$resolved"
    else
        command -v qemu-system-x86_64 2>/dev/null
    fi
}

# List host processes still holding a VM directory: the dstack.py launcher and
# the QEMU it runs as a child (both carry the VM directory on their command line).
cvm_pids() {
    local vm_dir="$1"
    pgrep -f "dstack\.py run ${vm_dir}(/|[[:space:]]|\$)" 2>/dev/null || true
    pgrep -f "qemu-system-x86_64.*${vm_dir}/" 2>/dev/null || true
}

# Function to download and extract OS image
download_os_image() {
    local temp_file="/tmp/dstack-os-image-$$.tar.gz"

    # Check if images directory already exists and has content
    if [ -d "$IMAGE_DIR" ] && [ -n "$(find "$IMAGE_DIR/$OS_IMAGE_NAME/" -name "*.img" -o -name "metadata.json" 2>/dev/null)" ]; then
        log_info "OS image already exists in $IMAGE_DIR"
        return 0
    fi

    log_info "Downloading DStack OS image from $OS_IMAGE_URL"

    # Create images directory
    mkdir -p "$IMAGE_DIR"

    # Download the image
    if command_exists wget; then
        if wget -O "$temp_file" "$OS_IMAGE_URL"; then
            log_success "Downloaded OS image successfully"
        else
            log_error "Failed to download OS image with wget"
            return 1
        fi
    elif command_exists curl; then
        if curl -L -o "$temp_file" "$OS_IMAGE_URL"; then
            log_success "Downloaded OS image successfully"
        else
            log_error "Failed to download OS image with curl"
            return 1
        fi
    else
        log_error "Neither wget nor curl is available for downloading"
        return 1
    fi

    # Extract the image
    log_info "Extracting OS image to $IMAGE_DIR"
    if tar -xzf "$temp_file" -C "$IMAGE_DIR"; then
        log_success "OS image extracted successfully"
        rm -f "$temp_file"
    else
        log_error "Failed to extract OS image"
        rm -f "$temp_file"
        return 1
    fi

    return 0
}

# Check command - verify system requirements
check_requirements() {
    log_info "Checking system requirements for DStack CVM..."

    local all_good=true

    # Check Docker
    log_info "Checking Docker installation..."
    if command_exists docker; then
        log_success "Docker is installed: $(docker --version)"

        # Check if Docker daemon is running
        if docker info >/dev/null 2>&1; then
            log_success "Docker daemon is running"
        else
            log_error "Docker daemon is not running"
            all_good=false
        fi

        # Check Docker Compose
        if command_exists docker-compose || docker compose version >/dev/null 2>&1; then
            log_success "Docker Compose is available"
        else
            log_error "Docker Compose is not installed"
            all_good=false
        fi
    else
        log_error "Docker is not installed"
        all_good=false
    fi

    # Check QEMU
    log_info "Checking QEMU installation..."
    local qemu_bin
    qemu_bin=$(resolve_qemu_path || true)
    if [ -n "$qemu_bin" ] && "$qemu_bin" --version >/dev/null 2>&1; then
        local qemu_version
        qemu_version=$("$qemu_bin" --version 2>/dev/null | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p' | head -n1)
        case "$qemu_version" in
        9.2.1)
            log_success "QEMU is the dstack 9.2.1 build: $qemu_bin"
            ;;
        1[0-9].*)
            log_error "QEMU $qemu_version at $qemu_bin is a distro build: the verifier's RTMR0 oracle is built from dstack QEMU 9.2.1, so attestation can never pass"
            log_info "See docs/host-setup.md section 3 — build kvinwang/qemu-tdx branch dstack-qemu-9.2.1 and point [qemu] path in client.conf at it"
            all_good=false
            ;;
        "")
            log_warning "Could not parse the QEMU version of $qemu_bin; attestation requires the dstack 9.2.1 build (docs/host-setup.md section 3)"
            ;;
        *)
            log_warning "QEMU $qemu_version at $qemu_bin is not the dstack 9.2.1 build attestation expects (docs/host-setup.md section 3)"
            ;;
        esac
    else
        log_error "QEMU (qemu-system-x86_64) is not installed or not executable${qemu_bin:+ at $qemu_bin}"
        all_good=false
    fi

    # Check KVM support
    log_info "Checking KVM support..."
    if [ -e /dev/kvm ]; then
        log_success "KVM device is available"
    else
        log_error "KVM device (/dev/kvm) is not available"
        all_good=false
    fi

    # Check Intel TDX support
    log_info "Checking Intel TDX support..."
    if grep -q "tdx" /proc/cpuinfo 2>/dev/null; then
        log_success "Intel TDX support detected in CPU"
    else
        log_warning "Intel TDX support not detected in CPU info"
    fi

    # Check the TDX module TCB. The 882 floor (2.0.08, minor SVN 5) is Intel's minimum for
    # the 2.0 series on Granite Rapids; the 1.5 series on Sapphire/Emerald Rapids numbers its
    # builds differently, so it only gets reported. Too old = "No matching TCB level found".
    log_info "Checking Intel TDX module build..."
    local tdx_line tdx_major tdx_build
    tdx_line=$(dmesg 2>/dev/null | grep -i 'virt/tdx' | grep -i 'build_num' | tail -n1)
    tdx_major=$(printf '%s' "$tdx_line" | sed -n 's/.*major_version[^0-9]*\([0-9][0-9]*\).*/\1/p')
    tdx_build=$(printf '%s' "$tdx_line" | sed -n 's/.*build_num[^0-9]*\([0-9][0-9]*\).*/\1/p')
    if [ -z "$tdx_build" ]; then
        log_warning "No TDX module build number in dmesg (kernel 7.0 prints none, and dmesg may need root); the proof is then the quote's tee_tcb_svn — see docs/host-setup.md section 1"
    elif [ "$tdx_major" != "2" ]; then
        log_info "TDX module ${tdx_major:-?}.x build $tdx_build (pre-Granite Rapids series); Intel's minimum is per platform — the quote's tee_tcb_svn is the proof, see docs/host-setup.md section 1"
    elif [ "$tdx_build" -ge 882 ]; then
        log_success "TDX module build $tdx_build meets Intel's minimum (882 = 2.0.08)"
    else
        log_error "TDX module build $tdx_build is below Intel's minimum 882 (2.0.08, minor SVN 5): DCAP verification fails with 'No matching TCB level found'"
        log_info "See docs/host-setup.md section 1 — update the module via /boot/efi/EFI/TDX/ or a vendor BIOS"
        all_good=false
    fi

    # Check SGX devices
    log_info "Checking SGX devices..."
    if [ -e /dev/sgx_enclave ] && [ -e /dev/sgx_provision ]; then
        log_success "SGX devices are available"
    else
        log_warning "SGX devices (/dev/sgx_enclave, /dev/sgx_provision) not found"
    fi

    # Check NUMA topology. The possible set is always a superset of the online one,
    # so any difference means the firmware over-declares nodes.
    log_info "Checking NUMA node topology..."
    if [ -r /sys/devices/system/node/possible ] && [ -r /sys/devices/system/node/online ]; then
        local numa_possible numa_online
        numa_possible=$(cat /sys/devices/system/node/possible)
        numa_online=$(cat /sys/devices/system/node/online)
        if [ "$numa_possible" = "$numa_online" ]; then
            log_success "NUMA nodes possible=$numa_possible online=$numa_online"
        else
            log_warning "NUMA nodes possible=$numa_possible but online=$numa_online: Gramine 1.5 (the pinned key-provider image) fails load_enclave with EINVAL on such firmware"
            log_info "See docs/host-setup.md section 4 — rebuild the key provider on gramine:1.9"
        fi
    else
        log_warning "Cannot read /sys/devices/system/node/{possible,online}; skipping the NUMA check"
    fi

    # Check key-provider
    log_info "Checking key-provider status..."
    if [ -f "$KEY_PROVIDER_DIR/docker-compose.yaml" ]; then
        log_success "Key-provider configuration found"

        # Check if key-provider process is running
        if pgrep -f "gramine-sealing-key-provider" >/dev/null 2>&1; then
            log_success "Key-provider (gramine-sealing-key-provider) is running"
        else
            log_warning "Key-provider process not running. Use 'docker-compose up -d' in $KEY_PROVIDER_DIR"
        fi
    else
        log_error "Key-provider configuration not found at $KEY_PROVIDER_DIR"
        all_good=false
    fi

    # Check the PCCS the key provider takes DCAP collateral from
    log_info "Checking PCCS reachability..."
    local qcnl_conf="$KEY_PROVIDER_DIR/sgx_default_qcnl.conf"
    if [ -f "$qcnl_conf" ]; then
        local pccs_url
        pccs_url=$(sed -n 's/.*"pccs_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$qcnl_conf" | head -n1)
        if [ -z "$pccs_url" ]; then
            log_warning "No pccs_url in $qcnl_conf"
        elif ! command_exists curl; then
            log_warning "curl is not installed; cannot check whether $pccs_url answers"
        elif curl -k -s -o /dev/null --max-time 5 "$pccs_url"; then
            log_success "PCCS answers: $pccs_url"
        else
            log_warning "PCCS $pccs_url does not answer; the key provider will fail DCAP collateral fetch"
        fi
        log_info "A reachable PCCS is not enough: the platform must be registered with Intel (multi-package platforms need it) — enable 'SGX Auto MP Registration' in BIOS or run PCKIDRetrievalTool against your PCCS, see docs/host-setup.md section 4"
    else
        log_warning "QCNL configuration not found at $qcnl_conf"
    fi

    # Check dstack-os-image
    log_info "Checking dstack-os-image..."
    if [ -d "$IMAGE_DIR" ] && [ -n "$(find "$IMAGE_DIR/$OS_IMAGE_NAME/" -name "*.img" -o -name "metadata.json" 2>/dev/null)" ]; then
        log_success "DStack OS images found in $IMAGE_DIR"
    else
        log_warning "DStack OS images not found in $IMAGE_DIR"
        log_info "You can download images or specify custom image path when creating CVMs"
    fi

    # Check Python dependencies
    log_info "Checking Python environment..."
    if command_exists python3; then
        log_success "Python3 is available"
        if [ -f "$SCRIPTS_DIR/dstack.py" ]; then
            log_success "DStack Python script found"
        else
            log_error "DStack Python script not found at $SCRIPTS_DIR/dstack.py"
            all_good=false
        fi
    else
        log_error "Python3 is not installed"
        all_good=false
    fi

    # Summary
    echo
    if [ "$all_good" = true ]; then
        log_success "All requirements are satisfied!"
    else
        log_error "Some requirements are missing. Please install missing components."
        return 1
    fi
}

# Create command - create new CVM using CLI arguments
new_cvm() {
    local cvm_name="$1"

    if [ -z "$cvm_name" ]; then
        echo "Usage: $0 new <cvm_name> [--env <environment>] [--enable-logs] [--enable-sysinfo]"
        echo
        echo "Configuration is read from .env file. Copy .env.example to .env and edit as needed."
        echo
        echo "Options:"
        echo "  --env <environment>      Target environment: local, staging, prod (default: prod)"
        echo "                             local   -> app/docker-compose.local.yml"
        echo "                             staging -> app/docker-compose.staging.yml"
        echo "                             prod    -> app/docker-compose.yml"
        echo "  --enable-logs            Enable public container logs"
        echo "  --enable-sysinfo         Enable public system and TCB info"
        echo
        echo "Examples:"
        echo "  $0 new my-cvm"
        echo "  $0 new my-cvm --env local"
        echo "  $0 new my-cvm --env staging"
        echo "  $0 new my-cvm --env prod --enable-logs --enable-sysinfo"
        echo "  $0 new gpu-cvm --env local"
        echo
        echo "Note: OS image will be automatically downloaded to $IMAGE_DIR if not present"
        return 1
    fi
    shift

    local logs_arg=""
    local sysinfo_arg=""
    local env_name="prod"

    # Parse optional new command flags
    while [ $# -gt 0 ]; do
        case "$1" in
        --env)
            if [ -z "$2" ]; then
                log_error "--env requires a value: local, staging, or prod"
                return 1
            fi
            env_name="$2"
            shift
            ;;
        --enable-logs)
            logs_arg="--enable-logs"
            ;;
        --enable-sysinfo)
            sysinfo_arg="--enable-sysinfo"
            ;;
        *)
            log_error "Unknown option for new command: $1"
            echo "Usage: $0 new <cvm_name> [--env <environment>] [--enable-logs] [--enable-sysinfo]"
            return 1
            ;;
        esac
        shift
    done

    # Configuration from .env file
    local env_file=".env"
    local image_path="$IMAGE_DIR/$OS_IMAGE_NAME" # Path to specific OS image

    log_info "Creating new CVM: $cvm_name"

    # Check if CVM already exists
    if [ -d "$VMS_DIR/$cvm_name" ]; then
        log_error "CVM '$cvm_name' already exists at $VMS_DIR/$cvm_name"
        return 1
    fi

    # Download OS image if not present
    if ! download_os_image; then
        log_error "Failed to download OS image"
        return 1
    fi

    # Resolve compose file based on environment
    case "$env_name" in
    local)
        local compose_file="$THIS_DIR/app/docker-compose.local.yml"
        ;;
    staging)
        local compose_file="$THIS_DIR/app/docker-compose.staging.yml"
        ;;
    prod)
        local compose_file="$THIS_DIR/app/docker-compose.yml"
        ;;
    *)
        log_error "Unknown environment: '$env_name'. Valid values are: local, staging, prod"
        return 1
        ;;
    esac
    local lkp_args="--local-key-provider"

    # Validate required files/directories
    if [ ! -f "$compose_file" ]; then
        log_error "Docker Compose file not found: $compose_file"
        return 1
    fi

    # Check if .env file exists and load configuration
    if ! [ -f "$env_file" ]; then
        log_error "Env file not found: $env_file"
        log_info "You can run 'cp .env.example .env' and edit it"
        return 1
    fi

    # Source .env file to load configuration
    set -a # automatically export all variables
    source "$env_file"
    set +a # disable automatic export

    # Set configuration from environment variables
    local vcpus="${CVM_VCPUS:-4}"
    local memory="${CVM_MEMORY:-4G}"
    local disk="${CVM_DISK:-20G}"
    local gpu_args=""
    local port_args=""

    # Configure GPU from environment
    if [ -n "$CVM_GPUS" ]; then
        if [ "$CVM_GPUS" = "all" ]; then
            gpu_args="--gpu all"
        else
            # Parse comma-separated GPU list
            IFS=',' read -ra GPUS <<<"$CVM_GPUS"
            for gpu in "${GPUS[@]}"; do
                gpu_args="$gpu_args --gpu $(echo $gpu | xargs)"
            done
        fi
    fi
    # Compute CVM_PORT from RENTING_PORT_MAPPINGS or RENTING_PORT_RANGE
    local computed_ports="tcp:$PORT_BIND_IP:$SSH_PUBLIC_PORT:$SSH_PORT,tcp:$PORT_BIND_IP:$EXTERNAL_PORT:$EXTERNAL_PORT"

    if [ -n "$RENTING_PORT_MAPPINGS" ]; then
        # Parse RENTING_PORT_MAPPINGS: "[[46681, 56681], [46682, 56682]]"
        # Extract port pairs and create tcp mappings
        local mappings=$(echo "$RENTING_PORT_MAPPINGS" | sed 's/\[\[//g' | sed 's/\]\]//g' | sed 's/\], \[/|/g')
        IFS='|' read -ra PORT_PAIRS <<<"$mappings"
        for pair in "${PORT_PAIRS[@]}"; do
            IFS=',' read -ra PORTS <<<"$pair"
            if [ ${#PORTS[@]} -eq 2 ]; then
                local internal_port=$(echo "${PORTS[0]}" | xargs)
                local external_port=$(echo "${PORTS[1]}" | xargs)
                if [ -n "$computed_ports" ]; then
                    computed_ports="$computed_ports,tcp:$PORT_BIND_IP:$external_port:$internal_port"
                else
                    computed_ports="tcp:$PORT_BIND_IP:$external_port:$internal_port"
                fi
            fi
        done
    elif [ -n "$RENTING_PORT_RANGE" ]; then
        # Parse RENTING_PORT_RANGE: "40000-65535" or "9001,9002,9003"
        if [[ "$RENTING_PORT_RANGE" == *"-"* ]]; then
            log_error "Range format is not supported"
            return 1
        else
            # Comma-separated format: "9001,9002,9003"
            IFS=',' read -ra PORTS <<<"$RENTING_PORT_RANGE"
            for port in "${PORTS[@]}"; do
                port=$(echo "$port" | xargs)
                if [ -n "$computed_ports" ]; then
                    computed_ports="$computed_ports,tcp:$PORT_BIND_IP:$port:$port"
                else
                    computed_ports="tcp:$PORT_BIND_IP:$port:$port"
                fi
            done
        fi
    else
        log_error "Neither RENTING_PORT_MAPPINGS nor RENTING_PORT_RANGE is configured in .env file"
        log_info "Please configure one of these variables to define available ports for CVM"
        return 1
    fi

    # Convert computed ports to port_args
    if [ -n "$computed_ports" ]; then
        IFS=',' read -ra PORT_LIST <<<"$computed_ports"
        for port in "${PORT_LIST[@]}"; do
            port_args="$port_args --port $(echo $port | xargs)"
        done
    fi

    # Display configuration
    log_info "Creating CVM with the following configuration:"
    echo "  Name: $cvm_name"
    echo "  Compose file: $compose_file"
    echo "  Image: $image_path"
    echo "  vCPUs: $vcpus"
    echo "  Memory: $memory"
    echo "  Disk: $disk"
    if [ -n "$gpu_args" ]; then
        echo "  GPU args: $gpu_args"
    fi
    if [ -n "$port_args" ]; then
        echo "  Port args: $port_args"
    fi
    if [ -n "$logs_arg" ]; then
        echo "  Public logs: enabled"
    fi
    if [ -n "$sysinfo_arg" ]; then
        echo "  Public sysinfo/tcbinfo: enabled"
    fi
    echo

    # Ensure VMs directory exists
    mkdir -p "$VMS_DIR"

    # G2 (CVM attestation) — stamp the pinned executor-runner digest into the
    # measured compose BEFORE dstack.py measures it into compose_hash. The
    # resolved copy (not the template) is what gets attested, so the runner
    # release is inside the trust boundary the validator whitelists.
    if grep -q '${EXECUTOR_RUNNER_IMAGE_DIGEST}' "$compose_file"; then
        if [ -z "$EXECUTOR_RUNNER_IMAGE_DIGEST" ]; then
            log_error "EXECUTOR_RUNNER_IMAGE_DIGEST is not set (required to pin the measured executor-runner)"
            log_info "Set it in $env_file to the approved runner release, e.g."
            log_info "  EXECUTOR_RUNNER_IMAGE_DIGEST=sha256:<digest>"
            return 1
        fi
        case "$EXECUTOR_RUNNER_IMAGE_DIGEST" in
        sha256:*) ;;
        *)
            log_error "EXECUTOR_RUNNER_IMAGE_DIGEST must start with 'sha256:' (got: $EXECUTOR_RUNNER_IMAGE_DIGEST)"
            return 1
            ;;
        esac
        local resolved_compose="$VMS_DIR/$cvm_name-docker-compose.yml"
        EXECUTOR_RUNNER_IMAGE_DIGEST="$EXECUTOR_RUNNER_IMAGE_DIGEST" \
            envsubst '${EXECUTOR_RUNNER_IMAGE_DIGEST}' <"$compose_file" >"$resolved_compose"
        log_info "Pinned executor-runner digest into measured compose: $EXECUTOR_RUNNER_IMAGE_DIGEST"
        compose_file="$resolved_compose"
    fi

    # Build the command
    python3 $SCRIPTS_DIR/dstack.py new "$compose_file" \
        --init-script "$INIT_SCRIPT" \
        --pre-launch-script "$PRE_LAUNCH_SCRIPT" \
        --dir "$VMS_DIR/$cvm_name" \
        --image "$image_path" \
        --vcpus "$vcpus" \
        --memory "$memory" \
        --disk "$disk" \
        --env-file "$env_file" \
        $gpu_args \
        $port_args \
        $lkp_args \
        $logs_arg \
        $sysinfo_arg

    if [ $? -eq 0 ]; then
        log_success "CVM '$cvm_name' created successfully at $VMS_DIR/$cvm_name"
        MY_IP=$(curl -s ifconfig.me)
        log_success "Executor endpoint: $MY_IP:$EXTERNAL_PORT"
    else
        log_error "Failed to create CVM '$cvm_name'"
        return 1
    fi
}

# Run command - run existing CVM
run_cvm() {
    local cvm_name="$1"
    local dry_run=""

    if [ -z "$cvm_name" ]; then
        echo "Usage: $0 run <cvm_name> [--dry-run]"
        return 1
    fi

    # Check for dry-run flag
    if [ "$2" = "--dry-run" ]; then
        dry_run="--dry-run"
        log_info "Running in dry-run mode"
    fi

    log_info "Running CVM: $cvm_name"

    # Check if CVM exists
    if [ ! -d "$VMS_DIR/$cvm_name" ]; then
        log_error "CVM '$cvm_name' not found at $VMS_DIR/$cvm_name"
        log_info "Available CVMs:"
        if [ -d "$VMS_DIR" ]; then
            ls -1 "$VMS_DIR" 2>/dev/null || echo "  No CVMs found"
        else
            echo "  No CVMs directory found"
        fi
        return 1
    fi

    # Check if manifest exists
    if [ ! -f "$VMS_DIR/$cvm_name/vm-manifest.json" ]; then
        log_error "VM manifest not found for CVM '$cvm_name'"
        return 1
    fi

    # Start key-provider if not running
    log_info "Ensuring key-provider is running..."
    if ! pgrep -f "gramine-sealing-key-provider" >/dev/null 2>&1; then
        log_info "Starting key-provider..."
        cd "$KEY_PROVIDER_DIR"
        docker-compose up -d
        sleep 3

        # Verify it started
        if pgrep -f "gramine-sealing-key-provider" >/dev/null 2>&1; then
            log_success "Key-provider started successfully"
        else
            log_warning "Key-provider may not have started properly"
        fi
    else
        log_success "Key-provider is already running"
    fi

    # Run the CVM
    log_info "Starting CVM '$cvm_name'..."
    cmd="python3 $SCRIPTS_DIR/dstack.py run '$VMS_DIR/$cvm_name' $dry_run"

    log_info "Executing: $cmd"

    if eval "$cmd"; then
        if [ -z "$dry_run" ]; then
            log_success "CVM '$cvm_name' started successfully"
        else
            log_success "Dry-run completed for CVM '$cvm_name'"
        fi
    else
        log_error "Failed to run CVM '$cvm_name'"
        return 1
    fi
}

# Stop command - gracefully shut down a running CVM
stop_cvm() {
    local cvm_name="$1"

    if [ -z "$cvm_name" ]; then
        echo "Usage: $0 stop <cvm_name> [--timeout N] [--force]"
        return 1
    fi

    log_info "Stopping CVM: $cvm_name"

    if [ ! -d "$VMS_DIR/$cvm_name" ]; then
        log_error "CVM '$cvm_name' not found at $VMS_DIR/$cvm_name"
        # the directory was removed under a running VM: the processes are the only trace left
        local orphan_pids
        orphan_pids=$(cvm_pids "$VMS_DIR/$cvm_name" | sort -u | xargs)
        if [ -n "$orphan_pids" ]; then
            log_error "but a VM from that directory is still running (pids: $orphan_pids) and keeps holding its GPUs and ports; kill those pids"
        fi
        return 1
    fi

    shift
    local stop_rc=0
    python3 "$SCRIPTS_DIR/dstack.py" stop "$VMS_DIR/$cvm_name" "$@" || stop_rc=$?

    if [ $stop_rc -ne 0 ]; then
        log_error "Failed to stop CVM '$cvm_name'"
        return 1
    fi

    # dstack.py stop exits 0 when runtime.json is missing — after 'rm -rf run/vms/<name>'
    # under a running VM it stops nothing, so only the process list proves the VM is gone.
    local leftover_pids attempt
    for attempt in 1 2 3 4 5; do
        leftover_pids=$(cvm_pids "$VMS_DIR/$cvm_name" | sort -u | xargs)
        [ -z "$leftover_pids" ] && break
        sleep 1
    done
    if [ -n "$leftover_pids" ]; then
        log_error "CVM '$cvm_name' is still running (pids: $leftover_pids) and keeps holding its GPUs and ports"
        log_info "Retry with '$0 stop $cvm_name --force', or kill those pids once you know what they are"
        return 1
    fi

    log_success "CVM '$cvm_name' stopped"
}

# List available GPUs
list_gpus() {
    log_info "Available GPUs:"
    python3 "$SCRIPTS_DIR/dstack.py" lsgpu
}

# List available CVMs
list_cvms() {
    log_info "Available CVMs:"
    if [ -d "$VMS_DIR" ]; then
        for cvm_dir in "$VMS_DIR"/*; do
            if [ -d "$cvm_dir" ]; then
                cvm_name=$(basename "$cvm_dir")
                if [ -f "$cvm_dir/vm-manifest.json" ]; then
                    echo "  $cvm_name (configured)"
                else
                    echo "  $cvm_name (incomplete)"
                fi
            fi
        done
    else
        echo "  No CVMs directory found"
    fi
}

# Show help
show_help() {
    echo "DStack CVM Management Script"
    echo
    echo "Usage: $0 <command> [arguments]"
    echo
    echo "Commands:"
    echo "  check                     Check if required software is installed"
    echo "  download                  Download and extract the OS image"
    echo "  new <name> [options]      Create a new CVM using .env configuration"
    echo "  run <name> [--dry-run]    Run the specified CVM"
    echo "  stop <name> [--timeout N] [--force]"
    echo "                            Gracefully stop a running CVM"
    echo "  list                      List available CVMs"
    echo "  lsgpu                     List available GPUs"
    echo "  help                      Show this help message"
    echo
    echo "Configuration:"
    echo "  All CVM settings are configured via .env file"
    echo "  Copy .env.example to .env and edit the CVM_* variables"
    echo
    echo "New command options:"
    echo "  --env <environment>       Target environment: local, staging, prod (default: prod)"
    echo "  --enable-logs             Enable public container logs"
    echo "  --enable-sysinfo          Enable public system and TCB info"
    echo
    echo "Examples:"
    echo "  $0 check"
    echo "  $0 download"
    echo "  $0 new my-cvm"
    echo "  $0 new my-cvm --env local"
    echo "  $0 new my-cvm --env staging"
    echo "  $0 new my-cvm --env prod --enable-logs --enable-sysinfo"
    echo "  $0 new gpu-cvm --env local"
    echo "  $0 run my-cvm"
    echo "  $0 run my-cvm --dry-run"
    echo "  $0 stop my-cvm"
    echo "  $0 stop my-cvm --force --timeout 60"
    echo "  $0 list"
    echo "  $0 lsgpu"
}

# Main script logic
case "$1" in
"check")
    check_requirements
    ;;
"download")
    log_info "Downloading OS image..."
    if download_os_image; then
        log_success "OS image download completed successfully"
    else
        log_error "Failed to download OS image"
        exit 1
    fi
    ;;
"new")
    shift
    new_cvm "$@"
    ;;
"run")
    run_cvm "$2" "$3"
    ;;
"stop")
    shift
    stop_cvm "$@"
    ;;
"list")
    list_cvms
    ;;
"lsgpu")
    list_gpus
    ;;
"help" | "-h" | "--help")
    show_help
    ;;
"")
    show_help
    ;;
*)
    log_error "Unknown command: $1"
    echo
    show_help
    exit 1
    ;;
esac
