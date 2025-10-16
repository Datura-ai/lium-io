#!/bin/sh
set -e

# Universal Python and Jupyter installation script
# Supports: Alpine (apk), Ubuntu/Debian (apt), CentOS/RHEL (yum/dnf), Arch (pacman)
#
# Usage: ./run_jupyter.sh [--password=PASSWORD] [--port=PORT] [--help]
#   --password: Jupyter password (required)
#   --port: Jupyter port (default: 8888)
#   --help: Show this help message

# Default values
JUPYTER_PORT=8888
JUPYTER_PASSWORD=""

# Parse command line arguments
parse_arguments() {
    while [ $# -gt 0 ]; do
        case $1 in
            --password=*)
                JUPYTER_PASSWORD="${1#*=}"
                shift
                ;;
            --port=*)
                JUPYTER_PORT="${1#*=}"
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--password=PASSWORD] [--port=PORT] [--help]"
                echo "  --password: Jupyter password (required)"
                echo "  --port: Jupyter port (default: 8888)"
                echo "  --help: Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Validate required parameters
    if [ -z "$JUPYTER_PASSWORD" ]; then
        echo "Error: --password is required"
        echo "Use --help for usage information"
        exit 1
    fi
}

# Function to detect package manager
detect_package_manager() {
    echo "Detecting Linux distribution and package manager..."
    if command -v apk >/dev/null 2>&1; then
        echo "apk"
    elif command -v apt >/dev/null 2>&1; then
        echo "apt"
    elif command -v dnf >/dev/null 2>&1; then
        echo "dnf"
    elif command -v yum >/dev/null 2>&1; then
        echo "yum"
    elif command -v pacman >/dev/null 2>&1; then
        echo "pacman"
    elif command -v zypper >/dev/null 2>&1; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

# Install system packages function
install_system_packages() {
    pkg_manager=$(detect_package_manager)
    
    if [ "$pkg_manager" = "unknown" ]; then
        echo "Error: Could not detect package manager. This script supports:"
        echo "  - Alpine (apk)"
        echo "  - Ubuntu/Debian (apt)"
        echo "  - CentOS/RHEL (yum/dnf)"
        echo "  - Arch (pacman)"
        echo "  - openSUSE (zypper)"
        exit 1
    fi
    
    echo "Detected package manager: $pkg_manager"
    
    # Install system packages based on package manager
    case $pkg_manager in
        "apk")
            echo "Installing packages with apk (Alpine)..."
            apk update
            apk add --no-cache python3 python3-dev python3-pip py3-pip build-base linux-headers gcc g++ make bash
            ;;
        "apt")
            echo "Installing packages with apt (Ubuntu/Debian)..."
            apt-get update
            apt-get install -y python3 python3-dev python3-pip python3-venv build-essential bash
            ;;
        "dnf")
            echo "Installing packages with dnf (Fedora/CentOS 8+)..."
            dnf install -y python3 python3-devel python3-pip gcc gcc-c++ make bash
            ;;
        "yum")
            echo "Installing packages with yum (CentOS/RHEL)..."
            yum install -y python3 python3-devel python3-pip gcc gcc-c++ make bash
            ;;
        "pacman")
            echo "Installing packages with pacman (Arch)..."
            pacman -Syu --noconfirm python python-pip base-devel bash
            ;;
        "zypper")
            echo "Installing packages with zypper (openSUSE)..."
            zypper install -y python3 python3-devel python3-pip gcc gcc-c++ make bash
            ;;
        *)
            echo "Error: Unsupported package manager: $pkg_manager"
            exit 1
            ;;
    esac
    
    echo "Verifying installation..."
    python3 --version
}

# Function to install Python packages with pip
install_python_packages() {
    echo "Installing Python packages with pip..."
    
    # Upgrade pip first
    python3 -m pip install --upgrade pip
    
    # Install Jupyter and common packages, ignoring system-installed packages
    echo "Installing Jupyter and Python packages..."
    python3 -m pip install --ignore-installed \
        jupyter \
        jupyterlab \
        notebook \
        ipykernel \
        matplotlib \
        numpy \
        pandas \
        requests

    echo "Verifying installation..."
    jupyter --version
}

# Install Python and Jupyter if needed
install_python_jupyter() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Python not found. Installing system packages and Python..."
        install_system_packages
        install_python_packages
    elif ! command -v jupyter >/dev/null 2>&1; then
        echo "Jupyter not found. Installing Python packages..."
        install_python_packages
    else
        echo "Python and Jupyter are already installed."
    fi
}

# Detect available shell for Jupyter terminal
detect_shell() {
    if command -v bash >/dev/null 2>&1; then
        echo "/bin/bash"
    elif command -v sh >/dev/null 2>&1; then
        echo "/bin/sh"
    else
        echo "/bin/sh"  # fallback
    fi
}

# Start jupyter lab
start_jupyter() {
    echo "Starting Jupyter Lab on port $JUPYTER_PORT..."
    
    # Install Python/Jupyter if needed
    install_python_jupyter
    
    # Detect available shell
    shell_cmd=$(detect_shell)
    echo "Using shell: $shell_cmd"
    
    nohup jupyter lab --allow-root --no-browser --port=$JUPYTER_PORT --ip=0.0.0.0 --FileContentsManager.delete_to_trash=False --ServerApp.terminado_settings="{\"shell_command\":[\"$shell_cmd\"]}" --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=/root --ServerApp.disable_check_xsrf=True --ServerApp.tornado_settings="{\"max_body_size\": 536870912}" &> /jupyter.log &
    
    echo "✅ Jupyter Lab started successfully on port $JUPYTER_PORT"
    echo "Access Jupyter at: http://localhost:$JUPYTER_PORT"
    echo "Password: $JUPYTER_PASSWORD"
    echo ""
    echo "Available commands:"
    echo "  - python3"
    echo "  - jupyter"
    echo "  - jupyter lab"
    echo "  - jupyter notebook"
}

# Parse command line arguments
parse_arguments "$@"

# Start Jupyter
start_jupyter