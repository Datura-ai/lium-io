#!/bin/bash
set -u

# enable command completion
set -o history -o histexpand

abort() {
  printf "%s\n" "$1"
  exit 1
}

getc() {
  local save_state
  save_state=$(/bin/stty -g)
  /bin/stty raw -echo
  IFS= read -r -n 1 -d '' "$@"
  /bin/stty "$save_state"
}

exit_on_error() {
    exit_code=$1
    last_command=${@:2}
    if [ $exit_code -ne 0 ]; then
        >&2 echo "\"${last_command}\" command failed with exit code ${exit_code}."
        exit $exit_code
    fi
}

shell_join() {
  local arg
  printf "%s" "$1"
  shift
  for arg in "$@"; do
    printf " "
    printf "%s" "${arg// /\ }"
  done
}

# string formatters
if [[ -t 1 ]]; then
  tty_escape() { printf "\033[%sm" "$1"; }
else
  tty_escape() { :; }
fi
tty_mkbold() { tty_escape "1;$1"; }
tty_underline="$(tty_escape "4;39")"
tty_blue="$(tty_mkbold 34)"
tty_red="$(tty_mkbold 31)"
tty_bold="$(tty_mkbold 39)"
tty_reset="$(tty_escape 0)"

ohai() {
  printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$(shell_join "$@")"
}

wait_for_user() {
  local c
  echo
  echo "Press RETURN to continue or any other key to abort"
  getc c
  # we test for \r and \n because some stuff does \r instead
  if ! [[ "$c" == $'\r' || "$c" == $'\n' ]]; then
    exit 1
  fi
}

#install pre
install_pre() {
    apt update
    apt install --no-install-recommends --no-install-suggests -y sudo apt-utils curl git cmake build-essential
    exit_on_error $?
}

# check if python is installed, if not install it
install_python() {
    # Check if python3.11 is installed
    if command -v python3.11 &> /dev/null
    then
        # Check the version
        PYTHON_VERSION=$(python3.11 --version 2>&1)
        if [[ $PYTHON_VERSION == *"Python 3.11"* ]]; then
            ohai "Python 3.11 is already installed."
        else
            ohai "Linking python to python 3.11"
            sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
            python -m pip install cffi
            python -m pip install cryptography
        fi
    else
        ohai "Installing Python 3.11"
        add-apt-repository ppa:deadsnakes/ppa
        apt install python3.11
        sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
        python -m pip install cffi
        python -m pip install cryptography
    fi

    # check if PDM is installed
    if command -v pdm &> /dev/null
    then
        ohai "PDM is already installed."
        echo "Checking PDM version..."
        pdm --version
    else
        ohai "Installing PDM..."
        curl -sSL https://pdm-project.org/install-pdm.py | python3 -

        local bashrc_file="/root/.bashrc"
        local path_string="export PATH=/root/.local/bin:\$PATH"

        if ! grep -Fxq "$path_string" $bashrc_file; then
            echo "$path_string" >> $bashrc_file
            echo "Added $path_string to $bashrc_file"
        else
            echo "$path_string already present in $bashrc_file"
        fi

        export PATH=/root/.local/bin:$PATH

        echo "Checking PDM version..."
        pdm --version
    fi
}

# install redis
install_redis() {
    if command -v redis-server &> /dev/null
    then
        ohai "Redis is already installed."
        echo "Checking Redis version..."
        redis-server --version
    else
        ohai "Installing Redis..."

        apt install -y redis-server

        echo "Starting Redis server..."
        service redis-server start

        echo "Checking Redis server status..."
        service redis-server status
    fi
}

# install postgresql
install_postgresql() {
    if command -v psql &> /dev/null
    then
        ohai "PostgreSQL is already installed."
        echo "Checking PostgreSQL version..."
        psql --version

        # Check if the database exists
        DB_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='compute_horde_miner'")
        if [ "$DB_EXISTS" == "1" ]; then
            echo "Database compute_horde_miner already exists."
        else
            echo "Creating database compute_horde_miner..."
            sudo -u postgres createdb compute_horde_miner
        fi
    else
        ohai "Installing PostgreSQL..."

        echo "Installing PostgreSQL..."
        sudo apt install -y postgresql postgresql-contrib

        echo "Starting PostgreSQL server..."
        sudo service postgresql start

        echo "Setting password for postgres user..."
        sudo -u postgres psql -c "ALTER USER postgres PASSWORD '12345';"

        echo "Creating database compute_horde_miner..."
        sudo -u postgres createdb compute_horde_miner
    fi
}

# install btcli
install_btcli() {
    if command -v btcli &> /dev/null
    then
        ohai "BtCLI is already installed."
    else
        ohai "Installing BtCLI..."

        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
    fi
}

# install hashcat
install_single_hashcat() {
    instance_dir=$1
    # URL to download Hashcat
    hashcat_url="https://hashcat.net/files/hashcat-6.2.6.7z"

    # Directory to store downloaded files
    download_dir="/tmp/hashcat_download"

    # Create the directory if it does not exist
    mkdir -p $instance_dir
    mkdir -p $download_dir

    # Download Hashcat
    wget -O $download_dir/hashcat.7z $hashcat_url

    # Install p7zip if not already installed
    if ! command -v 7z &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y p7zip-full
    fi

    # Extract the downloaded file
    7z x $download_dir/hashcat.7z -o$instance_dir

    # Make the hashcat executable
    chmod +x $instance_dir/hashcat-6.2.6/hashcat.bin
}

install_hashcat() {
    ohai "Installing Hashcat..."

    read -p "Enter Hashcat Instances: " HASHCAT_INSTANCES
    # Loop through the instances
    for i in $(seq 1 $HASHCAT_INSTANCES); do
        instance_dir="/opt/hashcat$i"
        if [ -f $instance_dir/hashcat-6.2.6/hashcat.bin ]; then
            echo "Hashcat already installed in $instance_dir"
        else
            echo "Hashcat not found in $instance_dir. Installing..."
            install_single_hashcat $instance_dir
        fi
    done
}

# generate .env file
generate_env() {
    PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    PROJECT_DIR=${PROJECT_DIR}/miner
    ENV_DIR="./envs/dev"
    cd "${PROJECT_DIR}"

    ohai "Generating env and Setting up dev..."

    read -p "Enter PORT_FOR_EXECUTORS: " PORT_FOR_EXECUTORS
    read -p "Enter BITTENSOR_MINER_ADDRESS: " BITTENSOR_MINER_ADDRESS
    read -p "Enter BITTENSOR_MINER_PORT: " BITTENSOR_MINER_PORT

    sed -e "s|{{PORT_FOR_EXECUTORS}}|$PORT_FOR_EXECUTORS|g" \
        -e "s|{{BITTENSOR_MINER_ADDRESS}}|$BITTENSOR_MINER_ADDRESS|g" \
        -e "s|{{BITTENSOR_MINER_PORT}}|$BITTENSOR_MINER_PORT|g" ${ENV_DIR}/.env.template > ${ENV_DIR}/.env

    sudo chmod +x ./setup-dev.sh
    /usr/bin/bash ./setup-dev.sh

    ohai "Running migrations..."
    cd "${PROJECT_DIR}/app/src"
    pdm run manage.py migrate

    mkdir -p /tmp/logs

    ohai "Creating services..."
    cd "${PROJECT_DIR}"
    sudo chmod +x ./app/envs/dev/*.*
    sudo cp ./app/envs/dev/celeryapp /etc/init.d/celeryapp
    sudo cp ./app/envs/dev/minerapp /etc/init.d/minerapp
    sudo cp ./app/envs/dev/minerbeat /etc/init.d/minerbeat

    sudo chmod +x /etc/init.d/celeryapp
    sudo chmod +x /etc/init.d/minerapp
    sudo chmod +x /etc/init.d/minerbeat
}

ohai "This script will install:"
echo "git"
echo "curl"
echo "python3.11 and pdm"
echo "python3-pip"
echo "redis"
echo "postgresql"
echo "bittensor"
echo "hashcat"

wait_for_user
install_pre
install_python
install_redis
install_postgresql
install_btcli
# install_hashcat
generate_env