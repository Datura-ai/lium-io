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
    sudo apt-get update
    sudo apt-get install --no-install-recommends --no-install-suggests -y sudo apt-utils curl git cmake build-essential
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
        sudo apt-get install -y python3.11
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
        sudo apt-get install -y python3.12-venv
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

        sudo apt-get install -y redis-server

        echo "Starting Redis server..."
        sudo systemctl start redis-server.service

        echo "Checking Redis server status..."
        sudo systemctl status redis-server.service
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
        DB_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='compute_subnet_db'")
        if [ "$DB_EXISTS" == "1" ]; then
            echo "Database compute_subnet_db already exists."
        else
            echo "Creating database compute_subnet_db..."
            sudo -u postgres createdb compute_subnet_db
        fi
    else
        echo "Installing PostgreSQL..."
        sudo apt-get install -y postgresql postgresql-contrib

        echo "Starting PostgreSQL server..."
        sudo systemctl start postgresql.service

        echo "Setting password for postgres user..."
        sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'password';"

        echo "Creating database compute_subnet_db..."
        sudo -u postgres createdb compute_subnet_db
    fi
}

# install btcli
install_btcli() {
    if command -v btcli &> /dev/null
    then
        ohai "BtCLI is already installed."
    else
        ohai "Installing BtCLI..."

        sudo apt-get install -y pipx 
        pipx install bittensor
        source ~/.bashrc
    fi
}

# install docker
install_docker() {
  if command -v docker &> /dev/null; then
    ohai "Docker is already installed."
    return 0
  else
    ohai "Installing Docker..."
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    
    # Add the repository to Apt sources:
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
  fi
}

ohai "This script will install:"
echo "bittensor"
echo "docker"


wait_for_user
install_pre
install_btcli
install_docker
