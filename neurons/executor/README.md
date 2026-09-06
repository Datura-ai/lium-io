# Executor

**[Node Quickstart on docs.lium.io](https://docs.lium.io/providers/nodes/quickstart)**

## Quick setup with `lium mine` (recommended)

Install [Sysbox](https://docs.lium.io/providers/nodes/sysbox) first — validators reject a node without the `sysbox-runc` runtime. The installer below also sets up the NVIDIA Container Toolkit:

```shell
curl -fsSL https://raw.githubusercontent.com/Datura-ai/lium-io/main/neurons/executor/nvidia_docker_sysbox_setup.sh | sudo bash
```

Then run the node in one command, with your miner hotkey:

```shell
curl -fsSL https://lium.io/mine.sh | bash -s -- -k <your_miner_hotkey_ss58>
```

The script installs the [`lium`](https://github.com/Datura-ai/lium) CLI and runs `lium mine`, which:

1. clones this repository into `./compute-subnet` (or pulls the branch if the directory already exists),
2. runs `scripts/install_executor_on_ubuntu.sh` (Docker, NVIDIA Container Toolkit and GPU checks),
3. renders `neurons/executor/.env` from `.env.template` with the hotkey and the ports — you are prompted for the service port (`8080`), the node SSH port (`2200`), an optional public SSH port and an optional renting port range; pass `--auto` to accept the defaults without prompts,
4. starts the executor with `docker compose up -d` and waits for the container to report `healthy`,
5. runs the validator's own check against the node (`daturaai/lium-validator:latest`).

At the end it prints the node's endpoint, GPU type and count, and a `provider.lium.io/nodes?action=add&…` link that pre-fills the **Add Node** form in the [Provider Portal](https://provider.lium.io) with those values. Other options: `-d/--dir` (checkout directory, default `compute-subnet`) and `-b/--branch` (default `main`); `lium mine --help` lists them.

## Manual setup

Use this path if you want to set every value yourself instead of running `lium mine`.

### Requirements
* Ubuntu machine
* install [docker](https://docs.docker.com/engine/install/ubuntu/)


### Step 1: Clone project

```
git clone https://github.com/Datura-ai/lium-io.git
```

### Step 2: Install Required Tools

Run following command to install required tools: 
```shell
cd lium-io && chmod +x scripts/install_executor_on_ubuntu.sh && scripts/install_executor_on_ubuntu.sh
```

if you don't have sudo on your machine, run
```shell
sed -i 's/sudo //g' scripts/install_executor_on_ubuntu.sh
```
to remove sudo from the setup script commands

### Step 3: Configure Docker for Nvidia

Please follow [this](https://stackoverflow.com/questions/72932940/failed-to-initialize-nvml-unknown-error-in-docker-after-few-hours) to setup docker for nvidia properly 


### Step 4: Install and Run

* Go to executor root
```shell
cd neurons/executor
```

* Add .env in the project
```shell
cp .env.template .env
```

* Install Required Tools 
```shell
./nvidia_docker_sysbox_setup.sh
```

Add the correct miner wallet hotkey for `MINER_HOTKEY_SS58_ADDRESS`.
You can change the ports for `INTERNAL_PORT`, `EXTERNAL_PORT`, `SSH_PORT` based on your need.

- **INTERNAL_PORT**: internal port of your executor docker container
- **EXTERNAL_PORT**: external expose port of your executor docker container
- **SSH_PORT**: ssh port map into 22 of your executor docker container
- **SSH_PUBLIC_PORT**: [Optional] ssh public access port of your executor docker container. If `SSH_PUBLIC_PORT` is equal to `SSH_PORT` then you don't have to specify this port.
- **MINER_HOTKEY_SS58_ADDRESS**: the miner hotkey address
- **RENTING_PORT_RANGE**: The port range that are publicly accessible. This can be empty if all ports are open. Available formats are: 
  - Range Specification(`from-to`): Miners can specify a range of ports, such as 2000-2005. This means ports from 2000 to 2005 will be open for the validator to select.
  - Specific Ports(`port1,port2,port3`): Miners can specify individual ports, such as 2000,2001,2002. This means only ports 2000, 2001, and 2002 will be available for the validator.
  - Default Behavior: If no ports are specified, the validator will assume that all ports on the executor are available.
- **RENTING_PORT_MAPPINGS**: Internal, external port mappings. Use this env when you are using proxy in front of your executors and the internal port and external port can't be the same. You can ignore this env, if all ports are open or the internal and external ports are the same. example:
  - if internal port 46681 is mapped to 56681 external port and internal port 46682 is mapped to 56682 external port, then RENTING_PORT_MAPPINGS="[[46681, 56681], [46682, 56682]]"

Note: Please use either **RENTING_PORT_RANGE** or **RENTING_PORT_MAPPINGS** and DO NOT use both of them if you have specific ports are available.


* Run project
```shell
docker compose up -d
```

## Recommended Setup For GPUs and Docker

### Step 1: Ensure `nvidia-container-toolkit` is installed. 

```shell
nvidia-container-cli --version
```

### Step 2: Ensure you installed latest `nvidia-container-cli` version. 
You can find latest version in [NVIDIA Container Toolkit Github Repository](https://github.com/NVIDIA/libnvidia-container). 

You can upgrade your `nvidia-container-toolkit` with following command:

```shell
sudo apt-get update && sudo apt-get install --only-upgrade nvidia-container-toolkit
```

### Step 3: Enable cgroups for docker. 

Go to `/etc/nvidia-container-runtime/config.toml` and enable `no-cgroups=false`. 

### Step 4: Update docker daemon.json file. 

Go to `/etc/docker/daemon.json` and add `"exec-opts": ["native.cgroupdriver=cgroupfs"]`. 

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "exec-opts": ["native.cgroupdriver=cgroupfs"]
}
```

### Step 5: Sysbox setup

#### System Requirments
| OS          | Version |
|-------------|---------|
| Ubuntu      | 22+     |
| Kernel      | 6.5+    |

Checking OS and Kernel version
```shell
hostnamectl
```

Get the latest kernel version on ubuntu 22.04 if the kernel version is less than 6.5
```shell
sudo apt update
sudo apt install --install-recommends linux-generic-hwe-22.04
sudo reboot
```

Installation of sysbox
```shell
./nvidia_docker_sysbox_setup.sh
```

Verify sysbox is working correctly with gpu
```shell
docker run --rm --runtime=sysbox-runc --gpus all daturaai/compute-subnet-executor:latest nvidia-smi
```

The above command should show the `nvidia-smi` result if sysbox is installed correctly.


### Step 6: Restart docker. 

```shell
sudo systemctl restart docker
```
