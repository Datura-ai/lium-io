# Executor

## Setup project
### Requirements
* Ubuntu machine
* install [docker](https://docs.docker.com/engine/install/ubuntu/)


### Step 1: Clone project

```
git clone https://github.com/Datura-ai/compute-subnet.git
```

### Step 2: Install Required Tools

Run following command to install required tools: 
```shell
cd compute-subnet && chmod +x scripts/install_executor_on_ubuntu.sh && scripts/install_executor_on_ubuntu.sh
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

## Intel TDX Attestation

### What is TDX Attestation?

Intel TDX (Trust Domain Extensions) attestation provides hardware-based security guarantees for your executor. When enabled, the executor generates cryptographic proofs (quotes) that validators can verify to ensure:

- The executor is running on genuine Intel TDX-capable hardware
- The execution environment hasn't been tampered with
- The SSH host key is authentic and bound to the secure enclave

### How It Works

1. **Quote Generation**: When a validator requests SSH access, the executor generates a TDX quote that includes a hash of the SSH host key
2. **Quote Transmission**: The quote is sent to the validator along with the SSH host key
3. **Verification**: The validator sends the quote to a TDX verifier service to validate its authenticity
4. **Attestation Digest**: The verifier extracts OS image hash and compose hash to create an attestation digest
5. **Whitelist Check**: (Optional) The validator checks if the attestation digest is in the approved whitelist

### Configuration

To enable TDX attestation on your executor:

1. Ensure your hardware supports Intel TDX
2. Install the dstack-sdk (included in dependencies)
3. Add to your `.env` file:
```bash
ENABLE_TDX_ATTESTATION=true
TDX_QUOTE_TIMEOUT=60
```
4. Restart your executor: `docker compose restart`

### Verification

To verify TDX is working correctly:

1. Check executor logs for TDX-related messages:
```bash
docker compose logs -f
```

2. Look for messages like:
   - "TDX attestation enabled"
   - "Generated TDX quote for SSH host key"
