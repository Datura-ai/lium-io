# Executor

**[Node Quickstart on docs.lium.io](https://docs.lium.io/providers/nodes/quickstart)**

## Quick setup with `lium mine` (recommended)

Before the first command, the machine needs:

- **Ubuntu 22.04 on x86_64**, kernel 5.19 or newer (6.x recommended; `hostnamectl` shows both), and root or passwordless `sudo`;
- **the NVIDIA driver loaded** — `nvidia-smi` lists the GPUs;
- **Docker Engine installed and running** — `docker ps` works. The Sysbox installer below refuses to run without it (`lium mine` runs the Docker install script too, but only after this step, and it is a no-op on a machine that already has Docker);
- **a public IP** with the service port (`8080`) and the node SSH port (`2200`) reachable, and the hotkey (SS58 address) of your registered provider account — see the [Provider Quickstart](https://docs.lium.io/providers/quickstart).

RAM, disk (including the 1.5× VRAM rule for the idle incentive) and the recommended XFS storage setup are in the [Node Quickstart requirements](https://docs.lium.io/providers/nodes/quickstart#requirements).

Install [Sysbox](https://docs.lium.io/providers/nodes/sysbox) first — validators reject a node without the `sysbox-runc` runtime. The installer below also sets up the NVIDIA Container Toolkit:

```shell
curl -fsSL https://raw.githubusercontent.com/Datura-ai/lium-io/main/neurons/executor/nvidia_docker_sysbox_setup.sh | sudo bash
```

The installer also sets up size-limited volumes, see [Volume plugin (vloopback)](#volume-plugin-vloopback). It exits `1` when this host cannot mount one.

Then run the node in one command, with your provider hotkey:

```shell
curl -fsSL https://lium.io/mine.sh | bash -s -- -k <your_provider_hotkey_ss58>
```

The script installs the [`lium`](https://github.com/Datura-ai/lium) CLI and runs `lium mine`, which:

1. clones this repository into `./compute-subnet` (or pulls the branch if the directory already exists),
2. runs `scripts/install_executor_on_ubuntu.sh` (Docker Engine and the compose plugin), then checks prerequisites — `nvidia-smi`, `nvidia-container-cli`, `docker info`,
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

* Install Sysbox and the NVIDIA Container Toolkit (root is required)
```shell
sudo ./nvidia_docker_sysbox_setup.sh
```

Put your provider hotkey (SS58 address) in `MINER_HOTKEY_SS58_ADDRESS` — the variable keeps its historical name.
You can change the ports for `INTERNAL_PORT`, `EXTERNAL_PORT`, `SSH_PORT` based on your need; the template's
defaults are `8001` / `8001` / `2200` (`lium mine` proposes `8080` for the service port instead).

- **INTERNAL_PORT**: internal port of your executor docker container
- **EXTERNAL_PORT**: external expose port of your executor docker container
- **SSH_PORT**: ssh port map into 22 of your executor docker container
- **SSH_PUBLIC_PORT**: [Optional] ssh public access port of your executor docker container. If `SSH_PUBLIC_PORT` is equal to `SSH_PORT` then you don't have to specify this port.
- **MINER_HOTKEY_SS58_ADDRESS**: your provider hotkey (SS58 address)
- **RENTING_PORT_RANGE**: The port range that are publicly accessible. This can be empty if all ports are open. Available formats are: 
  - Range Specification(`from-to`): a range of ports, such as 2000-2005. This means ports from 2000 to 2005 will be open for the validator to select.
  - Specific Ports(`port1,port2,port3`): individual ports, such as 2000,2001,2002. This means only ports 2000, 2001, and 2002 will be available for the validator.
  - Default Behavior: If no ports are specified, the validator will assume that all ports on the executor are available.
- **RENTING_PORT_MAPPINGS**: Internal, external port mappings. Use this env when you are using proxy in front of your executors and the internal port and external port can't be the same. You can ignore this env, if all ports are open or the internal and external ports are the same. example:
  - if internal port 46681 is mapped to 56681 external port and internal port 46682 is mapped to 56682 external port, then RENTING_PORT_MAPPINGS="[[46681, 56681], [46682, 56682]]"

Note: Please use either **RENTING_PORT_RANGE** or **RENTING_PORT_MAPPINGS** and DO NOT use both of them if you have specific ports are available.

Optional, commented out in the template (`src/core/config.py` has the defaults):

- **COMPUTE_REST_API_URL**: the Lium backend the executor pre-pulls its GPU's cache template image from (default `https://lium.io/api`; empty disables the pre-pull)
- **CACHE_TEMPLATE_REFRESH_SECONDS**: how often the template digest is re-checked (default `900`)
- **CONTAINER_SIGNATURE_MAX_AGE_SECONDS**: maximum clock skew accepted on signed pod-metrics and pod-log requests (default `300`; keep the host on NTP rather than widening this)
- **EXECUTOR_LOCAL_VERIFY_ENABLED**: answer the validator's one-call `POST /verify` (the verification suite from one signed intent, posted through the validator's SSH connection to the executor's loopback port; a request from the network is refused, and so is an intent that does not name this host's `MINER_HOTKEY_SS58_ADDRESS`) instead of 404 (default `false`; the validator falls back to its SSH checks either way)
- **LOCAL_VERIFY_MAX_DEADLINE_SECONDS**: the longest a `/verify` intent may keep the GPU before the executor answers with what finished (default `600`)
- **LOCAL_VERIFY_INTENT_WINDOW_SECONDS**: how far a `/verify` intent's `issued_at` may be from this host's clock (default `120`; NTP, as above)


* Run project
```shell
docker compose up -d
```

The stack then updates itself. How that works, how to make sure that a node is current, and the one-time restart for a node that stopped updating: [EXECUTOR_UPDATE.md](EXECUTOR_UPDATE.md).

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
| Ubuntu      | 22.04+  |
| Kernel      | 5.19+ (6.x recommended) |

Why 5.19: overlayfs on ID-mapped mounts, which Sysbox needs for GPUs, landed in 5.19; `nvidia_docker_sysbox_setup.sh` checks it.

Checking OS and Kernel version
```shell
hostnamectl
```

Get the HWE kernel on Ubuntu 22.04 if the kernel version is older than 5.19
```shell
sudo apt update
sudo apt install --install-recommends linux-generic-hwe-22.04
sudo reboot
```

Installation of sysbox (as root; `sudo ./nvidia_docker_sysbox_setup.sh --check` only runs the host checks)
```shell
sudo ./nvidia_docker_sysbox_setup.sh
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

### Volume plugin (vloopback)

A rental with a disk limit gets its volume from the `vloopback` Docker volume plugin ([`ashald/docker-volume-loopback`](https://github.com/ashald/docker-volume-loopback)), mounted into the pod's Sysbox container. Once Sysbox runs a GPU container, `nvidia_docker_sysbox_setup.sh` does this itself, and re-running it on a working host changes nothing:

1. installs the plugin when it is missing, with `DATA_DIR=<Docker data-root>/loopback` (the data-root is `docker info --format '{{.DockerRootDir}}'`), the same way the validators install it on the first rental;
2. on a plugin whose `DATA_DIR` is not an absolute path, stops if a rental is running, then sets `DATA_DIR=<Docker data-root>/loopback` and enables the plugin;
3. creates a 1 GB test volume, checks that `docker volume inspect` shows a `Mountpoint` starting with `/`, mounts the volume into a Sysbox container, writes a file and reads it back, lets the container exit (which unmounts the volume) and removes the volume.

When a step fails, the installer prints one message with the cause and this link, restarts the executor it stopped, and exits `1`. `--check` runs step 3 on an installed plugin and prints `FIX` lines for steps 1 and 2 without changing the plugin.

The same commands by hand:

```shell
root=$(docker info --format '{{.DockerRootDir}}')
docker plugin install ashald/docker-volume-loopback --alias vloopback --grant-all-permissions DATA_DIR=$root/loopback   # when the plugin is missing
docker plugin disable vloopback && docker plugin set vloopback DATA_DIR=$root/loopback && docker plugin enable vloopback   # when DATA_DIR is not absolute; stop rentals first
docker volume create -d vloopback -o size=10G t && docker volume inspect t --format '{{.Mountpoint}}'   # must start with /
docker run --rm --runtime=sysbox-runc -v t:/lium-vol alpine sh -c 'echo ok > /lium-vol/probe && cat /lium-vol/probe'   # prints ok
docker volume rm t
```

Causes the installer names:

- **`docker plugin disable` fails with `plugin vloopback:latest is in use`**: volumes left by earlier rentals still use the plugin. With no rental running, list them with `docker volume ls --filter driver=vloopback`, remove them, and run the installer again.
- **`error setting up ID-mapped mount on path …`**, or a `Mountpoint` that does not start with `/`: Sysbox mounts volumes through ID-mapped mounts, and the filesystem under Docker's data-root does not support them. Put the data-root on ext4 or XFS ([Docker Storage Setup](https://docs.lium.io/providers/nodes/docker-storage)), then run the installer again.
- **`docker plugin install` fails**: the host cannot pull `ashald/docker-volume-loopback` from Docker Hub. Check the host's proxy or DNS, then run the installer again.

Validators run the same test (plugin, `Mountpoint`, a volume mounted into a Sysbox container) in every machine scrape. A node that fails it reports `storage_limit_supported: false` with a reason code such as `VLOOPBACK_SYSBOX_MOUNT_FAILED` in `storage_limit_scrape_error`, and its rentals start without a disk limit.

## The executor image

`Dockerfile` starts from `python:3.11-slim` pinned by digest; the comment above the `FROM` line names the tag and the date the digest was taken. To move to a newer base, resolve the tag (`docker buildx imagetools inspect python:3.11-slim`), put the new digest on that line and rebuild. The build ends with `sshd_setup.sh`: it turns sshd's `PerSourcePenalties` off through `/etc/ssh/sshd_config.d/lium.conf` when the base's OpenSSH knows the directive (9.8 and later), and fails the build when `sshd -T` rejects the rendered configuration.

### Validator hotkeys and uploaded ssh keys

The executor accepts a request signed by the validator hotkey only. The hotkey is compiled into the image (`src/core/config.py`, `VALIDATOR_HOTKEY_SS58`, overridable at build time with `VALIDATOR_HOTKEY_SS58=<ss58> bash docker_build.sh`). During a hotkey rotation the image accepts a second one, `VALIDATOR_NEXT_HOTKEY_SS58` (same file, or `VALIDATOR_HOTKEY_SS58=<ss58> VALIDATOR_NEXT_HOTKEY_SS58=<ss58>` at build time; `docker_build.sh` refuses `next` without `current`), and a signature by either is valid; with it empty only the first hotkey is accepted. Neither hotkey is read from the environment: the signers an executor trusts are fixed by its image.

An ssh key the validator installs through `/upload_ssh_key` is appended to the container's `~/.ssh/authorized_keys` with a `lium-uploaded-at=<unix time>` comment. The validator removes its key through `/remove_ssh_key` when its job is done; a key still present `EXECUTOR_UPLOADED_KEY_TTL_S` seconds after the upload (default 900) is removed by the executor itself, checked every `EXECUTOR_UPLOADED_KEY_PURGE_INTERVAL_S` seconds (default 60). Only lines carrying that comment are expired. `authorized_keys` lives on the disk reserve (`setup_disk_reserve.sh`), so keys uploaded before this release survive the upgrade without the comment. At its first start on this release the executor adds the comment, with the start time, to every key line that lacks it (blank lines and `#` comments are skipped; a stamped line is rewritten as the key, one space, the comment) and logs `legacy keys stamped: N`; those keys then expire `EXECUTOR_UPLOADED_KEY_TTL_S` seconds later like any upload. Lines that already carry the comment keep their own time. The stamp runs once: it writes `~/.ssh/.lium-uploaded-keys-stamped` next to `authorized_keys`, and a later start that finds that file does nothing.
