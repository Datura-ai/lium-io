# Validator

**[Validator Documentation](https://docs.lium.io/validators)**

Subnet 51 has one validator, operated by the Lium team (hotkey `5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13p`).
This directory is its source. If you want to verify what the validator does, use the community
[sn51-auditor](https://github.com/Datura-ai/sn51-auditor) instead of running a second validator.
The steps below are for the Lium team's own deployments and for testnet.

## System Requirements

For validation, a validator machine will need:

- **CPU**: 4 cores
- **RAM**: 8 GB

Ensure that your machine meets these requirements before proceeding with the setup.

---

First, register and regen your bittensor wallet and validator hotkey onto the machine. 

For installation of btcli, check [this guide](https://github.com/opentensor/bittensor/blob/master/README.md#install-bittensor-sdk)
```
btcli s register --netuid 51
```
```
btcli w regen_coldkeypub
```
```
btcli w regen_hotkey
```

## Installation

### Using Docker

#### Step 1: Clone Git repo

```
git clone https://github.com/Datura-ai/lium-io.git
```

#### Step 2: Install Required Tools

```
cd lium-io && chmod +x scripts/install_validator_on_ubuntu.sh && ./scripts/install_validator_on_ubuntu.sh
```

Verify docker installation

```
docker --version
```
If did not correctly install, follow [this link](https://docs.docker.com/engine/install/)

#### Step 3: Setup ENV
```
cp neurons/validators/.env.template neurons/validators/.env
```

Replace with your information for `BITTENSOR_WALLET_NAME`, `BITTENSOR_WALLET_HOTKEY_NAME`, `HOST_WALLET_DIR`.
If you want you can use different port for `INTERNAL_PORT`, `EXTERNAL_PORT`.

`LOG_LEVEL` sets the validator's log level (default `INFO`). At `INFO`, an INFO check outcome that repeats the previous cycle is logged only at `DEBUG`, and only its step time stays at `INFO`; a repeated provider-state warning (such as missing collateral) is `DEBUG` too; other warnings, errors and the run's last or halting event (the one carrying the step summary) still log every cycle; set `LOG_LEVEL=DEBUG` and restart to see every outcome again. `DEBUG` makes the logs much larger, so use it for a short time.

#### Step 4: Docker Compose Up

```
cd neurons/validators && docker compose up -d
```

## How the port check samples declared ports

Each cycle the validator checks from outside that an executor's declared ports are forwarded. An executor needs at least `MIN_PORT_COUNT` (3) ports that answer to be listed.

Ports already held by a rental or a filler are skipped.

- **Which ports:** the lowest `BATCH_PORT_VERIFICATION_SIZE` (300) free declared ports: one `--network=host` batch container (two attempts), then the first 50 ports published in chunks of 10, then the first 10 one by one. A declaration of 300 ports or fewer is probed in full. The DinD probe then takes one port. **Declare only the ports you forward:** ports above the lowest 300 are never probed.
- **Per-range tally:** the port-connectivity event carries `port_ranges`, a list of `{range, declared, probed, answered}` entries. Declared ports are grouped into buckets of `PORT_RANGE_BUCKET_WIDTH` (5000) ports by external port, so a partial forward shows which part of a wide range answered.

Code: `src/services/executor_connectivity/port_selector.py` (`select`, `tally_port_ranges`) and `orchestrator.py`.

## Download speed from inside a container

The VerifyX speed test runs on the host network. A renter's container goes through the docker bridge and NAT and can get far less, so on an idle node (no pod, no filler) the validator also starts a throwaway `alpine:3.19` container (with `sysbox-runc` where the node has it) once every `CONTAINER_DOWNLOAD_SPEED_INTERVAL_HOURS` (6) and reads up to 16 MiB of a pinned PyPI wheel (files.pythonhosted.org) and a pinned Hugging Face file, 30 s each. Each run logs `CONTAINER_DOWNLOAD_SPEED` with endpoint, bytes, seconds and Mbps, or an error kind (`dns_failed`, `unreachable`, `timeout`, `http_error`, `no_output`, `other`).

A reading fails when an endpoint reads under `CONTAINER_DOWNLOAD_SPEED_MIN_MBPS` (10), or the container cannot reach an endpoint that the validator itself reaches. Two failing readings in a row (the second 30 min after the first) are the finding: `CONTAINER_DOWNLOAD_SLOW_OBSERVED` (logged only) by default, and `CONTAINER_DOWNLOAD_SLOW` (score 0, until a reading passes) with `CONTAINER_DOWNLOAD_SPEED_ENFORCEMENT_ENABLED=true`. An endpoint the validator cannot reach either, an HTTP error or a container that did not start is no verdict.

Code: `src/services/task/checks/container_download_speed.py`.
