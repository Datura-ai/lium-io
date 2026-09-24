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

#### Step 4: Docker Compose Up

```
cd neurons/validators && docker compose up -d
```

## How the port check samples declared ports

Each cycle the validator checks from outside that an executor's declared ports are forwarded. An executor needs at least `MIN_PORT_COUNT` (3) ports that answer to be listed.

- **Budget:** at most `BATCH_PORT_VERIFICATION_SIZE` (300) ports are probed per executor per cycle, whatever the size of the declaration. Ports already held by a rental or a filler are skipped.
- **Small declarations:** 300 ports or fewer are all probed.
- **Wide declarations:** the lowest 150 declared ports are probed, and the other 150 probes are spread evenly over the rest of the declaration, always including the highest declared port. A range such as `40000-65535` is probed across its whole span, so ports forwarded only at the top of it are still found.
- **Deterministic:** an unchanged declaration is probed on the same ports every cycle, so the result does not flip between cycles by chance.
- **Per-range tally:** the port-connectivity event carries `port_ranges`, a list of `{range, declared, probed, answered}` entries. Declared ports are grouped into buckets of `PORT_RANGE_BUCKET_WIDTH` (5000) ports by external port, so a partial forward shows which part of a wide range answered.

Code: `src/services/executor_connectivity/port_selector.py` (`sample_ports`, `tally_port_ranges`).
