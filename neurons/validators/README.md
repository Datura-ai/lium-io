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
- **Wide declarations:** the lowest 150 available ports are probed, and the other 150 probes are spread evenly over the rest, always including the highest available port. A range such as `40000-65535` is probed across its whole span, so a wide enough block forwarded only at the top of it is still found.
- **Limits of the spread:** on `40000-65535` the spread probes about every 170th port (about every 305th on the default `20000-65535`). A block forwarded at the top verifies 3 ports only when it is about 342 ports or wider (about 611 on the default range, 513 on `40000-65535` when the DinD probe on one of them fails). A block of 3-150 open ports lying only at positions 151-299 of a wide declaration is not found, and a low block of 151-300 open ports verifies about 151 ports. **Declare only the ports you forward:** a declaration of 300 ports or fewer is probed in full.
- **Deterministic:** for the same declaration and rental set, the same ports are probed every cycle, so the result does not flip between cycles by chance. Renting or releasing a port shifts the spread picks.
- **Per-range tally:** the port-connectivity event carries `port_ranges`, a list of `{range, declared, probed, answered}` entries. Declared ports are grouped into buckets of `PORT_RANGE_BUCKET_WIDTH` (5000) ports by external port, so a partial forward shows which part of a wide range answered.

Code: `src/services/executor_connectivity/port_selector.py` (`sample_ports`, `tally_port_ranges`).
