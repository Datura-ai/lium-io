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

Ports already held by a rental or a filler are skipped in both passes.

- **Pass one (every executor, every cycle):** the lowest `BATCH_PORT_VERIFICATION_SIZE` (300) free declared ports, probed as before: one `--network=host` batch container (two attempts), then the first 50 ports published in chunks of 10, then the first 10 one by one. A declaration of 300 ports or fewer is probed in full. The DinD probe then takes one port, as before.
- **Below the floor (`PORT_PROBE_TOPUP_BELOW_FLOOR`, off by default):** when the batch answered but verified fewer than 3, the top-up first re-probes pass one's failed ports through published ports. It never reaches a port above pass one's 300, so a host that forwards only higher ports stays below 3. When the batch completed with no answers the top-up does not run. Either way, after the DinD probe and any top-up, if fewer than 3 are verified and pass one's batch container completed, pass two probes up to 300 more ports, spread evenly over the free declared ports pass one did not test, always including the highest. It uses the batch tier only, with one attempt, so it costs at most one extra container. Its answers are added to the verified ports; the DinD result does not change which ports count.
- **Limits of pass two:** on `40000-65535` it probes every 84th or 85th port from 40300 up (every 151st or 152nd from 20300 up on the default `20000-65535`). Starting from no verified ports, a block forwarded at the top verifies 3 ports when it is 170 ports or wider (304 on the default range). A block anywhere above the lowest 300 ports needs 254 ports (454). **Declare only the ports you forward:** a declaration of 300 ports or fewer is probed in full by pass one.
- **Deterministic:** for the same declaration and rental set, pass two probes the same ports every cycle. Renting or releasing a port shifts its picks.
- **Per-range tally:** the port-connectivity event carries `port_ranges`, a list of `{pass, range, declared, probed, answered}` entries, and `second_pass`: null with the flag off, otherwise `not_needed`, `skipped_batch_failed`, `no_ports_left`, `ran` or `batch_failed`. Declared ports are grouped into buckets of `PORT_RANGE_BUCKET_WIDTH` (5000) ports by external port, so a partial forward shows which part of a wide range answered.

Code: `src/services/executor_connectivity/port_selector.py` (`select`, `select_spread`, `tally_port_ranges`) and `orchestrator.py`.
