# Validator

**[Validator Documentation](https://docs.lium.io/bittensor-subnet/validator)**

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
git clone https://github.com/Datura-ai/compute-subnet.git
```

#### Step 2: Install Required Tools

```
cd compute-subnet && chmod +x scripts/install_validator_on_ubuntu.sh && ./scripts/install_validator_on_ubuntu.sh
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
