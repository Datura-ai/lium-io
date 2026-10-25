# Miner

**[Provider Documentation](https://docs.lium.io/providers)** — docs.lium.io calls a miner a *provider*, the
central miner the *self-hosted provider* ([setup guide](https://docs.lium.io/providers/self-hosted-provider)),
and an executor a *node*. Most providers do not run this service: the
[Lium.io Central Provider Server](https://docs.lium.io/providers/provider-configuration) runs it for them and is
ON by default in the [Provider Portal](https://provider.lium.io); nodes are set up with `lium mine`
([Node Quickstart](https://docs.lium.io/providers/nodes/quickstart)).

## Overview

This miner allows you to contribute your GPU resources to the Compute Subnet and earn compensation for providing computational power. You will run a central miner on a CPU server, which manages multiple executors running on GPU-equipped machines.

### Central Miner Server Requirements

To run the central miner, you only need a CPU server with the following specifications:

- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 50GB available disk space
- **OS**: Ubuntu (recommended)

### Executors

Executors are GPU-equipped machines that perform the computational tasks. The central miner manages these executors, which can be easily added or removed from the network.

To see the compatible GPUs to mine with and their relative rewards, see this dict [here](https://github.com/Datura-ai/lium-io/blob/main/neurons/validators/src/services/const.py).

## Installation

### Using Docker

#### Step 1: Clone the Git Repository

```
git clone https://github.com/Datura-ai/lium-io.git
```

#### Step 2: Install Required Tools

```
cd lium-io && chmod +x scripts/install_miner_on_ubuntu.sh && ./scripts/install_miner_on_ubuntu.sh
```

Verify if bittensor and docker installed: 
```
btcli --version
```

```
docker --version
```

If one of them isn't installed properly, install using following link:     
For bittensor, use [This Link](https://github.com/opentensor/bittensor/blob/master/README.md#install-bittensor-sdk) \
For docker, use [This Link](https://docs.docker.com/engine/install/)

#### Step 3: Setup ENV
```
cp neurons/miners/.env.template neurons/miners/.env
```

Fill in your information for:

`BITTENSOR_WALLET_NAME`: Your wallet name for Bittensor. You can check this with `btcli wallet list`

`BITTENSOR_WALLET_HOTKEY_NAME`: The hotkey name of your wallet's registered hotkey. If it is not registered, run `btcli subnet register --netuid 51`. 

`EXTERNAL_IP_ADDRESS`: The external IP address of your central miner server. Make sure it is open to external connections on the `EXTERNAL PORT`

`RENTAL_REQUEST_HOOK`: (Optional) The URL for the call hook. Miners will terminate chute programs within seconds using this hook.

`HOST_WALLET_DIR`: The directory path of your wallet on the machine.

`INTERNAL_PORT` and `EXTERNAL_PORT`: Optionally customize these ports. Make sure the `EXTERNAL PORT` is open for external connections to connect to the validators.

`COMPUTE_REST_API_URL`: The Lium backend the miner talks to (template default `https://lium.io/api`).


#### Step 4: Start the Miner

```
cd neurons/miners && docker compose up -d
```

## Managing Ethereum Address

### Associating Your Ethereum Address into Subnet

To associate an Ethereum address with your Bittensor hotkey (a one-time step), use the following command:

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py associate-eth --private-key <ethereum-private-key>
```

- `<ethereum-private-key>`: Your Ethereum private key that will be associated with your Bittensor hotkey.

### Get the associated Ethereum Address

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py get-associated-evm-address
```

## Managing Executors

### What is a Validator Hotkey?

The **validator hotkey** is a unique identifier tied to a validator that authenticates and verifies the performance of your executor machines. When you specify a validator hotkey during executor registration, it ensures that your executor is validated by this specific validator.

### Adding an Executor

Executors are machines running on GPUs that you can add to your central miner. The more executors (GPUs) you have, the greater your compensation will be. Here's how to add them:

1. Ensure the executor machine is set up and running Docker. For more information, follow the [executor README.md here](../executor/README.md)
2. Use the following command to add an executor to the central miner:

    ```bash
    docker exec -it <container-id or name> pdm run /root/app/src/cli.py add-executor --address <executor-ip-address> --port <executor-port> --price <gpu-price> [--validator <validator-hotkey>]
    ```

    **Required parameters:**
    - `<executor-ip-address>`: The IP address of the executor machine.
    - `<executor-port>`: The port number used for the executor (default: `8001`).
    - `<gpu-price>`: GPU price per hour in USD for the executor.

    **Optional parameters:**
    - `<validator-hotkey>`: The validator hotkey that you want to give access to this executor. If not provided, our validator_hotkey will be used as default.

### List Executors

To list added executors from the central miner, follow these steps:

1. run following command:

    ```bash
    docker exec -it <docker instance> pdm run /root/app/src/cli.py show-executors
    ```

### Switch validator

To switch validator you can use the following command

```bash
docker exec -it <docker instance> pdm run /root/app/src/cli.py switch-validator --address <executor-ip-address> --port <executor-port> --validator <validator-hotkey>
```

- `<executor-ip-address>`: The IP address of the executor machine.
- `<executor-port>`: The port number used for the executor.
- `<validator-hotkey>`: The validator hotkey you want to switch

### Set/Update Executor GPU Price

Run the following command to set or update the executor price:

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py update-executor-price --address <executor-ip-address> --port <executor-port> --price <gpu-price>
```

- `<executor-ip-address>`: The IP address of the executor machine.
- `<executor-port>`: The port number used for the executor.
- `<gpu-price>`: GPU price per hour in USD for the executor.

### Removing an Executor

To remove an executor from the central miner, follow these steps:
1. Run the following command to remove the executor:

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py remove-executor --address <executor-ip-address> --port <executor-port>
```

- `<executor-ip-address>`: The IP address of the executor machine.
- `<executor-port>`: The port number used for the executor.

2. Type "y" and click enter in the interactive shell.

## Validator hotkey rotation

`migrate-validator-hotkey` is the data step of a validator hotkey rotation, run once per miner after the config pair swap (`DEFAULT_VALIDATOR_HOTKEY=<new>`, `VALIDATOR_NEXT_HOTKEY=<old>`): it re-keys this miner's executor rows from the previous validator hotkey to the active one. The defaults read those two settings; `--from-hotkey` / `--to-hotkey` name them explicitly and `--dry-run` prints the row count without changing anything. Lium announces the rotation date; the current swap (to `5F1aH9124FPvDNMWKgPCCJ8WeBBV3B3vpijDCyxZVyZgrEV9`) is pending.

```shell
docker exec -it <container-id or name> pdm run /root/app/src/cli.py migrate-validator-hotkey --dry-run
docker exec -it <container-id or name> pdm run /root/app/src/cli.py migrate-validator-hotkey
```

## Monitoring earnings

To monitor your earnings, use [Taomarketcap.com](https://taomarketcap.com/subnets/51/miners)'s subnet 51 miner page to track your daily rewards, and relative performance with other miners.
