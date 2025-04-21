# Miner

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

To see the compatible GPUs to mine with and their relative rewards, see this dict [here](https://github.com/Datura-ai/compute-subnet/blob/main/neurons/validators/src/services/const.py#L3).

## Installation

### Using Docker

#### Step 1: Clone the Git Repository

```
git clone https://github.com/Datura-ai/compute-subnet.git
```

#### Step 2: Install Required Tools

```
cd compute-subnet && chmod +x scripts/install_miner_on_ubuntu.sh && ./scripts/install_miner_on_ubuntu.sh
```

Verify if bittensor and docker installed: 
```
btcli --version
```

```
docker --version
```

If one of them isn't installed properly, install using following link:     
For bittensor, use [This Link](https://github.com/opentensor/bittensor/blob/master/README.md#install-bittensor-sdk)
For docker, use [This Link](https://docs.docker.com/engine/install/)

#### Step 3: Setup ENV
```
cp neurons/miners/.env.template neurons/miners/.env
```

Fill in your information for:

`BITTENSOR_WALLET_NAME`: Your wallet name for Bittensor. You can check this with `btcli wallet list`

`BITTENSOR_WALLET_HOTKEY_NAME`: The hotkey name of your wallet's registered hotkey. If it is not registered, run `btcli subnet register --netuid 51`. 

`EXTERNAL_IP_ADDRESS`: The external IP address of your central miner server. Make sure it is open to external connections on the `EXTERNAL PORT`

`HOST_WALLET_DIR`: The directory path of your wallet on the machine.

`INTERNAL_PORT` and `EXTERNAL_PORT`: Optionally customize these ports. Make sure the `EXTERNAL PORT` is open for external connections to connect to the validators.


#### Step 4: Start the Miner

```
cd neurons/miners && docker compose up -d
```

## Managing Executors

### What is a Validator Hotkey?

The **validator hotkey** is a unique identifier tied to a validator that authenticates and verifies the performance of your executor machines. When you specify a validator hotkey during executor registration, it ensures that your executor is validated by this specific validator.

To switch to a different validator first follow the instructions for removing an executor. After removing the executor, you need to re-register executors by running the add-executor command again (Step 2 of Adding an Executor).

### Adding an Executor

Executors are machines running on GPUs that you can add to your central miner. The more executors (GPUs) you have, the greater your compensation will be. Here's how to add them:

1. Ensure the executor machine is set up and running Docker. For more information, follow the [executor README.md here](../executor/README.md)
2. Use the following command to add an executor to the central miner:

    ```bash
    docker exec <container-id or name> pdm run /root/app/src/cli.py add-executor --address <executor-ip-address> --port <executor-port> --validator <validator-hotkey>
    ```

    - `<executor-ip-address>`: The IP address of the executor machine.
    - `<executor-port>`: The port number used for the executor (default: `8001`).
    - `<validator-hotkey>`: The validator hotkey that you want to give access to this executor. Which validator hotkey should you pick? Follow [this guide](assigning_validator_hotkeys.md)

### List Executors

To list added executors from the central miner, follow these steps:

1. run following command:

    ```bash
    docker exec <docker instance> pdm run /root/app/src/cli.py show-executors
    ```

### Removing an Executor

To remove an executor from the central miner, follow these steps:
 1. Run the following command to remove the executor:

    ```bash
    docker exec -i <docker instance> pdm run /root/app/src/cli.py remove-executor --address <executor public ip> --port <executor external port>
    ```
2. Type "y" and click enter in the interactive shell

### Switch Validator

If you need to change validator hotkey of an executor, follow these steps:
1. run following command to switch validator hotkey:

    ```bash
    docker exec -i <docker instance> pdm run /root/app/src/cli.py switch-validator --address <executor-ip-address> --port <executor-port> --validator <validator-hotkey>
    ```
2. Type "y" and click enter in the interactive shell

Note: We are not recommending to switch validator hotkey and this may lead to unexpected results.

### Monitoring earnings

To monitor your earnings, use [Taomarketcap.com](https://taomarketcap.com/subnets/51/miners)'s subnet 51 miner page to track your daily rewards, and relative performance with other miners.
