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


#### Step 4: Start the Miner

```
cd neurons/miners && docker compose up -d
```

## Managing Ethereum Address

### Associating Your Ethereum Address into Subnet

Before managing executors, you need to associate your Ethereum address with your Bittensor hotkey. This is a one-time setup requirement. Use the following command:

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py associate-eth --private-key <ethereum-private-key>
```

- `<ethereum-private-key>`: Your Ethereum private key that will be associated with your Bittensor hotkey.

### Get the associated Ethereum Address

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py get-associated-evm-address
```

### Transfer TAO to Ethereum Address from your Miner Wallet

You need to fund/transfer TAO into your Ethereum Address to deposit collateral (TAO) for your executors.

To transfer TAO from your miner wallet (the wallet in your env config), you can use the following command 

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py transfer-tao-to-eth-address --amount <tao-amount> --private-key <ethereum-private-key>
```
- `<tao-amount>`: TAO amount you are going to transfer.
- `<ethereum-private-key>`: Your Ethereum private key that is used for association.

You will be prompted to enter your Bittensor wallet password to proceed the transfer.

### Convert Ethereum Address to SS58 format
You can transfer TAO to the Ethereum Address manually with btcli.

To do so, you need to have SS58 format address of your Ethereum Address.

You can transfer TAO to this address from any wallet.

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py get-eth-ss58-address --private-key <ethereum-private-key>
```

- `<ethereum-private-key>`: Your Ethereum private key that is used for association.

### Get Balance of Ethereum Address
```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py get-balance-of-eth-address --private-key <ethereum-private-key>
```

- `<ethereum-private-key>`: Your Ethereum private key that is used for association.


## Managing Executors

### What is a Validator Hotkey?

The **validator hotkey** is a unique identifier tied to a validator that authenticates and verifies the performance of your executor machines. When you specify a validator hotkey during executor registration, it ensures that your executor is validated by this specific validator.

### Adding an Executor

Executors are machines running on GPUs that you can add to your central miner. The more executors (GPUs) you have, the greater your compensation will be. Here's how to add them:

1. Ensure the executor machine is set up and running Docker. For more information, follow the [executor README.md here](../executor/README.md)
2. Use the following command to add an executor to the central miner:

    ```bash
    docker exec -it <container-id or name> pdm run /root/app/src/cli.py add-executor --address <executor-ip-address> --port <executor-port> --validator <validator-hotkey> [--price <price-per-hour>] [--gpu-type <gpu-type>] [--gpu-count <gpu-count>] [--deposit-amount <deposit-amount>] [--private-key <ethereum-private-key>]
    ```

    **Required parameters:**
    - `<executor-ip-address>`: The IP address of the executor machine.
    - `<executor-port>`: The port number used for the executor (default: `8001`).

    **Optional parameters:**
    - `<validator-hotkey>`: The validator hotkey that you want to give access to this executor. If not provided, our validator_hotkey will be used as default.
    - `<price-per-hour>`: Price per hour in USD for the executor.
    - `<gpu-type>`: Type of GPU available on the executor.
    - `<gpu-count>`: Number of GPUs available on the executor.
    - `<deposit-amount>`: The amount of TAO to deposit as collateral for this executor (must meet minimum required collateral).
    - `<ethereum-private-key>`: The Ethereum private key for the miner (used for collateral transactions).

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

### Set/Update Executor Price

Run the following command to set or update the executor price:

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py update-executor-price --address <executor public ip> --port <executor external port> --price <executor price>
```

- `<executor public ip>`: The public IP address of the executor machine.
- `<executor external port>`: The external port number used for the executor.
- `<executor price>`: Price per hour in USD for the executor.

### Removing an Executor

To remove an executor from the central miner, follow these steps:
1. Run the following command to remove the executor:

    ```bash
    docker exec -it <container-id or name> pdm run /root/app/src/cli.py remove-executor --address <executor public ip> --port <executor external port>
    ```

    - `<executor public ip>`: The public IP address of the executor machine.
    - `<executor external port>`: The external port number used for the executor.

2. Type "y" and click enter in the interactive shell.

### Getting Miner Collateral

To check the total collateral deposited by the miner, use the following command:

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py get-miner-collateral
```

This will display the total TAO collateral that miner has deposited.


### Depositing Collateral for an Executor

To deposit additional collateral for an existing executor, use the following command:

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py deposit-collateral --address <executor-ip-address> --port <executor-port> --deposit_amount <deposit-amount> --private-key <ethereum-private-key>
```

- `<executor-ip-address>`: The IP address of the executor machine.
- `<executor-port>`: The port number used for the executor.
- `<deposit-amount>`: The amount of TAO to deposit as additional collateral for this executor.
- `<ethereum-private-key>`: The Ethereum private key for the miner (used for collateral transactions).

This command allows you to increase the collateral for an executor already registered in the database.

### Getting Executor Collateral

To check the collateral amount for a specific executor, use the following command:

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py get-executor-collateral --address <executor-ip-address> --port <executor-port>
```

- `<executor-ip-address>`: The IP address of the executor machine.
- `<executor-port>`: The port number used for the executor.

This will display the TAO collateral associated with the specified executor.

### Getting Miner Reclaim Requests

To view all reclaim requests for the current miner, use the following command:

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py get-reclaim-requests
```

This will print a JSON list of all reclaim requests made by the miner, including their status and details.

### Finalizing a Reclaim Request

To finalize a reclaim request and reclaim your collateral, use the following command:

```bash
docker exec -it <container-id or name> pdm run /root/app/src/cli.py finalize-reclaim-request --reclaim-request-id <reclaim-request-id> --private-key <ethereum-private-key>
```

- `<reclaim-request-id>`: The ID of the reclaim request you wish to finalize.
- `<ethereum-private-key>`: The Ethereum private key for the miner (used for collateral contract transactions).

This command will finalize the reclaim request and return the collateral to your account.

### Monitoring earnings

To monitor your earnings, use [Taomarketcap.com](https://taomarketcap.com/subnets/51/miners)'s subnet 51 miner page to track your daily rewards, and relative performance with other miners.
