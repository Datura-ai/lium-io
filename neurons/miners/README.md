# Miners

## Installation

### Using Docker

#### Step 1: Clone Git repo

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

Replace with your information for `BITTENSOR_WALLET_NAME`, `BITTENSOR_WALLET_HOTKEY_NAME`, `BITTENSOR_NETUID`, `BITTENSOR_NETWORK`, `EXTERNAL_IP_ADDRESS`. 

#### Step 4: Docker Compose Up

```
cd neurons/miners && docker compose up -d
```

#### Step 5: Register Executors


```shell
docker exec <container-id or name> python /root/app/src/cli.py add-executor --address <executor-ip-address> --port <executor-port> --validator <validator-hotkey>
```
