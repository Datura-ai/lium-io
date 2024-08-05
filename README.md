# Datura Compute Subnet

## Setup project
### Requirements
* Ubuntu machine
* install [pdm](https://pdm-project.org/latest/)
* python version v3.11.*
* For ssh connection test on local machine, need to install openssh-client and openssh-server
```
sudo apt-get install openssh-client openssh-server
```

### Install and Run
#### Miner
* Go to miner project root (./neurons/miners)
* Install python dependencies 
```
pdm install
```
* create miner wallet coldkey and hotkey with btcli. here's the [docs](https://docs.bittensor.com/getting-started/wallets).
```
btcli wallet new_coldkey --wallet.name <miner_coldkey>
btcli wallet new_hotkey --wallet.name <miner_coldkey> --wallet.hotkey <miner_hotkey>
```
* add .env in the project root
```
BITTENSOR_WALLET_NAME=m<miner_coldkey>
BITTENSOR_WALLET_HOTKEY_NAME=<miner_hotkey>
SQLALCHEMY_DATABASE_URI=postgresql://postgres:password@localhost:7432/compute-subnet-db
```
* Run miner postgresql with docker-compose
```
docker compose up
```
* run project with following command
```
bash run.sh
```

#### Validator
* Go to validator project root (./neurons/validators)
* Install python dependencies 
```
pdm install
```
* create validator wallet coldkey and hotkey with btcli. here's the [docs](https://docs.bittensor.com/getting-started/wallets).
```
btcli wallet new_coldkey --wallet.name <validator_coldkey>
btcli wallet new_hotkey --wallet.name <validator_coldkey> --wallet.hotkey <validator_hotkey>
```
* add .env in the project root
```
BITTENSOR_WALLET_NAME=<validator_coldkey>
BITTENSOR_WALLET_HOTKEY_NAME=<validator_hotkey>
SQLALCHEMY_DATABASE_URI=postgresql://postgres:password@localhost:6432/compute-subnet-db
```
* Run validator postgresql with docker-compose in the project root
```
docker compose up
```
* run project with following command
```
bash run.sh
```