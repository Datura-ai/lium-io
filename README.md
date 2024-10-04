# Datura Compute Subnet

# Compute Subnet on Bittensor

Welcome to the **Compute Subnet on Bittensor**! This project enables a decentralized, peer-to-peer GPU rental marketplace, connecting miners who contribute GPU resources with users who need computational power. Our frontend interface is available at [computenet.ai](https://computenet.ai), where you can easily rent machines from the subnet.

## Table of Contents

- [Introduction](#introduction)
- [High-Level Architecture](#high-level-architecture)
- [Getting Started](#getting-started)
  - [For Renters](#for-renters)
  - [For Miners](#for-miners)
  - [For Validators](#for-validators)
- [Contact and Support](#contact-and-support)

## Introduction

The Compute Subnet on Bittensor is a decentralized network that allows miners to contribute their GPU resources to a global pool. Users can rent these resources for computational tasks, such as machine learning, data analysis, and more. The system ensures fair compensation for miners based on the quality and performance of their GPUs.


## High-Level Architecture

- **Miners**: Provide GPU resources to the network, evaluated and scored by validators.
- **Validators**: Securely connect to miner machines to verify hardware specs and performance. They maintain the network's integrity.
- **Renters**: Rent computational resources from the network to run their tasks.
- **Frontend (computenet.ai)**: The web interface facilitating easy interaction between miners and renters.
- **Bittensor Network**: The decentralized blockchain in which the compensation is managed and paid out by the validatators to the miners through its native token, $TAO.

## Getting Started

### For Renters

If you are looking to rent computational resources, you can easily do so through the Compute Subnet. Renters can:

1. Visit [computenet.ai](https://computenet.ai) and sign up.
2. **Browse** available GPU resources.
3. **Select** machines based on GPU type, performance, and price.
4. **Deploy** and monitor your computational tasks using the platform's tools.

To start renting machines, visit [computenet.ai](https://computenet.ai) and access the resources you need.

### For Miners

Miners can contribute their GPU-equipped machines to the network. The machines are scored and validated based on factors like GPU type, number of GPUs, bandwidth, and overall GPU performance. Higher performance results in better compensation for miners.

If you are a miner and want to contribute GPU resources to the subnet, please refer to the [Miner Setup Guide](neurons/miners/README.md) for instructions on how to:

- Set up your environment.
- Install the miner software.
- Register your miner and connect to the network.
- Get compensated for providing GPUs!

### For Validators

Validators play a crucial role in maintaining the integrity of the Compute Subnet by verifying the hardware specifications and performance of miners’ machines. Validators ensure that miners are fairly compensated based on their GPU contributions and prevent fraudulent activities.

For more details, visit the [Validator Setup Guide](neurons/validators/README.md).


## Contact and Support

If you need assistance or have any questions, feel free to reach out:

- **Discord Support**: [Dedicated Channel within the Bittensor Discord](https://discord.com/channels/799672011265015819/1291754566957928469)
