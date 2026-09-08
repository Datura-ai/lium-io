# lium.io

**[Lium Documentation](https://docs.lium.io)** — providers: <https://docs.lium.io/providers>, validators: <https://docs.lium.io/validators>

<img width="469" height="468" alt="image" src="https://github.com/user-attachments/assets/69550b83-91a9-492a-bd7a-09d35c6106d3" />

Welcome to **Lium.io powered by Bittensor Subnet 51**! This project enables a decentralized, peer-to-peer GPU rental marketplace, connecting miners who contribute GPU resources with users who need computational power. Our frontend interface is available at [lium.io](https://lium.io), where you can easily rent machines from the subnet.

## Table of Contents

- [Introduction](#introduction)
- [High-Level Architecture](#high-level-architecture)
- [Getting Started](#getting-started)
  - [For Renters](#for-renters)
  - [For Miners](#for-miners)
  - [For Validators](#for-validators)
- [Repository Layout](#repository-layout)
- [Running the Tests](#running-the-tests)
- [Releases](#releases)
- [Contact and Support](#contact-and-support)

## Introduction

The Compute Subnet on Bittensor is a decentralized network that allows miners to contribute their GPU resources to a global pool. Users can rent these resources for computational tasks, such as machine learning, data analysis, and more. The system ensures fair compensation for miners based on the quality and performance of their GPUs.

## High-Level Architecture

- **Miners**: Provide GPU resources to the network, evaluated and scored by validators.
- **Validators**: Securely connect to miner machines to verify hardware specs and performance. They maintain the network's integrity.
- **Renters**: Rent computational resources from the network to run their tasks.
- **Frontend (lium.io)**: The web interface facilitating easy interaction between miners and renters.
- **Bittensor Network**: The decentralized blockchain in which the compensation is managed and paid out by the validators to the miners through its native token, $TAO.

## Getting Started

### For Renters

If you are looking to rent computational resources, you can easily do so through the Compute Subnet. Renters can:

1. Visit [lium.io](https://lium.io) and sign up.
2. **Browse** available GPU resources.
3. **Select** machines based on GPU type, performance, and price.
4. **Deploy** and monitor your computational tasks using the platform's tools.

To start renting machines, visit [lium.io](https://lium.io) and access the resources you need.

**Command-Line Alternative**: For renters who prefer working from the terminal, you can also use the [Lium CLI](https://github.com/Datura-ai/lium) - a command-line interface that allows you to manage GPU pods, SSH into machines, transfer files, and execute commands directly from your terminal. Install it with `pip install lium.io`.

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

## Repository Layout

Three neurons, each with its own `pyproject.toml` / `pdm.lock`, Dockerfile and compose files; `watchtower/`, its own pdm project with a Dockerfile; one shared pdm package (`datura/`, a `pyproject.toml` only); and `packages/lium-core/`, the library published to PyPI. The root `pyproject.toml` (`compute-subnet`, Python 3.11) holds the shared `[tool.ruff]` config and the repo-wide dev tools (`ruff`, `pre-commit`), plus one declared runtime dependency (`aiohttp`):

- `neurons/validators/` — the validator: scores miners, verifies executors over SSH, creates and manages rental containers on them (`src/services/docker_service.py`), sets weights. `src/miner_jobs/` holds the scripts the validator uploads to an executor and runs there (`machine_scrape.py` hardware scrape, `backup_storage.py` / `restore_storage.py`, `workspace_mount.py`); `machine_scrape.py` is obfuscated per job by `src/services/file_encrypt_service.py` before upload, so its key order is load-bearing (see the comment at the top of that file).
- `neurons/miners/` — the miner: registers executors with the network and answers validator requests; its database schema is Alembic migrations under `migrations/`.
- `neurons/executor/` — the agent installed on a GPU machine: exposes the machine to its miner's validators, runs the containers. `dstacktee/` runs it inside an Intel TDX confidential VM with attestation (its own README).
- `datura/` — the protocol shared by the three: request/response models (`datura/requests`), consumers, errors.
- `packages/lium-core/` — `lium_core.shared_config`, the shared-config client the validator and the miner install from PyPI as `lium-core` (its own README; CI `lium-core-ci.yml`, release by hand through `lium-core-release.yml`).
- `lium_protocol/` — the validator↔backend wire protocol as one versioned package: every WebSocket message in both directions and every backend HTTP body the validator reads, as pydantic models, with a committed JSON-schema snapshot (`lium_protocol/lium_protocol/snapshots/lium_protocol.v1.json`) that CI compares with the models. Consumers pin it by tag (`lium_protocol/README.md`); `neurons/validators/tests/test_protocol_compat.py` replays the recorded messages through it and through the validator's own models.
- `watchtower/` — pulls validator-signed image updates and restarts containers (its own README).
- `e2e/` — the whole loop on one machine without the chain: a real executor on its own dockerd, a real miner, the validator's services as the tester (`e2e/README.md`; `./gate.sh` is what CI runs).
- `scripts/` — the `install_*_on_ubuntu.sh` installers referenced by the setup guides; `docs/` — operator notes; `contrib/` — contribution and style guides.

## Running the Tests

Python 3.11 and [pdm](https://pdm-project.org). Each service is its own pdm project; the validator suite is the large one (~1,950 tests, about two minutes, SQLite — no Postgres needed). From the repository root, each service in its own subshell so the block runs top to bottom:

```bash
(cd neurons/validators && pdm install && \
 BITTENSOR_WALLET_NAME=test_wallet BITTENSOR_WALLET_HOTKEY_NAME=test_hotkey \
 SQLALCHEMY_DATABASE_URI=sqlite:///test.db ASYNC_SQLALCHEMY_DATABASE_URI=sqlite+aiosqlite:///test.db \
 ENABLE_TDX_ATTESTATION=True TDX_VERIFIER_URL=http://localhost:8000/verify \
 pdm run pytest tests/ -v --tb=short --strict-markers)

(cd neurons/executor && pdm install && mkdir -p tmp && pdm run pytest tests/ -v --tb=short)
(cd neurons/miners && pdm install && pdm run pytest tests/ -v --tb=short)
(cd lium_protocol && pip install . pytest && python -m lium_protocol.schema --check && python -m pytest tests -v --tb=short)
```

These are the commands `.github/workflows/test.yml` (**Tests**) runs on every pull request, every push to `main` and every merge-queue run. The workflow has no path filter; its jobs decide for themselves:

- `route` reads the changed files (`.github/actions/changed-packages`), after dropping `*.md`, `.gitignore` and the root `docs/` tree — a README-only PR runs no neuron job.
- Each neuron's test job runs only when that neuron, `datura/` or `.github/` changed.
- `ruff-check` (`ruff check --select F,ASYNC210,ASYNC251 --ignore F541 --extend-exclude migrations neurons datura watchtower`) always runs and is part of `tests-ok`.
- `lint` (`ruff format --check`, report-only, not required) runs per changed neuron.
- `e2e-gate` (`cd e2e && ./gate.sh`) runs when a neuron, `datura/`, `.github/` or `e2e/` changed.
- `tests-ok` is the one status check to require; it reports on every PR.

Tests follow Arrange-Act-Assert, one behaviour per function; `ruff format` (pre-commit hook in `.pre-commit-config.yaml`) is the formatter.

## Releases

Images are built and pushed to Docker Hub by the `*_cd_prod` and `*_cd_dev` workflows from each neuron's `docker_build.sh` / `docker_publish.sh` (and the `*_runner_*` pair for the auto-updating runner image):

| Tag pushed | Workflow | Images |
|---|---|---|
| `executor-v*` | `executor_cd_prod.yml` | `daturaai/compute-subnet-executor`, `daturaai/compute-subnet-executor-runner` |
| `validator-v*` | `validator_cd_prod.yml` | `daturaai/compute-subnet-validator`, `daturaai/compute-subnet-validator-runner` |
| `miner-v*` | `miner_cd_prod.yml` | `daturaai/compute-subnet-miner`, `daturaai/compute-subnet-miner-runner` |

The `*_cd_dev.yml` and `*_cd_staging.yml` workflows are started by hand (`workflow_dispatch`); the two `*_cd_staging.yml` use `docker/build-push-action` to publish `ghcr.io/datura-ai/lium-validator:staging` and `ghcr.io/datura-ai/lium-miner:staging`. The deploy of the validator and central miner lives in the private `lium-io-deployment` repository.

## Contact and Support

If you need assistance or have any questions, feel free to reach out:

- **Discord Support**: [Dedicated Channel within the Bittensor Discord](https://discord.com/channels/799672011265015819/1291754566957928469)
