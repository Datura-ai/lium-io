# Watchtower

Monitors a Docker image for validator-signed updates and automatically pulls and restarts containers when a new version is available. It is the updater of both executor stacks: the CVM stack and the standard stack (`neurons/executor/docker-compose.yml`, service `watchtower`, image `daturaai/lium-watchtower:<version>`).

## Architecture

1. Finds the runner container: `executor-runner` (the CVM stack), else the one container with the compose label `com.docker.compose.service=executor-runner` (the standard stack, `executor-executor-runner-1`)
2. Reads the digest of the image that container runs
3. Fetches the latest authorized digest from the validator-signed endpoint
4. Verifies the signature using the validator's public key (hotkey)
5. If digests differ, pulls `image@digest` (never a tag), then recreates the container from it (10 s stop timeout)

### Pull by digest, and the mirror bypass

A pull by tag goes through the host's Docker daemon, and the daemon asks a registry mirror first when `/etc/docker/daemon.json` has `registry-mirrors`. A mirror can keep serving an old copy of a tag (DAH-3419: 99 of 493 nodes stayed on an old runner that way). A pull by digest is content-addressed: the daemon checks the hash of the manifest it receives, so a mirror can return the right image or an error, never an old image. When that pull fails or returns an image without the digest, the same digest is pulled as `registry-1.docker.io/<image>@<digest>`. The daemon applies `registry-mirrors` to `docker.io` names only, so this reference reaches Docker Hub directly. The container is then created from whichever reference succeeded.

### Recreating the container

An existing runner is rebuilt with its own configuration: name, command and entrypoint (unless inherited from the old image), environment, labels, working directory, user, the whole HostConfig (binds, restart policy, privileges) and its network endpoints. So a runner that compose created keeps its compose labels and its `.env` bind. The order keeps a runner on the host at every step: the old container is renamed aside, the new one is created under the old name, the old one is stopped, the new one is started, and the old one is removed last. When the create fails, the old container gets its name back. When the new container does not start, it is removed and the old one is started again. When no runner exists, one named `executor-runner` is created with the CVM stack's configuration (docker socket and `WATCHTOWER_ENV_FILE_PATH` bound to `/root/executor/.env`). When more than one container carries the runner label (for a moment during `docker compose up -d`), the cycle ends without a pull.

## Installation

**Prerequisites:** Python 3.11+, Docker daemon, PDM

```bash
cd watchtower
pdm install
cp .env.template .env
```

## Configuration

Edit `.env`:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `WATCHTOWER_ENABLED` | Enable/disable the service | `true` | No |
| `WATCHTOWER_IMAGE` | Docker image to monitor | `daturaai/compute-subnet-executor-runner` | No |
| `WATCHTOWER_INTERVAL` | Check interval in seconds | `300` | No |
| `WATCHTOWER_ENV_FILE_PATH` | Path to the executor's `.env` file, used only when no runner container exists yet | `~/.env` | No |

### Endpoint Response Format

The validator endpoint must return:

```json
{
  "digest": "sha256:abc123...",
  "timestamp": 1234567890,
  "signature": "0xabcdef..."
}
```

The signature is produced by signing the string `<digest>:<timestamp>` (for example `sha256:abc123...:1234567890`) with the validator's hotkey. `verify_watchtower_signature` checks it with `WATCHTOWER_VALIDATOR_HOTKEY` and refuses a timestamp more than 10 minutes from now.

## Build & Deploy

`WATCHTOWER_ENDPOINT_URL` and `WATCHTOWER_VALIDATOR_HOTKEY` are **baked into the image at build time** for non-prod environments. They are written into `src/config_override.py` by `docker_build.sh` and cannot be overridden at runtime.

```bash
DEPLOY_ENV=staging \
WATCHTOWER_ENDPOINT_URL=https://staging.lium.io/api/watchtower/digest \
WATCHTOWER_VALIDATOR_HOTKEY=<ss58-address> \
./docker_build.sh
```

For `DEPLOY_ENV=prod`, the script writes no override and the prod values in `src/config.py` (`https://lium.io/api/watchtower/digest`, the Lium validator hotkey) apply.

Every build is tagged twice: `daturaai/lium-watchtower:<env tag>` (`latest`, `staging`, `local`) and `daturaai/lium-watchtower:<version>` for prod or `<version>-<env tag>` for the others, where `<version>` is `project.version` in `pyproject.toml`. The executor compose files pull the version tag (`1.1.0` in `docker-compose.yml`, `1.1.0-staging` in `docker-compose.dev.yml`), and `neurons/executor/tests/test_stack_updater_by_digest.py` fails when the two drift. Bumping the version means: edit `pyproject.toml`, edit both compose files, build and push both tags.

> **Important:** This image is the entrypoint deployed to executor CVM machines. Any rebuild requires all executor operators to restart their CVM machines — treat rebuilds as significant, infrequent events.

## Usage

### Run directly

```bash
cd watchtower
pdm run python src/watchtower.py
```

### Run via Docker Compose

```yaml
services:
  watchtower:
    build: ./watchtower
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    env_file:
      - ./watchtower/.env
    restart: always
```

### Disable temporarily

```bash
# .env
WATCHTOWER_ENABLED=false
```

## Development

```bash
# Run all tests
pdm run pytest tests/test_watchtower.py -v

# Run a specific test class
pdm run pytest tests/test_watchtower.py::TestVerifyWatchtowerSignature -v

# Run with coverage
pdm run pytest tests/test_watchtower.py --cov=src --cov-report=html
```

## Project Structure

```
watchtower/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── models.py
│   └── watchtower.py
├── tests/
│   └── test_watchtower.py
├── .env.template
├── docker_build.sh
├── pyproject.toml
└── README.md
```
