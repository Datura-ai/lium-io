# Watchtower

Monitors a Docker image for validator-signed updates and automatically pulls and restarts containers when a new version is available.

## Architecture

1. Queries the digest of the currently running Docker image
2. Fetches the latest authorized digest from the validator-signed endpoint
3. Verifies the signature using the validator's public key (hotkey)
4. If digests differ, pulls the new image, then gracefully restarts each affected container (10s timeout)

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
| `WATCHTOWER_ENV_FILE_PATH` | Path to the executor's `.env` file | `~/.env` | No |

### Endpoint Response Format

The validator endpoint must return:

```json
{
  "digest": "sha256:abc123...",
  "timestamp": 1234567890,
  "signature": "0xabcdef..."
}
```

The signature is produced by signing `{"digest": "sha256:...", "timestamp": ...}` (keys sorted) with the validator's private key.

## Build & Deploy

`WATCHTOWER_ENDPOINT_URL` and `WATCHTOWER_VALIDATOR_HOTKEY` are **baked into the image at build time** for non-prod environments. They are written into `src/config_override.py` by `docker_build.sh` and cannot be overridden at runtime.

```bash
DEPLOY_ENV=staging \
WATCHTOWER_ENDPOINT_URL=https://staging.lium.io/api/watchtower/digest \
WATCHTOWER_VALIDATOR_HOTKEY=<ss58-address> \
./docker_build.sh
```

For `DEPLOY_ENV=prod`, the script skips embedding and expects these values to be supplied at runtime.

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
