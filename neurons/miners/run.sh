#!/bin/sh -e

# Set mode (default: miner)
MODE=${1:-miner}

# Initialize Bittensor wallet from mnemonic
if [ -n "$BITTENSOR_HOTKEY_MNEMONIC" ]; then
    # Validate required environment variables
    if [ -z "$BITTENSOR_WALLET_DIRECTORY" ] || [ -z "$BITTENSOR_WALLET_NAME" ] || [ -z "$BITTENSOR_WALLET_HOTKEY_NAME" ]; then
        echo "Error: BITTENSOR_HOTKEY_MNEMONIC is set but missing required variables:"
        echo "  BITTENSOR_WALLET_DIRECTORY, BITTENSOR_WALLET_NAME, BITTENSOR_WALLET_HOTKEY_NAME"
        exit 1
    fi

    echo "Initializing Bittensor wallet..."

    pdm run btcli wallet create \
        --wallet.name "$BITTENSOR_WALLET_NAME" \
        --wallet.hotkey "$BITTENSOR_WALLET_HOTKEY_NAME" \
        --wallet.path "$BITTENSOR_WALLET_DIRECTORY" \
        --n-words 12 \
        --no-use-password \
        --overwrite \
        --quiet > /dev/null

    pdm run btcli wallet regen_hotkey \
        --wallet-name "$BITTENSOR_WALLET_NAME" \
        --hotkey "$BITTENSOR_WALLET_HOTKEY_NAME" \
        --wallet-path "$BITTENSOR_WALLET_DIRECTORY" \
        --mnemonic "$BITTENSOR_HOTKEY_MNEMONIC" \
        --no-use-password \
        --overwrite \
        --quiet > /dev/null

    echo "Bittensor wallet initialized successfully"
else
    echo "BITTENSOR_HOTKEY_MNEMONIC not set, skipping wallet initialization from seed phrase"
fi

if [ "$MODE" = "miner" ]; then
    # db migrate, retried while the database is still starting (a first start that races Postgres would
    # otherwise serve without its tables); after 5 minutes go on, as before, with the tables there are
    tries=0
    until pdm run alembic upgrade head; do
        tries=$((tries + 1))
        [ "$tries" -ge 30 ] && { echo "alembic upgrade head failed 30 times, starting anyway"; break; }
        sleep 10
    done
    # migrate validator hotkey
    pdm run src/cli.py migrate-validator-hotkey
    # run fastapi app
    pdm run src/miner.py
elif [ "$MODE" = "connector" ]; then
    pdm run src/connector.py
else
    echo "Error: Invalid mode '$MODE'. Use 'miner' or 'connector'."
    exit 1
fi
