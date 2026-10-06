#!/bin/sh -e

# Set mode (default: validator)
MODE=${1:-validator}

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

if [ "$MODE" = "validator" ]; then
    pdm run alembic upgrade head
    pdm run src/validator.py
elif [ "$MODE" = "connector" ]; then
    pdm run src/connector.py
else
    echo "Error: Invalid mode '$MODE'. Use 'validator' or 'connector'."
    exit 1
fi
