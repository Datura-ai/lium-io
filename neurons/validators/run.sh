#!/bin/sh

# Initialize Bittensor wallet from mnemonic
if [ -n "$BITTENSOR_HOTKEY_MNEMONIC" ]; then
    # Validate required environment variables
    if [ -z "$BITTENSOR_WALLET_DIR" ] || [ -z "$BITTENSOR_WALLET_NAME" ] || [ -z "$BITTENSOR_WALLET_HOTKEY_NAME" ]; then
        echo "Error: BITTENSOR_HOTKEY_MNEMONIC is set but missing required variables:"
        echo "  BITTENSOR_WALLET_DIR, BITTENSOR_WALLET_NAME, BITTENSOR_WALLET_HOTKEY_NAME"
        exit 1
    fi

    echo "Initializing Bittensor wallet..."

    pdm run btcli wallet create \
        --wallet.name $BITTENSOR_WALLET_NAME \
        --wallet.hotkey $BITTENSOR_WALLET_HOTKEY_NAME \
        --wallet.path $BITTENSOR_WALLET_DIR \
        --n-words 12 \
        --no-use-password \
        --overwrite
        --quiet

    pdm run btcli wallet regen_hotkey \
        --wallet-name "$BITTENSOR_WALLET_NAME" \
        --hotkey "$BITTENSOR_WALLET_HOTKEY_NAME" \
        --wallet-path "$BITTENSOR_WALLET_DIR" \
        --mnemonic "$BITTENSOR_HOTKEY_MNEMONIC" \
        --no-use-password \
        --overwrite \
        --quiet

    echo "Bittensor wallet initialized successfully"
else
    echo "BITTENSOR_HOTKEY_MNEMONIC not set, skipping wallet initialization from seed phrase"
fi


# db migrate
pdm run alembic upgrade head

# run fastapi app
pdm run src/validator.py
