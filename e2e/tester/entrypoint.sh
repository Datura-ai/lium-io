#!/bin/sh -e
# Tester entrypoint: the validator's wallet from the stack's public test mnemonic (the same regen path run.sh uses),
# the validator schema on its database, then whatever pytest command compose passes.
cd /app
if [ ! -f "$BITTENSOR_WALLET_DIRECTORY/$BITTENSOR_WALLET_NAME/hotkeys/$BITTENSOR_WALLET_HOTKEY_NAME" ]; then
  pdm run btcli wallet create --wallet.name "$BITTENSOR_WALLET_NAME" --wallet.hotkey "$BITTENSOR_WALLET_HOTKEY_NAME" \
    --wallet.path "$BITTENSOR_WALLET_DIRECTORY" --n-words 12 --no-use-password --overwrite --quiet >/dev/null
  pdm run btcli wallet regen_hotkey --wallet-name "$BITTENSOR_WALLET_NAME" --hotkey "$BITTENSOR_WALLET_HOTKEY_NAME" \
    --wallet-path "$BITTENSOR_WALLET_DIRECTORY" --mnemonic "$BITTENSOR_HOTKEY_MNEMONIC" --no-use-password --overwrite --quiet >/dev/null
fi
pdm run alembic upgrade head >/dev/null 2>&1 || echo "tester: alembic upgrade head failed (the suites do not need the validator DB; continuing)"
cd /e2e
exec /app/.venv/bin/python -m "$@"
