# Securely extract whitelisted environment variables from .user-config JSON
# This prevents code execution and only allows variables defined in .env.example
WHITELIST="\
INTERNAL_PORT \
EXTERNAL_PORT \
SSH_PORT \
SSH_PUBLIC_PORT \
RENTING_PORT_RANGE \
RENTING_PORT_MAPPINGS \
MINER_HOTKEY_SS58_ADDRESS \
VALIDATOR_HOTKEY_SS58_ADDRESS \
ENABLE_TDX_ATTESTATION \
"

if [ -f .host-shared/.user-config ]; then
    # Rebuild .env file from JSON with whitelist filtering
    >.env # Clear/create .env file

    for var_name in $WHITELIST; do
        # Use jq to safely extract value from JSON object
        value=$(cat .host-shared/.user-config | jq -r ".env_vars.${var_name} // empty")

        if [ -n "$value" ]; then
            # Write to .env file and emit event to RTMR
            echo "${var_name}=${value}" >>.env
            dstack-util extend --event "env-${var_name}" --payload "$(echo -n $value | dstack-util hex)"
        fi
    done

    echo "INFO: Environment variables loaded from .user-config"
else
    echo "WARNING: .user-config not found, skipping environment variable loading"
fi
dstack-util extend --event "env_end" --payload ""
