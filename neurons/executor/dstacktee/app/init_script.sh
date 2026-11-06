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
ENABLE_GPU_ATTESTATION \
GPU_ATTESTATION_ARCH \
EXECUTOR_LOCAL_VERIFY_ENABLED \
LOCAL_VERIFY_MAX_DEADLINE_SECONDS \
LOCAL_VERIFY_INTENT_WINDOW_SECONDS \
EXECUTOR_SSH_SOURCE_PIN \
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
# QEMU user networking shows sshd 10.0.2.2, never the validator's address, so a CVM pins no source unless its config asks
grep -q '^EXECUTOR_SSH_SOURCE_PIN=' .env 2>/dev/null || echo "EXECUTOR_SSH_SOURCE_PIN=false" >>.env
dstack-util extend --event "env_end" --payload ""
