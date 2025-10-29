# Validator

## System Requirements

For validation, a validator machine will need:

- **CPU**: 4 cores
- **RAM**: 8 GB

Ensure that your machine meets these requirements before proceeding with the setup.

---

First, register and regen your bittensor wallet and validator hotkey onto the machine. 

For installation of btcli, check [this guide](https://github.com/opentensor/bittensor/blob/master/README.md#install-bittensor-sdk)
```
btcli s register --netuid 51
```
```
btcli w regen_coldkeypub
```
```
btcli w regen_hotkey
```

## Installation

### Using Docker

#### Step 1: Clone Git repo

```
git clone https://github.com/Datura-ai/compute-subnet.git
```

#### Step 2: Install Required Tools

```
cd compute-subnet && chmod +x scripts/install_validator_on_ubuntu.sh && ./scripts/install_validator_on_ubuntu.sh
```

Verify docker installation

```
docker --version
```
If did not correctly install, follow [this link](https://docs.docker.com/engine/install/)

#### Step 3: Setup ENV
```
cp neurons/validators/.env.template neurons/validators/.env
```

Replace with your information for `BITTENSOR_WALLET_NAME`, `BITTENSOR_WALLET_HOTKEY_NAME`, `HOST_WALLET_DIR`.
If you want you can use different port for `INTERNAL_PORT`, `EXTERNAL_PORT`.

##### Optional: Intel TDX Attestation Configuration

To enable Intel TDX attestation verification for executors, add the following environment variables to your `.env` file:

```bash
# Enable TDX attestation verification
ENABLE_TDX_ATTESTATION=true

# Enable attestation digest whitelist checking (optional)
ENABLE_ATTESTATION_WHITELIST=false
```

**Configuration Options:**

- **ENABLE_TDX_ATTESTATION**: Set to `true` to enable TDX attestation verification (default: `false`)
- **ENABLE_ATTESTATION_WHITELIST**: Set to `true` to only allow executors with whitelisted attestation digests (default: `false`)

**How TDX Attestation Works:**

1. When an executor with TDX enabled connects, it provides a TDX quote along with its SSH host key
2. The validator sends the quote to the configured TDX verifier service
3. The verifier validates the quote and returns attestation details (OS image hash, compose hash)
4. If whitelist is enabled, the validator checks if the attestation digest is approved
5. Only executors that pass verification are allowed to connect

**Note**: TDX attestation is optional. If disabled, validators will accept all executors regardless of TDX capabilities.

#### Step 4: Docker Compose Up

```
cd neurons/validators && docker compose up -d
```

## Managing TDX Attestation Whitelist

If you have enabled `ENABLE_ATTESTATION_WHITELIST=true`, you need to manage which attestation digests are allowed to connect to your validator.

### What is an Attestation Digest?

An attestation digest is a unique identifier created from the TDX quote's OS image hash and compose hash. It represents a specific trusted execution environment configuration.

### Adding Attestation Digests to Whitelist

To add an attestation digest to the whitelist, you need to insert it directly into the `attestation_whitelist` database table:

```sql
INSERT INTO attestation_whitelist (tee_type, attestation_digest, is_active, created_at, updated_at)
VALUES ('dstack/tdx', '<attestation-digest-value>', true, NOW(), NOW());
```

Replace `<attestation-digest-value>` with the actual attestation digest you want to whitelist.

### Viewing Attestation Digests

When executors attempt to connect with TDX enabled, the validator logs will show the attestation digest:

```
Attestation verified: executor=<ip>:<port>, attestation_digest=<digest>, tee_type=dstack/tdx
```

You can use these logs to identify which attestation digests to whitelist.

### Disabling an Attestation Digest

To disable an attestation digest without deleting it:

```sql
UPDATE attestation_whitelist 
SET is_active = false, updated_at = NOW()
WHERE attestation_digest = '<attestation-digest-value>';
```
